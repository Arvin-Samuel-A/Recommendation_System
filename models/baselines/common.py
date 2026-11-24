"""Building blocks shared by the external baselines (SASRec, BERT4Rec, DuoRec, BSARec).

Reimplemented to match the behaviour of the BSARec benchmark code (Shin et al., AAAI 2024,
https://github.com/yehjin-shin/BSARec, `src/model/_modules.py` and
`src/model/_abstract_model.py`, itself based on FMLP-Rec). That repository has no license
file, so nothing here is copied from it; the arithmetic is reproduced so the published
hyperparameters mean the same thing in our pipeline:

- post-LN blocks: `LayerNorm(dropout(sublayer(x)) + x)`, TF-style LayerNorm with eps 1e-12;
- the feed-forward is 4x wide with the exact (erf) GELU;
- attention masks are ADDITIVE with -10000 on masked keys (not -inf), so a query whose
  keys are all masked (a left-padded position under the causal mask) attends uniformly
  instead of producing NaN -- the padded positions' outputs feed BSARec's Fourier filter,
  so this matters for that model;
- the input is `dropout(LayerNorm(item_emb + pos_emb))`;
- every Linear and Embedding weight is drawn from N(0, 0.02) (the padding row included,
  as `module.weight.data.normal_` does there), biases are zero.

Item ids follow steering/conventions.md: 0 = padding, 1..num_items = items,
num_items + 1 = MASK (BERT4Rec only).
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Additive mask value of the reference implementation (`(1 - mask) * -10000.0`).
MASK_VALUE = -10000.0


class TFLayerNorm(nn.Module):
    """LayerNorm with the epsilon inside the square root (TF style), eps 1e-12."""

    def __init__(self, hidden: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.weight * (x - u) / torch.sqrt(s + self.eps) + self.bias


def gelu(x: torch.Tensor) -> torch.Tensor:
    """The exact GELU, x * Phi(x)."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


_ACTIVATIONS = {"gelu": gelu, "relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}


class PointwiseFeedForward(nn.Module):
    def __init__(self, hidden: int, dropout: float, act: str = "gelu"):
        super().__init__()
        self.dense_1 = nn.Linear(hidden, 4 * hidden)
        self.dense_2 = nn.Linear(4 * hidden, hidden)
        self.act = _ACTIVATIONS[act]
        self.dropout = nn.Dropout(dropout)
        self.norm = TFLayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.dense_2(self.act(self.dense_1(x))))
        return self.norm(h + x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with an additive `(B, 1, S, S)` mask, post-LN residual."""

    def __init__(self, hidden: int, heads: int, attn_dropout: float, dropout: float):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden size {hidden} is not a multiple of {heads} heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.dense = nn.Linear(hidden, hidden)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.norm = TFLayerNorm(hidden)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.heads, self.head_dim).transpose(1, 2)   # (B, h, S, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self._split(self.query(x)), self._split(self.key(x)), self._split(self.value(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim) + mask
        probs = self.attn_dropout(torch.softmax(scores, dim=-1))
        ctx = (probs @ v).transpose(1, 2).reshape(x.shape)
        return self.norm(self.out_dropout(self.dense(ctx)) + x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, attn_dropout: float, dropout: float, act: str):
        super().__init__()
        self.attention = SelfAttention(hidden, heads, attn_dropout, dropout)
        self.feed_forward = PointwiseFeedForward(hidden, dropout, act)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.attention(x, mask))


def additive_mask(seq: torch.Tensor, causal: bool, dtype=torch.float32) -> torch.Tensor:
    """`(B, 1, S, S)` additive mask: key `j` is visible to query `i` iff `seq[:, j] > 0`
    (and `j <= i` when causal); hidden keys get -10000."""
    keep = (seq > 0)[:, None, None, :]                                  # (B, 1, 1, S)
    if causal:
        S = seq.shape[1]
        keep = keep & torch.ones(S, S, dtype=torch.bool, device=seq.device).tril()[None, None]
    return (1.0 - keep.to(dtype)) * MASK_VALUE


class BaselineModel(nn.Module):
    """Common interface of every baseline.

    - `encode(seq)` -> `(B, S, H)` hidden states;
    - `user_embedding(seq)` -> `(B, H)`, the representation that scores the next item;
    - `training_loss(batch, gen)` -> `(loss, parts)`, the model's own published loss on a
      batch of left-padded prefixes (`batch["input"]`, `(B, S)`) whose next item is
      `batch["target"]` (`(B,)`);
    - `full_scores(seq)` -> `(B, num_items + 2)` fp32 dot-product scores over the whole
      embedding vocabulary (columns = item ids; padding and MASK are removed by
      `evaluation.metrics.catalogue_ranking_scores`).

    `needs_negatives` / `needs_same_target` tell the training feed which extra inputs to
    build (one uniform negative per example; DuoRec's same-target sequence).
    """

    needs_negatives = False
    needs_same_target = False

    def __init__(self, num_items: int, params: Dict, vocab_rows: Optional[int] = None,
                 position_embeddings: bool = True):
        super().__init__()
        self.num_items = int(num_items)
        self.params = dict(params)
        hidden = int(params["hidden_size"])
        self.hidden = hidden
        self.max_seq_length = int(params["max_seq_length"])
        # The reference sizes the table at `max_item + 1` (padding + items); BERT4Rec adds
        # the MASK row. Logits over the whole table, padding included, are what its
        # cross-entropy losses use (BSARec repo issue #8).
        rows = self.num_items + 1 if vocab_rows is None else int(vocab_rows)
        self.item_embeddings = nn.Embedding(rows, hidden, padding_idx=0)
        if position_embeddings:
            self.position_embeddings = nn.Embedding(self.max_seq_length, hidden)
            self.emb_norm = TFLayerNorm(hidden)
        self.emb_dropout = nn.Dropout(float(params["hidden_dropout"]))

    # ---- shared pieces ------------------------------------------------------------------
    def init_weights(self, module: nn.Module) -> None:
        std = float(self.params.get("initializer_range", 0.02))
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, TFLayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.GRU):
            # As the reference: Xavier on the first layer's weights only; deeper layers keep
            # PyTorch's default init.
            nn.init.xavier_uniform_(module.weight_hh_l0)
            nn.init.xavier_uniform_(module.weight_ih_l0)

    def embed(self, seq: torch.Tensor) -> torch.Tensor:
        S = seq.shape[1]
        positions = torch.arange(S, device=seq.device)
        x = self.item_embeddings(seq) + self.position_embeddings(positions)[None]
        return self.emb_dropout(self.emb_norm(x))

    def encode(self, seq: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def user_embedding(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encode(seq)[:, -1, :]

    def training_loss(self, batch: Dict[str, torch.Tensor], gen: torch.Generator):
        raise NotImplementedError  # pragma: no cover - abstract

    def _last_position_ce(self, out_last: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of the last position over every embedding row (reference)."""
        logits = out_last @ self.item_embeddings.weight.T
        return F.cross_entropy(logits.float(), target)

    @torch.no_grad()
    def full_scores(self, seq: torch.Tensor) -> torch.Tensor:
        user = self.user_embedding(seq).float()
        scores = user @ self.item_embeddings.weight.float().T
        V = self.num_items + 2
        if scores.shape[1] < V:   # no MASK row: pad its column, ranked out anyway
            pad = scores.new_full((scores.shape[0], V - scores.shape[1]), float("-inf"))
            scores = torch.cat([scores, pad], dim=1)
        return scores


class TransformerStack(nn.Module):
    def __init__(self, params: Dict, layer_factory=None):
        super().__init__()
        hidden = int(params["hidden_size"])
        make = layer_factory or (lambda: TransformerLayer(
            hidden, int(params["num_heads"]), float(params["attn_dropout"]),
            float(params["hidden_dropout"]), params.get("hidden_act", "gelu"),
        ))
        self.layers = nn.ModuleList([make() for _ in range(int(params["num_layers"]))])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


def bce_pos_neg(user: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
    """SASRec's binary cross-entropy: positive logit -> 1 plus negative logit -> 0."""
    pos = (user * pos_emb).sum(-1).float()
    neg = (user * neg_emb).sum(-1).float()
    return (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
