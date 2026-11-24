"""The five external baselines, reimplemented after the BSARec benchmark code.

Reference: Shin et al., "An Attentive Inductive Bias for Sequential Recommendation beyond
the Self-Attention", AAAI 2024 (arXiv 2312.10325), code at
https://github.com/yehjin-shin/BSARec (commit c80bdc0, `src/model/*.py`). The repository
has no license, so this is a reimplementation of its behaviour, not a copy. Original
papers: SASRec (Kang & McAuley, ICDM 2018), BERT4Rec (Sun et al., CIKM 2019), GRU4Rec
(Hidasi et al., ICLR 2016; the improved variant of Tan et al., DLRS 2016, as in RecBole),
DuoRec (Qiu et al., WSDM 2022).

Every model is trained on left-padded prefixes and scores the next item from the LAST
position, as in the reference (its training set is every prefix of each user's sequence,
each predicting its final item). Each docstring states the loss exactly as the reference
computes it, including where that differs from the original paper.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines.common import (
    BaselineModel, TFLayerNorm, TransformerStack, SelfAttention, PointwiseFeedForward,
    additive_mask, bce_pos_neg,
)


class SASRec(BaselineModel):
    """Causal Transformer. Loss (reference `sasrec.py`): binary cross-entropy of the last
    position against the true next item (label 1) and ONE uniformly sampled negative not
    in the prefix (label 0), averaged over the batch."""

    needs_negatives = True

    def __init__(self, num_items: int, params: Dict):
        super().__init__(num_items, params)
        self.encoder = TransformerStack(params)
        self.apply(self.init_weights)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed(seq), additive_mask(seq, causal=True))

    def training_loss(self, batch, gen):
        user = self.user_embedding(batch["input"])
        loss = bce_pos_neg(user, self.item_embeddings(batch["target"]),
                           self.item_embeddings(batch["negative"]))
        return loss, {"bce": loss.detach()}


class GRU4Rec(BaselineModel):
    """Embedding -> dropout -> multi-layer GRU (no bias) -> Linear back to the embedding
    size. Loss (reference `gru4rec.py`, from RecBole): BPR, `-log(1e-10 + sigmoid(pos -
    neg))`, on the last position with one uniform negative. No position embeddings."""

    needs_negatives = True

    def __init__(self, num_items: int, params: Dict):
        super().__init__(num_items, params, position_embeddings=False)
        self.gru = nn.GRU(
            input_size=self.hidden, hidden_size=int(params["gru_hidden_size"]),
            num_layers=int(params["num_layers"]), bias=False, batch_first=True,
        )
        self.dense = nn.Linear(int(params["gru_hidden_size"]), self.hidden)
        self.apply(self.init_weights)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.emb_dropout(self.item_embeddings(seq)))
        return self.dense(out)

    def training_loss(self, batch, gen):
        user = self.user_embedding(batch["input"])
        pos = (user * self.item_embeddings(batch["target"])).sum(-1).float()
        neg = (user * self.item_embeddings(batch["negative"])).sum(-1).float()
        loss = -torch.log(1e-10 + torch.sigmoid(pos - neg)).mean()
        return loss, {"bpr": loss.detach()}


class BERT4Rec(BaselineModel):
    """Bidirectional Transformer with a MASK token (id num_items + 1).

    Training, as the reference `bert4rec.py` actually computes it: `int(S * mask_ratio)`
    positions of each row, drawn uniformly without replacement from ALL S positions
    (padding included), are replaced by MASK; the loss is the cross-entropy of the LAST
    position's output against the next item, over every embedding row. The reference also
    builds the Cloze (masked-position) loss but overwrites it before returning, so it never
    reaches the gradient; it is not computed here. Inference, as the reference: drop the
    oldest position, append MASK, score from the MASK position.
    """

    def __init__(self, num_items: int, params: Dict):
        super().__init__(num_items, params, vocab_rows=num_items + 2)
        self.mask_token = num_items + 1
        self.mask_ratio = float(params["mask_ratio"])
        self.encoder = TransformerStack(params)
        self.apply(self.init_weights)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed(seq), additive_mask(seq, causal=False))

    def mask_positions(self, seq: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        B, S = seq.shape
        n = int(S * self.mask_ratio)
        noise = torch.rand(B, S, generator=gen, device=seq.device)
        idx = noise.argsort(dim=1)[:, :n]
        return seq.clone().scatter_(1, idx, self.mask_token)

    def user_embedding(self, seq: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.encode(seq)[:, -1, :]
        mask = torch.full_like(seq[:, :1], self.mask_token)
        return self.encode(torch.cat([seq[:, 1:], mask], dim=1))[:, -1, :]

    def training_loss(self, batch, gen):
        out = self.encode(self.mask_positions(batch["input"], gen))[:, -1, :]
        loss = self._last_position_ce(out, batch["target"])
        return loss, {"ce": loss.detach()}


class DuoRec(BaselineModel):
    """Causal Transformer (SASRec encoder) with DuoRec's contrastive regulariser.

    Loss (reference `duorec.py`, `ssl = "us_x"`): cross-entropy of the last position over
    every embedding row, plus `lmd_sem` x InfoNCE between a second, independently dropped-
    out forward of the same input ("unsupervised" view) and the forward of a different
    training sequence with the same next item ("supervised" view). InfoNCE uses the last
    position, dot-product (`sim = "dot"`) or cosine similarity over temperature `tau`, the
    other 2(B - 1) views in the batch as negatives.
    """

    needs_same_target = True

    def __init__(self, num_items: int, params: Dict):
        super().__init__(num_items, params)
        if params.get("ssl", "us_x") != "us_x":
            raise ValueError("only DuoRec's published ssl='us_x' variant is implemented")
        self.tau = float(params["tau"])
        self.lmd_sem = float(params["lmd_sem"])
        self.sim = params.get("sim", "dot")
        self.encoder = TransformerStack(params)
        self.apply(self.init_weights)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed(seq), additive_mask(seq, causal=True))

    def info_nce(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """NT-Xent over the 2B views: each view's positive is its pair, every other view
        except itself is a negative (the reference's `mask_correlated_samples`)."""
        B = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0).float()
        if self.sim == "cos":
            sim = F.cosine_similarity(z[:, None], z[None], dim=-1) / self.tau
        else:
            sim = z @ z.T / self.tau
        sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(sim, labels)

    def training_loss(self, batch, gen):
        seq = batch["input"]
        out = self.encode(seq)[:, -1, :]
        ce = self._last_position_ce(out, batch["target"])
        aug = self.encode(seq)[:, -1, :]
        sem = self.encode(batch["same_target"])[:, -1, :]
        nce = self.info_nce(aug, sem)
        loss = ce + self.lmd_sem * nce
        return loss, {"ce": ce.detach(), "nce": nce.detach()}


class FrequencyLayer(nn.Module):
    """BSARec's inductive-bias branch: keep the lowest `c // 2 + 1` rFFT frequencies along
    the sequence, rescale the rest by a learned per-dimension `beta = sqrt_beta^2`, then
    dropout and a post-LN residual. Note the FFT runs over all S positions, padding
    included, so this branch is not causal."""

    def __init__(self, hidden: int, c: int, dropout: float):
        super().__init__()
        self.keep = c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden))
        self.dropout = nn.Dropout(dropout)
        self.norm = TFLayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        spec = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        keep = (torch.arange(spec.shape[1], device=x.device) < self.keep)[None, :, None]
        low = torch.fft.irfft(spec * keep, n=S, dim=1, norm="ortho").to(x.dtype)
        mixed = low + self.sqrt_beta ** 2 * (x - low)
        return self.norm(self.dropout(mixed) + x)


class BSARecLayer(nn.Module):
    """`alpha * FrequencyLayer(x) + (1 - alpha) * SelfAttention(x)`, then the FFN."""

    def __init__(self, params: Dict):
        super().__init__()
        hidden = int(params["hidden_size"])
        self.alpha = float(params["alpha"])
        self.filter = FrequencyLayer(hidden, int(params["c"]), float(params["hidden_dropout"]))
        self.attention = SelfAttention(hidden, int(params["num_heads"]),
                                       float(params["attn_dropout"]), float(params["hidden_dropout"]))
        self.feed_forward = PointwiseFeedForward(hidden, float(params["hidden_dropout"]),
                                                 params.get("hidden_act", "gelu"))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.alpha * self.filter(x) + (1.0 - self.alpha) * self.attention(x, mask)
        return self.feed_forward(h)


class BSARec(BaselineModel):
    """BSARec. Loss (reference `bsarec.py`): cross-entropy of the last position over every
    embedding row. The attention branch uses the causal mask."""

    def __init__(self, num_items: int, params: Dict):
        super().__init__(num_items, params)
        self.encoder = TransformerStack(params, layer_factory=lambda: BSARecLayer(params))
        self.apply(self.init_weights)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed(seq), additive_mask(seq, causal=True))

    def training_loss(self, batch, gen):
        loss = self._last_position_ce(self.user_embedding(batch["input"]), batch["target"])
        return loss, {"ce": loss.detach()}


BASELINE_CLASSES = {
    "sasrec": SASRec,
    "bert4rec": BERT4Rec,
    "gru4rec": GRU4Rec,
    "duorec": DuoRec,
    "bsarec": BSARec,
}
