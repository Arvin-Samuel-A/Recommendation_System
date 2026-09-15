import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.transformer import l2_normalize


def _maybe_normalize(t: torch.Tensor, normalize: bool) -> torch.Tensor:
    """REV-044: FR-8 requires L2-normalizing both sides of every dot product.

    Normalization lives at the scoring sites (here and in `score_candidates`) rather
    than inside a `TransformerModel.forward` branch nothing calls, so the [-1, 1] bound
    actually holds on the production path.
    """
    return l2_normalize(t) if normalize else t


def normalize_item_table(
    item_emb_weight: torch.Tensor, normalize: bool,
) -> Optional[torch.Tensor]:
    """The L2-normalized item table for one step, or None when normalization is off.

    REV-077: `MaskedItemLoss` and `MultiStepLoss` each normalized the whole
    `(vocab_size, hidden_dim)` table on every call, and `l2_normalize` casts to fp32
    first -- a 26 MB fp32 intermediate at `vocab_size=20722, hidden_dim=320`, allocated
    and discarded twice per step. Both objectives take a full-vocabulary softmax, so
    every row IS needed; what was avoidable is doing it twice. Callers compute this once
    per step and hand the result to both.

    Deliberately NOT cached across steps: the result is part of the step's autograd
    graph. A `_version`-keyed cache would serve a freed graph wherever the embedding is
    not updated between calls -- `evaluation/gradient_diagnostics.py` calls these losses
    repeatedly with no optimizer step at all, and LoRA personalization freezes the
    backbone entirely.
    """
    return l2_normalize(item_emb_weight) if normalize else None


def _shared_normalized_table(
    item_emb_weight: torch.Tensor,
    normalized_item_emb: Optional[torch.Tensor],
    normalize: bool,
) -> torch.Tensor:
    """Use the caller's per-step normalized table when given, else normalize here."""
    if not normalize:
        return item_emb_weight
    if normalized_item_emb is not None:
        return normalized_item_emb
    return l2_normalize(item_emb_weight)


class SampledSoftmaxLoss(nn.Module):
    def __init__(self, normalize_embeddings: bool = False):
        # REV-049: the number of negatives is implied by `neg_ids.shape[-1]` and is not a
        # constructor argument. `training/local_train.py` used to pass it positionally,
        # which raised TypeError on the first call and blocked every personalization run.
        super().__init__()
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        user_emb: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        item_emb_weight: torch.Tensor,
    ) -> torch.Tensor:
        pos_emb = item_emb_weight[pos_ids]
        neg_emb = item_emb_weight[neg_ids]

        user_emb = _maybe_normalize(user_emb, self.normalize_embeddings)
        pos_emb = _maybe_normalize(pos_emb, self.normalize_embeddings)
        neg_emb = _maybe_normalize(neg_emb, self.normalize_embeddings)

        pos_score = (user_emb * pos_emb).sum(dim=-1, keepdim=True)
        neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([pos_score, neg_scores], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class MaskedItemLoss(nn.Module):
    def __init__(self, normalize_embeddings: bool = False):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask_positions: torch.Tensor,
        true_item_ids: torch.Tensor,
        item_emb_weight: torch.Tensor,
        normalized_item_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        masked_hidden = hidden_states[mask_positions]
        if masked_hidden.shape[0] == 0:
            # REV-048: this tensor is detached from the graph. Callers must exclude the
            # objective from the GradNorm active set for this step -- see
            # `has_valid_signal` and `training/pretrain.py`.
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        masked_targets = true_item_ids[mask_positions]
        h = _maybe_normalize(masked_hidden, self.normalize_embeddings)
        w = _shared_normalized_table(
            item_emb_weight, normalized_item_emb, self.normalize_embeddings,
        )
        logits = h @ w.T
        return F.cross_entropy(logits, masked_targets)


class MultiStepLoss(nn.Module):
    def __init__(self, hidden_dim: int, multi_step_k: int, normalize_embeddings: bool = False):
        super().__init__()
        self.step_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(multi_step_k)
        ])
        self.multi_step_k = multi_step_k
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        all_hidden_states: torch.Tensor,
        targets_k: torch.Tensor,
        item_emb_weight: torch.Tensor,
        normalized_item_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = all_hidden_states.shape
        total_loss = torch.tensor(0.0, device=all_hidden_states.device, dtype=all_hidden_states.dtype)
        valid_steps = 0

        w = _shared_normalized_table(
            item_emb_weight, normalized_item_emb, self.normalize_embeddings,
        )

        for step in range(self.multi_step_k):
            pos = S - 1 - step
            if pos < 0:
                continue

            hidden_at_pos = all_hidden_states[:, pos, :]
            projected = self.step_projections[step](hidden_at_pos)

            step_targets = targets_k[:, step]
            valid_mask = step_targets != 0

            if valid_mask.sum() == 0:
                continue

            valid_hidden = _maybe_normalize(projected[valid_mask], self.normalize_embeddings)
            valid_targets = step_targets[valid_mask]

            logits = valid_hidden @ w.T
            step_loss = F.cross_entropy(logits, valid_targets)
            total_loss = total_loss + step_loss
            valid_steps += 1

        if valid_steps == 0:
            # REV-048: detached zero -- caller must drop this objective for this step.
            return torch.tensor(0.0, device=all_hidden_states.device, dtype=all_hidden_states.dtype)
        return total_loss / valid_steps


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_orig: torch.Tensor, z_aug: torch.Tensor) -> torch.Tensor:
        z_orig = F.normalize(z_orig, p=2, dim=-1)
        z_aug = F.normalize(z_aug, p=2, dim=-1)

        z = torch.cat([z_orig, z_aug], dim=0)
        B2 = z.shape[0]

        sim = z @ z.T / self.temperature

        mask = torch.eye(B2, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)

        B = B2 // 2
        labels = torch.cat([
            torch.arange(B, B2, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)


def has_valid_signal(loss: torch.Tensor) -> bool:
    """Whether a loss carries gradient signal for this step.

    REV-048: the spec's Edge Cases require a degenerate batch (zero masked positions, or
    all multi-step targets invalid) to drop that objective from the GradNorm active set.
    A detached `0.0` passed to `torch.autograd.grad` raises
    "The differentiated Tensor ... appears to not have been used in the graph", which
    aborts training mid-run rather than degrading gracefully.
    """
    return isinstance(loss, torch.Tensor) and loss.requires_grad


def apply_masking(
    seq: torch.Tensor,
    mask_prob: float,
    mask_id: int,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_seq = seq.clone()
    true_item_ids = seq.clone()

    real_items = seq != 0

    # REV-054: a run-scoped generator is held on its own device (CPU, so the masking
    # stream is reproducible regardless of the accelerator). torch.rand requires the
    # generator and the output device to agree, so draw on the generator's device and
    # move the result rather than forcing the caller to match devices.
    gen_device = rng.device if rng is not None else seq.device
    rand_vals = torch.rand(seq.shape, generator=rng, device=gen_device).to(seq.device)
    mask_positions = (rand_vals < mask_prob) & real_items

    # REV-069: the "never mask every real item in a row" guard used to be a Python loop
    # over the batch, comparing two `.sum()` tensors in an `if`. Each comparison forces a
    # device->host synchronisation, so at batch_size=256 that is >=512 syncs per step on
    # the hot path, before every forward. Vectorised: one row-wise count, one batched
    # draw, one scatter, no sync.
    n_real = real_items.sum(dim=1)
    n_mask = mask_positions.sum(dim=1)
    needs_fix = (n_real > 0) & (n_mask >= n_real)

    # Run unconditionally: an `if bool(needs_fix.any())` would still force one
    # device->host sync per step, and would make the generator stream depend on batch
    # CONTENT, so an identical seed could diverge. The work is a handful of (B, S)
    # kernels, cheaper than the sync it replaces.
    #
    # Pick one real position per row uniformly: rank real positions by a random key and
    # take the argmax. Non-real positions get -1 so they can never win.
    keys = torch.rand(seq.shape, generator=rng, device=gen_device).to(seq.device)
    keys = torch.where(real_items, keys, torch.full_like(keys, -1.0))
    keep_idx = keys.argmax(dim=1)
    unmask = torch.zeros_like(mask_positions)
    unmask.scatter_(1, keep_idx.unsqueeze(1), True)
    mask_positions = mask_positions & ~(unmask & needs_fix.unsqueeze(1))

    masked_seq[mask_positions] = mask_id
    return masked_seq, mask_positions, true_item_ids


def mask_metadata(
    metadata: Optional[torch.Tensor], mask_positions: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Zero the metadata rows at masked positions.

    REV-059: the model is forwarded on `masked_seq`, but `batch["metadata"]` was gathered
    from the ORIGINAL sequence, so a masked position received the MASK token's embedding
    PLUS the true item's 39-dimensional genome+genre vector -- a near-unique descriptor of
    the very item the masked-item objective is asked to predict. Zeroing those rows is
    exactly `metadata_matrix[masked_seq]`, because MASK_ID's row is all-zero (REV-072),
    and it contributes nothing through the metadata projection.
    """
    if metadata is None or mask_positions is None:
        return metadata
    return metadata.masked_fill(mask_positions.unsqueeze(-1).to(metadata.device), 0.0)


def static_weighted_sum(active_losses: dict, config) -> torch.Tensor:
    total = active_losses.get("next_item", torch.tensor(0.0))
    if "masked_item" in active_losses:
        total = total + config.masked_lm_weight * active_losses["masked_item"]
    if "multi_step" in active_losses:
        total = total + config.multi_step_weight * active_losses["multi_step"]
    if "contrastive" in active_losses:
        total = total + config.contrastive_weight * active_losses["contrastive"]
    return total
