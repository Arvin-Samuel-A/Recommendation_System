import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple

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


class LogitScale(nn.Module):
    """Multiplies an objective's logits by 1/temperature, fixed or learnable.

    Fixed (the default): a Python float. At temperature 1.0 `forward` returns its input
    unchanged, so the default adds no op and no parameter -- training is bit-identical to
    the code before this module existed.

    Learnable (CLIP-style): `log_scale` is an fp32 parameter initialised to
    log(1/temperature). The forward clamps its value at log(max_scale) straight-through
    (gradient kept), and `clamp_logit_scales_` clamps the stored value after every
    optimizer step, so the scale sits at the bound with a live gradient and can come
    back down from it. The parameter stays fp32 even when the model is pure bf16: a
    bf16 log-scale near 2.66 has a resolution of 0.016, coarser than one AdamW step.

    Evaluation never applies it: ranking candidates is invariant to a positive scalar, so
    `score_candidates_batch` and the ANN retrieval rank raw cosines (see there).
    """

    def __init__(self, temperature: float = 1.0, learnable: bool = False,
                 max_scale: float = 100.0):
        super().__init__()
        if not temperature > 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.learnable = learnable
        self.max_log_scale = math.log(max_scale)
        if learnable:
            init = math.log(1.0 / temperature)
            if init > self.max_log_scale + 1e-12:
                raise ValueError(
                    f"initial scale 1/{temperature:g} exceeds max_scale {max_scale:g}"
                )
            self.log_scale = nn.Parameter(torch.tensor(init, dtype=torch.float32))
            self.fixed_scale = None
        else:
            self.fixed_scale = 1.0 / temperature

    def scale(self):
        """The current multiplier: a 0-d tensor (learnable) or a float (fixed)."""
        if self.learnable:
            # Straight-through clamp: the value never exceeds the bound, but the gradient
            # is that of the unclamped log-scale. A plain `clamp` has zero gradient at and
            # past the bound (fp32 rounding of log(max) puts a clamped value just past
            # it), so the scale could never be pulled back down once it got there.
            over = (self.log_scale - self.max_log_scale).clamp(min=0).detach()
            return (self.log_scale - over).exp()
        return self.fixed_scale

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.learnable:
            if self.fixed_scale == 1.0:
                return logits
            return logits * self.fixed_scale
        # A 0-d fp32 tensor times a bf16 tensor stays bf16, so the loss dtypes are those
        # of the unscaled path.
        return logits * self.scale()

    @torch.no_grad()
    def clamp_(self) -> None:
        if self.learnable:
            self.log_scale.clamp_(max=self.max_log_scale)

    def extra_repr(self) -> str:
        if self.learnable:
            return f"learnable, max_scale={math.exp(self.max_log_scale):g}"
        return f"fixed scale={self.fixed_scale:g}"


def logit_scales(loss_fns: dict) -> dict:
    """`{objective: LogitScale}` for every loss module that has one."""
    out = {}
    for name, fn in loss_fns.items():
        scale = getattr(fn, "logit_scale", None)
        if isinstance(scale, LogitScale):
            out[name] = scale
    return out


def learnable_logit_scale_parameters(loss_fns: dict) -> List[nn.Parameter]:
    return [s.log_scale for s in logit_scales(loss_fns).values() if s.learnable]


def clamp_logit_scales_(loss_fns: dict) -> None:
    """Clamp every learnable scale to its bound; call after each optimizer step."""
    for scale in logit_scales(loss_fns).values():
        if scale.learnable:
            scale.clamp_()


def learnable_logit_scale_values(loss_fns: dict) -> dict:
    """`{objective: current scale}` of the learnable scales, as floats (one sync each)."""
    return {
        name: float(scale.scale().detach())
        for name, scale in logit_scales(loss_fns).items() if scale.learnable
    }


def _scale_or_identity(logit_scale: Optional[LogitScale]) -> LogitScale:
    return logit_scale if logit_scale is not None else LogitScale()


#: Most logits `full_softmax_cross_entropy` materialises at once. A full-catalogue softmax
#: at every training position is `(B * S) x vocab` -- ~1.4G logits at batch 256, width
#: 200 and ML-20M's ~26.7k items -- so rows are processed in chunks of
#: `FULL_SOFTMAX_CHUNK_ELEMENTS // vocab` (~630 rows, ~67 MB of fp32 logits). Measured on
#: MPS, t-base, one step with every objective: 2**24 instead of 2**26 took batch 16 from
#: 6.5 to 3.3 GB and batch 64 from 11.1 to 7.2 GB.
FULL_SOFTMAX_CHUNK_ELEMENTS = 2 ** 24
#: Below this many logits a call skips checkpointing. Above it, even one chunk is
#: checkpointed: up to 11 objectives (next-item, masked-item, nine multi-step horizons)
#: each keep an un-checkpointed call's logits alive until backward, which at batch 16
#: was ~2.8 GB -- more than the large-batch, chunked path holds.
FULL_SOFTMAX_UNCHECKPOINTED_ELEMENTS = 2 ** 22


def _cross_entropy_chunk(
    h: torch.Tensor, w: torch.Tensor, targets: torch.Tensor, logit_scale: "LogitScale",
) -> torch.Tensor:
    return F.cross_entropy(logit_scale(h @ w.T), targets)


def full_softmax_cross_entropy(
    h: torch.Tensor,
    targets: torch.Tensor,
    w: torch.Tensor,
    logit_scale: "LogitScale",
) -> torch.Tensor:
    """Mean cross-entropy of `h @ w.T` (every row of `w` a class) against `targets`.

    Above `FULL_SOFTMAX_UNCHECKPOINTED_ELEMENTS` logits, rows are processed in chunks
    under activation checkpointing, so only one chunk's logits are alive at a time in the
    forward AND the backward (which recomputes them), and none are kept between the two. The result is the mean over all rows, as a chunk-size-weighted sum of chunk
    means -- per-chunk means keep a pure-bf16 run's accumulator at loss magnitude.
    """
    n = h.shape[0]
    if n * w.shape[0] <= FULL_SOFTMAX_UNCHECKPOINTED_ELEMENTS:
        return _cross_entropy_chunk(h, w, targets, logit_scale)
    rows = max(1, FULL_SOFTMAX_CHUNK_ELEMENTS // max(1, w.shape[0]))
    total = None
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        part = checkpoint(
            _cross_entropy_chunk, h[start:stop], w, targets[start:stop], logit_scale,
            use_reentrant=False, preserve_rng_state=False,
        ) * ((stop - start) / n)
        total = part if total is None else total + part
    return total


class NextItemLoss(nn.Module):
    """LLM-style next-item objective: a full-catalogue softmax at EVERY position.

    `targets[b, t]` is the item following position `t` (0 at padding). Positions are
    pooled into one mean, so every training interaction counts equally whatever chunk it
    sits in. Hidden states must be routed per position (see `models.moe.route_objective`)
    so no position sees the items after it.
    """

    def __init__(self, normalize_embeddings: bool = False,
                 logit_scale: Optional[LogitScale] = None):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.logit_scale = _scale_or_identity(logit_scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        item_emb_weight: torch.Tensor,
        normalized_item_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid = targets != 0
        h = hidden_states[valid]
        if h.shape[0] == 0:
            # REV-048: detached zero -- the caller must drop this objective for this step.
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        h = _maybe_normalize(h, self.normalize_embeddings)
        w = _shared_normalized_table(
            item_emb_weight, normalized_item_emb, self.normalize_embeddings,
        )
        return full_softmax_cross_entropy(h, targets[valid], w, self.logit_scale)


class SampledSoftmaxLoss(nn.Module):
    """Last-position next-item loss over sampled negatives. Used by per-user LoRA
    training (local and federated); pretraining uses `NextItemLoss`."""

    def __init__(self, normalize_embeddings: bool = False,
                 logit_scale: Optional[LogitScale] = None):
        # REV-049: the number of negatives is implied by `neg_ids.shape[-1]` and is not a
        # constructor argument. `training/local_train.py` used to pass it positionally,
        # which raised TypeError on the first call and blocked every personalization run.
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.logit_scale = _scale_or_identity(logit_scale)

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

        logits = self.logit_scale(torch.cat([pos_score, neg_scores], dim=-1))
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class MaskedItemLoss(nn.Module):
    def __init__(self, normalize_embeddings: bool = False,
                 logit_scale: Optional[LogitScale] = None):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.logit_scale = _scale_or_identity(logit_scale)

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
        return full_softmax_cross_entropy(h, masked_targets, w, self.logit_scale)


class MultiStepLoss(nn.Module):
    def __init__(self, hidden_dim: int, multi_step_k: int, normalize_embeddings: bool = False,
                 logit_scale: Optional[LogitScale] = None):
        super().__init__()
        self.step_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(multi_step_k)
        ])
        self.multi_step_k = multi_step_k
        self.normalize_embeddings = normalize_embeddings
        # One scale for the objective, shared by its k horizons. A learnable scale must be
        # attached AFTER any `.to(dtype=...)` of this module, so it stays fp32 (see
        # `training.pretrain._build_loss_fns`).
        self.logit_scale = _scale_or_identity(logit_scale)

    def forward(
        self,
        all_hidden_states: torch.Tensor,
        targets: torch.Tensor,
        item_emb_weight: torch.Tensor,
        normalized_item_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the items 2, 3, ..., k+1 steps ahead from EVERY position.

        `targets` is the training batch's `(B, S)` next-item targets, so the item `h`
        steps after position `t` is `targets[:, t + h - 1]`. Horizon 1 is the next-item
        objective's and is deliberately not duplicated; step `s` (0-based) predicts
        horizon `s + 2`. A position counts when it is real and its horizon target lies
        inside the chunk -- every target is strictly in the future of its position, and
        the per-position routing keeps the hidden state causal.
        """
        B, S, D = all_hidden_states.shape
        total_loss = torch.tensor(0.0, device=all_hidden_states.device, dtype=all_hidden_states.dtype)
        valid_steps = 0

        w = _shared_normalized_table(
            item_emb_weight, normalized_item_emb, self.normalize_embeddings,
        )

        for step in range(self.multi_step_k):
            shift = step + 1
            if shift >= S:
                break
            step_targets = targets[:, shift:]
            valid = (targets[:, :S - shift] != 0) & (step_targets != 0)

            hidden = all_hidden_states[:, :S - shift][valid]
            if hidden.shape[0] == 0:
                continue

            projected = _maybe_normalize(
                self.step_projections[step](hidden), self.normalize_embeddings,
            )
            step_loss = full_softmax_cross_entropy(
                projected, step_targets[valid], w, self.logit_scale,
            )
            total_loss = total_loss + step_loss
            valid_steps += 1

        if valid_steps == 0:
            # REV-048: detached zero -- caller must drop this objective for this step.
            return torch.tensor(0.0, device=all_hidden_states.device, dtype=all_hidden_states.dtype)
        return total_loss / valid_steps


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, logit_scale: Optional[LogitScale] = None):
        """`logit_scale`, when given (`shared_logit_temperature` with a learnable scale),
        replaces the fixed division by `temperature`."""
        super().__init__()
        self.temperature = temperature
        if logit_scale is not None:
            self.logit_scale = logit_scale

    def forward(self, z_orig: torch.Tensor, z_aug: torch.Tensor) -> torch.Tensor:
        # InfoNCE is computed in fp32 by design; under mixed precision autocast would
        # otherwise run `z @ z.T` in fp16, where the -1e9 diagonal fill overflows.
        with torch.autocast(device_type=z_orig.device.type, enabled=False):
            return self._forward_fp32(z_orig.float(), z_aug.float())

    def _forward_fp32(self, z_orig: torch.Tensor, z_aug: torch.Tensor) -> torch.Tensor:
        z_orig = F.normalize(z_orig, p=2, dim=-1)
        z_aug = F.normalize(z_aug, p=2, dim=-1)

        z = torch.cat([z_orig, z_aug], dim=0)
        B2 = z.shape[0]

        if hasattr(self, "logit_scale"):
            sim = self.logit_scale(z @ z.T)
        else:
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


def mean_pool_real(hidden: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """`(B, D)` mean of `hidden` over each row's non-padding positions (`seq != 0`).

    The contrastive objective compares sequences, so it reads this rather than the last
    position. Under LLM-style training the next-item loss is spread over every position
    (~53k per batch), so a last-position InfoNCE averaged over the B rows weighted the
    scored position ~100x more than next-item did; measured, it drove the last
    position's next-item loss to chance (10.25 vs 7.8 elsewhere) and val NDCG@10 below
    popularity. Pooled, its gradient spreads over the row like every other objective's.
    """
    real = (seq != 0).unsqueeze(-1).to(hidden.dtype)
    return (hidden * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)


def static_weighted_sum(active_losses: dict, config) -> torch.Tensor:
    total = active_losses.get("next_item", torch.tensor(0.0))
    if "masked_item" in active_losses:
        total = total + config.masked_lm_weight * active_losses["masked_item"]
    if "multi_step" in active_losses:
        total = total + config.multi_step_weight * active_losses["multi_step"]
    if "contrastive" in active_losses:
        total = total + config.contrastive_weight * active_losses["contrastive"]
    return total
