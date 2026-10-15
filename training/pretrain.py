import json
import logging
import math
import os
import pickle
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from configs.config import Config, OBJECTIVE_ORDER
from models.transformer import TransformerModel
from models.moe import MoELayer
from training.losses import (
    mask_metadata,
    SampledSoftmaxLoss,
    MaskedItemLoss,
    MultiStepLoss,
    InfoNCELoss,
    apply_masking,
    static_weighted_sum,
    has_valid_signal,
    normalize_item_table,
)
from training.gradnorm import (
    GradNormController, get_anchor_layer_params,
)
from data.augmentation import augment_sequence
from data.sequence_generator import (
    load_tensorised,
    load_metadata_matrix,
    build_dataloader,
    GlobalNegativePool,
    generate_eval_candidates,
    load_eval_candidates,
    sample_training_negatives,
)
from evaluation.routing_diagnostics import save_routing_diagnostics
from evaluation.gradient_diagnostics import run_gradient_diagnostics
from evaluation.metrics import evaluate_model
from utils import (
    get_device, resolve_dtype, resolve_param_dtype, make_grad_scaler, get_source_revision,
    atomic_torch_save,
)

logger = logging.getLogger(__name__)


def build_scheduler(optimizer: torch.optim.Optimizer, config: Config, total_steps: int) -> LambdaLR:
    warmup_steps = int(total_steps * config.warmup_ratio)
    # specs/ablation-run FR-5 / D5 (owner decision 4): the cosine decays to a floor of
    # `min_lr_ratio` x peak instead of 0. `min_lr_ratio = 0` is the previous schedule for
    # every step <= total_steps; the clamp only affects steps past the end, which
    # `run_pretrain` never takes.
    min_lr_ratio = config.min_lr_ratio

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = min(1.0, float(step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def build_model(config: Config) -> Tuple[TransformerModel, MoELayer]:
    device = get_device(config)
    dtype = resolve_param_dtype(config)
    model = TransformerModel(config)
    moe = MoELayer(config).to(device=device, dtype=dtype)
    return model, moe


def save_checkpoint(
    path: str,
    epoch: int,
    model: TransformerModel,
    moe: MoELayer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    gradnorm: Optional[GradNormController],
    val_metrics: Dict[str, Dict[str, float]],
    config: Config,
    loss_fns: Optional[dict] = None,
    is_best: bool = False,
    resume_state: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "moe_state_dict": moe.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "gradnorm_state_dict": gradnorm.state_dict() if gradnorm else None,
        # REV-047: the MultiStepLoss step projections are trainable parameters; without
        # this they would not survive a reload even once they are optimized.
        "loss_fns_state_dict": {
            name: fn.state_dict()
            for name, fn in (loss_fns or {}).items()
            if any(True for _ in fn.parameters())
        },
        "val_ndcg": val_metrics.get("ndcg", {}).get("mean", 0.0),
        "val_mrr": val_metrics.get("mrr", {}).get("mean", 0.0),
        "val_recall": val_metrics.get("recall", {}).get("mean", 0.0),
        "val_precision": val_metrics.get("precision", {}).get("mean", 0.0),
        "config": asdict(config),
        # FR-3 / AC-09 (owner, 2026-09-18): which code produced this checkpoint, answerable
        # from the file. `config` does NOT substitute -- the multi-step objective change at
        # the centre of REV-129 left `config` byte-identical, which is why that dispute
        # could not be settled from the repository at all.
        "source_revision": get_source_revision(),
    }
    if is_best:
        # REV-037: FR-3 requires these as named fields on the best checkpoint rather
        # than only being recoverable by parsing the embedded config dict.
        checkpoint.update({
            "best_epoch": epoch,
            "best_val_ndcg": checkpoint["val_ndcg"],
            "best_val_mrr": checkpoint["val_mrr"],
            "best_val_recall": checkpoint["val_recall"],
            "best_val_precision": checkpoint["val_precision"],
            "seed": config.seed,
            "model_name": config.model_name,
            "moe_variant": config.moe_variant,
            "use_gradnorm": config.use_gradnorm,
            "num_experts": config.num_experts,
        })
    if resume_state is not None:
        # specs/ablation-run D8: `last.pt` carries everything needed to continue exactly.
        checkpoint["resume_state"] = resume_state
    # specs/ablation-run FR-8: a crash mid-write never leaves a truncated checkpoint.
    atomic_torch_save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: TransformerModel,
    moe: MoELayer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    gradnorm: Optional[GradNormController] = None,
    loss_fns: Optional[dict] = None,
    checkpoint: Optional[dict] = None,
) -> dict:
    if checkpoint is None:
        checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    moe.load_state_dict(checkpoint["moe_state_dict"])
    if loss_fns is not None:
        for name, state in (checkpoint.get("loss_fns_state_dict") or {}).items():
            if name in loss_fns:
                loss_fns[name].load_state_dict(state)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if gradnorm is not None and checkpoint.get("gradnorm_state_dict") is not None:
        gradnorm.load_state_dict(checkpoint["gradnorm_state_dict"])
    return checkpoint


def train_one_step(
    model: TransformerModel,
    moe: MoELayer,
    batch: dict,
    config: Config,
    loss_fns: dict,
    gradnorm: Optional[GradNormController],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    batch_idx: int,
    rng: random.Random,
    mask_rng: Optional[torch.Generator] = None,
    metadata_matrix: Optional[torch.Tensor] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    device = get_device(config)
    mask_id = config.num_items + 1
    # Mixed precision keeps fp32 master weights; without an active autocast the step
    # would silently run in fp32 instead of the configured compute dtype.
    if config.mixed_precision:
        assert torch.is_autocast_enabled(device.type), (
            "mixed_precision=True but autocast is not active; wrap the call in "
            "utils.autocast_context(config)"
        )
    use_scaler = scaler is not None and scaler.is_enabled()

    seq = batch["sequences"]
    targets = batch["targets"]
    neg_ids = batch["neg_ids"]
    targets_k = batch.get("targets_k")
    metadata = batch.get("metadata")

    if config.use_masked_lm:
        # REV-054: masking now draws from a run-scoped generator rather than the
        # global torch RNG, as T2.1 specifies.
        masked_seq, mask_positions, true_ids = apply_masking(
            seq, config.mask_prob, mask_id, rng=mask_rng,
        )
        # REV-059: the metadata block was gathered from the UNMASKED sequence, so a
        # masked position still carried the true item's genome vector into the forward
        # pass -- the MIM objective was being handed the answer. Zeroing the masked rows
        # is what `metadata_matrix[masked_seq]` yields, since MASK_ID's row is all-zero.
        metadata = mask_metadata(metadata, mask_positions)
    else:
        masked_seq = seq
        mask_positions = None
        true_ids = None

    all_hidden = model(masked_seq, metadata=metadata, return_all_hidden=True)
    pooled = all_hidden[:, -1, :]

    moe.clear_cache()
    active_losses = {}
    routing_snapshots = {}

    Z_next, r_next = moe(all_hidden, pooled, 0)
    user_emb = Z_next[:, -1, :]
    active_losses["next_item"] = loss_fns["next_item"](user_emb, targets, neg_ids, model.item_emb_weight)
    if r_next is not None:
        routing_snapshots["next_item"] = r_next.detach()

    # REV-048: the spec's Edge Cases require an objective with no valid positions /
    # targets in this batch to be EXCLUDED from the GradNorm active set. Its loss is a
    # detached 0.0, and passing that to torch.autograd.grad raises "The differentiated
    # Tensor ... appears to not have been used in the graph", aborting a long run
    # mid-training. `has_valid_signal` is the exclusion test.
    skipped_objectives = []

    # REV-077: both full-vocabulary objectives need the SAME normalized item table, and
    # `l2_normalize` builds a 26 MB fp32 intermediate at t-base dimensions. Compute it
    # once per step and share it, rather than once inside each objective.
    normalized_item_emb = normalize_item_table(
        model.item_emb_weight, config.normalize_embeddings,
    )

    if config.use_masked_lm and mask_positions is not None:
        Z_mim, r_mim = moe(all_hidden, pooled, 1)
        mim_loss = loss_fns["masked_item"](
            Z_mim, mask_positions, true_ids, model.item_emb_weight,
            normalized_item_emb=normalized_item_emb,
        )
        if has_valid_signal(mim_loss):
            active_losses["masked_item"] = mim_loss
            if r_mim is not None:
                routing_snapshots["masked_item"] = r_mim.detach()
        else:
            skipped_objectives.append("masked_item")

    if config.use_multi_step and targets_k is not None:
        Z_ms, r_ms = moe(all_hidden, pooled, 2)
        ms_loss = loss_fns["multi_step"](
            Z_ms, targets_k, model.item_emb_weight,
            normalized_item_emb=normalized_item_emb,
        )
        if has_valid_signal(ms_loss):
            active_losses["multi_step"] = ms_loss
            if r_ms is not None:
                routing_snapshots["multi_step"] = r_ms.detach()
        else:
            skipped_objectives.append("multi_step")

    is_contrastive_batch = config.use_contrastive and (batch_idx % config.contrastive_interval == 0)
    if is_contrastive_batch:
        Z_cl, r_cl = moe(all_hidden, pooled, 3)
        z_orig = torch.nn.functional.normalize(Z_cl[:, -1, :].float(), p=2, dim=-1)

        B, S = seq.shape
        aug_seqs = []
        # One device->host copy for the batch, not one sync per row.
        seq_rows = seq.cpu().tolist()
        for i in range(B):
            real = [item for item in seq_rows[i] if item != 0]
            if len(real) == 0:
                real = [1]
            aug = augment_sequence(real, mask_id, rng, config.crop_ratio, config.mask_ratio)
            padded = [0] * (S - len(aug)) + aug
            padded = padded[-S:]
            aug_seqs.append(padded)
        aug_seq_t = torch.tensor(aug_seqs, dtype=torch.long, device=device)

        # REV-038: the primary view is forwarded with metadata; without this the two
        # InfoNCE views differed by metadata presence as well as by the augmentation,
        # so the objective partly learned to bridge a metadata-present/absent gap.
        aug_meta = None
        if metadata is not None and metadata_matrix is not None:
            # The full (vocab, F) metadata matrix stays on CPU; index there and move
            # only the gathered (B, S, F) block to the accelerator.
            # REV-072: the matrix now covers MASK_ID with an all-zero row, so the clamp
            # that used to give augmented mask tokens the LAST REAL ITEM's genome vector
            # is gone -- a masked position contributes nothing, as it should.
            aug_meta = metadata_matrix[aug_seq_t.to(metadata_matrix.device)].to(device=device)

        aug_hidden = model(aug_seq_t, metadata=aug_meta, return_all_hidden=True)
        aug_pooled = aug_hidden[:, -1, :]
        Z_aug, _ = moe(aug_hidden, aug_pooled, 3)
        z_aug = torch.nn.functional.normalize(Z_aug[:, -1, :].float(), p=2, dim=-1)

        active_losses["contrastive"] = loss_fns["contrastive"](z_orig, z_aug)
        if r_cl is not None:
            routing_snapshots["contrastive"] = r_cl.detach()

    # REV-022b: T5.2 requires dtype assertions on outputs as well as parameters.
    # Under mixed precision autocast picks per-op dtypes (layer norm and the losses run
    # in fp32), so the output check applies to the pure-dtype path only and every loss
    # is expected in fp32.
    compute_dtype = resolve_dtype(config)
    if not config.mixed_precision:
        assert all_hidden.dtype == compute_dtype, (
            f"Model output dtype {all_hidden.dtype}, expected {compute_dtype}"
        )
    for name, loss in active_losses.items():
        # The contrastive objective is computed on fp32-normalized embeddings
        # (InfoNCE's similarity matrix is numerically fragile in bf16); this is a
        # deliberate, documented exception recorded in design.md.
        fp32_loss = name == "contrastive" or config.mixed_precision
        expected = torch.float32 if fp32_loss else compute_dtype
        assert loss.dtype == expected, (
            f"Loss '{name}' dtype {loss.dtype}, expected {expected}"
        )

    if gradnorm is not None:
        L_total = gradnorm.weighted_sum(active_losses)
    else:
        L_total = static_weighted_sum(active_losses, config)

    # Check A (specs/ablation-run D4): every active loss and the total, finite, in ONE
    # stacked check (one host sync instead of one per loss), still before backward. Only
    # on failure are the per-loss values read, to name the offending objective.
    finite = torch.isfinite(torch.stack(
        [loss.detach().float() for loss in active_losses.values()]
        + [L_total.detach().float()]
    )).all()
    if not bool(finite):
        for name, loss in active_losses.items():
            assert torch.isfinite(loss), f"{name} loss is not finite: {loss.item()}"
        raise AssertionError(f"Total loss not finite: {L_total.item()}")

    # REV-022a: code-quality.md requires set_to_none=True on the hot path.
    optimizer.zero_grad(set_to_none=True)
    # fp16 only: scale the loss so small gradients do not underflow. GradNorm below
    # differentiates the UNSCALED losses -- it needs only norms, which are dominated by
    # the large components that fp16 represents fine, and scaling would add overflow.
    (scaler.scale(L_total) if use_scaler else L_total).backward(
        retain_graph=(gradnorm is not None),
    )

    if gradnorm is not None:
        anchor_params = get_anchor_layer_params(model, config)
        gradnorm.update(active_losses, anchor_params)

    trainable = list(model.parameters()) + list(moe.parameters())
    for fn in loss_fns.values():
        trainable += list(fn.parameters())

    # REV-022c: FR-7 and AC-08 both require finite-value assertions on GRADIENTS.
    # A NaN appearing in gradients rather than in the loss scalar previously passed
    # every assertion and silently corrupted the weights.
    # The total norm is finite iff every gradient is, so one check replaces a per-tensor
    # assert that forced a device->host sync for each of the ~91 parameter tensors.
    # REV-047: clipping must cover the loss functions' own parameters too.
    # With a GradScaler, an fp16 overflow is expected while the scale calibrates: the
    # scaler detects the non-finite gradients and skips that optimizer step instead.
    if use_scaler:
        scaler.unscale_(optimizer)
    total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
    # Check B (specs/ablation-run D4): the gradient norm (non-scaler path; with a scaler
    # an fp16 overflow is expected while the scale calibrates and the scaler skips that
    # step) and the GradNorm lambdas, in ONE stacked check, before `optimizer.step()`.
    _assert_finite_grads_and_lambdas(total_grad_norm, gradnorm, check_grad_norm=not use_scaler)
    if use_scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()

    # specs/ablation-run D4: 0-d device tensors, not `.item()` (one sync each). They
    # support float()/format()/math.isfinite(); `run_pretrain` materialises them once per
    # epoch, producing the same floats.
    loss_values = {name: loss.detach() for name, loss in active_losses.items()}
    loss_values["total"] = L_total.detach()
    if skipped_objectives:
        loss_values["_skipped"] = skipped_objectives

    # REV-071: release the cached expert outputs HERE, not only at the start of the next
    # step. `_cached_expert_outputs` is a (B, S, K, D) tensor still attached to this
    # step's autograd graph -- ~131 MB of activations plus the graph at batch_size=256,
    # S=50, K=4, D=320 in bf16 -- and the start-of-step call meant it survived from
    # `optimizer.step()` until the next step began, coexisting with the new forward's
    # allocation. The cache can never be hit across steps anyway: every step brings a
    # fresh `all_hidden`. The start-of-step `clear_cache()` is kept, so the invariant
    # holds even if a caller enters mid-loop.
    moe.clear_cache()

    return loss_values, routing_snapshots if routing_snapshots else None


def _assert_finite_grads_and_lambdas(
    total_grad_norm: torch.Tensor,
    gradnorm: Optional[GradNormController],
    check_grad_norm: bool,
) -> None:
    """Check B of D4: one host sync for the grad norm and the active lambdas together."""
    parts = []
    if check_grad_norm:
        parts.append(total_grad_norm.detach().float().reshape(1))
    if gradnorm is not None:
        lambdas = gradnorm.lambdas.detach().float().reshape(-1)
        parts.append(lambdas.to(parts[0].device) if parts else lambdas)
    if not parts:
        return
    if bool(torch.isfinite(torch.cat(parts)).all()):
        return
    if check_grad_norm:
        assert torch.isfinite(total_grad_norm), "Non-finite gradient detected"
    lambdas = gradnorm.lambdas.detach()
    raise AssertionError(f"GradNorm lambdas not finite: {lambdas}")


def assert_param_dtype(model: TransformerModel, moe: MoELayer, dtype: torch.dtype) -> None:
    for name, p in model.named_parameters():
        assert p.dtype == dtype, f"Model param {name} has dtype {p.dtype}, expected {dtype}"
    for name, p in moe.named_parameters():
        if "log_lambdas" in name:
            continue
        assert p.dtype == dtype, f"MoE param {name} has dtype {p.dtype}, expected {dtype}"


def assert_finite_step(loss_values: Dict[str, float], gradnorm: Optional[GradNormController]) -> None:
    """Finite check over a step's loss values and lambdas, for callers outside the loop.

    specs/ablation-run D4: `run_pretrain` no longer calls this per step -- the same checks
    run inside `train_one_step` (checks A and B), batched, before `optimizer.step()`.
    """
    for name, val in loss_values.items():
        if name.startswith("_"):
            continue
        assert math.isfinite(val), f"Loss '{name}' is not finite: {val}"
    if gradnorm is not None:
        lambdas = gradnorm.lambdas.detach()
        assert torch.all(torch.isfinite(lambdas)), f"GradNorm lambdas not finite: {lambdas}"


def setup_epoch_logger(
    config: Config, model_name: str, append: bool = False,
) -> logging.Logger:
    """The per-epoch log. `append=True` (a resumed run) keeps the existing lines."""
    log_path = os.path.join(config.log_dir, f"pretrain_{model_name.replace('-', '_')}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    epoch_logger = logging.getLogger(f"pretrain.{model_name}")
    epoch_logger.setLevel(logging.INFO)
    for handler in list(epoch_logger.handlers):
        handler.close()
    epoch_logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a" if append else "w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    epoch_logger.addHandler(fh)
    return epoch_logger


def log_epoch(
    epoch_logger: logging.Logger,
    epoch: int,
    train_losses: Dict[str, float],
    val_metrics: Dict[str, Dict[str, float]],
    lr: float,
    gradnorm: Optional[GradNormController],
) -> None:
    parts = [f"epoch={epoch}"]
    parts.append(f"train_loss_total={train_losses.get('total', 0.0):.6f}")
    for name in OBJECTIVE_ORDER:
        if name in train_losses:
            key = name.replace("_", "")
            parts.append(f"train_loss_{key}={train_losses[name]:.6f}")
    for metric_name in ["ndcg", "mrr", "recall", "precision"]:
        val = val_metrics.get(metric_name, {}).get("mean", 0.0)
        parts.append(f"val_{metric_name}@10={val:.6f}")
    parts.append(f"lr={lr:.8f}")
    if gradnorm is not None:
        lambdas = gradnorm.lambdas.detach().cpu()
        for i, obj_name in enumerate(OBJECTIVE_ORDER):
            parts.append(f"lambda_{obj_name}={lambdas[i].item():.6f}")
    # AC-09: the per-epoch log is the artifact REV-129 had to reconstruct a timeline to
    # date. Stamping each line makes "which code wrote this number?" a lookup.
    revision = get_source_revision()
    parts.append(f"source_commit={revision.get('commit')}")
    parts.append(f"source_hash={revision.get('source_hash')}")
    parts.append(f"source_dirty={revision.get('dirty')}")
    epoch_logger.info(" | ".join(parts))


def log_gradnorm_lambdas(
    config: Config,
    model_name: str,
    step: int,
    gradnorm: GradNormController,
) -> None:
    path = os.path.join(config.log_dir, f"gradnorm_{model_name.replace('-', '_')}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lambdas = gradnorm.lambdas.detach().cpu()
    record = {
        "step": step,
        "lambdas": {name: lambdas[i].item() for i, name in enumerate(OBJECTIVE_ORDER)},
        "source_revision": get_source_revision(),
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def plot_gradnorm_trajectory(config: Config, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(config.log_dir, f"gradnorm_{model_name.replace('-', '_')}.jsonl")
    if not os.path.exists(path):
        return
    steps, lambda_series = [], {name: [] for name in OBJECTIVE_ORDER}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            steps.append(rec["step"])
            for name in OBJECTIVE_ORDER:
                lambda_series[name].append(rec["lambdas"].get(name, 0.0))

    os.makedirs(config.plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in OBJECTIVE_ORDER:
        if any(v > 0 for v in lambda_series[name]):
            ax.plot(steps, lambda_series[name], label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Lambda")
    ax.set_title("GradNorm Lambda Trajectory")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(config.plot_dir, f"gradnorm_lambda_trajectory_{model_name.replace('-', '_')}.png")
    fig.savefig(out, dpi=config.plot_dpi)
    plt.close(fig)


def _build_loss_fns(config: Config, device: torch.device) -> dict:
    dtype = resolve_param_dtype(config)
    norm = config.normalize_embeddings
    loss_fns = {
        "next_item": SampledSoftmaxLoss(normalize_embeddings=norm),
    }
    if config.use_masked_lm:
        loss_fns["masked_item"] = MaskedItemLoss(normalize_embeddings=norm)
    if config.use_multi_step:
        loss_fns["multi_step"] = MultiStepLoss(
            config.hidden_dim, config.multi_step_k, normalize_embeddings=norm,
        ).to(device=device, dtype=dtype)
    if config.use_contrastive:
        loss_fns["contrastive"] = InfoNCELoss(config.contrastive_temperature)
    return loss_fns


ROUTING_BUFFER_MAX_BATCHES = 256


def _append_routing_snapshots(
    buffer: Dict[str, List[torch.Tensor]],
    snapshots: Dict[str, torch.Tensor],
    max_batches: int = ROUTING_BUFFER_MAX_BATCHES,
) -> None:
    """Accumulate detached (B, K) routing snapshots under a hard cap.

    REV-010: the buffer previously grew for a whole epoch with no cap -- on
    MovieLens-20M at batch_size 256 that is tens of thousands of retained tensors per
    objective. Reservoir-free FIFO truncation keeps the memory bounded while leaving the
    distribution estimate representative.
    """
    for obj_name, snap in snapshots.items():
        entries = buffer.setdefault(obj_name, [])
        entries.append(snap.detach().to("cpu", torch.float32))
        if len(entries) > max_batches:
            del entries[0]


def _emit_routing_diagnostics(
    buffer: Dict[str, List[torch.Tensor]], config: Config, model_name: str,
) -> Optional[dict]:
    """Concatenate the buffer and hand it to `save_routing_diagnostics`, then clear it.

    REV-010 / evaluation-diagnostics REV-008: the buffer used to be filled and then
    garbage-collected at epoch end, so `save_routing_diagnostics` had no call site and
    `routing_diagnostics_{model}.json` could never be produced.
    """
    per_objective = {
        name: torch.cat(entries, dim=0)
        for name, entries in buffer.items()
        if entries
    }
    buffer.clear()
    if not per_objective:
        return None
    return save_routing_diagnostics(
        per_objective, config, model_name, source_revision=get_source_revision(),
    )


def collect_validation_routing(
    model: TransformerModel,
    moe: MoELayer,
    val_dataset,
    config: Config,
    max_batches: int = ROUTING_BUFFER_MAX_BATCHES,
) -> Dict[str, torch.Tensor]:
    """Comprehensive end-of-training routing pass over the validation set.

    evaluation-diagnostics FR-8 requires this alongside the periodic in-training passes.
    Runs under `no_grad` in eval mode; only objectives enabled by the config contribute.
    """
    device = get_device(config)
    loader = build_dataloader(val_dataset, config, shuffle=False)
    collected: Dict[str, List[torch.Tensor]] = {}

    was_training = model.training
    model.eval()
    moe.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            seq = batch["sequences"].to(device)
            metadata = batch.get("metadata")
            if isinstance(metadata, torch.Tensor):
                metadata = metadata.to(device)
            all_hidden = model(seq, metadata=metadata, return_all_hidden=True)
            pooled = all_hidden[:, -1, :]
            moe.clear_cache()
            enabled = {
                "next_item": True,
                "masked_item": config.use_masked_lm,
                "multi_step": config.use_multi_step,
                "contrastive": config.use_contrastive,
            }
            for obj_idx, obj_name in enumerate(OBJECTIVE_ORDER):
                if not enabled[obj_name]:
                    continue
                _, r = moe(all_hidden, pooled, obj_idx)
                if r is not None:
                    collected.setdefault(obj_name, []).append(
                        r.detach().to("cpu", torch.float32)
                    )
    if was_training:
        model.train()
        moe.train()

    return {name: torch.cat(v, dim=0) for name, v in collected.items() if v}


def _host_numpy(t: torch.Tensor) -> np.ndarray:
    """A NumPy view of a host tensor; a device tensor is copied back (one sync).

    specs/ablation-run D4: the training loop now samples negatives on the CPU batch
    BEFORE the host->device copy, so this is a zero-copy view there.
    """
    if t.device.type == "cpu":
        return t.detach().numpy()
    return t.detach().cpu().numpy()


def attach_training_negatives(
    batch: dict,
    pool: np.ndarray,
    history_arrays: Dict[int, np.ndarray],
    config: Config,
    rng: np.random.RandomState,
    device: torch.device,
    buffer: Optional[np.ndarray] = None,
) -> dict:
    """Attach filtered per-example negatives to a batch, in place.

    REV-012 / REV-009a: this is the single definition of how training negatives are
    drawn. The training loop and `run_gradient_diagnostics` both go through it, so the
    diagnostic pass measures the same objective the optimizer sees, and a dataloader
    batch (which carries no `neg_ids`) is usable at either site.
    """
    neg_ids = sample_training_negatives(
        pool=pool,
        targets=_host_numpy(batch["targets"]),
        user_ids=_host_numpy(batch["user_ids"]),
        history_arrays=history_arrays,
        num_negatives=config.num_sampled_negatives,
        num_items=config.num_items,
        rng=rng,
        _buffer=buffer,
    )
    batch["neg_ids"] = torch.from_numpy(neg_ids).to(device=device, dtype=torch.long)
    return batch


def collect_trainable_parameters(
    model: TransformerModel, moe: MoELayer, loss_fns: dict,
) -> List[torch.nn.Parameter]:
    """Every nn.Parameter reachable from a training step.

    REV-047: `MultiStepLoss` owns k learnable `nn.Linear(D, D)` step projections that
    FR-3 requires to be learned. The optimizer used to be built from
    `model.parameters() + moe.parameters()` BEFORE `_build_loss_fns` ran, so 0 of 3 were
    ever optimized -- the multi-step objective trained the backbone through a frozen
    random projection, and the projections accumulated ungated, unclipped gradients.
    """
    params = list(model.parameters()) + list(moe.parameters())
    for fn in loss_fns.values():
        params += list(fn.parameters())
    return params


def _load_or_build_val_candidates(
    config: Config,
    processed: str,
    val_dataset,
    neg_array: np.ndarray,
    eval_user_history: dict,
    device: torch.device,
) -> torch.Tensor:
    """Load the pre-stored validation candidate sets, building them only as a fallback.

    REV-041 / T8.3: the fixed candidate sets are generated once during preprocessing and
    reused by every model evaluation. Regenerating them per run (which is what used to
    happen, on every epoch of every run) made cross-model identity incidental rather
    than structural.

    REV-063: returns ONE `(N, C)` CPU tensor. This used to explode the matrix into N
    individual device tensors -- 993,372 small accelerator allocations held for the whole
    run. `evaluate_model` slices the batch it needs and moves only that across.
    """
    path = os.path.join(processed, "val_candidates.npy")
    if os.path.exists(path):
        matrix = load_eval_candidates(path)
        if matrix.shape[0] != len(val_dataset):
            raise RuntimeError(
                f"Pre-stored val candidate sets have {matrix.shape[0]} rows but the "
                f"validation split has {len(val_dataset)} examples. Re-run "
                f"--stage preprocess."
            )
        return torch.as_tensor(matrix, dtype=torch.long)

    logger.warning(
        "%s not found -- regenerating validation candidate sets. Re-run "
        "--stage preprocess so every model evaluates against the same stored sets.",
        path,
    )
    val_examples = {
        "targets": val_dataset.targets.numpy().tolist(),
        "user_ids": val_dataset.user_ids.numpy().tolist(),
    }
    candidates = generate_eval_candidates(
        val_examples, neg_array, eval_user_history,
        config.num_sampled_negatives,
        padding_id=0, mask_id=config.num_items + 1, seed=config.seed,
    )
    return torch.as_tensor(np.asarray(candidates), dtype=torch.long)


# ---------------- epoch-granular exact resume (specs/ablation-run FR-8, design D8) ----------

LAST_CHECKPOINT = "last.pt"
#: GradNorm lambdas are appended to `gradnorm_{model}.jsonl` every this many global steps.
GRADNORM_LOG_INTERVAL_STEPS = 100
RESUME_FORMAT = 1
_REVISION_KEYS = ("commit", "source_hash", "dirty")


def _capture_rng_state(
    rng: random.Random,
    neg_rng: np.random.RandomState,
    mask_rng: torch.Generator,
    device: torch.device,
) -> dict:
    return {
        "python": rng.getstate(),
        "numpy_neg": neg_rng.get_state(),
        "torch_mask": mask_rng.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "torch_mps": torch.mps.get_rng_state() if device.type == "mps" else None,
        # Not read by the training step today; saved so a future reader cannot drift.
        "python_global": random.getstate(),
        "numpy_global": np.random.get_state(),
    }


def _restore_rng_state(
    state: dict,
    rng: random.Random,
    neg_rng: np.random.RandomState,
    mask_rng: torch.Generator,
    device: torch.device,
) -> None:
    rng.setstate(state["python"])
    neg_rng.set_state(state["numpy_neg"])
    mask_rng.set_state(state["torch_mask"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if device.type == "mps" and state.get("torch_mps") is not None:
        torch.mps.set_rng_state(state["torch_mps"])
    if state.get("python_global") is not None:
        random.setstate(state["python_global"])
    if state.get("numpy_global") is not None:
        np.random.set_state(state["numpy_global"])


def _training_fingerprint_of(config: Config) -> str:
    # Lazy: `ablation_runner` imports this module inside its functions only.
    from training.ablation_runner import _training_fingerprint

    return _training_fingerprint(config)


def _load_resume_checkpoint(path: str, config: Config, total_steps: int) -> dict:
    """Load `last.pt` and refuse it unless it belongs to THIS run (D8)."""
    checkpoint = torch.load(path, weights_only=False)
    state = checkpoint.get("resume_state")
    remedy = f"delete {path} or pass --no_resume to train from scratch"
    if not isinstance(state, dict) or state.get("format") != RESUME_FORMAT:
        raise RuntimeError(f"Cannot resume from {path}: no resume_state (format {RESUME_FORMAT}); {remedy}.")
    current_fp = _training_fingerprint_of(config)
    if state.get("training_fingerprint") != current_fp:
        raise RuntimeError(
            f"Refusing to resume from {path}: training_fingerprint differs (checkpoint "
            f"{state.get('training_fingerprint')}, this run {current_fp}) -- the config "
            f"changed; {remedy}."
        )
    if state.get("total_steps") != total_steps:
        raise RuntimeError(
            f"Refusing to resume from {path}: total_steps differs (checkpoint "
            f"{state.get('total_steps')}, this run {total_steps}); {remedy}."
        )
    stored_rev = checkpoint.get("source_revision") or {}
    current_rev = get_source_revision()
    for key in _REVISION_KEYS:
        if stored_rev.get(key) != current_rev.get(key):
            raise RuntimeError(
                f"Refusing to resume from {path}: source_revision.{key} differs "
                f"(checkpoint {stored_rev.get(key)}, this code {current_rev.get(key)}) -- "
                f"a run assembled from two code revisions has no single provenance; {remedy}."
            )
    return checkpoint


def _truncate_epoch_log(path: str, start_epoch: int) -> None:
    """Drop per-epoch log lines for epochs >= start_epoch (they will be rewritten)."""
    if not os.path.exists(path):
        return
    import re

    kept = []
    with open(path) as f:
        for line in f:
            match = re.search(r"\bepoch=(\d+)\b", line)
            if match and int(match.group(1)) >= start_epoch:
                continue
            kept.append(line)
    with open(path, "w") as f:
        f.writelines(kept)


def _truncate_gradnorm_log(path: str, global_step: int) -> None:
    """Keep only GradNorm JSONL records for steps already taken."""
    if not os.path.exists(path):
        return
    kept = []
    with open(path) as f:
        for line in f:
            try:
                if json.loads(line)["step"] < global_step:
                    kept.append(line)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    with open(path, "w") as f:
        f.writelines(kept)


def run_pretrain(config: Config, model_name: str) -> Dict[str, float]:
    config.apply_model_config(model_name)
    config.validate()
    # specs/ablation-run FR-7 / AC-20: never train on a different user sample than asked.
    from data.preprocessing import check_processed_dir
    check_processed_dir(config)
    device = get_device(config)
    dtype = resolve_param_dtype(config)
    rng = random.Random(config.seed)

    processed = config.processed_data_dir

    # The dataset statistics must be read BEFORE the metadata matrix: REV-072's
    # alignment pads the matrix out to `vocab_size`, which is derived from `num_items`.
    with open(os.path.join(processed, "dataset_stats.json")) as f:
        stats = json.load(f)
    config.num_items = stats["num_items"]
    config.num_users = stats["num_users"]
    config.vocab_size = config.num_items + 2

    # REV-058/REV-072: the aligned (vocab, F) matrix. `RecDataset` keeps it and gathers
    # per item; nothing materialises a dense (N, S, F) block any more.
    metadata_matrix = load_metadata_matrix(processed, config)
    if metadata_matrix is not None:
        config.feature_dim = metadata_matrix.shape[-1]

    train_dataset = load_tensorised(
        os.path.join(processed, "train_tensorised.pt"), metadata_matrix,
        max_seq_length=config.max_seq_length,
    )
    val_dataset = load_tensorised(
        os.path.join(processed, "val_tensorised.pt"), metadata_matrix,
        max_seq_length=config.max_seq_length,
    )

    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    num_batches_per_epoch = len(train_loader)
    if config.max_steps_per_epoch > 0:
        num_batches_per_epoch = min(num_batches_per_epoch, config.max_steps_per_epoch)
    total_steps = num_batches_per_epoch * config.pretrain_epochs

    model, moe = build_model(config)
    assert_param_dtype(model, moe, resolve_param_dtype(config))

    # REV-047: loss functions are built BEFORE the optimizer so their parameters are
    # registered in it.
    loss_fns = _build_loss_fns(config, device)
    optimizer = torch.optim.AdamW(
        collect_trainable_parameters(model, moe, loss_fns),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=True,
    )
    optimizer.zero_grad(set_to_none=True)
    scheduler = build_scheduler(optimizer, config, total_steps)

    _opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    for fn_name, fn in loss_fns.items():
        for pname, p in fn.named_parameters():
            assert id(p) in _opt_ids, (
                f"Loss parameter {fn_name}.{pname} is not registered in the optimizer"
            )

    # Enabled only for fp16 mixed precision; a no-op otherwise.
    scaler = make_grad_scaler(config)

    gradnorm = None
    if config.use_gradnorm:
        gradnorm = GradNormController(config.gradnorm_alpha, config.gradnorm_lr)
        gradnorm.log_lambdas = torch.nn.Parameter(gradnorm.log_lambdas.data.to(device=device))
        gradnorm.optimizer = torch.optim.Adam([gradnorm.log_lambdas], lr=config.gradnorm_lr)

    # AC-05: loaded once, outside the epoch loop, never regenerated per epoch.
    neg_pool_path = os.path.join(processed, "global_negative_pool.npy")
    neg_pool = GlobalNegativePool.load(neg_pool_path, set())
    neg_array = neg_pool.pool

    # REV-039: steering/testing.md lists this among the assertions that must be active
    # during all training/evaluation. Both histories are loaded so the guard is real.
    with open(os.path.join(processed, "train_user_history.pkl"), "rb") as f:
        train_user_history = pickle.load(f)
    with open(os.path.join(processed, "eval_user_history.pkl"), "rb") as f:
        eval_user_history = pickle.load(f)
    assert train_user_history is not eval_user_history, (
        "train_user_history and eval_user_history must be distinct objects"
    )

    # REV-012: training negatives are drawn per example, excluding the target, pad,
    # mask and the user's TRAINING history. Arrays are prebuilt once for the bitmap
    # membership test in `sample_training_negatives`.
    train_history_arrays = {
        int(u): np.asarray(items, dtype=np.int64)
        for u, items in train_user_history.items()
    }

    # REV-012: a run-scoped RNG (seeded from config.seed) drives negative sampling, and
    # the membership bitmap is allocated once and reused across steps.
    neg_rng = np.random.RandomState(config.seed)
    neg_buffer = np.zeros((config.batch_size, config.num_items + 2), dtype=bool)

    # REV-054: run-scoped generator for masking instead of the global torch RNG.
    mask_rng = torch.Generator(device="cpu")
    mask_rng.manual_seed(config.seed)

    # REV-038: the contrastive augmented view needs the same metadata lookup the primary
    # view gets, so the two views differ only by the augmentation.
    metadata_matrix_t = None
    if metadata_matrix is not None:
        metadata_matrix_t = metadata_matrix.to(dtype)

    def attach_negatives(batch: dict) -> dict:
        return attach_training_negatives(
            batch, neg_array, train_history_arrays, config, neg_rng, device, neg_buffer,
        )

    # REV-063: the stored `(N, S)` tensor IS the batch source; slicing it into N
    # per-example tensors up front cost ~1M allocations and bought nothing.
    val_sequences = val_dataset.sequences
    val_candidate_tensors = _load_or_build_val_candidates(
        config, processed, val_dataset, neg_array, eval_user_history, device,
    )

    best_val_ndcg = -1.0
    best_epoch = -1
    global_step = 0
    start_epoch = 0
    ckpt_dir = config.model_checkpoint_dir()
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, LAST_CHECKPOINT)

    # REV-010: the buffer spans the diagnostic interval, not a single epoch, and is
    # cleared by `_emit_routing_diagnostics` once consumed.
    routing_buffer: Dict[str, List[torch.Tensor]] = {}
    diag_loader = build_dataloader(train_dataset, config, shuffle=True)
    epoch_loss_windows: List[Dict[str, float]] = []
    epoch_mean_losses: List[float] = []

    gradnorm_log_path = os.path.join(config.log_dir, f"gradnorm_{model_name.replace('-', '_')}.jsonl")
    epoch_log_path = os.path.join(config.log_dir, f"pretrain_{model_name.replace('-', '_')}.log")

    # specs/ablation-run FR-8 / D8: resume from `last.pt` at the next epoch. Every object
    # above is constructed first; restoring the RNGs AFTER construction is what makes the
    # next epoch see exactly the state an uninterrupted run would.
    resumed = None
    if config.resume_training and os.path.exists(last_path):
        resumed = _load_resume_checkpoint(last_path, config, total_steps)
        state = resumed["resume_state"]
        load_checkpoint(
            last_path, model, moe, optimizer=optimizer, scheduler=scheduler,
            gradnorm=gradnorm, loss_fns=loss_fns, checkpoint=resumed,
        )
        if state.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(state["scaler"])
        _restore_rng_state(state["rng"], rng, neg_rng, mask_rng, device)
        train_loader.generator.set_state(state["loader_generators"]["train"])
        diag_loader.generator.set_state(state["loader_generators"]["diag"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_val_ndcg = float(state["best_val_ndcg"])
        best_epoch = int(state["best_epoch"])
        epoch_mean_losses = list(state["epoch_mean_losses"])
        epoch_loss_windows = list(state["epoch_loss_windows"])
        assert scheduler.last_epoch == global_step, (
            f"restored scheduler.last_epoch={scheduler.last_epoch} != global_step={global_step}"
        )
        logger.info(
            "Resuming %s from %s at epoch %d (global step %d)",
            model_name, last_path, start_epoch, global_step,
        )
        _truncate_epoch_log(epoch_log_path, start_epoch)
        _truncate_gradnorm_log(gradnorm_log_path, global_step)
    elif os.path.exists(gradnorm_log_path):
        os.remove(gradnorm_log_path)

    epoch_logger = setup_epoch_logger(config, model_name, append=resumed is not None)

    non_blocking = device.type == "cuda" and config.pin_memory

    for epoch in range(start_epoch, config.pretrain_epochs):
        model.train()
        moe.train()
        epoch_losses: Dict[str, List[float]] = {}

        for batch_idx, batch in enumerate(train_loader):
            if config.max_steps_per_epoch > 0 and batch_idx >= config.max_steps_per_epoch:
                break
            # specs/ablation-run D4: negatives are drawn from the CPU batch the loader
            # yielded (same inputs, same `neg_rng` draws, same output), so the step no
            # longer copies `targets`/`user_ids` device->host. Then one host->device move,
            # asynchronous from pinned memory on CUDA.
            batch = attach_negatives(batch)
            batch = {
                k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            loss_values, snapshots = train_one_step(
                model, moe, batch, config, loss_fns, gradnorm,
                optimizer, scheduler, batch_idx, rng,
                mask_rng=mask_rng, metadata_matrix=metadata_matrix_t, scaler=scaler,
            )

            # specs/ablation-run D4: the finite checks ran inside `train_one_step`;
            # values stay on the device until the epoch ends.
            for name, val in loss_values.items():
                if name.startswith("_"):
                    continue
                epoch_losses.setdefault(name, []).append(val)

            if snapshots is not None:
                _append_routing_snapshots(routing_buffer, snapshots)

            # REV-009: gradient diagnostics on FRESH diagnostic batches every
            # `grad_diag_interval_steps`, which was never read before.
            if (
                config.grad_diag_interval_steps > 0
                and global_step > 0
                and global_step % config.grad_diag_interval_steps == 0
            ):
                run_gradient_diagnostics(
                    model, moe, diag_loader, loss_fns,
                    get_anchor_layer_params(model, config), config, model_name,
                    negative_sampler=attach_negatives,
                    metadata_matrix=metadata_matrix_t,
                    augment_rng=rng,
                    source_revision=get_source_revision(),
                )
                optimizer.zero_grad(set_to_none=True)
                model.train()
                moe.train()

            if gradnorm is not None and global_step % GRADNORM_LOG_INTERVAL_STEPS == 0:
                log_gradnorm_lambdas(config, model_name, global_step, gradnorm)

            global_step += 1

        # One materialisation per objective per epoch (was one `.item()` per loss per
        # step); the floats are identical to the per-step `.item()` values.
        epoch_losses = {
            name: torch.stack([v.float() for v in vals]).tolist()
            for name, vals in epoch_losses.items()
        }
        avg_losses = {name: sum(vals) / len(vals) for name, vals in epoch_losses.items()}

        # AC-04 / REV-005: record the first- and last-window means of the per-batch total
        # loss so a run can be checked for an actual decrease. Without this the only
        # observable was the whole-epoch mean, which cannot show a trend. Note that over
        # a few-step gate run this window is dominated by LR warmup and is NOT evidence
        # for or against AC-04 -- that criterion needs real multi-epoch training.
        totals = epoch_losses.get("total", [])
        if totals:
            window = max(1, len(totals) // 5)
            loss_windows = {
                "first_window_loss": sum(totals[:window]) / window,
                "last_window_loss": sum(totals[-window:]) / window,
                "window_size": window,
                "num_batches": len(totals),
            }
        else:
            loss_windows = {}
        epoch_loss_windows.append(loss_windows)

        model.eval()
        moe.eval()
        with torch.no_grad():
            val_metrics = evaluate_model(
                model, val_sequences, val_candidate_tensors, config,
                moe_layer=moe, metadata_matrix=val_dataset.metadata_matrix,
            )

        epoch_mean_losses.append(avg_losses.get("total", float("nan")))

        current_lr = scheduler.get_last_lr()[0]
        log_epoch(epoch_logger, epoch, avg_losses, val_metrics, current_lr, gradnorm)

        if (
            config.routing_diag_interval_epochs > 0
            and (epoch + 1) % config.routing_diag_interval_epochs == 0
        ):
            _emit_routing_diagnostics(routing_buffer, config, model_name)

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        save_checkpoint(
            ckpt_path, epoch, model, moe, optimizer, scheduler, gradnorm,
            val_metrics, config, loss_fns=loss_fns,
        )

        val_ndcg = val_metrics.get("ndcg", {}).get("mean", 0.0)
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            best_path = os.path.join(ckpt_dir, "best.pt")
            # REV-037: is_best=True writes the named FR-3 metadata block; REV-047: the
            # loss functions own trainable parameters that must survive a reload.
            save_checkpoint(
                best_path, epoch, model, moe, optimizer, scheduler, gradnorm,
                val_metrics, config, loss_fns=loss_fns, is_best=True,
            )

        # specs/ablation-run FR-8 / D8: the full resume state, written last (after best.pt)
        # and atomically. A crash before this line redoes the epoch on resume, which
        # reproduces the same best.pt because training is deterministic given this state.
        resume_state = {
            "format": RESUME_FORMAT,
            "epoch": epoch,
            "global_step": global_step,
            "total_steps": total_steps,
            "training_fingerprint": _training_fingerprint_of(config),
            "best_val_ndcg": best_val_ndcg,
            "best_epoch": best_epoch,
            "epoch_mean_losses": list(epoch_mean_losses),
            "epoch_loss_windows": list(epoch_loss_windows),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "rng": _capture_rng_state(rng, neg_rng, mask_rng, device),
            "loader_generators": {
                "train": train_loader.generator.get_state(),
                "diag": diag_loader.generator.get_state(),
            },
        }
        save_checkpoint(
            last_path, epoch, model, moe, optimizer, scheduler, gradnorm,
            val_metrics, config, loss_fns=loss_fns, resume_state=resume_state,
        )

        logger.info(
            f"Epoch {epoch}: train_loss={avg_losses.get('total', 0.0):.4f} "
            f"val_ndcg={val_ndcg:.4f} best_ndcg={best_val_ndcg:.4f} (epoch {best_epoch})"
        )

    # evaluation-diagnostics FR-8: comprehensive end-of-training routing pass over the
    # validation set, which supersedes whatever residue is left in the interval buffer.
    routing_buffer.clear()
    val_routing = collect_validation_routing(model, moe, val_dataset, config)
    if val_routing:
        save_routing_diagnostics(
            val_routing, config, model_name,
            source_revision=get_source_revision(),
        )

    if gradnorm is not None:
        plot_gradnorm_trajectory(config, model_name)

    return {
        "best_val_ndcg": best_val_ndcg,
        "best_epoch": best_epoch,
        "epoch_mean_losses": epoch_mean_losses,
        "epoch_loss_windows": epoch_loss_windows,
    }
