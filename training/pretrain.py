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
    GradNormController, get_anchor_layer_params, probe_double_backward,
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
from utils import get_device, resolve_dtype

logger = logging.getLogger(__name__)


def build_scheduler(optimizer: torch.optim.Optimizer, config: Config, total_steps: int) -> LambdaLR:
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def build_model(config: Config) -> Tuple[TransformerModel, MoELayer]:
    device = get_device(config)
    dtype = resolve_dtype(config)
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
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: TransformerModel,
    moe: MoELayer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    gradnorm: Optional[GradNormController] = None,
    loss_fns: Optional[dict] = None,
) -> dict:
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
) -> Tuple[Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    device = get_device(config)
    mask_id = config.num_items + 1

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
        for i in range(B):
            real = seq[i][seq[i] != 0].tolist()
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

    for name, loss in active_losses.items():
        assert torch.isfinite(loss), f"{name} loss is not finite: {loss.item()}"

    # REV-022b: T5.2 requires dtype assertions on outputs as well as parameters.
    assert all_hidden.dtype == torch.bfloat16, (
        f"Model output dtype {all_hidden.dtype}, expected bfloat16"
    )
    for name, loss in active_losses.items():
        # The contrastive objective is computed on fp32-normalized embeddings
        # (InfoNCE's similarity matrix is numerically fragile in bf16); this is a
        # deliberate, documented exception recorded in design.md.
        expected = torch.float32 if name == "contrastive" else torch.bfloat16
        assert loss.dtype == expected, (
            f"Loss '{name}' dtype {loss.dtype}, expected {expected}"
        )

    if gradnorm is not None:
        L_total = gradnorm.weighted_sum(active_losses)
    else:
        L_total = static_weighted_sum(active_losses, config)

    assert torch.isfinite(L_total), "Total loss not finite"

    # REV-022a: code-quality.md requires set_to_none=True on the hot path.
    optimizer.zero_grad(set_to_none=True)
    L_total.backward(retain_graph=(gradnorm is not None))

    if gradnorm is not None:
        anchor_params = get_anchor_layer_params(model, config)
        gradnorm.update(active_losses, anchor_params, model=model)

    trainable = list(model.parameters()) + list(moe.parameters())
    for fn in loss_fns.values():
        trainable += list(fn.parameters())

    # REV-022c: FR-7 and AC-08 both require finite-value assertions on GRADIENTS.
    # A NaN appearing in gradients rather than in the loss scalar previously passed
    # every assertion and silently corrupted the weights.
    for p in trainable:
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Non-finite gradient detected"

    # REV-047: clipping must cover the loss functions' own parameters too.
    torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
    optimizer.step()
    scheduler.step()

    loss_values = {name: loss.detach().item() for name, loss in active_losses.items()}
    loss_values["total"] = L_total.detach().item()
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


def assert_bf16_params(model: TransformerModel, moe: MoELayer) -> None:
    for name, p in model.named_parameters():
        assert p.dtype == torch.bfloat16, f"Model param {name} has dtype {p.dtype}, expected bfloat16"
    for name, p in moe.named_parameters():
        if "log_lambdas" in name:
            continue
        assert p.dtype == torch.bfloat16, f"MoE param {name} has dtype {p.dtype}, expected bfloat16"


def assert_finite_step(loss_values: Dict[str, float], gradnorm: Optional[GradNormController]) -> None:
    for name, val in loss_values.items():
        if name.startswith("_"):
            continue
        assert math.isfinite(val), f"Loss '{name}' is not finite: {val}"
    if gradnorm is not None:
        lambdas = gradnorm.lambdas.detach()
        assert torch.all(torch.isfinite(lambdas)), f"GradNorm lambdas not finite: {lambdas}"


def setup_epoch_logger(config: Config, model_name: str) -> logging.Logger:
    log_path = os.path.join(config.log_dir, f"pretrain_{model_name.replace('-', '_')}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    epoch_logger = logging.getLogger(f"pretrain.{model_name}")
    epoch_logger.setLevel(logging.INFO)
    epoch_logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
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
    record = {"step": step, "lambdas": {name: lambdas[i].item() for i, name in enumerate(OBJECTIVE_ORDER)}}
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
    dtype = resolve_dtype(config)
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
    return save_routing_diagnostics(per_objective, config, model_name)


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
        targets=batch["targets"].detach().cpu().numpy(),
        user_ids=batch["user_ids"].detach().cpu().numpy(),
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


def run_pretrain(config: Config, model_name: str) -> Dict[str, float]:
    config.apply_model_config(model_name)
    config.validate()
    device = get_device(config)
    dtype = resolve_dtype(config)
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
    assert_bf16_params(model, moe)

    # REV-047: loss functions are built BEFORE the optimizer so their parameters are
    # registered in it.
    loss_fns = _build_loss_fns(config, device)
    optimizer = torch.optim.AdamW(
        collect_trainable_parameters(model, moe, loss_fns),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    scheduler = build_scheduler(optimizer, config, total_steps)

    _opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    for fn_name, fn in loss_fns.items():
        for pname, p in fn.named_parameters():
            assert id(p) in _opt_ids, (
                f"Loss parameter {fn_name}.{pname} is not registered in the optimizer"
            )

    gradnorm = None
    if config.use_gradnorm:
        gradnorm = GradNormController(config.gradnorm_alpha, config.gradnorm_lr)
        gradnorm.log_lambdas = torch.nn.Parameter(gradnorm.log_lambdas.data.to(device=device))
        gradnorm.optimizer = torch.optim.Adam([gradnorm.log_lambdas], lr=config.gradnorm_lr)
        # REV-092: GradNorm needs a double-backward through the anchor block's attention
        # kernel, which not every backend implements (SDPA on CPU does not). Probe it
        # here -- before the dataset load -- so an unsupported combination fails at
        # second zero with a legible message instead of on the first optimizer step.
        probe_double_backward(model, config)

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

    epoch_logger = setup_epoch_logger(config, model_name)

    best_val_ndcg = -1.0
    best_epoch = -1
    global_step = 0
    ckpt_dir = config.model_checkpoint_dir()
    os.makedirs(ckpt_dir, exist_ok=True)

    # REV-010: the buffer spans the diagnostic interval, not a single epoch, and is
    # cleared by `_emit_routing_diagnostics` once consumed.
    routing_buffer: Dict[str, List[torch.Tensor]] = {}
    diag_loader = build_dataloader(train_dataset, config, shuffle=True)
    epoch_loss_windows: List[Dict[str, float]] = []
    epoch_mean_losses: List[float] = []

    gradnorm_log_path = os.path.join(config.log_dir, f"gradnorm_{model_name.replace('-', '_')}.jsonl")
    if os.path.exists(gradnorm_log_path):
        os.remove(gradnorm_log_path)

    for epoch in range(config.pretrain_epochs):
        model.train()
        moe.train()
        epoch_losses: Dict[str, List[float]] = {}

        for batch_idx, batch in enumerate(train_loader):
            if config.max_steps_per_epoch > 0 and batch_idx >= config.max_steps_per_epoch:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            batch = attach_negatives(batch)

            loss_values, snapshots = train_one_step(
                model, moe, batch, config, loss_fns, gradnorm,
                optimizer, scheduler, batch_idx, rng,
                mask_rng=mask_rng, metadata_matrix=metadata_matrix_t,
            )

            assert_finite_step(loss_values, gradnorm)

            for name, val in loss_values.items():
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
                )
                optimizer.zero_grad(set_to_none=True)
                model.train()
                moe.train()

            if gradnorm is not None and global_step % 100 == 0:
                log_gradnorm_lambdas(config, model_name, global_step, gradnorm)

            global_step += 1

        avg_losses = {name: sum(vals) / len(vals) for name, vals in epoch_losses.items()}

        # AC-04 / REV-005: record the first- and last-window means of the per-batch total
        # loss so a one-epoch run can be checked for an actual decrease. Without this the
        # only observable was the whole-epoch mean, which cannot show a trend.
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

        logger.info(
            f"Epoch {epoch}: train_loss={avg_losses.get('total', 0.0):.4f} "
            f"val_ndcg={val_ndcg:.4f} best_ndcg={best_val_ndcg:.4f} (epoch {best_epoch})"
        )

    # evaluation-diagnostics FR-8: comprehensive end-of-training routing pass over the
    # validation set, which supersedes whatever residue is left in the interval buffer.
    routing_buffer.clear()
    val_routing = collect_validation_routing(model, moe, val_dataset, config)
    if val_routing:
        save_routing_diagnostics(val_routing, config, model_name)

    if gradnorm is not None:
        plot_gradnorm_trajectory(config, model_name)

    return {
        "best_val_ndcg": best_val_ndcg,
        "best_epoch": best_epoch,
        "epoch_mean_losses": epoch_mean_losses,
        "epoch_loss_windows": epoch_loss_windows,
    }
