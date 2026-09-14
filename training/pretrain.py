import json
import logging
import math
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from configs.config import Config, OBJECTIVE_ORDER
from models.transformer import TransformerModel
from models.moe import MoELayer
from training.losses import (
    SampledSoftmaxLoss,
    MaskedItemLoss,
    MultiStepLoss,
    InfoNCELoss,
    apply_masking,
    static_weighted_sum,
)
from training.gradnorm import GradNormController, get_anchor_layer_params
from data.augmentation import augment_sequence
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
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "moe_state_dict": moe.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "gradnorm_state_dict": gradnorm.state_dict() if gradnorm else None,
        "val_ndcg": val_metrics.get("ndcg", {}).get("mean", 0.0),
        "val_mrr": val_metrics.get("mrr", {}).get("mean", 0.0),
        "val_recall": val_metrics.get("recall", {}).get("mean", 0.0),
        "val_precision": val_metrics.get("precision", {}).get("mean", 0.0),
        "config": asdict(config),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: TransformerModel,
    moe: MoELayer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    gradnorm: Optional[GradNormController] = None,
) -> dict:
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    moe.load_state_dict(checkpoint["moe_state_dict"])
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
) -> Tuple[Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    device = get_device(config)
    mask_id = config.num_items + 1

    seq = batch["sequences"]
    targets = batch["targets"]
    neg_ids = batch["neg_ids"]
    targets_k = batch.get("targets_k")
    metadata = batch.get("metadata")

    if config.use_masked_lm:
        masked_seq, mask_positions, true_ids = apply_masking(seq, config.mask_prob, mask_id)
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

    if config.use_masked_lm and mask_positions is not None:
        Z_mim, r_mim = moe(all_hidden, pooled, 1)
        active_losses["masked_item"] = loss_fns["masked_item"](Z_mim, mask_positions, true_ids, model.item_emb_weight)
        if r_mim is not None:
            routing_snapshots["masked_item"] = r_mim.detach()

    if config.use_multi_step and targets_k is not None:
        Z_ms, r_ms = moe(all_hidden, pooled, 2)
        active_losses["multi_step"] = loss_fns["multi_step"](Z_ms, targets_k, model.item_emb_weight)
        if r_ms is not None:
            routing_snapshots["multi_step"] = r_ms.detach()

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
        aug_seq_t = torch.tensor(aug_seqs, device=device)

        aug_hidden = model(aug_seq_t, return_all_hidden=True)
        aug_pooled = aug_hidden[:, -1, :]
        moe.clear_cache()
        Z_aug, _ = moe(aug_hidden, aug_pooled, 3)
        z_aug = torch.nn.functional.normalize(Z_aug[:, -1, :].float(), p=2, dim=-1)

        active_losses["contrastive"] = loss_fns["contrastive"](z_orig.to(seq.dtype if seq.is_floating_point() else z_orig.dtype), z_aug.to(z_orig.dtype))
        if r_cl is not None:
            routing_snapshots["contrastive"] = r_cl.detach()

    for name, loss in active_losses.items():
        assert torch.isfinite(loss), f"{name} loss is not finite: {loss.item()}"

    if gradnorm is not None:
        L_total = gradnorm.weighted_sum(active_losses)
    else:
        L_total = static_weighted_sum(active_losses, config)

    assert torch.isfinite(L_total), f"Total loss not finite"

    optimizer.zero_grad()
    L_total.backward(retain_graph=(gradnorm is not None))

    if gradnorm is not None:
        anchor_params = get_anchor_layer_params(model)
        gradnorm.update(active_losses, anchor_params)

    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(moe.parameters()),
        config.max_grad_norm,
    )
    optimizer.step()
    scheduler.step()

    loss_values = {name: loss.detach().item() for name, loss in active_losses.items()}
    loss_values["total"] = L_total.detach().item()

    return loss_values, routing_snapshots if routing_snapshots else None
