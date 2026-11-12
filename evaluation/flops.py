import json
import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from configs.config import Config
from data.sequence_generator import load_metadata_matrix
from models.moe import MoELayer
from utils import autocast_context, get_device

logger = logging.getLogger(__name__)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(
    model: torch.nn.Module,
    moe_layer: Optional[MoELayer],
    config: Config,
    metadata_matrix: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    device = get_device(config)
    model.eval()
    if moe_layer is not None:
        moe_layer.eval()

    # REV-035b: this used to be `torch.zeros(...)` -- an all-PADDING sequence. Because
    # `TransformerModel.forward` computes `padding_mask = (seq != 0)`, an all-zero input
    # makes `has_padding` True, so every block took the explicit-mask branch and built an
    # (B, H, S, S) float mask. A real serving request is full of valid item IDs and takes
    # the `is_causal=True` fast path, so FR-6's "full serving path" was never the path
    # being timed. Build the request from valid IDs instead.
    num_items = max(int(getattr(config, "num_items", 0) or 0), 0)
    if num_items > 0:
        generator = torch.Generator().manual_seed(config.seed)
        dummy_seq = torch.randint(
            1, num_items + 1, (1, config.max_seq_length),
            generator=generator, dtype=torch.long,
        ).to(device)
    else:
        # No vocabulary resolved yet. Item ID 1 is still a valid, non-padding token, so
        # the timed path stays the serving path.
        dummy_seq = torch.ones(1, config.max_seq_length, dtype=torch.long, device=device)
    # REV-110: this was `torch.zeros(...)`. A real request gathers `metadata_matrix[seq]`,
    # whose rows are dense genome-relevance floats; an all-zero block is exactly what a
    # metadata-FREE model produces, which left the timed input ambiguous between the two
    # configurations on a path FR-6 calls the "full serving path". Gather from the real
    # matrix when one is available and fall back to noise otherwise -- never zeros. The
    # matmul shape is unchanged either way, so `mean_ms` is not materially affected; the
    # point is that the timed input is the serving input.
    dummy_meta = None
    metadata_source = None
    if config.use_metadata and config.feature_dim > 0:
        if metadata_matrix is None:
            metadata_matrix = load_metadata_matrix(config.processed_data_dir, config)
        if metadata_matrix is not None:
            gathered = metadata_matrix[dummy_seq.to(metadata_matrix.device)]
            dummy_meta = gathered.to(device=device, dtype=torch.bfloat16)
            metadata_source = "item_metadata"
        else:
            generator = torch.Generator().manual_seed(config.seed)
            dummy_meta = torch.randn(
                1, config.max_seq_length, config.feature_dim,
                generator=generator, dtype=torch.float32,
            ).to(device=device, dtype=torch.bfloat16)
            metadata_source = "randn"

    # specs/eval-integrity FR-3.2: each warm-up and timed forward owns an autocast region
    # (inside the timed window, so latency includes the casts, as in serving).
    region_device = next(model.parameters()).device.type
    for _ in range(config.warmup_runs):
        with torch.no_grad(), autocast_context(config, device_type=region_device):
            all_hidden = model(dummy_seq, metadata=dummy_meta, return_all_hidden=True)
            pooled = all_hidden[:, -1, :]
            if moe_layer is not None and moe_layer.variant != "none":
                moe_layer.clear_cache()
                moe_layer(all_hidden, pooled, 0)

    _synchronize(device)
    times = []
    for _ in range(config.timed_runs):
        _synchronize(device)
        start = time.perf_counter()
        with torch.no_grad(), autocast_context(config, device_type=region_device):
            all_hidden = model(dummy_seq, metadata=dummy_meta, return_all_hidden=True)
            pooled = all_hidden[:, -1, :]
            if moe_layer is not None and moe_layer.variant != "none":
                moe_layer.clear_cache()
                moe_layer(all_hidden, pooled, 0)
        _synchronize(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times_arr = np.array(times)
    # REV-035: this was `times_arr.mean()`, identical to mean_ms. That is only correct
    # for a batch of 1, and encoded that assumption implicitly -- if the benchmark input
    # ever changed shape the field became silently wrong. Divide by the number of
    # recommendations actually produced per timed run.
    recs_per_run = int(dummy_seq.shape[0])
    return {
        "mean_ms": float(times_arr.mean()),
        "std_ms": float(times_arr.std()),
        "min_ms": float(times_arr.min()),
        "p95_ms": float(np.percentile(times_arr, 95)),
        "recommendations_per_run": recs_per_run,
        "ms_per_recommendation": float(times_arr.mean()) / recs_per_run,
        # REV-035b: record the shape that was actually timed, so a reader can tell which
        # attention path the numbers describe.
        "timed_sequence_length": int(dummy_seq.shape[1]),
        "timed_padding_positions": int((dummy_seq == 0).sum().item()),
        "timed_with_metadata": dummy_meta is not None,
        # REV-110: which metadata the numbers describe -- real gathered rows or the
        # noise fallback when `item_metadata.npy` is not on disk.
        "timed_metadata_source": metadata_source,
    }


def measure_routing_overhead(
    model: torch.nn.Module,
    moe_full: MoELayer,
    moe_none: MoELayer,
    config: Config,
) -> Dict[str, float]:
    full_stats = benchmark_latency(model, moe_full, config)
    none_stats = benchmark_latency(model, moe_none, config)
    return {
        "full_mean_ms": full_stats["mean_ms"],
        "none_mean_ms": none_stats["mean_ms"],
        "overhead_ms": full_stats["mean_ms"] - none_stats["mean_ms"],
    }


def estimate_flops_per_inference(config: Config, moe_layer: Optional[MoELayer] = None) -> int:
    """Analytic forward-pass FLOPs for one sequence (multiply-add counted as 2).

    REV-021b: FR-7 names `flops_per_inference`, and ablations-cli FR-14's
    `accuracy_vs_flops.png` has no other source for it. Counts the dominant terms:
    per-block attention projections and score/context matmuls, the FFN, the final
    item-score projection, and the MoE experts plus router when one is present.
    """
    S = config.max_seq_length
    D = config.hidden_dim
    L = config.num_layers
    F = D * config.ffn_expansion

    # 4 projections (q, k, v, out): S x D x D each
    attn_proj = 4 * 2 * S * D * D
    # scores (S x S x D) and context (S x S x D)
    attn_mat = 2 * 2 * S * S * D
    ffn = 2 * 2 * S * D * F
    per_block = attn_proj + attn_mat + ffn
    total = L * per_block

    # tied-weight scoring against the full item table
    total += 2 * D * config.vocab_size

    if moe_layer is not None and getattr(moe_layer, "variant", "none") != "none":
        K = config.num_experts
        # each expert is a two-layer MLP applied to every position
        expert_hidden = D * config.expert_expansion
        total += K * 2 * (2 * S * D * expert_hidden)
        # router gate over the pooled state
        total += 2 * D * K

    return int(total)


def count_parameters(model: torch.nn.Module, moe_layer: Optional[MoELayer] = None) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    router_params = 0
    expert_params = 0
    if moe_layer is not None:
        for name, p in moe_layer.named_parameters():
            if "router" in name or "objective_embeddings" in name or "gates" in name:
                router_params += p.numel()
            elif "experts" in name or "shared_experts" in name:
                expert_params += p.numel()

    moe_total = sum(p.numel() for p in moe_layer.parameters()) if moe_layer else 0

    all_params = total + moe_total
    return {
        "total_parameters": all_params,
        "trainable_parameters": trainable + (sum(p.numel() for p in moe_layer.parameters() if p.requires_grad) if moe_layer else 0),
        "router_parameters": router_params,
        "expert_parameters": expert_params,
        # REV-021b: FR-7 requires BOTH sizes as named fields. `model_size_mb` was a
        # single bf16-only number, which read as fp32 to anyone not checking the source.
        "model_size_bf16_mb": all_params * 2 / (1024 * 1024),
        "model_size_fp32_mb": all_params * 4 / (1024 * 1024),
        "model_size_mb": all_params * 2 / (1024 * 1024),  # retained: bf16, as before
    }


def save_efficiency_report(
    config: Config,
    model_name: str,
    latency: Dict[str, float],
    param_counts: Dict[str, int],
    overhead: Optional[Dict[str, float]] = None,
    source_revision: Optional[Dict] = None,
) -> None:
    os.makedirs(config.log_dir, exist_ok=True)
    mn = model_name.replace("-", "_")
    report = {**latency, **{k: int(v) if isinstance(v, (int, np.integer)) else float(v) for k, v in param_counts.items()}}
    # REV-021b: FR-7 names these alongside the sizes; they identify which configuration
    # the numbers describe without having to cross-reference a run config.
    report["moe_variant"] = config.moe_variant
    report["num_experts"] = int(config.num_experts)
    report["model_name"] = model_name
    if overhead:
        report["routing_overhead"] = overhead
    # AC-09: echo the source_revision of the checkpoint these numbers came from.
    if source_revision is not None:
        report["source_revision"] = source_revision
    path = os.path.join(config.log_dir, f"efficiency_{mn}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
