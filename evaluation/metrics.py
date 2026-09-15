import math
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from configs.config import Config
from models.transformer import l2_normalize


def compute_metrics_batch(
    scores: torch.Tensor, target_idx: int = 0, k: int = 10
) -> Dict[str, torch.Tensor]:
    """Ranking metrics for a `(B, C)` score matrix, one row per example.

    REV-076: the scalar path sorted the full candidate vector and then called `.item()`
    to locate the target, forcing a device synchronisation once per example -- ~1M syncs
    per evaluation pass. The target's rank is `1 + |{candidates scoring strictly higher}|`,
    which needs no sort and no sync, and ties resolve in the target's favour exactly as
    a descending sort that places equal scores after it would.
    """
    if scores.dim() != 2:
        raise ValueError(f"compute_metrics_batch expects (B, C) scores, got {tuple(scores.shape)}")
    scores = scores.float()
    target_scores = scores[:, target_idx : target_idx + 1]
    rank = (scores > target_scores).sum(dim=1) + 1
    rank_f = rank.float()
    hit = (rank <= k).float()

    return {
        "ndcg": hit / torch.log2(rank_f + 1.0),
        "mrr": hit / rank_f,
        "recall": hit,
        "precision": hit / k,
    }


def compute_metrics(scores: torch.Tensor, target_idx: int = 0, k: int = 10) -> Dict[str, float]:
    """Single-example ranking metrics. Kept for toy tests and per-example callers."""
    batched = compute_metrics_batch(scores.reshape(1, -1), target_idx=target_idx, k=k)
    return {name: float(val[0].item()) for name, val in batched.items()}


def aggregate_metrics(
    per_example: Union[List[Dict[str, float]], Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, float]]:
    """Mean/std per metric, over either a list of per-example dicts or stacked tensors."""
    if isinstance(per_example, dict):
        result = {}
        for key, t in per_example.items():
            t = t.detach().float().cpu()
            result[key] = {
                "mean": t.mean().item() if t.numel() else 0.0,
                "std": t.std().item() if t.numel() > 1 else 0.0,
            }
        return result

    if not per_example:
        return {}

    keys = per_example[0].keys()
    result = {}
    for key in keys:
        vals = [ex[key] for ex in per_example]
        t = torch.tensor(vals)
        result[key] = {
            "mean": t.mean().item(),
            "std": t.std().item() if len(vals) > 1 else 0.0,
        }
    return result


def score_candidates_batch(
    model: torch.nn.Module,
    seqs: torch.Tensor,
    candidate_ids: torch.Tensor,
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Score `(B, C)` candidates for `(B, S)` sequences in ONE forward pass.

    REV-063: evaluation used to forward a batch of exactly 1 per example. At 2.26 ms per
    example the 993,372-example validation split cost ~37 minutes per pass, so a 20-epoch
    run spent ~12 hours in validation alone and the full-scale gates could not be
    demonstrated. Everything here is the same arithmetic as the single-example path, run
    over a batch dimension.
    """
    model.eval()
    with torch.no_grad():
        # The eval datasets are held on CPU while the model lives on the accelerator, so
        # move each batch across here. CUDA rejects a cross-device index outright, and
        # MPS rejects it as "Passed CPU tensor to MPS op" -- this is the single place
        # every scoring caller (validation, test, cold-start, long-tail) funnels through.
        param_device = next(model.parameters()).device
        seqs = seqs.to(param_device)
        candidate_ids = candidate_ids.to(param_device)
        if metadata is not None:
            metadata = metadata.to(device=param_device, dtype=model.item_emb_weight.dtype)

        all_hidden = model(seqs, metadata=metadata, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]

        if moe_layer is not None and moe_layer.variant != "none":
            # REV-045: MoELayer invalidates its expert cache on a new input, so scoring
            # batch N no longer reuses batch 0's expert outputs.
            Z_ut, _ = moe_layer(all_hidden, pooled, 0)
            user_emb = Z_ut[:, -1, :]
        else:
            user_emb = pooled

        item_embs = model.item_emb_weight[candidate_ids]  # (B, C, D)

        # REV-044: FR-8 / AC-07 -- L2-normalize both sides before the dot product so
        # scores are bounded to [-1, 1]. This is the production scoring path.
        if config.normalize_embeddings:
            user_emb = l2_normalize(user_emb)
            item_embs = l2_normalize(item_embs)

        scores = torch.einsum("bd,bcd->bc", user_emb, item_embs)

        if temperature != 1.0:
            scores = scores / temperature

    return scores


def score_candidates(
    model: torch.nn.Module,
    seq: torch.Tensor,
    candidate_ids: torch.Tensor,
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Single-example scoring. Thin wrapper over the batched path."""
    if metadata is not None and metadata.dim() == 2:
        metadata = metadata.unsqueeze(0)
    scores = score_candidates_batch(
        model, seq.unsqueeze(0), candidate_ids.unsqueeze(0), config,
        moe_layer=moe_layer, metadata=metadata, temperature=temperature,
    )
    return scores.squeeze(0)


def _stack_sequences(eval_sequences, indices) -> torch.Tensor:
    if isinstance(eval_sequences, torch.Tensor):
        return eval_sequences[indices[0] : indices[-1] + 1]
    return torch.stack([
        s if isinstance(s, torch.Tensor) else torch.as_tensor(s, dtype=torch.long)
        for s in (eval_sequences[i] for i in indices)
    ]).long()


def _stack_candidates(candidate_sets, indices) -> torch.Tensor:
    if isinstance(candidate_sets, torch.Tensor):
        return candidate_sets[indices[0] : indices[-1] + 1].long()
    return torch.stack([
        c if isinstance(c, torch.Tensor) else torch.as_tensor(c, dtype=torch.long)
        for c in (candidate_sets[i] for i in indices)
    ]).long()


def evaluate_model(
    model: torch.nn.Module,
    eval_sequences: Union[torch.Tensor, Sequence[torch.Tensor]],
    candidate_sets: Union[torch.Tensor, Sequence[torch.Tensor]],
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata_per_example: Optional[Sequence[torch.Tensor]] = None,
    metadata_matrix: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Batched evaluation over an aligned set of sequences and candidate sets.

    `eval_sequences` and `candidate_sets` may each be a single stacked `(N, S)` / `(N, C)`
    tensor -- which is what `load_eval_candidates` and `RecDataset` already hold -- or a
    sequence of per-example tensors.

    REV-058 / REV-063: prefer `metadata_matrix`, the `(vocab, F)` matrix, over
    `metadata_per_example`. Passing the matrix lets each batch gather only its own
    `(B, S, F)` block instead of the caller materialising a dense `(N, S, F)` list.
    """
    temperature = config.eval_temperature if config.use_logit_calibration else 1.0
    n = eval_sequences.shape[0] if isinstance(eval_sequences, torch.Tensor) else len(eval_sequences)
    if n == 0:
        return {}

    bs = batch_size or max(1, config.eval_batch_size)
    collected: Dict[str, List[torch.Tensor]] = {}

    for start in range(0, n, bs):
        indices = list(range(start, min(start + bs, n)))
        seqs = _stack_sequences(eval_sequences, indices)
        cands = _stack_candidates(candidate_sets, indices)

        meta = None
        if metadata_matrix is not None:
            meta = metadata_matrix[seqs.to(metadata_matrix.device)]
        elif metadata_per_example is not None:
            meta = torch.stack([
                torch.as_tensor(metadata_per_example[i]) for i in indices
            ])

        scores = score_candidates_batch(
            model, seqs, cands, config, moe_layer=moe_layer,
            metadata=meta, temperature=temperature,
        )
        for name, val in compute_metrics_batch(scores, target_idx=0, k=config.eval_k).items():
            collected.setdefault(name, []).append(val.detach().cpu())

    return aggregate_metrics({name: torch.cat(vals) for name, vals in collected.items()})
