import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from configs.config import Config


def compute_metrics(scores: torch.Tensor, target_idx: int = 0, k: int = 10) -> Dict[str, float]:
    ranked = scores.argsort(descending=True)
    rank_pos = (ranked == target_idx).nonzero(as_tuple=True)[0]
    rank = rank_pos.item() + 1

    return {
        "ndcg": 1.0 / math.log2(rank + 1) if rank <= k else 0.0,
        "mrr": 1.0 / rank if rank <= k else 0.0,
        "recall": 1.0 if rank <= k else 0.0,
        "precision": 1.0 / k if rank <= k else 0.0,
    }


def aggregate_metrics(per_example: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
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


def score_candidates(
    model: torch.nn.Module,
    seq: torch.Tensor,
    candidate_ids: torch.Tensor,
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        all_hidden = model(seq.unsqueeze(0), metadata=metadata, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]

        if moe_layer is not None and moe_layer.variant != "none":
            Z_ut, _ = moe_layer(all_hidden, pooled, 0)
            user_emb = Z_ut[:, -1, :]
        else:
            user_emb = pooled

        item_embs = model.item_emb_weight[candidate_ids]
        scores = (user_emb @ item_embs.T).squeeze(0)

        if temperature != 1.0:
            scores = scores / temperature

    return scores


def evaluate_model(
    model: torch.nn.Module,
    eval_sequences: List[torch.Tensor],
    candidate_sets: List[torch.Tensor],
    config: Config,
    moe_layer: Optional[torch.nn.Module] = None,
    metadata_per_example: Optional[List[torch.Tensor]] = None,
) -> Dict[str, Dict[str, float]]:
    temperature = config.eval_temperature if config.use_logit_calibration else 1.0
    all_metrics = []

    for i, (seq, candidates) in enumerate(zip(eval_sequences, candidate_sets)):
        meta = metadata_per_example[i] if metadata_per_example is not None else None
        scores = score_candidates(model, seq, candidates, config, moe_layer, meta, temperature)
        metrics = compute_metrics(scores, target_idx=0, k=config.eval_k)
        all_metrics.append(metrics)

    return aggregate_metrics(all_metrics)
