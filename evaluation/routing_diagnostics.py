import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from configs.config import OBJECTIVE_ORDER


def routing_entropy(r: torch.Tensor) -> torch.Tensor:
    return -(r * torch.log(r.clamp(min=1e-12))).sum(dim=-1)


def expert_utilization(r: torch.Tensor) -> torch.Tensor:
    return r.mean(dim=0)


def objective_routing_diff(
    per_objective_routing: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    names = list(per_objective_routing.keys())
    means = {n: r.mean(dim=0) for n, r in per_objective_routing.items()}

    T = len(names)
    K = next(iter(means.values())).shape[0]

    cos_sim = torch.zeros(T, T)
    kl_div = torch.zeros(T, T)

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            mi = means[ni]
            mj = means[nj]
            cos_sim[i, j] = F.cosine_similarity(mi.unsqueeze(0), mj.unsqueeze(0)).item()
            kl_div[i, j] = F.kl_div(
                mj.clamp(min=1e-12).log(), mi, reduction="sum",
            ).item()

    return {"cosine_similarity": cos_sim, "kl_divergence": kl_div, "objective_names": names}


def collapse_tests(
    r: torch.Tensor,
    cv_threshold: float = 0.1,
    domination_threshold: float = 0.9,
) -> Dict[str, float]:
    mean_r = r.mean(dim=-1)
    std_r = r.std(dim=-1)
    cv = std_r / (mean_r + 1e-12)
    frac_uniform = (cv < cv_threshold).float().mean().item()

    max_r = r.max(dim=-1).values
    frac_dominated = (max_r > domination_threshold).float().mean().item()

    return {
        "fraction_near_uniform": frac_uniform,
        "fraction_single_expert_dominated": frac_dominated,
    }


def objective_expert_heatmap(
    per_objective_routing: Dict[str, torch.Tensor],
) -> torch.Tensor:
    ordered = []
    for name in OBJECTIVE_ORDER:
        if name in per_objective_routing:
            ordered.append(per_objective_routing[name].mean(dim=0))
    return torch.stack(ordered)


def save_routing_diagnostics(
    per_objective_routing: Dict[str, torch.Tensor],
    config,
    model_name: str,
) -> dict:
    report = {}

    all_r = torch.cat(list(per_objective_routing.values()), dim=0)
    report["overall_entropy_mean"] = routing_entropy(all_r).mean().item()
    report["overall_utilization"] = expert_utilization(all_r).tolist()

    per_obj = {}
    for name, r in per_objective_routing.items():
        per_obj[name] = {
            "entropy_mean": routing_entropy(r).mean().item(),
            "utilization": expert_utilization(r).tolist(),
        }
    report["per_objective"] = per_obj

    diff = objective_routing_diff(per_objective_routing)
    report["objective_cosine_similarity"] = diff["cosine_similarity"].tolist()
    report["objective_kl_divergence"] = diff["kl_divergence"].tolist()

    collapse = collapse_tests(all_r)
    report["collapse"] = collapse

    heatmap = objective_expert_heatmap(per_objective_routing)
    report["heatmap"] = heatmap.tolist()
    report["heatmap_shape"] = list(heatmap.shape)

    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"routing_diagnostics_{model_name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    return report
