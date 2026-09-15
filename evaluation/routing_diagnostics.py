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
    # conventions.md: CLI model names are "t-small", file names are "t_small".
    # The plots already normalized; the JSON report did not, so the two halves of
    # the same report landed under different spellings.
    mn = model_name.replace("-", "_")
    path = os.path.join(log_dir, f"routing_diagnostics_{mn}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    plot_routing_diagnostics(report, config, model_name)
    return report


def plot_routing_diagnostics(report: dict, config, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(config.plot_dir, exist_ok=True)
    mn = model_name.replace("-", "_")

    per_obj = report.get("per_objective", {})
    if per_obj:
        names = list(per_obj.keys())
        entropies = [per_obj[n]["entropy_mean"] for n in names]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, entropies)
        ax.set_ylabel("Mean Entropy")
        ax.set_title("Routing Entropy by Objective")
        fig.tight_layout()
        fig.savefig(os.path.join(config.plot_dir, f"routing_entropy_{mn}.png"), dpi=config.plot_dpi)
        plt.close(fig)

    utilization = report.get("overall_utilization", [])
    if utilization:
        K = len(utilization)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(K), utilization)
        ax.axhline(y=1.0 / K, color="red", linestyle="--", label=f"1/K = {1.0/K:.3f}")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Utilization")
        ax.set_title("Expert Utilization")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(config.plot_dir, f"expert_utilization_{mn}.png"), dpi=config.plot_dpi)
        plt.close(fig)

    heatmap = report.get("heatmap")
    if heatmap:
        hm = np.array(heatmap)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(hm, aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Objective")
        obj_names = [n for n in OBJECTIVE_ORDER if n in per_obj]
        ax.set_yticks(range(len(obj_names)))
        ax.set_yticklabels(obj_names)
        ax.set_xticks(range(hm.shape[1]))
        ax.set_title("Objective-Expert Heatmap")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(config.plot_dir, f"objective_expert_heatmap_{mn}.png"), dpi=config.plot_dpi)
        plt.close(fig)
