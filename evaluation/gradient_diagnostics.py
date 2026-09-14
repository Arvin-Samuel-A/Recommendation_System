import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_cosine_similarity(
    per_objective_losses: Dict[str, torch.Tensor],
    shared_params: List[nn.Parameter],
) -> torch.Tensor:
    names = list(per_objective_losses.keys())
    T = len(names)

    grads = {}
    for name, loss in per_objective_losses.items():
        g = torch.autograd.grad(
            loss, shared_params, retain_graph=True, allow_unused=True,
        )
        flat = torch.cat([
            gi.flatten() if gi is not None else torch.zeros_like(p).flatten()
            for gi, p in zip(g, shared_params)
        ])
        grads[name] = flat

    cos_sim = torch.zeros(T, T)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            cos_sim[i, j] = F.cosine_similarity(
                grads[ni].unsqueeze(0), grads[nj].unsqueeze(0),
            ).item()

    return cos_sim


def per_objective_grad_norms(
    per_objective_losses: Dict[str, torch.Tensor],
    shared_params: List[nn.Parameter],
) -> Dict[str, float]:
    norms = {}
    for name, loss in per_objective_losses.items():
        g = torch.autograd.grad(
            loss, shared_params, retain_graph=True, allow_unused=True,
        )
        flat = torch.cat([
            gi.flatten() if gi is not None else torch.zeros_like(p).flatten()
            for gi, p in zip(g, shared_params)
        ])
        norms[name] = flat.norm(p=2).item()

    vals = list(norms.values())
    t = torch.tensor(vals)
    mean_norm = t.mean().item()
    std_norm = t.std().item() if len(vals) > 1 else 0.0
    cv = std_norm / (mean_norm + 1e-12)

    return {
        "per_objective": norms,
        "mean": mean_norm,
        "std": std_norm,
        "coefficient_of_variation": cv,
    }


def save_gradient_diagnostics(
    cos_sim: torch.Tensor,
    grad_norms: dict,
    objective_names: List[str],
    config,
    model_name: str,
) -> dict:
    report = {
        "cosine_similarity": cos_sim.tolist(),
        "objective_names": objective_names,
        "gradient_norms": grad_norms,
    }

    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"gradient_diagnostics_{model_name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    return report
