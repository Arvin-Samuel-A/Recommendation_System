"""Next-item-protected gradient projection (an asymmetric PCGrad).

Each auxiliary objective's gradient that conflicts with the primary objective's (negative
dot product) loses its component along the primary gradient; one that agrees is kept
as is. The primary gradient itself is never changed. Every auxiliary is projected against
the ORIGINAL primary gradient, independently of the others, so the result does not depend
on the order of the objectives.

The dot products run over every trainable parameter. A parameter only one objective
reaches (an objective-specific gate, say) has a zero entry in every other objective's
gradient, so it adds nothing to a dot product with them: the projection acts on the
shared parameters, as PCGrad's does.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from configs.config import OBJECTIVE_ORDER

PROJECTION_MODES = ("", "next_item")


def static_objective_weights(config) -> Dict[str, float]:
    """The fixed objective weights `static_weighted_sum` applies (next-item at 1)."""
    return {
        "next_item": 1.0,
        "masked_item": config.masked_lm_weight,
        "multi_step": config.multi_step_weight,
        "contrastive": config.contrastive_weight,
    }


def project_conflicting_gradients(
    active_losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    params: List[nn.Parameter],
    primary: str = "next_item",
) -> Dict[str, torch.Tensor]:
    """Write the projected sum of the weighted objectives' gradients into `p.grad`.

    Takes one backward pass per active objective. Returns, per auxiliary objective, the
    cosine between its weighted gradient and the primary's before projection
    (`proj_cos_<name>`) and whether it was projected (`proj_frac_<name>`, 1.0 or 0.0), as
    0-d tensors left on the device.
    """
    if primary not in active_losses:
        raise ValueError(f"primary objective '{primary}' is not active this step")
    names = [n for n in OBJECTIVE_ORDER if n in active_losses]
    grads: Dict[str, List[torch.Tensor]] = {}
    for i, name in enumerate(names):
        g = torch.autograd.grad(
            weights[name] * active_losses[name].float(), params,
            retain_graph=i < len(names) - 1, allow_unused=True,
        )
        grads[name] = [torch.zeros_like(p) if gi is None else gi for p, gi in zip(params, g)]

    g_p = grads[primary]
    pp = torch.stack([(g.float() * g.float()).sum() for g in g_p]).sum()
    total = [g.clone() for g in g_p]
    stats: Dict[str, torch.Tensor] = {}
    for name in names:
        if name == primary:
            continue
        g_a = grads[name]
        dot = torch.stack([(a.float() * b.float()).sum() for a, b in zip(g_a, g_p)]).sum()
        aa = torch.stack([(a.float() * a.float()).sum() for a in g_a]).sum()
        # Branch-free (no host sync): the coefficient is 0 when the two agree.
        coef = torch.clamp(dot, max=0.0) / (pp + 1e-20)
        for t, a, b in zip(total, g_a, g_p):
            t.add_(a - coef.to(a.dtype) * b)
        stats[f"proj_cos_{name}"] = (dot / (aa.sqrt() * pp.sqrt() + 1e-20)).detach()
        stats[f"proj_frac_{name}"] = (dot < 0).float().detach()
    for p, t in zip(params, total):
        p.grad = t
    return stats
