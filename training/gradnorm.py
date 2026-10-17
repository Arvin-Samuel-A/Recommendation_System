from typing import Dict, List, Optional

import torch
import torch.nn as nn

from configs.config import OBJECTIVE_ORDER


ANCHOR_CHOICES = ("last_shared_transformer", "all_transformer_blocks")


def get_anchor_layer_params(model: nn.Module, config=None) -> List[nn.Parameter]:
    """Resolve the GradNorm anchor parameters.

    REV-055: T3.3 specifies a config-driven anchor. `"last_shared_transformer"` (the
    default) is the last Transformer block, which is shared by every MoE variant because
    the MoE layer is a separate module.
    """
    anchor = getattr(config, "gradnorm_anchor", "last_shared_transformer") if config else \
        "last_shared_transformer"
    if anchor not in ANCHOR_CHOICES:
        raise ValueError(
            f"Unknown gradnorm_anchor '{anchor}'. Choose from {list(ANCHOR_CHOICES)}"
        )
    if anchor == "all_transformer_blocks":
        return [p for p in model.blocks.parameters() if p.requires_grad]
    last_block = model.blocks[-1]
    return [p for p in last_block.parameters() if p.requires_grad]


RENORM_SCOPES = ("active", "enabled")


def enabled_objectives(config) -> List[str]:
    """The objectives a config trains, in OBJECTIVE_ORDER (next_item is always on)."""
    flags = {
        "next_item": True,
        "masked_item": config.use_masked_lm,
        "multi_step": config.use_multi_step,
        "contrastive": config.use_contrastive,
    }
    return [name for name in OBJECTIVE_ORDER if flags[name]]


class GradNormController:
    def __init__(
        self, alpha: float, lr: float, renorm_scope: str = "active",
        enabled_objectives: Optional[List[str]] = None,
    ):
        """`renorm_scope`: which lambdas `_renormalize` rescales after each update.

        "active" (default, the original behaviour): only this step's active objectives,
        to sum to their count. "enabled": every objective in `enabled_objectives`
        (default: all of OBJECTIVE_ORDER), active this step or not, to sum to their
        count -- so an intermittent objective's lambda is not reset by the steps it
        skips. Disabled objectives keep lambda 1 either way and are never rescaled.
        """
        if renorm_scope not in RENORM_SCOPES:
            raise ValueError(
                f"Unknown renorm_scope '{renorm_scope}'. Choose from {list(RENORM_SCOPES)}"
            )
        self.alpha = alpha
        self.num_objectives = len(OBJECTIVE_ORDER)
        self.log_lambdas = nn.Parameter(torch.zeros(self.num_objectives))
        self.optimizer = torch.optim.Adam([self.log_lambdas], lr=lr)
        self.initial_losses: Dict[str, float] = {}
        self._obj_to_idx = {name: i for i, name in enumerate(OBJECTIVE_ORDER)}
        self.renorm_scope = renorm_scope
        enabled = list(OBJECTIVE_ORDER) if enabled_objectives is None else list(enabled_objectives)
        unknown = [n for n in enabled if n not in self._obj_to_idx]
        if unknown:
            raise ValueError(f"Unknown objectives {unknown}; choose from {OBJECTIVE_ORDER}")
        self._enabled_indices = [self._obj_to_idx[n] for n in OBJECTIVE_ORDER if n in enabled]

    @property
    def lambdas(self) -> torch.Tensor:
        return self.log_lambdas.exp()

    def weighted_sum(self, active_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.log_lambdas.device, dtype=torch.float32)
        for name, loss in active_losses.items():
            idx = self._obj_to_idx[name]
            lam = self.lambdas[idx]
            total = total + lam * loss.float()
        return total

    def update(
        self,
        active_losses: Dict[str, torch.Tensor],
        anchor_params: List[nn.Parameter],
    ) -> Optional[torch.Tensor]:
        if len(active_losses) == 0:
            return None

        for name, loss in active_losses.items():
            if name not in self.initial_losses:
                self.initial_losses[name] = loss.detach().float().item()

        active_names = list(active_losses.keys())
        active_indices = [self._obj_to_idx[n] for n in active_names]
        num_active = len(active_names)

        # REV-120 / design D7: G_t is built in its FACTORED form, `lambda_t * ||grad_W L_t||`,
        # with the norm detached and `create_graph=False`. lambda_t is a scalar held in
        # `log_lambdas` with no dependence on the anchor parameters W, so
        # `grad_W(lambda_t * L_t) == lambda_t * grad_W L_t` exactly -- this is a change of
        # expression, not of algorithm (verified bit-identical). The previous form
        # differentiated a graph whose nodes were themselves derivative computations, which
        # FR-5 now forbids: it was the single largest cost in a training step and it cannot
        # run at all on a host whose attention kernel implements only a first-order
        # backward.
        G_t_list = []
        for name in active_names:
            idx = self._obj_to_idx[name]
            grads = torch.autograd.grad(
                active_losses[name].float(), anchor_params,
                retain_graph=True, create_graph=False,
            )
            grad_norm = torch.cat([g.flatten() for g in grads]).norm(p=2).detach()
            G_t_list.append(self.lambdas[idx] * grad_norm)

        G_t = torch.stack(G_t_list)
        G_mean = G_t.detach().mean()

        Lhat_list = []
        for name in active_names:
            L0 = self.initial_losses[name]
            if L0 < 1e-12:
                Lhat_list.append(torch.tensor(1.0, device=self.log_lambdas.device))
            else:
                Lhat_list.append(active_losses[name].detach().float() / L0)
        Lhat = torch.stack(Lhat_list)
        r_t = Lhat / (Lhat.mean() + 1e-12)
        G_star = G_mean * (r_t ** self.alpha)

        gradnorm_loss = (G_t - G_star.detach()).abs().sum()

        # REV-057: FR-5 requires the GradNorm optimizer to be separate from the main model
        # optimizer. `gradnorm_loss.backward()` would accumulate into every `.grad` the
        # expression reaches, on top of the task gradients the caller is about to step on;
        # taking the lambda gradient in isolation with `torch.autograd.grad` writes into
        # `log_lambdas.grad` ONLY and leaves every model `.grad` untouched.
        # REV-075: code-quality.md requires set_to_none=True on every zero_grad.
        # REV-120 / design D7: with G_t factored, `gradnorm_loss` is a scalar multiply over a
        # num_active-element tensor whose only graph runs through `log_lambdas`, so this is
        # a first-order grad costing ~0.45 ms (it was 276 ms as a second-order traversal).
        # D7 admits the closed form too; `autograd.grad` is preferred because it stays
        # correct if FR-5's `gradnorm_loss` expression ever changes.
        # `.detach()` before the assignment is the trap D7 names: a tensor derived from
        # `self.lambdas[idx] * ...` carries a graph through `log_lambdas`, and PyTorch
        # SILENTLY accepts assigning it to `.grad`, pinning that graph alive across steps.
        (lambda_grad,) = torch.autograd.grad(gradnorm_loss, [self.log_lambdas])
        self.optimizer.zero_grad(set_to_none=True)
        self.log_lambdas.grad = lambda_grad.detach()
        self.optimizer.step()

        if self.renorm_scope == "enabled":
            # Active objectives are always enabled; the union guards a caller that passes
            # an objective it did not declare.
            self._renormalize(sorted(set(self._enabled_indices) | set(active_indices)))
        else:
            self._renormalize(active_indices)

        return gradnorm_loss.detach()

    def _renormalize(self, active_indices: List[int]) -> None:
        """Rescale the given lambdas (the active ones, or every enabled one under
        renorm_scope "enabled") to sum to their count.

        specs/ablation-run D4: branch-free. `if renorm > 0:` was a host sync every step.
        Same elementwise arithmetic as the previous per-index loop (each index reads the
        lambdas before any of them is rewritten), and when renorm <= 0 the old values are
        kept exactly, as the branch did.
        """
        num_active = len(active_indices)
        with torch.no_grad():
            idx = torch.tensor(active_indices, device=self.log_lambdas.device)
            lam = self.lambdas[idx]
            renorm = lam.sum()
            scale = num_active / renorm
            self.log_lambdas[idx] = torch.where(
                renorm > 0, (lam * scale).log(), self.log_lambdas[idx],
            )

    def state_dict(self) -> dict:
        return {
            "log_lambdas": self.log_lambdas.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
            "initial_losses": dict(self.initial_losses),
        }

    def load_state_dict(self, state: dict) -> None:
        self.log_lambdas.data.copy_(state["log_lambdas"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.initial_losses = dict(state["initial_losses"])
