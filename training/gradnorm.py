from typing import Dict, List, Optional

import torch
import torch.nn as nn

from configs.config import OBJECTIVE_ORDER


def get_anchor_layer_params(model: nn.Module) -> List[nn.Parameter]:
    last_block = model.blocks[-1]
    return [p for p in last_block.parameters() if p.requires_grad]


class GradNormController:
    def __init__(self, alpha: float, lr: float):
        self.alpha = alpha
        self.num_objectives = len(OBJECTIVE_ORDER)
        self.log_lambdas = nn.Parameter(torch.zeros(self.num_objectives))
        self.optimizer = torch.optim.Adam([self.log_lambdas], lr=lr)
        self.initial_losses: Dict[str, float] = {}
        self._obj_to_idx = {name: i for i, name in enumerate(OBJECTIVE_ORDER)}

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

        G_t_list = []
        for name in active_names:
            idx = self._obj_to_idx[name]
            weighted_loss = self.lambdas[idx] * active_losses[name].float()
            grads = torch.autograd.grad(
                weighted_loss, anchor_params, retain_graph=True, create_graph=True,
            )
            grad_norm = torch.cat([g.flatten() for g in grads]).norm(p=2)
            G_t_list.append(grad_norm)

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

        self.optimizer.zero_grad()
        gradnorm_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            active_lams = self.lambdas[active_indices]
            renorm = active_lams.sum()
            if renorm > 0:
                scale = num_active / renorm
                for idx in active_indices:
                    self.log_lambdas[idx] = (self.lambdas[idx] * scale).log()

        return gradnorm_loss.detach()

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
