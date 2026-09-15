from typing import Dict, List, NoReturn, Optional

import torch
import torch.nn as nn

from configs.config import OBJECTIVE_ORDER


ANCHOR_CHOICES = ("last_shared_transformer", "all_transformer_blocks")


class DoubleBackwardUnsupportedError(RuntimeError):
    """The attention kernel under the GradNorm anchor has no double-backward.

    REV-092: GradNorm needs a differentiable gradient of each weighted loss w.r.t. the
    anchor parameters, i.e. a second backward through the anchor block's attention
    kernel. Not every kernel implements one -- on CPU, SDPA raises "derivative for
    aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented".
    Left unwrapped, an operator sees an ATen symbol rather than the actual problem.
    """


def _raise_double_backward(exc: Exception, model: Optional[nn.Module]) -> "NoReturn":
    backend = "unknown"
    device = "unknown"
    try:
        backend = getattr(getattr(model, "backend", None), "backend_name", "unknown")
        device = str(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        pass
    raise DoubleBackwardUnsupportedError(
        f"GradNorm requires a double-backward (create_graph=True) through the anchor "
        f"block, and the attention backend '{backend}' on device '{device}' does not "
        f"support one. Underlying error: {type(exc).__name__}: {exc}\n"
        f"Remedies: run with use_gradnorm=False (static objective weights), or choose a "
        f"device/attention_backend whose kernel implements double-backward "
        f"(SDPA on MPS and CUDA do; SDPA on CPU does not)."
    ) from exc


def probe_double_backward(model: nn.Module, config=None) -> None:
    """Fail at startup, not after dataset loading, when GradNorm cannot run here.

    REV-092: runs one double-backward through the anchor block on a tiny synthetic
    batch. Called from `run_pretrain` when `use_gradnorm=True`, alongside the attention
    backend self-test.
    """
    anchor_params = get_anchor_layer_params(model, config)
    if not anchor_params:
        return
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    block = model.blocks[-1]
    x = torch.randn(1, 4, model.hidden_dim, device=device, dtype=dtype, requires_grad=True)
    was_training = block.training
    block.train()
    try:
        out = block(x, None).float().sum()
        grads = torch.autograd.grad(out, anchor_params, create_graph=True)
        torch.cat([g.flatten() for g in grads]).norm(p=2).backward()
    except (RuntimeError, NotImplementedError) as exc:
        _raise_double_backward(exc, model)
    finally:
        block.train(was_training)
        for p in anchor_params:
            p.grad = None


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
        model: Optional[nn.Module] = None,
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
            try:
                grads = torch.autograd.grad(
                    weighted_loss, anchor_params, retain_graph=True, create_graph=True,
                )
            except (RuntimeError, NotImplementedError) as exc:
                # REV-092: re-raise as something an operator can act on.
                _raise_double_backward(exc, model)
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

        # REV-057: `gradnorm_loss.backward()` propagated through the `create_graph=True`
        # double-backward graph above, which depends on the model parameters, so it
        # accumulated second-order gradients into the anchor block (and everything
        # upstream of it) on top of the task gradients the caller is about to step on.
        # FR-5 requires the GradNorm optimizer to be separate from the main model
        # optimizer; computing the lambda gradient in isolation with `torch.autograd.grad`
        # writes into `log_lambdas.grad` ONLY and leaves every model `.grad` untouched.
        # REV-075: code-quality.md requires set_to_none=True on every zero_grad.
        # REV-092: this is where the double-backward actually runs -- the grad above
        # only BUILDS the graph with create_graph=True; differentiating through it is
        # what needs the kernel's second derivative. Both sites are wrapped so the
        # diagnostic fires wherever the unsupported kernel surfaces.
        try:
            (lambda_grad,) = torch.autograd.grad(gradnorm_loss, [self.log_lambdas])
        except (RuntimeError, NotImplementedError) as exc:
            _raise_double_backward(exc, model)
        self.optimizer.zero_grad(set_to_none=True)
        self.log_lambdas.grad = lambda_grad
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
