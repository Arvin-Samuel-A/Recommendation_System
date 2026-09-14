import json
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config, OBJECTIVE_ORDER


class Expert(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int, dropout: float):
        super().__init__()
        ffn_dim = hidden_dim * expansion
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class SharedExperts(nn.Module):
    def __init__(self, num_experts: int, hidden_dim: int, expansion: int, dropout: float):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(hidden_dim, expansion, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.stack([e(h) for e in self.experts], dim=2)


class ARVRouter(nn.Module):
    def __init__(self, hidden_dim: int, objective_embedding_dim: int,
                 num_experts: int, router_conditioning: str):
        super().__init__()
        self.router_conditioning = router_conditioning
        if router_conditioning == "input_only":
            input_dim = hidden_dim
        else:
            input_dim = hidden_dim + objective_embedding_dim
        self.linear = nn.Linear(input_dim, num_experts)

    def forward(self, pooled_hidden: torch.Tensor,
                objective_emb: torch.Tensor) -> torch.Tensor:
        if self.router_conditioning == "input_only":
            logits = self.linear(pooled_hidden)
        else:
            if objective_emb.dim() == 1:
                objective_emb = objective_emb.unsqueeze(0).expand(pooled_hidden.shape[0], -1)
            logits = self.linear(torch.cat([pooled_hidden, objective_emb], dim=-1))
        return F.softmax(logits, dim=-1)


class ConventionalMMoEGates(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, num_objectives: int):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Linear(hidden_dim, num_experts)
            for _ in range(num_objectives)
        ])

    def forward(self, pooled_hidden: torch.Tensor,
                objective_idx: int) -> torch.Tensor:
        return F.softmax(self.gates[objective_idx](pooled_hidden), dim=-1)


def route_objective(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    return (routing_weights[:, None, :, None] * expert_outputs).sum(dim=2)


class MoELayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.variant = config.moe_variant
        self._cached_expert_outputs: Optional[torch.Tensor] = None

        if self.variant == "none":
            return

        self.experts = SharedExperts(
            config.num_experts, config.hidden_dim,
            config.expert_expansion, config.dropout,
        )

        if self.variant == "arv_mmoe":
            self.objective_embeddings = nn.Embedding(
                len(OBJECTIVE_ORDER), config.objective_embedding_dim,
            )
            self.router = ARVRouter(
                config.hidden_dim, config.objective_embedding_dim,
                config.num_experts, config.router_conditioning,
            )
        elif self.variant == "mmoe":
            self.gates = ConventionalMMoEGates(
                config.hidden_dim, config.num_experts, len(OBJECTIVE_ORDER),
            )

    def clear_cache(self) -> None:
        self._cached_expert_outputs = None

    def forward(
        self,
        all_hidden_states: torch.Tensor,
        pooled_hidden: torch.Tensor,
        objective_idx: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.variant == "none":
            return (all_hidden_states, None)

        if self._cached_expert_outputs is None:
            self._cached_expert_outputs = self.experts(all_hidden_states)
        expert_outputs = self._cached_expert_outputs

        if self.variant == "arv_mmoe":
            e_t = self.objective_embeddings.weight[objective_idx]
            r_ut = self.router(pooled_hidden, e_t)
        else:
            r_ut = self.gates(pooled_hidden, objective_idx)

        Z_ut = route_objective(expert_outputs, r_ut)
        return (Z_ut, r_ut)


def count_router_parameters(config: Config) -> dict:
    D = config.hidden_dim
    K = config.num_experts
    E = config.objective_embedding_dim
    T = len(OBJECTIVE_ORDER)

    mmoe_gate_params = T * (D * K + K)

    if config.router_conditioning == "input_only":
        arv_router_params = D * K + K
    else:
        arv_router_params = (D + E) * K + K
    arv_objective_embedding_params = T * E
    arv_total = arv_router_params + arv_objective_embedding_params
    difference = mmoe_gate_params - arv_total

    report = {
        "model": config.model_name,
        "hidden_dim": D,
        "num_experts": K,
        "objective_embedding_dim": E,
        "num_objectives": T,
        "router_conditioning": config.router_conditioning,
        "mmoe_gate_params": mmoe_gate_params,
        "arv_router_params": arv_router_params,
        "arv_objective_embedding_params": arv_objective_embedding_params,
        "arv_total_router_params": arv_total,
        "difference": difference,
    }

    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"moe_parameter_report_{config.model_dir_name()}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    return report
