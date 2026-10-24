import json
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config, OBJECTIVE_ORDER
from utils import get_source_revision


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
                objective_emb = objective_emb.expand(*pooled_hidden.shape[:-1], -1)
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
    pooled_hidden: torch.Tensor,
    all_expert_outputs: torch.Tensor,
    objective_emb: Optional[torch.Tensor],
    router: nn.Module,
    objective_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the routing distribution and mix the experts with it.

    A ``(B, D)`` ``pooled_hidden`` gives one distribution per sequence, broadcast across
    positions: returns ``(Z_ut, r_ut)`` of shapes ``(B, S, D)`` and ``(B, K)``. A
    ``(B, S, D)`` one routes every position from its own (causal) hidden state:
    ``r_ut`` is ``(B, S, K)``. The per-position form is what the objectives that predict
    at every position use, since routing from the last position would hand earlier
    positions information about the items after them; at the last position the two
    forms agree exactly.
    ``router`` is either an :class:`ARVRouter` (then ``objective_emb`` is used) or a
    :class:`ConventionalMMoEGates` (then ``objective_idx`` selects the gate).

    REV-052: this takes the arguments and returns the pair T2.1 specifies. It previously
    took ``(expert_outputs, routing_weights)`` and returned only ``Z_ut``, with the
    routing distribution computed by the caller -- a divergence between tasks.md and the
    source rather than a defect.
    """
    if isinstance(router, ConventionalMMoEGates):
        r_ut = router(pooled_hidden, objective_idx)
    else:
        r_ut = router(pooled_hidden, objective_emb)

    if r_ut.dim() == 3:
        Z_ut = (r_ut[..., None] * all_expert_outputs).sum(dim=2)
    else:
        Z_ut = (r_ut[:, None, :, None] * all_expert_outputs).sum(dim=2)
    return Z_ut, r_ut


class MoELayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.variant = config.moe_variant
        self._cached_expert_outputs: Optional[torch.Tensor] = None
        self._cache_key: Optional[tuple] = None
        #: The exact input the cache was computed from. Retained so its identity cannot
        #: be recycled by the allocator while the cache is live (see forward()).
        self._cache_input: Optional[torch.Tensor] = None

        if self.variant == "none":
            self.num_experts = 0
            return

        # REV-046: `match_mmoe_expert_count` makes the conventional-MMoE arm of
        # Ablation C use the same number of experts as the ARV-MMoE arm, so the two
        # differ only in the routing formulation. When it is False the conventional
        # arm is instead given `mmoe_num_experts` experts of its own.
        if self.variant == "mmoe" and not config.match_mmoe_expert_count:
            self.num_experts = config.mmoe_num_experts
        else:
            self.num_experts = config.num_experts

        self.experts = SharedExperts(
            self.num_experts, config.hidden_dim,
            config.expert_expansion, config.dropout,
        )

        if self.variant == "arv_mmoe":
            self.objective_embeddings = nn.Embedding(
                len(OBJECTIVE_ORDER), config.objective_embedding_dim,
            )
            self.router = ARVRouter(
                config.hidden_dim, config.objective_embedding_dim,
                self.num_experts, config.router_conditioning,
            )
        elif self.variant == "mmoe":
            self.gates = ConventionalMMoEGates(
                config.hidden_dim, self.num_experts, len(OBJECTIVE_ORDER),
            )

    def clear_cache(self) -> None:
        self._cached_expert_outputs = None
        self._cache_key = None
        self._cache_input = None

    @staticmethod
    def _make_cache_key(t: torch.Tensor) -> tuple:
        """Version/shape fingerprint, used ALONGSIDE the retained input reference.

        This alone is NOT sufficient to identify a tensor: see `_cache_input` below.
        """
        return (tuple(t.shape), t._version, t.dtype, str(t.device))

    def forward(
        self,
        all_hidden_states: torch.Tensor,
        pooled_hidden: torch.Tensor,
        objective_idx: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.variant == "none":
            return (all_hidden_states, None)

        # REV-045 (second pass): the cache key used to be
        # `(id(t), t.data_ptr(), shape, _version, dtype, device)`. Both `id()` and
        # `data_ptr()` are RECYCLED -- once a tensor is freed, CPython can hand its id to
        # the next object and the caching allocator can hand its address to the next
        # allocation. In `score_candidates`, which builds a fresh `all_hidden` per
        # example and drops the previous one, a new example could therefore produce a key
        # identical to the previous example's and silently reuse its expert outputs.
        # Under the full test suite this reproduced at 5/25 examples.
        #
        # Holding a REFERENCE to the cached input makes identity trustworthy: while the
        # cache is populated that tensor cannot be freed, so neither its id nor its
        # address can be reused by anything else. `is` is then exact, and the fingerprint
        # additionally catches in-place mutation of that same tensor. `clear_cache()`
        # releases the reference.
        key = self._make_cache_key(all_hidden_states)
        if (
            self._cached_expert_outputs is None
            or self._cache_input is not all_hidden_states
            or self._cache_key != key
        ):
            self._cached_expert_outputs = self.experts(all_hidden_states)
            self._cache_input = all_hidden_states
            self._cache_key = key
        expert_outputs = self._cached_expert_outputs

        if self.variant == "arv_mmoe":
            e_t = self.objective_embeddings.weight[objective_idx]
            return route_objective(
                pooled_hidden, expert_outputs, e_t, self.router, objective_idx,
            )
        return route_objective(
            pooled_hidden, expert_outputs, None, self.gates, objective_idx,
        )


def count_router_parameters(config: Config, run_tag: Optional[str] = None) -> dict:
    D = config.hidden_dim
    K = config.num_experts
    E = config.objective_embedding_dim
    T = len(OBJECTIVE_ORDER)

    # REV-046: the conventional-MMoE arm may carry its own expert count when
    # `match_mmoe_expert_count` is False; the gate accounting must follow it.
    K_mmoe = K if config.match_mmoe_expert_count else config.mmoe_num_experts
    mmoe_gate_params = T * (D * K_mmoe + K_mmoe)

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
        "mmoe_num_experts": K_mmoe,
        "match_mmoe_expert_count": config.match_mmoe_expert_count,
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
    # REV-078: the filename used to depend only on the model SIZE, but the report's
    # contents vary with moe_variant, num_experts, router_conditioning and
    # match_mmoe_expert_count. Ablation C runs mmoe and arv_mmoe at one size and G runs
    # four expert counts at one size, so every arm wrote to the same path and only the
    # last survived -- including the report that is supposed to back C's
    # parameter-matching claim. `run_tag` (which `_run_single_config` already builds)
    # takes precedence; otherwise the distinguishing fields go in the name.
    if run_tag:
        suffix = f"{config.model_dir_name()}_{run_tag}"
    else:
        suffix = (
            f"{config.model_dir_name()}_{config.moe_variant}"
            f"_K{K}_{config.router_conditioning}"
        )
        if not config.match_mmoe_expert_count:
            suffix += f"_mmoeK{K_mmoe}"
    report["run_tag"] = run_tag
    report["report_suffix"] = suffix
    # Provenance (steering/code-quality.md, "Artifact Provenance"; REV-136's closing note,
    # owner-approved 2026-09-18). This report is a pure function of `config` and is NOT
    # derived from a checkpoint, so it echoes the RUNNING revision rather than a
    # checkpoint's -- `artifact_source_revision` would be the wrong call here, since there
    # is no model whose provenance this could be reporting.
    report["source_revision"] = get_source_revision()
    path = os.path.join(log_dir, f"moe_parameter_report_{suffix}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    return report
