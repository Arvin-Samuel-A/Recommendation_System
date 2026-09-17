# MoE Routing — Requirements

## Overview

Implement the Adaptive Routing Vector Mixture-of-Experts (ARV-MMoE) layer, the conventional MMoE baseline, and the `moe_variant="none"` passthrough — the central architectural contribution of the paper. The layer sits after the Transformer backbone and before per-objective loss heads, producing objective-conditioned expert mixtures.

## Problem Statement

Multi-objective training causes gradient interference between objectives sharing a single representation. ARV-MMoE addresses this by routing each objective through a differentiated mixture of shared experts, conditioned on both the user representation and an objective embedding — using a single shared router instead of independent per-objective gates.

## Goals

- G1: Shared experts computed once, reused across all objectives
- G2: ARV-MMoE: single shared router conditioned on `[H_u; e_t]` producing per-objective mixture weights
- G3: Conventional MMoE: independent per-objective gates for ablation comparison
- G4: `moe_variant="none"` passthrough for baselines without routing
- G5: Correct parameter accounting for fair MMoE vs ARV-MMoE comparison
- G6: Inference-time serving uses `e_rec ≡ e_next` (no separate untrained embedding)

## Non-Goals

- Sparse top-k routing (all experts contribute with soft weights)
- Per-position routing (routing is per-user, broadcast across positions)
- Persistent per-client routing personalization (LoRA handles client adaptation)

## Functional Requirements

### FR-1: Shared Experts
- `K = config.num_experts` identical-shaped FFN experts applied position-wise
- Expert architecture: Linear(hidden_dim, hidden_dim * expansion) → GELU → Dropout → Linear(hidden_dim * expansion, hidden_dim)
- All K expert outputs computed once: `(B, S, K, D)`
- Reused by every objective — only routing weights differ

### FR-2: Objective Embeddings
- `nn.Embedding(num_objectives=4, objective_embedding_dim=32)`
- Fixed order: [0]=next_item, [1]=masked_item, [2]=multi_step, [3]=contrastive
- These are the ONLY place objective identity enters routing computation
- Learnable; receive gradients through their respective objective losses

### FR-3: ARV Router (Proposed)
- Single shared router: `nn.Linear(hidden_dim + emb_dim, num_experts)`
- Input: `[pooled_hidden; objective_emb]` → softmax → routing distribution `r_{u,t}` of shape `(B, K)`
- Broadcast routing across positions: `r_{u,t}.unsqueeze(1).unsqueeze(-1) * expert_outputs` → sum over experts → `Z_{u,t}` of shape `(B, S, D)`
- Different users get different routing distributions (user-adaptive)
- Different objectives get different routing distributions (objective-conditioned)

### FR-4: Ablation B — Router Conditioning
- `config.router_conditioning = "objective_conditioned"` (default): router sees `[H_u; e_t]`
- `config.router_conditioning = "input_only"`: router sees `H_u` alone (no objective embedding)
- Isolates whether conditioning on the objective provides measurable benefit

### FR-5: Conventional MMoE (Ablation C Baseline)
- Each objective gets its own independently parameterized gate: `nn.Linear(hidden_dim, num_experts)` per objective
- No parameter sharing across objectives' routing policies
- Gate never sees an objective embedding — task-specificity from separate weights
- Same expert-mixing mechanics as ARV, different routing computation

### FR-6: Parameter Accounting
- `count_router_parameters(config)` returns: mmoe_gate_params, arv_router_params, arv_objective_embedding_params, arv_total_router_params, difference
- Log to `outputs/logs/moe_parameter_report_{model}_{run_tag}.json` for every `moe_variant` run. When no run tag is supplied the suffix carries the distinguishing fields instead: `{model}_{moe_variant}_K{K}_{router_conditioning}` (plus `_mmoeK{K}` when `match_mmoe_expert_count=False`)
  - *(REV-104, owner 2026-09-17: the path was previously `{model}.json` alone. Ablation C runs mmoe and arv_mmoe at one size and Ablation G sweeps four expert counts at one size, so every arm overwrote a single file and only the last survived — including the report that is supposed to back C's parameter-matched claim.)*
- When `match_mmoe_expert_count=True` (default), K is identical across mmoe and arv_mmoe
- Never claim "parameter-matched" without the report backing it

### FR-7: `moe_variant="none"` Passthrough
- MoE layer returns `(all_hidden_states, None)` — no routing, no experts
- Loss heads read hidden states directly from Transformer output
- Must remain fully working for Ablation A configs 1–2

### FR-8: Top-Level MoE Module
- `MoELayer.forward(all_hidden_states, pooled_hidden, objective_idx)` → `(Z_u_t, routing_weights)`
- `objective_idx` follows OBJECTIVE_ORDER
- Returns both the mixed representation and the routing distribution (for diagnostics)

### FR-9: Inference Serving Path (Step 8.6)
- At inference: `e_rec = objective_embeddings.weight[0]` (e_rec ≡ e_next by identity)
- Never create a separate untrained inference-only embedding
- Serving: Transformer → pooled H_u → ARV routing with e_next → expert mixture Z_rec → scoring

### FR-10: Per-Objective Head Consumption
- Next-item head: `Z_{u,next}[:, last_pos, :]`
- MIM head: `Z_{u,mim}` at all masked positions
- Multi-step head: `Z_{u,forecast}[:, S-1, :]` — the same last position for every step
  - *(REV-105, owner 2026-09-17: was `S-1-step`. See `specs/loss-functions` FR-3: all k horizon heads now read one fixed context representation.)*
- Contrastive head: `Z_{u,contrast}[:, last_pos, :]`, L2-normalized

## Edge Cases

- `moe_variant="none"` with GradNorm: GradNorm still operates on the Transformer anchor, even without MoE (Ablation D config iii)
- Single expert (`num_experts=1`): valid configuration, should produce trivial routing
- Objective disabled (e.g., `use_contrastive=False`): its routing is never computed; no loss contribution

## Constraints

- MoE parameters (router, experts, objective embeddings, gates) are global/frozen during local/federated training
- MoE parameters must NEVER appear in LoRA state dicts or federated payloads
- Expert outputs computed once per forward pass, not per objective
- Routing distribution accumulated in per-epoch buffer for diagnostics (never discarded after loss)

## Dependencies

- Transformer backbone (`models/transformer.py`) for hidden states and pooled representations
- `configs/config.py` for MoE parameters
- Consumed by `training/losses.py`, `training/pretrain.py`, and all evaluation modules

## Acceptance Criteria

- **AC-01:** ARV-MMoE produces different routing distributions for different objectives on the same input
- **AC-02:** ARV-MMoE produces different routing distributions for different users on the same objective
- **AC-03:** Expert outputs are computed once and reused (verified by identity or computation count)
- **AC-04:** `moe_variant="none"` returns raw hidden states with no MoE computation
- **AC-05:** `count_router_parameters()` matches manual formula computation for all variants
- **AC-06:** Inference routing uses `objective_embeddings.weight[0]` (not a separate parameter)
- **AC-07:** Output shapes: `Z_u_t` is `(B, S, D)` and `routing_weights` is `(B, K)` for MoE variants
- **AC-08:** Conventional MMoE gates are independent per-objective (no shared parameters)
- **AC-09:** Router conditioning ablation correctly includes/excludes objective embedding from router input
