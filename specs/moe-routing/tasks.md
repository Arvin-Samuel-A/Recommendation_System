# MoE Routing — Tasks

Ordered tasks. Depends on transformer-backbone (complete) and data-pipeline T1.1 (Config).

## Phase 1: Core Components

- [x] **T1.1: Implement `Expert` class in `models/moe.py`** — FFN: Linear(D, D*expansion) → GELU → Dropout → Linear(D*expansion, D). Position-wise application. (FR-1)

- [x] **T1.2: Implement `SharedExperts` class** — `nn.ModuleList` of K `Expert` instances. `forward(h)` returns `torch.stack([e(h) for e in experts], dim=2)` → `(B, S, K, D)`. (FR-1)

- [x] **T1.3: Implement `ARVRouter` class** — `nn.Linear(hidden_dim + emb_dim, num_experts)`. `forward(pooled_hidden, objective_emb)`: handle broadcasting of objective_emb, concat, softmax → `(B, K)`. Support `router_conditioning="input_only"` variant where input dim is `hidden_dim` only. (FR-3, FR-4)

- [x] **T1.4: Implement objective embeddings** — `nn.Embedding(4, objective_embedding_dim)` with fixed order: [0]=next_item, [1]=masked_item, [2]=multi_step, [3]=contrastive. (FR-2)

- [x] **T1.5: Implement `ConventionalMMoEGates` class** — `nn.ModuleList` of `num_objectives` independent `nn.Linear(hidden_dim, num_experts)`. `forward(pooled_hidden, objective_idx)` → softmax → `(B, K)`. (FR-5)

## Phase 2: Assembly

- [x] **T2.1: Implement `route_objective` function** — Takes pooled_hidden, all_expert_outputs, objective_emb, router. Computes routing distribution, broadcasts across positions, sums expert outputs. Returns `(Z_ut, r_ut)` where Z_ut is `(B, S, D)` and r_ut is `(B, K)`. (FR-3, FR-10)

- [x] **T2.2: Implement `MoELayer` class** — `__init__` builds components based on `config.moe_variant`: "none" → nothing, "arv_mmoe" → SharedExperts + ARVRouter + ObjectiveEmbeddings, "mmoe" → SharedExperts + ConventionalMMoEGates. `forward(all_hidden_states, pooled_hidden, objective_idx)` returns `(Z_ut, r_ut)`. For "none" variant: return `(all_hidden_states, None)`. Implement expert output caching (compute once per batch, clear between batches). (FR-7, FR-8)

- [x] **T2.3: Implement `count_router_parameters` function** — Compute: mmoe_gate_params, arv_router_params, arv_objective_embedding_params, arv_total, difference. Uses config dimensions. Save to `outputs/logs/moe_parameter_report_{model}.json`. (FR-6, AC-05)
  - **Provenance stamp added 2026-09-18** (owner-approved; driven by `training-pipeline`
    REV-136's closing note, not by a finding in this spec). The report now carries
    `source_revision`. It echoes the **running** revision via `get_source_revision()`, not
    `artifact_source_revision()`: the report is a pure function of `config` and is not
    derived from a checkpoint, so there is no model whose provenance it could be
    reporting. Verified by reading the field back off a written file, not off the returned
    dict — `tests/test_moe.py::TestParameterReportNamespacing::test_report_carries_the_running_source_revision`,
    which fails if the stamping line is removed.
  - The two already-published reports
    (`moe_parameter_report_t_base_A_arv_mmoe_gradnorm{,_lora}.json`, written 23:32) were
    **not** regenerated: they predate the field, and REV-138's instruction not to
    re-stamp existing artifacts applies. Future arms carry it.

## Phase 3: Testing

- [x] **T3.1: Shape tests** — Verify Z_ut is `(B, S, D)` and r_ut is `(B, K)` for arv_mmoe and mmoe. Verify `(all_hidden_states, None)` for "none". (AC-07)

- [x] **T3.2: Routing differentiation test (objectives)** — Same input batch, different objective_idx → verify ARV-MMoE produces different r_ut. (AC-01)

- [x] **T3.3: Routing differentiation test (users)** — Different input batches, same objective_idx → verify ARV-MMoE produces different r_ut. (AC-02)

- [x] **T3.4: Expert sharing test** — Call MoELayer.forward twice with different objectives → verify expert outputs computed only once (check cache hit or computation count). (AC-03)

- [x] **T3.5: Passthrough test** — `moe_variant="none"` → verify output is identical to input all_hidden_states by tensor equality. (AC-04)

- [x] **T3.6: Parameter accounting test** — Verify `count_router_parameters()` output matches hand-calculated values for known dimensions. (AC-05)

- [x] **T3.7: Inference identity test** — Verify `objective_embeddings.weight[0]` is used at inference, not a separate parameter. (AC-06)

- [x] **T3.8: Conventional MMoE independence test** — Verify gate parameters for different objectives are distinct `nn.Linear` modules with no shared weight tensors. (AC-08)

- [x] **T3.9: Ablation B router conditioning test** — `input_only` variant: verify router input dimension is `hidden_dim` (not `hidden_dim + emb_dim`). (AC-09)
