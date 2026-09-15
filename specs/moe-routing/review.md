# Review — MoE Routing

## Review Status

**PASS_WITH_WARNINGS**

## Feature

`specs/moe-routing` — `models/moe.py`: `Expert`, `SharedExperts`, `ARVRouter`, `ConventionalMMoEGates`, `route_objective`, `MoELayer`, `count_router_parameters`.

## Review Scope

- Spec directory: `specs/moe-routing/` (requirements.md, design.md, tasks.md)
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`
- Baseline: working tree vs `HEAD` = `32b0904` on `master`. All feature work is uncommitted.
- Files inspected: `models/moe.py` (255 lines, modified), `tests/test_moe.py` (348 lines, modified), and the consuming call sites in `training/pretrain.py`, `training/local_train.py`, `evaluation/metrics.py`, `evaluation/gradient_diagnostics.py`, `models/ann_retrieval.py`, `training/ablation_runner.py`
- Re-review of the prior `review.md`. `moe.py` and `tests/test_moe.py` were both modified after it (21:45–21:46).
- Date: 2026-09-15

## Summary

Both prior findings against this spec are resolved. REV-071's cache-lifetime issue is fixed at the right place: `moe.clear_cache()` is now called at the *end* of `train_one_step` ([pretrain.py:335](training/pretrain.py:335)) as well as at the start, so the `(B, S, K, D)` expert tensor and its autograd graph are released when the step returns rather than surviving until the next step begins — and `tests/test_training.py::TestMoECacheLifetime` asserts exactly that, including that the cache still serves all four objectives *within* one step. REV-078's report-overwriting is fixed: the filename now carries either the ablation `run_tag` or the distinguishing config fields, and four tests in `TestParameterReportNamespacing` cover the C and G arm collisions specifically.

All nine acceptance criteria hold with real test evidence. The one remaining item is REV-079, an INFO-level consequence of FR-1's own "computed once" requirement.

## Git Changes

| File | State |
|---|---|
| `models/moe.py` | modified (+~120 lines vs `32b0904`) |
| `tests/test_moe.py` | modified (+~210 lines) |
| `configs/config.py` | modified — `match_mmoe_expert_count`, `mmoe_num_experts` |
| `specs/moe-routing/review.md` | untracked (this file) |

`specs/moe-routing/tasks.md` is unchanged from `32b0904` — all 17 tasks were already `[x]` at that commit and none needed re-marking.

**Scope assessment:** in scope. No new dependencies; `models/moe.py` imports only `torch` and `configs.config`.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Shared experts | PASS | `SharedExperts.forward` returns `torch.stack([...], dim=2)` → `(B, S, K, D)` ([moe.py:32](models/moe.py:32)). Computed once per distinct `all_hidden_states` and reused by all four objectives via the identity-checked cache at [moe.py:175-184](models/moe.py:175). |
| FR-2 Objective embeddings | PASS | `nn.Embedding(len(OBJECTIVE_ORDER), objective_embedding_dim)` ([moe.py:127](models/moe.py:127)); `OBJECTIVE_ORDER` is the single global list in `configs/config.py`. They are the only place objective identity enters the ARV router. Learnable — they sit in `moe.parameters()` and so in the optimizer via `collect_trainable_parameters`. |
| FR-3 ARV router | PASS | `nn.Linear(hidden_dim + emb_dim, num_experts)` → softmax → `(B, K)`; broadcast as `r_ut[:, None, :, None] * all_expert_outputs` summed over dim 2 → `(B, S, D)` ([moe.py:94](models/moe.py:94)). User- and objective-differentiation both tested. |
| FR-4 Ablation B conditioning | PASS | `router_conditioning="input_only"` narrows the router input to `hidden_dim` ([moe.py:41-45](models/moe.py:41)); `Config.validate` restricts the flag to the two legal values; `TestAblationBRouterConditioning` asserts both input dimensions. |
| FR-5 Conventional MMoE | PASS | `ConventionalMMoEGates` holds one independent `nn.Linear` per objective, never sees an objective embedding, and shares the expert-mixing path through `route_objective`. |
| FR-6 Parameter accounting | PASS | `count_router_parameters` ([moe.py:196](models/moe.py:196)) returns all five named fields plus the K/matching context; `TestParameterAccounting::test_count_matches_manual` checks against the hand formula. Called from `_run_single_config` for every non-`none` arm, with `run_tag`, so FR-6's "for every moe_variant run" now has a caller. Filename deviates from FR-6's literal `moe_parameter_report_{model}.json` — see Design Compliance. |
| FR-7 `none` passthrough | PASS | `MoELayer.forward` returns `(all_hidden_states, None)` before touching any submodule ([moe.py:158](models/moe.py:158)); `__init__` builds nothing for that variant. `TestPassthrough::test_none_returns_input_unchanged` asserts tensor identity. |
| FR-8 Top-level module | PASS | `forward(all_hidden_states, pooled_hidden, objective_idx) -> (Z_ut, r_ut)`; `objective_idx` indexes `OBJECTIVE_ORDER` at every call site (0/1/2/3 in `train_one_step`, `run_gradient_diagnostics`, `collect_validation_routing`). |
| FR-9 Inference serving path | PASS | `score_candidates_batch` and `models/ann_retrieval.py::_user_embeddings` both call `moe_layer(all_hidden, pooled, 0)` — objective index 0 is `next_item`, so `e_rec ≡ e_next` by construction. No separate inference embedding exists in the module. |
| FR-10 Per-objective head consumption | PASS | `train_one_step`: next-item reads `Z_next[:, -1, :]`, MIM reads `Z_mim` at `mask_positions`, multi-step reads `Z_ms` at `S-1-step` inside `MultiStepLoss`, contrastive reads `Z_cl[:, -1, :]` L2-normalized. |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 different routing per objective | PASS | `TestRoutingDifferentiation::test_different_objectives_different_routing`. |
| AC-02 different routing per user | PASS | `TestRoutingDifferentiation::test_different_users_different_routing`. |
| AC-03 expert outputs computed once and reused | PASS | `TestExpertSharing::test_expert_outputs_cached`, plus `test_cache_invalidates_on_new_input_without_clear` and `test_cache_invalidates_across_a_scoring_loop` — the last of which is the regression test for the recycled-`id()`/`data_ptr()` bug the cache now guards against by retaining `_cache_input`. `TestMoECacheLifetime::test_cache_still_serves_the_objectives_within_one_step` confirms REV-071's fix did not break reuse. |
| AC-04 `none` returns raw hidden states | PASS | `TestPassthrough::test_none_returns_input_unchanged` (tensor identity, not allclose). |
| AC-05 parameter count matches manual formula | PASS | `TestParameterAccounting::test_count_matches_manual`; `TestParameterReportNamespacing::test_report_contents_are_unchanged_by_the_naming` confirms the REV-078 rename did not alter the numbers. |
| AC-06 inference uses `objective_embeddings.weight[0]` | PASS | `TestInferenceIdentity::test_e_rec_is_e_next`. |
| AC-07 output shapes | PASS | `TestShapes` covers `arv_mmoe`, `mmoe` and `none`. |
| AC-08 MMoE gates independent | PASS | `TestConventionalMMoEIndependence::test_gates_are_independent` checks the gate modules hold distinct weight tensors. |
| AC-09 router conditioning includes/excludes the embedding | PASS | `TestAblationBRouterConditioning::test_input_only_dimension` and `test_objective_conditioned_dimension`. |

## Findings

```
REV-079
Severity: INFO | Status: OPEN | Requirement: FR-1 | AC: —
File: models/moe.py:18-21 (`Expert`), 175-184 (cache reuse across objectives)
Problem: `Expert` contains `nn.Dropout`. Because the K expert outputs are computed once
  per forward pass and reused by all four objectives (which is what FR-1 requires), every
  objective in a training step sees the SAME dropout mask on the expert outputs. Four
  independent forwards would each draw their own.
Impact: None on correctness or on any acceptance criterion. It slightly correlates the
  four objectives' gradients through a shared noise realisation, which is worth knowing
  when reading the Ablation D gradient-cosine numbers -- part of the measured inter-
  objective cosine similarity is attributable to the shared mask rather than to genuine
  representational conflict.
Correction: None required -- the alternative contradicts FR-1 and G1. Worth one sentence
  in the paper's method section, or in `design.md`, so the diagnostic is read correctly.
```

```
REV-104
Severity: INFO | Status: OPEN | Requirement: FR-6 | AC: AC-05
File: models/moe.py:240-251 (`count_router_parameters`)
Problem: FR-6 names the output path literally as
  `outputs/logs/moe_parameter_report_{model}.json`. The implementation writes
  `moe_parameter_report_{model}_{run_tag}.json`, or
  `moe_parameter_report_{model}_{variant}_K{K}_{conditioning}.json` when no run tag is
  supplied. `outputs/logs/` currently holds
  `moe_parameter_report_t_base_A_arv_mmoe_gradnorm.json` and
  `..._A_arv_mmoe_gradnorm_lora.json`, confirming the suffixed form is what reaches disk.
Impact: None -- this was the previous review's own recommended correction for REV-078
  ("include the distinguishing fields in the filename"), and it is strictly better than
  every Ablation C and G arm overwriting one path. Recorded only so the filename
  divergence from FR-6's literal wording is on the record.
Correction: None for BUILD. If the spec owner wants the spec and the artifacts to agree,
  FR-6's path should be reworded to `moe_parameter_report_{model}_{run_tag}.json`.
```

## Design Compliance

Matches `design.md` and `steering/architecture.md`'s key decisions:

- **"Shared experts, differentiated routing"** — expert FFNs computed once per forward, only `r_{u,t}` differs per objective. Implemented exactly.
- **"Pooled-then-broadcast routing"** — routing computed from `pooled_hidden` once, broadcast across S. Implemented at [moe.py:94](models/moe.py:94).
- **"`e_rec ≡ e_next` at inference"** — no separate embedding exists.
- **MoE variant boundary** (`conventions.md`) — `"none"` is a genuinely working direct path, `"mmoe"` uses independent gates, `"arv_mmoe"` uses the shared conditioned router.
- **MoE parameters are global** — `MoELayer` is never a submodule of `TransformerModel`, so it cannot appear in a LoRA state dict by construction; `training/federated.py::assert_no_moe_keys` additionally checks against the MoE's real key space.

Two additions beyond `design.md`, both benign and documented:
- `match_mmoe_expert_count` / `mmoe_num_experts` (REV-046). This is what makes FR-6's "K is identical across mmoe and arv_mmoe when `match_mmoe_expert_count=True`" a real flag rather than an assumption, and `count_router_parameters` follows it.
- The `_cache_input` reference held alongside the fingerprint key. Necessary for correctness — `id()` and `data_ptr()` are both recycled — and explained in a long comment at [moe.py:161-174](models/moe.py:161).

No unapproved dependencies. No sparse top-k routing, no per-position routing, no per-client routing state — all three Non-Goals respected.

## Task Completion

All 17 tasks are `[x]`. Each was verified against the source and its named test:

| Task | Verdict |
|---|---|
| T1.1 `Expert` | Complete — Linear→GELU→Dropout→Linear at `expert_expansion`. |
| T1.2 `SharedExperts` | Complete — exactly the `torch.stack(..., dim=2)` the task specifies. |
| T1.3 `ARVRouter` | Complete including the `input_only` variant and the 1-D objective-embedding broadcast. |
| T1.4 objective embeddings | Complete, order taken from `OBJECTIVE_ORDER`. |
| T1.5 `ConventionalMMoEGates` | Complete. |
| T2.1 `route_objective` | Complete — takes and returns what the task specifies (REV-052's signature fix holds). |
| T2.2 `MoELayer` | Complete, including caching and clearing. The "clear between batches" wording is now satisfied at both ends of the step. |
| T2.3 `count_router_parameters` | Complete; filename divergence noted as REV-104. |
| T3.1–T3.9 tests | All nine exist, are named for the property they check, and genuinely check it. `test_none_returns_input_unchanged` uses identity rather than `allclose`, as T3.5 asks. |

No false-complete task claims found.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` (full suite) | 231 passed, 2 failed, 135 s. Neither failure is in this spec — both are `transformer-backbone` REV-062's armed MPS backend gate. |
| `uv run python -m pytest tests/test_moe.py -q` | All 17 pass (included in the run above). |
| `uv run python -m pytest tests/test_training.py::TestMoECacheLifetime -q` | Passes — confirms REV-071's release-at-end-of-step fix. |
| Direct inspection of `outputs/logs/` | Two distinct `moe_parameter_report_t_base_A_*.json` files present, confirming the REV-078 namespacing reaches disk with different arms writing different paths. |
| Scratch: `count_parameters(model, moe)` on a t-small build | Returns `router_parameters` and `expert_parameters` split correctly by the `"router"`/`"objective_embeddings"`/`"gates"` vs `"experts"` name test. |

## Final Assessment

This spec is complete and correct. Both prior findings are resolved with test evidence, all nine acceptance criteria pass, and the design decisions in `steering/architecture.md` are implemented as written.

Nothing is required of BUILD. The two open items are INFO-only:

- **REV-079** — no code change; worth one sentence in `design.md` or the paper so the Ablation D gradient-cosine numbers are read with the shared dropout mask in mind.
- **REV-104** — no code change; if the spec owner wants FR-6's literal path to match the artifacts, that is a one-line spec edit, not a BUILD edit.

Status `PASS_WITH_WARNINGS` reflects those two INFO findings only. Note that this spec's own tests all pass, but the repository-wide suite does not, for a reason owned by `specs/transformer-backbone` (REV-062).
