# Review — ARV-MMoE & Conventional MMoE Routing

## Review Status

**PASS**

## Feature

`specs/moe-routing` — `models/moe.py`: `Expert`, `SharedExperts`, `ARVRouter`,
`ConventionalMMoEGates`, `route_objective`, `MoELayer`, `count_router_parameters`.

## Review Scope

- Spec directory: `specs/moe-routing/` (requirements.md, design.md, tasks.md). None
  modified in the working tree.
- Steering: `architecture.md` (shared experts, pooled-then-broadcast routing,
  `e_rec ≡ e_next`, MoE-variant component boundary), `conventions.md`
  (`OBJECTIVE_ORDER`), `code-quality.md`.
- Baseline: `master` at `0bc67ee`, `origin/master` at the same commit. No file owned by
  this spec is modified in the working tree.
- Files inspected: `models/moe.py` (255 lines, read in full), `tests/test_moe.py`,
  the MoE call sites in `training/pretrain.py`, `training/local_train.py`,
  `evaluation/flops.py`, and the MoE-prefix constants in `models/lora.py` and
  `training/federated.py`.
- Re-review. Prior findings carried in: REV-079, REV-104 (both RESOLVED).
- Date: 2026-09-18

## Summary

Nothing for BUILD. Every requirement and acceptance criterion holds, and both prior
findings were re-verified against the code and the written artifacts rather than carried
forward on the previous review's word.

The two ACs most worth re-deriving independently — parameter accounting (AC-05) and the
expert-sharing cache (AC-03) — were recomputed by hand this pass and match.

## Git Changes

No file owned by this spec appears in `git status`. The four modified files belong to
`evaluation-diagnostics`.

**Scope assessment:** nothing unexpected, nothing missing.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Shared experts | PASS | `Expert` is `Linear(D, D·expansion) → GELU → Dropout → Linear(D·expansion, D)`, applied position-wise. `SharedExperts.forward` stacks on `dim=2`, giving `(B, S, K, D)`; verified `(4, 10, 4, 320)` at t-base. Computed once per input and reused by every objective |
| FR-2 Objective embeddings | PASS | `nn.Embedding(len(OBJECTIVE_ORDER), objective_embedding_dim)` — the count comes from `OBJECTIVE_ORDER` imported from `configs/config.py`, so it cannot drift from the order `steering/conventions.md` fixes. `named_parameters()` on an `arv_mmoe` layer shows `objective_embeddings.weight` as the only place objective identity enters routing |
| FR-3 ARV router | PASS | Single `nn.Linear(hidden_dim + emb_dim, num_experts)` over `[pooled; e_t]` → softmax → `(B, K)`. Broadcast is `r_ut[:, None, :, None] * all_expert_outputs` summed on `dim=2` — one routing distribution per user per objective, broadcast across every position, matching architecture decision 2 |
| FR-4 Ablation B conditioning | PASS | `router_conditioning="input_only"` sets `input_dim = hidden_dim` and the forward path omits the concat entirely rather than zeroing the embedding. Both dimensions asserted by `TestAblationBRouterConditioning` |
| FR-5 Conventional MMoE | PASS | `nn.ModuleList` of `num_objectives` independent `nn.Linear(hidden_dim, num_experts)`; the gate signature takes `objective_idx` and never an embedding, so task-specificity comes only from separate weights. `test_gates_are_independent` checks distinct weight tensors |
| FR-6 Parameter accounting | PASS | `count_router_parameters` returns all five required fields. Recomputed by hand at t-base (D=320, K=4, E=32, T=4): `mmoe_gate_params = 4·(320·4+4) = 5136` ✓, `arv_router_params = (320+32)·4+4 = 1412` ✓, `arv_objective_embedding_params = 4·32 = 128` ✓, total 1540, difference 3596 ✓. The filename carries the distinguishing fields — measured `moe_parameter_report_t_base_arv_mmoe_K4_objective_conditioned.json` — and `run_tag` takes precedence when supplied |
| FR-7 `"none"` passthrough | PASS | Early return `(all_hidden_states, None)` before any expert or router is constructed; `MoELayer.__init__` returns before allocating them, so the variant carries no MoE parameters at all. Verified: the returned object **is** the input tensor (identity, not equality) and the routing weight is `None` |
| FR-8 Top-level module | PASS | `MoELayer.forward(all_hidden_states, pooled_hidden, objective_idx) → (Z_ut, r_ut)`, with `objective_idx` following `OBJECTIVE_ORDER` at all five production call sites (`pretrain.py:193/214/227/241/542`, `local_train.py:154`) |
| FR-9 Inference serving path | PASS | `e_t = self.objective_embeddings.weight[objective_idx]`, and the inference/scoring call sites pass `0` — so `e_rec` is literally `weight[0]`, the trained next-item embedding. `named_parameters()` confirms no inference-only embedding exists |
| FR-10 Per-objective head consumption | PASS | Each objective's `Z` is routed with its own index and handed to its own head. The multi-step clause (amended 2026-09-17 to `S-1` for every step) is satisfied downstream: `MultiStepLoss` reads `pos = S - 1` from the `Z_ms` it is given |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 different objectives → different routing | PASS | Measured directly: `r_ut` for `objective_idx=0` and `1` on the same input are not `allclose`. `test_different_objectives_different_routing` |
| AC-02 different users → different routing | PASS | Measured directly: a different `pooled_hidden` at the same objective gives a different `r_ut`. `test_different_users_different_routing` |
| AC-03 experts computed once and reused | PASS | The cache is keyed on a **retained reference** to the input (`self._cache_input is all_hidden_states`) *plus* a `(shape, _version, dtype, device)` fingerprint. Holding the reference is what makes identity trustworthy — `id()` and `data_ptr()` are both recycled, which is the REV-045 defect. Three tests cover it, including `test_cache_invalidates_across_a_scoring_loop`, which reproduces the recycling scenario |
| AC-04 `"none"` returns raw hidden states | PASS | Verified by object identity (`out is h`), not tensor equality — the stronger check, and the one that proves no computation happened |
| AC-05 accounting matches manual formula | PASS | Recomputed by hand this pass for the objective-conditioned case; all four quantities match exactly. `test_count_matches_manual` covers it in CI, and the `input_only` and `match_mmoe_expert_count=False` branches are both handled in the formula |
| AC-06 inference uses `objective_embeddings.weight[0]` | PASS | `test_e_rec_is_e_next`; plus the parameter listing shows no candidate separate embedding to have used instead |
| AC-07 shapes `(B, S, D)` and `(B, K)` | PASS | Measured `(4, 10, 320)` and `(4, 4)` at t-base. `TestShapes` covers `arv_mmoe`, `mmoe` and `none` |
| AC-08 MMoE gates independent | PASS | `test_gates_are_independent` |
| AC-09 conditioning ablation includes/excludes the embedding | PASS | `test_input_only_dimension` and `test_objective_conditioned_dimension` assert the router's actual input dimension in each mode |

## Findings

No open findings.

```
REV-079
Severity: INFO | Status: RESOLVED | Requirement: FR-1 | AC: —
Re-confirmed. The expert FFN matches FR-1 clause for clause, including the dropout
placement between GELU and the second projection.
```

```
REV-104
Severity: INFO | Status: RESOLVED | Requirement: FR-6 | AC: AC-05
File: models/moe.py:225-248; specs/moe-routing/requirements.md FR-6
Re-verified against both the amended requirement and the artifacts. FR-6 now specifies
  the `{model}_{run_tag}` path with the `{model}_{moe_variant}_K{K}_{router_conditioning}`
  (+ `_mmoeK{K}`) fallback, and `count_router_parameters` implements exactly that. Measured
  on a scratch log dir: an untagged t-base arv_mmoe run writes
  `moe_parameter_report_t_base_arv_mmoe_K4_objective_conditioned.json`. The two reports
  already in `outputs/logs/` carry run tags
  (`..._t_base_A_arv_mmoe_gradnorm.json`, `..._t_base_A_arv_mmoe_gradnorm_lora.json`),
  confirming the tagged branch is the one the ablation runner exercises.
  `TestParameterReportNamespacing` has four tests covering Ablation C's two arms,
  Ablation G's four expert counts, the untagged fallback, and that the naming change did
  not alter the report's contents.
```

## Design Compliance

Compliant, and compliant with `steering/architecture.md` on the three decisions this
module owns:

- **Shared experts, differentiated routing** (architecture decision 1) — the K expert
  outputs are computed once per forward pass and only the mixture weights differ per
  objective. The cache is what makes this true in practice, and it is correct.
- **Pooled-then-broadcast routing** (decision 2) — routing is computed once from
  `pooled_hidden` and broadcast across positions via `r_ut[:, None, :, None]`. No
  per-position routing anywhere, matching the Non-Goals.
- **`e_rec ≡ e_next` at inference** (decision 3) — satisfied by indexing the same table.
- **MoE-variant component boundary** — `"none"` is a genuinely working path that
  constructs no MoE parameters; `"mmoe"` uses independent gates; `"arv_mmoe"` uses the
  shared conditioned router.
- **Global/frozen constraint** — the four MoE prefixes (`moe.router`, `moe.experts`,
  `moe.objective_embeddings`, `moe.gates`) are declared in `models/lora.py` and
  `training/federated.py` and enforced there. That enforcement is assessed under
  `specs/personalization-federated`, but the constant agrees with this spec's constraint
  and with the parameter names the module actually produces.
- **Routing distributions retained for diagnostics** — `pretrain.py` snapshots every
  objective's `r_ut` detached, accumulates them in a capped per-epoch buffer and emits
  them through `save_routing_diagnostics`; they are not discarded after the loss.

Soft routing over all experts, with no top-k sparsity, matches the Non-Goals.

## Task Completion

All tasks T1.1–T3.9 are marked `[x]` and all are genuinely complete. Verified this pass:

| Task | Verdict |
|---|---|
| T2.1 `route_objective` | Complete with the signature and return pair T2.1 specifies, handling both router types |
| T2.2 `MoELayer` incl. expert caching | Complete. The caching requirement ("compute once per batch, clear between batches") is met by a cache that is safe against pointer recycling, with `clear_cache()` releasing the retained reference |
| T2.3 `count_router_parameters` | Complete. The task text still names the old `{model}.json` path, but FR-6 was amended by the owner on 2026-09-17 and the code follows the amended requirement — requirements are authoritative over the older task text, and the amendment is recorded in FR-6 itself |
| T3.1–T3.9 tests | Complete; 18 tests pass. Several are stronger than the task text asks — T3.5 checks object identity rather than equality, and T3.4 is backed by three cache tests rather than one |

No false-complete tasks.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **275 passed**, 0 failed, 55.74 s |
| `uv run python -m pytest tests/test_moe.py -q` | **18 passed**, 0.72 s |
| `count_router_parameters` at t-base vs hand-computed formula (direct) | 5136 / 1412 / 128 / 1540 / 3596 — all four match |
| Report filename on a scratch log dir (direct) | `moe_parameter_report_t_base_arv_mmoe_K4_objective_conditioned.json` |
| `MoELayer` forward at t-base, objectives 0 and 1 (direct) | `Z` `(4, 10, 320)`, `r` `(4, 4)`; routing differs across objectives and across users |
| `MoELayer(moe_variant="none")` forward (direct) | returned tensor **is** the input object; routing weights `None` |
| `named_parameters()` on an `arv_mmoe` layer (direct) | experts + `objective_embeddings.weight` + `router.linear.{weight,bias}` — no inference-only embedding |
| `grep` MoE call sites across `main.py`, `training/`, `evaluation/` | six sites, objective indices 0–3 in `OBJECTIVE_ORDER` order; inference/scoring paths pass 0 |
| `ls outputs/logs/moe_parameter_report_*` | two run-tagged reports, confirming the tagged naming branch is live |

## Final Assessment

**Nothing for BUILD.** This spec is accepted and has been stable across rounds.

The one thing worth noting for the record rather than as a defect: T2.3's task text still
names the pre-amendment `moe_parameter_report_{model}.json` path. FR-6 was amended on
2026-09-17 to the namespaced form and the code implements the amendment, so requirements
and code agree and only the older task description lags. BUILD may sync that line, but
nothing depends on it — FR-6 carries the rationale inline, so the decision is not at risk
of being lost.
