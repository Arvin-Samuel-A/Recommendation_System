# Review — Evaluation & Diagnostics

## Review Status

**CHANGES_REQUIRED**

## Feature

`specs/evaluation-diagnostics` — `evaluation/metrics.py`, `cold_start.py`, `longtail.py`, `flops.py`, `routing_diagnostics.py`, `gradient_diagnostics.py`, and `models/ann_retrieval.py`: fixed candidate protocol, four metrics, ANN retrieval, cold-start, long-tail, latency/efficiency, routing diagnostics, gradient-interference diagnostics.

## Review Scope

- Spec directory: `specs/evaluation-diagnostics/` (requirements.md, design.md, tasks.md)
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`
- Baseline: working tree vs `HEAD` = `32b0904` on `master`. All feature work is uncommitted.
- Files inspected: `evaluation/metrics.py` (214), `cold_start.py` (194, untracked), `longtail.py` (184, untracked), `flops.py` (203, untracked), `routing_diagnostics.py` (166), `gradient_diagnostics.py` (299), `models/ann_retrieval.py` (163, untracked), `tests/test_metrics.py`, `tests/test_diagnostics.py`, `tests/test_ann_retrieval.py`, plus the call sites in `training/pretrain.py` and `main.py`
- Re-review of the prior `review.md`. `metrics.py` (15:37), `flops.py` (21:36), `ann_retrieval.py` (21:37) and `tests/test_diagnostics.py` (21:36) were modified after it.
- Date: 2026-09-15

## Summary

All five prior findings are fixed. REV-063's per-example forward loop is gone — `score_candidates_batch` scores `(B, C)` candidates for `(B, S)` sequences in one pass, and `evaluate_model` batches at `eval_batch_size`. REV-076 is fixed better than recommended: `compute_metrics_batch` computes the rank as `(scores > target_score).sum() + 1`, so there is no sort and no per-example device sync at all. REV-064 is fixed — `models/ann_retrieval.py` now exists, implements the full force-inclusion protocol, is wired to `config.use_ann` in `run_evaluate_stage`, and has 9 tests. REV-059b and REV-035b are both closed with tests that check the property rather than the shape.

What is still wrong is the candidate-set protocol. The spec's Constraints say "Fixed evaluation candidate sets generated once and reused — never regenerated per model", and `data/processed/test_candidates.npy` exists for exactly that purpose. `--stage evaluate` and the ablation runner load it. `cold_start.py` and `longtail.py` do not — they rebuild candidates per call, and `longtail.py` does so without an RNG, which makes `filter_negatives` seed from the target item. I verified the consequence directly: **every test example sharing a target gets byte-identical negatives**, and the resulting sets differ from the stored ones. The long-tail bucket comparison is the one measurement most sensitive to that.

## Git Changes

| File | State |
|---|---|
| `evaluation/metrics.py` | modified |
| `evaluation/routing_diagnostics.py` | modified |
| `evaluation/gradient_diagnostics.py` | modified |
| `evaluation/cold_start.py`, `longtail.py`, `flops.py` | untracked (new) |
| `models/ann_retrieval.py` | untracked (new) |
| `tests/test_diagnostics.py` | modified |
| `tests/test_ann_retrieval.py` | untracked (new) |
| `specs/evaluation-diagnostics/tasks.md` | modified |
| `specs/evaluation-diagnostics/review.md` | untracked (this file) |

**Scope assessment:** in scope. `models/ann_retrieval.py` lands under `models/` exactly where `conventions.md`'s file layout puts it. No external recommendation libraries — retrieval is brute-force `torch.topk` over the item table, which respects the "no RecBole/Surprise/pytorch-fm" anti-pattern and is strictly stronger than an approximate index.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Fixed candidate protocol | PARTIAL | Sets are pre-stored (`val_candidates.npy`, `test_candidates.npy`, fingerprinted) and loaded by `main.py::run_evaluate_stage` ([main.py:194](main.py:194)), `pretrain.py::_load_or_build_val_candidates` and `ablation_runner.py:313`. Each set is one target at column 0 plus `num_sampled_negatives` filtered against pad/mask/target/eval-history. But `cold_start.py`, `longtail.py`, `local_train.py` and `federated.py` regenerate instead of loading — REV-093. |
| FR-2 ANN retrieval | PASS | `retrieve_candidates` ([ann_retrieval.py](models/ann_retrieval.py)) scores the full vocabulary, masks pad/MASK/out-of-range, filters seen items with the target exempt, takes top-K, force-includes the target into the lowest-scoring slot, moves it to column 0, and returns `force_inclusion_rate`. Wired at [main.py:228-242](main.py:228) behind `config.use_ann`; the rate is written into `test_metrics_{model}.json`. |
| FR-3 Metrics | PASS | `compute_metrics_batch` ([metrics.py:11](evaluation/metrics.py:11)) implements all four single-target formulas; `aggregate_metrics` returns mean and std over either stacked tensors or a list of dicts. Multi-seed mean±std across independent runs is `main.py::run_multi_seed_stage` (ablations-cli FR-15). |
| FR-4 Cold-start evaluation | PASS | `run_cold_start_evaluation` asserts cold∩train is empty against the real `train_user_history.pkl` before doing anything, classifies each target, runs a genuine metadata-ON and metadata-OFF arm (REV-001's fix holds — the two arms now differ by an actual `metadata=` argument), reports cold/warm × ON/OFF, writes `cold_start_{model}.json` and `cold_vs_warm_{model}.png`. |
| FR-5 Long-tail evaluation | PARTIAL | Buckets are 20/20/60 from training popularity with zero-training items in tail and a partition assertion (REV-007's fix holds); report and plot written. Two problems in the evaluation itself: candidates are regenerated with target-seeded negatives (REV-093) and the scoring path omits metadata (REV-094). |
| FR-6 Latency benchmarking | PASS | `benchmark_latency` ([flops.py:25](evaluation/flops.py:25)) discards `warmup_runs`, times `timed_runs` with `_synchronize` before and after each, and reports mean/std/min/p95/ms_per_recommendation. REV-035b's fix holds: the dummy request is built from valid item IDs, so it takes the `is_causal=True` fast path that real serving takes. `measure_routing_overhead` has a caller ([main.py:358](main.py:358)). |
| FR-7 Efficiency report | PASS | `save_efficiency_report` writes every named field; `flops_per_inference` is injected by `run_latency_stage` ([main.py:365](main.py:365)) before the call, and both `model_size_fp32_mb` and `model_size_bf16_mb` are present. `moe_variant`, `num_experts` and `routing_overhead` are recorded alongside. |
| FR-8 Routing diagnostics | PASS | Entropy, utilization, pairwise cosine + KL, `(T, K)` heatmap and both collapse tests are all implemented. Periodic in-training passes via `_emit_routing_diagnostics` every `routing_diag_interval_epochs`, plus the comprehensive end-of-training validation pass via `collect_validation_routing` ([pretrain.py:889-894](training/pretrain.py:889)) and the standalone `--stage routing_diagnostics`. All three plots written at `config.plot_dpi`. |
| FR-9 Gradient diagnostics | PASS | `(T, T)` cosine matrix and per-objective norms with CV, computed on freshly-drawn batches from a separate loader (never optimizer-step batches), every `grad_diag_interval_steps` over `grad_diag_num_batches`. The contrastive objective participates (REV-009b), and the masked-item arm now sees masked metadata (REV-059b). Both plots written. See REV-109 for a mode-restoration nit. |
| FR-10 Calibration | PASS | `use_logit_calibration=False` and `eval_temperature=1.0` by default; when enabled, `score_candidates_batch` divides by temperature. No temperature is ever fitted, so the "fit on validation only, never on test" clause cannot be violated. Nothing in the code claims temperature changes ranking. |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 same candidate sets across model evaluations | PARTIAL | Holds for `--stage evaluate` and every ablation arm, which load the same fingerprinted `test_candidates.npy`. Does not hold across evaluation *types*: I confirmed `longtail.py`'s regenerated sets differ from the stored ones, while `cold_start.py`'s happen to coincide. The test named for this criterion does not involve models at all — REV-095. |
| AC-02 metrics correct on toy inputs | PASS | `tests/test_metrics.py` covers rank 1, rank 10, rank 11 (miss), an exact NDCG value and the single-candidate edge case. |
| AC-03 cold items disjoint from training in cold-start eval | PASS | Asserted at runtime ([cold_start.py:53-57](evaluation/cold_start.py:53)) with a message naming the offenders; independently confirmed against the real artifacts (2,072 cold items, zero overlap with `train_user_history.pkl`). |
| AC-04 long-tail buckets partition all item IDs | PASS | Asserted in `compute_popularity_buckets`; `TestLongtailBucketPartition::test_disjoint_complete_correct_percentiles` and `test_exact_20_20_60_proportions`. Confirmed against the real `popularity_buckets.pkl`. |
| AC-05 latency includes device synchronization | PASS | `_synchronize(device)` before and after every timed run ([flops.py:71-80](evaluation/flops.py:71)); MPS and CUDA branches both present. |
| AC-06 heatmap shape `(num_objectives, num_experts)` | PASS | `TestHeatmapShape::test_shape_matches_objectives_experts`. |
| AC-07 gradient cosine matrix symmetric, 1s on diagonal | PASS | `TestGradientCosineSimilarity::test_symmetric_with_ones_diagonal`. |
| AC-08 all JSON and PNG in correct locations | PASS (by inspection) | Every writer uses `config.log_dir` / `config.plot_dir` with the `{model}` name normalised to underscores and `dpi=config.plot_dpi`. No run has produced these artifacts yet (training-pipeline REV-106), so this is code-level verification only. |
| AC-09 ANN force-inclusion rate reported | PASS | `retrieve_candidates` returns and logs it; `run_evaluate_stage` writes it into `test_metrics_{model}.json` under `ann_retrieval`. `TestForceInclusion` (3 tests) covers the replace-lowest and target-at-column-0 behaviour. |

## Findings

```
REV-093
Severity: MEDIUM | Status: OPEN | Requirement: FR-1, FR-5 | AC: AC-01
File: evaluation/longtail.py:111-114 (`run_longtail_evaluation`);
       evaluation/cold_start.py:112-114 (`run_cold_start_evaluation`);
       training/local_train.py:266 (`run_local_personalization`)
Problem: The spec's Constraints say "Fixed evaluation candidate sets generated once and
  reused -- never regenerated per model", and `data/processed/test_candidates.npy`
  (1,059,494 x 101, fingerprinted) exists for that. `--stage evaluate` and
  `ablation_runner._run_single_config` load it. These three call sites rebuild candidates
  per call with `filter_negatives` instead. Worse, `longtail.py` and `local_train.py` pass
  no `rng=`, so `filter_negatives` falls back to `RandomState(int(target) % ...)` -- seeded
  from the TARGET ITEM.
  Reproduced on a toy pool: two examples with the same target and different users/histories
  received byte-identical negative sets, and neither matched the stored sets:
      stored[0]   : [10, 103, 436, 349, 271, 107,  72, 189,  21]
      longtail[0] : [10, 266, 126, 485,  16, 321, 370, 124, 157]
      longtail rows 0 and 2 (same target, different user): identical negatives -> True
  `cold_start.py` does pass a shared `RandomState(config.seed)` and therefore happens to
  reproduce the stored sets exactly -- but only by coincidence of iteration order, not by
  construction.
Impact: The head/mid/tail numbers in `longtail_{model}.json` are computed against a
  different candidate protocol than the headline `test_metrics_{model}.json`, so the two
  are not on the same scale and cannot be quoted side by side. Within the long-tail study
  itself, an item's difficulty is now a deterministic function of its ID: a head item
  appearing thousands of times in the test split is scored against the SAME 100 negatives
  every time, so per-bucket errors are correlated across examples and the reported std
  understates the true variance -- in the one study whose entire point is comparing across
  item-popularity buckets.
Correction: Load the stored sets. `longtail.py` and `cold_start.py` should read
  `test_candidates.npy` via `load_eval_candidates` and slice the rows for the batch, the
  way `ablation_runner.py:313` already does -- the stored rows are already filtered against
  pad, mask, target and eval history, which is exactly what these sites recompute. Keep
  `filter_negatives` only for the genuine fallback path, and when it is used, always pass
  an explicit `rng`. Add a test asserting that `run_longtail_evaluation` and
  `run_evaluate_stage` score against the same candidate rows for the same example indices.
```

```
REV-094
Severity: LOW | Status: OPEN | Requirement: FR-5 | AC: —
File: evaluation/longtail.py:124 (`score_candidates_batch(model, seqs, cands, config,
       moe_layer=moe_layer)` -- no `metadata=`)
Problem: The backbone is pretrained with `use_metadata=True`, the best checkpoint is
  selected on a validation pass that supplies metadata, `--stage evaluate` supplies
  metadata (REV-017's fix), and `cold_start.py`'s meta_on arm supplies metadata. The
  long-tail evaluation supplies none (`evaluation/longtail.py:124`), so the model runs
  with the metadata projection contributing nothing.
Impact: Long-tail metrics come from a different serving path than every other reported
  number -- the same defect class as REV-062b, which was fixed for local personalization.
  Absolute NDCG per bucket is not comparable with the headline test metrics, and the
  metadata channel is precisely what is expected to help tail items most, so the study
  systematically understates the model it is evaluating.
Correction: Load the aligned matrix with `load_metadata_matrix(processed, config)` and
  pass `metadata=meta_matrix[seqs]` to `score_candidates_batch`, exactly as
  `cold_start.py:126-133` does for its meta_on arm. If a metadata-free long-tail arm is
  wanted, make it an explicit second arm rather than the only one.
```

```
REV-095
Severity: LOW | Status: OPEN | Requirement: FR-1 | AC: AC-01
File: tests/test_diagnostics.py:261-275 (`TestCandidateSetConsistency`);
       specs/evaluation-diagnostics/tasks.md:61 (T7.1)
Problem: T7.1 is marked [x] -- "Load candidate sets from two 'different model' evaluation
  calls -> verify sets are identical. (AC-01)". The test constructs no model and performs
  no evaluation. It calls `generate_eval_candidates` twice with identical arguments and
  asserts the two results match, i.e. it tests that a seeded generator is deterministic --
  which is a different property, and one already covered by
  `tests/test_sequence_generator.py::TestEvalCandidates::test_identical_across_configs`.
Impact: AC-01 is unverified. The real risk it exists to catch is exactly REV-093: two
  evaluation entry points scoring the same example against different candidate sets. The
  current test cannot detect that, and indeed does not.
Correction: Test the property at the evaluation boundary. Build a small processed
  directory, run two `evaluate_model` calls with different model configs against the
  loaded `test_candidates.npy`, and assert the candidate rows consumed are identical --
  then extend it to `run_longtail_evaluation` and `run_cold_start_evaluation` once REV-093
  is fixed. Leave T7.1 unchecked until such a test exists.
```

```
REV-109
Severity: LOW | Status: OPEN | Requirement: FR-9 | AC: —
File: evaluation/gradient_diagnostics.py:136 (`model.eval()`), no restore on exit
Problem: `run_gradient_diagnostics` puts the model into eval mode and never restores the
  previous mode. The in-training caller compensates immediately afterwards
  (`training/pretrain.py:821-822` calls `model.train()` / `moe.train()`), so the live path
  is correct -- but the invariant is held by the caller rather than by the function that
  broke it. `moe_layer` is never switched at all, so its dropout stays in whatever mode it
  arrived in while the model's does not; the two are inconsistent for the duration of the
  diagnostic.
Impact: None today. It is a trap: any future caller that runs the diagnostic mid-training
  and does not know to re-enter train mode will silently continue training with dropout
  disabled.
Correction: Save `model.training` and `moe_layer.training` on entry and restore both on
  exit, the way `training/pretrain.py::collect_validation_routing:517-547` already does.
  The explicit `model.train()` in the training loop can stay as belt-and-braces.
```

```
REV-110
Severity: LOW | Status: OPEN | Requirement: FR-6 | AC: AC-05
File: evaluation/flops.py:52-57 (`dummy_meta = torch.zeros(...)`)
Problem: REV-035b correctly replaced the all-padding dummy sequence with valid item IDs so
  the timed path is the serving path. The metadata tensor beside it is still
  `torch.zeros(1, S, feature_dim)`. A real request gathers `metadata_matrix[seq]`, whose
  rows are dense genome-relevance floats.
Impact: Small. The matmul shape and cost are identical, so `mean_ms` is essentially
  unaffected; only the values differ. Recorded because FR-6 says "full serving path" and
  this is the last remaining synthetic stand-in on that path -- and because an all-zero
  metadata block is exactly what a metadata-free model produces, which makes the timed
  input ambiguous between the two configurations.
Correction: Build the dummy metadata by gathering from the real matrix when it is
  available -- `metadata_matrix[dummy_seq]` -- and fall back to `torch.randn` otherwise.
  The report already records `timed_with_metadata`; add the source so a reader can tell.
```

## Design Compliance

Matches `design.md` and the steering constraints, with one architectural addition that was the previous review's own recommendation.

- **Serving path identity** — `score_candidates_batch` and `ann_retrieval._user_embeddings` both build the user embedding the same way (`all_hidden[:, -1, :]` → `moe(…, objective_idx=0)` → `Z[:, -1, :]` → optional L2), so `e_rec ≡ e_next` holds on every scoring path (moe-routing FR-9).
- **Diagnostics use declared diagnostic data** — `run_gradient_diagnostics` draws from its own `diag_loader`, never the optimizer's batch, satisfying the Constraint and FR-9.
- **No editorializing** — the routing report emits entropy, utilization, cosine, KL and the two collapse fractions as numbers with no interpretation, as the Constraints require.
- **`conventions.md` naming** — every JSON and PNG uses the `t_small`/`t_base`/`t_large` underscore spelling, and plots are 150 DPI via `config.plot_dpi`.
- **New component:** `models/ann_retrieval.py`. `design.md` did not scope a module here, but `conventions.md`'s file layout names it and T1.3 required it; the prior review's REV-064 explicitly recommended implementing it with brute-force retrieval. Not a silent redesign.

One boundary worth flagging for the owner rather than for BUILD: `objective_expert_heatmap` returns one row per *enabled* objective, so under Ablation F (one objective disabled) the matrix is `(3, K)` rather than `(4, K)`. AC-06 says `(num_objectives, num_experts)`. The Edge Cases already anticipate disabled objectives, so this reads as intended behaviour, but the AC wording does not say so.

## Task Completion

25 of 25 tasks are `[x]`. Twenty-three verify; two do not.

| Task | Verdict |
|---|---|
| T1.1 metric functions | Complete. |
| T1.2 evaluation loop | Complete, and now batched. |
| T1.3 ANN retrieval path | Complete — this was the previous review's REV-064 false-complete and the module now genuinely exists, is wired, and has 10 tests. |
| T1.4 calibration | Complete (applied when enabled; disabled by default; no fitting performed, so no leakage risk). |
| T1.5 metrics tests | Complete, including the rank-11 miss edge case. |
| T2.1–T2.2 cold-start | Complete, including the runtime disjointness assertion and a real ON/OFF comparison. |
| T3.1 popularity buckets | Complete. |
| **T3.2 long-tail evaluation** | **Partially false-complete.** The bucket classification, report and plot are correct, but the evaluation scores against regenerated, target-seeded candidates (REV-093) and without metadata (REV-094), so it is not the protocol FR-1 and FR-5 describe. |
| T4.1–T4.3 latency, overhead, efficiency | Complete; `measure_routing_overhead` now has a caller. |
| T5.1–T5.5 routing diagnostics | Complete, with both the periodic and comprehensive passes wired. |
| T6.1–T6.4 gradient diagnostics | Complete; contrastive included, fresh batches, both plots. |
| **T7.1 candidate set consistency test** | **False-complete.** The test exercises generator determinism, not cross-evaluation consistency (REV-095). |
| T7.2–T7.4 diagnostics tests | Complete and non-vacuous. |

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` (full suite) | 231 passed, 2 failed, 135 s. Neither failure is in this spec — both are `transformer-backbone` REV-062's armed MPS backend gate. |
| `uv run python -m pytest tests/test_metrics.py tests/test_diagnostics.py tests/test_ann_retrieval.py -q` | All pass (included in the run above). |
| Scratch: compare `generate_eval_candidates` output against `cold_start.py`'s and `longtail.py`'s regeneration patterns on a toy pool | `cold_start == stored: True`; `longtail == stored: False`; two rows sharing a target got identical negatives — REV-093 confirmed. |
| Direct artifact check: `test_candidates.npy` | `(1059494, 101)`, matches `test_examples.pkl`; target at column 0 for the first 1,000 rows; no duplicates, no pad/mask in row 0. |
| Direct artifact check: `popularity_buckets.pkl` vs `cold_items.pkl` | Buckets partition the catalogue; cold set disjoint from all training interactions — AC-03 and AC-04 confirmed on real data. |
| Scratch: `count_parameters(model, moe)` on a t-small build | Returns every FR-7 field; `router_parameters` and `expert_parameters` split correctly by name. |
| `grep -rn "load_eval_candidates\|generate_eval_candidates\|filter_negatives("` across non-test source | Confirms the split: `main.py` and `ablation_runner.py` load; `cold_start.py`, `longtail.py`, `local_train.py`, `federated.py` regenerate. |

## Final Assessment

The evaluation and diagnostics code is in good shape — every prior finding is fixed, the batching rewrite is correct rather than merely faster, and the ANN module closes the one requirement that previously had no code behind it. The remaining work is the candidate-set protocol, which is a genuine correctness issue for the long-tail study rather than a style point.

BUILD should, in order:

1. **REV-093** — make `longtail.py` and `cold_start.py` load `test_candidates.npy` through `load_eval_candidates` instead of rebuilding, and pass an explicit `rng` anywhere `filter_negatives` survives as a fallback. This is the one finding that changes reported numbers.
2. **REV-094** — pass metadata in `run_longtail_evaluation`, matching every other scoring path.
3. **REV-095** — replace `TestCandidateSetConsistency` with a test at the evaluation boundary, then re-mark T7.1. Leave it unchecked until then.
4. **REV-109** and **REV-110** — small, independent, do whenever convenient.

FR-1, FR-5 and AC-01 stay PARTIAL until REV-093 and REV-094 land. AC-08 is code-verified only; it will need a real run's artifacts, which is blocked behind `specs/transformer-backbone` REV-062 and tracked in `specs/training-pipeline/review.md` REV-106.
