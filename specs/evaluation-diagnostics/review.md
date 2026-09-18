# Review — Evaluation & Diagnostics

## Review Status

**PASS_WITH_WARNINGS**

(REV-133 is now resolved too, so every finding this spec owns is closed. What remains is one
INFO (REV-134). `training-pipeline` REV-138 is now closed too, so the artifacts this spec
produces are stamped with a revision that actually moves when `evaluation/` changes.)

## Feature

`specs/evaluation-diagnostics` — `evaluation/metrics.py`, `cold_start.py`, `longtail.py`,
`flops.py`, `routing_diagnostics.py`, `gradient_diagnostics.py`.

## Review Scope

- Spec directory: `specs/evaluation-diagnostics/` — `tasks.md` modified (T3.3/T3.4, and
  T8.1's figures marked superseded). `requirements.md` and `design.md` unmodified.
- Steering: `code-quality.md` (no silent substitution, mandatory sync before timing, and
  the new "Artifact Provenance" section), `conventions.md`, `testing.md`, `architecture.md`.
- Baseline: `master` at `0bc67ee`; 34 files modified after BUILD's round. This spec owns
  `evaluation/*.py` (+91 total) and `tests/test_diagnostics.py`.
- Real artifacts re-measured after the 2026-09-18 cascade: `longtail_t_small.json`,
  `gradient_diagnostics_t_small.json`, `efficiency_t_small.json`, `cold_start_t_small.json`,
  `routing_diagnostics_t_small.json`, `test_metrics_t_small.json`,
  `data/processed/popularity_buckets.pkl`.
- Re-review (third pass). Prior findings carried in: REV-133 (LOW, OPEN), REV-134 (INFO,
  OPEN), REV-132, REV-128, REV-129, REV-119 (RESOLVED).
- Date: 2026-09-18 (post-BUILD-fix)

## Summary

All of this spec's findings are closed. The artifacts were regenerated as part of the
2026-09-18 cascade, off the new checkpoint and the REV-137 training data, and every figure
was re-measured here.

- **REV-133 resolved.** T8.1's stale long-tail figures are now explicitly listed as
  *superseded values … retained so the supersession is checkable rather than merely
  asserted*, with the current values above them. That is a better resolution than deleting
  them: a reader can verify the change rather than take it on trust.
- **Long-tail (REV-132) holds through the data change.** `popularity_buckets.pkl` re-measured
  at 4,144 / 4,144 / 12,432 over 20,720, and `longtail_t_small.json` reports
  `nominal_fractions` identical to `realized_fractions_of_catalogue` at 0.20/0.20/0.60 with
  `disjoint: true`. The catalogue-wide split survived REV-137 regenerating the buckets.
- **REV-128 holds on the regenerated artifact.** Gradient cosine diagonal
  `[1.00000107, 1.00000095, 1.00000095, 1.00000155]`, max deviation **1.55e-06**, and the
  matrix maximum is 1.0000015 — no entry above 1.0, where the bf16 artifact had three.
- **Cold-start is no longer vacuous.** All four cells present on a *natural* cold set: cold
  NDCG 0.04165 / 0.04143 (metadata on / off) against warm 0.08658 / 0.07695.
- **Provenance.** All six artifacts this spec produces echo the checkpoint's
  `source_revision` exactly — verified by comparison, and their mtimes postdate the
  checkpoint.

One caveat, and it lands harder here than anywhere: the provenance enumeration does not
cover `evaluation/`, so these six artifacts carry a revision that does not move when the
code producing them changes. **That is now fixed** (`training-pipeline` REV-138): the owner
widened the coverage rule and BUILD implemented it as an exclusion rule. Verified here —
editing `evaluation/longtail.py` now MOVES `source_hash`, where before it left it
byte-identical, and all 7 `evaluation/` modules are among the 27 covered files.

REV-134 (pytest writing into the real `outputs/logs/`) is still open, unchanged and still
INFO.

## Git Changes

`evaluation/` (+91 across six files): the REV-137-driven longtail changes plus
`artifact_source_revision` stamping in `cold_start.py`, `flops.py`,
`gradient_diagnostics.py`, `routing_diagnostics.py` and `longtail.py`.
`tests/test_diagnostics.py` (+11): the catalogue-denominator bucket assertions.

**Scope assessment:** proportionate. The evaluation modules gained provenance plumbing and
nothing else; the metric, diagnostic and timing logic is untouched, confirmed by diff and by
the 38 tests across `test_diagnostics.py` / `test_metrics.py` / `test_ann_retrieval.py`
passing unchanged.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Fixed candidate protocol | PASS | Every consumer reads the stored fingerprinted rows through `load_or_build_eval_candidates` / `load_eval_candidates`; the loaders recompute and compare the block fingerprint on every load, including through the memory-mapped path |
| FR-2 ANN retrieval | PASS | `models/ann_retrieval.py` retrieves, filters seen/invalid, force-includes the target by replacing the lowest-scoring candidate, and returns `num_force_included` / `num_examples`; `main.py` surfaces the rate |
| FR-3 Metrics | PASS | Single-target formulas implemented exactly; per-example values plus mean and std. `tests/test_metrics.py` pins rank 1, rank 10 and rank 11 |
| FR-4 Cold-start evaluation | PASS | `cold_start_t_small.json` carries all four cells — `cold_meta_on`, `cold_meta_off`, `warm_meta_on`, `warm_meta_off` — and `cold_vs_warm_t_small.png` renders |
| FR-5 Long-tail evaluation | PASS | **Now satisfied clause by clause for the first time.** Measured on the artifact: `bucket_populations` 4,144 / 4,144 / 12,432; `bucket_partition` reports `disjoint: true`, `complete_union_size: 20720`, and `nominal_fractions` identical to `realized_fractions_of_catalogue` at 0.20/0.20/0.60. Popularity is computed from the training history only |
| FR-6 Latency benchmarking | PASS | `_synchronize` dispatches `torch.mps.synchronize()` / `torch.cuda.synchronize(device)` and is called before the timed loop and around **each** timed run. Warmup runs discarded; all five required statistics reported |
| FR-7 Efficiency report | PASS | Every field present in `efficiency_t_small.json`, including `routing_overhead` and the REV-110 provenance fields (`timed_metadata_source`, `timed_with_metadata`, `timed_padding_positions`, `timed_sequence_length`) |
| FR-8 Routing diagnostics | PASS | Entropy, utilization, pairwise cosine/KL, the `(T, K)` heatmap and both collapse tests are implemented and present in `routing_diagnostics_t_small.json`; three plots render |
| FR-9 Gradient diagnostics | PASS | `(T, T)` cosine matrix and per-objective norms with CV, computed on freshly drawn diagnostic batches rather than optimizer-step batches, now in fp32 |
| FR-10 Calibration | PASS | `use_logit_calibration=False` and `eval_temperature=1.0` by default; the temperature divisor is applied only when calibration is enabled, and nothing claims it changes ranking |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 same candidate sets for all evaluations | PASS | One stored matrix per split, fingerprint-verified on every load; `TestCandidateSetConsistency` asserts the rows the long-tail and cold-start evaluators hand to the scorer are the stored rows |
| AC-02 metrics correct on toy inputs | PASS | `tests/test_metrics.py` |
| AC-03 cold items disjoint from training | PASS | Asserted in `evaluation/cold_start.py` and re-measured against the regenerated artifacts under `specs/data-pipeline`: **3,010 naturally-cold items**, `cold ∩ train_interacted = ∅`, and `train_interactions_removed_by_holdout: 0`. `cold_start_t_small.json` was regenerated off the new checkpoint (15:38) and now reports all four cells on a natural cold set: cold NDCG 0.04165 / 0.04143 (metadata on / off) against warm 0.08658 / 0.07695 — the cold-start question is no longer vacuous |
| AC-04 long-tail buckets partition all item IDs | PASS | Measured on both the pickle and the JSON: sizes sum to 20,720 = `num_items`, `disjoint: true`, `complete_union_size: 20720`. Pinned by two tests, one of which I verified fails on the superseded scheme |
| AC-05 latency includes device synchronization | PASS | Verified in code at three call sites in the timing loop |
| AC-06 heatmap shape `(num_objectives, num_experts)` | PASS | `TestRoutingDiagnostics` checks the shape and the entropy extremes (uniform → max, delta → zero) |
| AC-07 cosine matrix symmetric with 1s on diagonal | PASS | Measured on the real artifact: diagonal within **1.79e-06** of 1.0, no entry above 1.0. The test now asserts `< 1e-4` on the diagonal, a documented `FP32_ULP = 1e-5` range tolerance, and `< 1e-6` symmetry — replacing a `< 0.01` bound wide enough to accept the 1.0039 the bf16 artifact actually contained |
| AC-08 all JSON reports and PNG plots generated | PASS | All six logs and all eight plots present in the expected locations |
| AC-09 ANN force-inclusion rate reported | PASS | Reported when `use_ann=True`; `tests/test_ann_retrieval.py` covers the force-inclusion path |

## Findings

```
REV-132
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-5 | AC: AC-04
File: evaluation/longtail.py:36-86 (compute_popularity_buckets)
This is the long-tail bucketing question carried as an open item for the spec owner across
  several rounds, now decided (2026-09-18) and implemented. The split is 20/20/60 of the
  WHOLE CATALOGUE. No requirements change was needed -- FR-5 meant what it literally says,
  and the code now matches it.
Resolution, measured on the regenerated artifacts rather than on the diff:
  head 3,356 -> 4,144, mid 3,357 -> 4,144, tail 14,007 -> 12,432; realized catalogue
  fractions 0.162/0.162/0.676 -> exactly 0.200/0.200/0.600. `popularity_buckets.pkl` and
  `longtail_t_small.json` agree, and the JSON's `nominal_fractions` and
  `realized_fractions_of_catalogue` are identical -- the two fields T3.3 added exist
  precisely so this is checkable from the file.
Verification (independent): the superseded scheme was reinstated in the working copy and
  the suite re-run. `test_disjoint_complete_correct_percentiles` FAILS on it and passes on
  the fix; 21 other tests unaffected. Source restored.
Design note, and the right call: the zero-interaction clause is now a CONSEQUENCE of the
  ordering rather than a separate step, so it is asserted rather than enforced -- with an
  explicit `ValueError` naming the incompatibility for the one dataset shape where FR-5's
  first and second clauses genuinely conflict. Unreachable here (19.0% zero-interaction
  against a 60% tail), but it fails loudly instead of silently choosing a clause, which is
  what `steering/code-quality.md` requires.
```

```
REV-128
Severity: LOW | Status: RESOLVED | Requirement: FR-9 | AC: AC-07
File: evaluation/gradient_diagnostics.py (gradient_cosine_similarity,
      per_objective_grad_norms)
Resolution verified on the artifact, which is where the defect was visible: the cosine
  diagonal now reads [1.00000107, 1.00000095, 1.00000095, 1.00000179] -- maximum deviation
  1.79e-06, against the bf16 artifact's 3.91e-03, with three of four entries previously
  ABOVE 1.0 and none above now. The flattened gradients are cast with `.float()` at the
  measurement site, matching the pattern `routing_diagnostics` and `local_train` already
  use.
The test tightening matters as much as the fix: the old bound was `< 0.01`, wide enough to
  accept the 1.0039 that was actually being produced. It is now `< 1e-4` on the diagonal, a
  documented `FP32_ULP = 1e-5` range assertion, and `< 1e-6` on symmetry.
```

```
REV-129
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-4, FR-5, FR-7 | AC: AC-08
File: outputs/logs/, outputs/plots/
Resolution: all missing artifacts produced and verified against the files --
  `test_metrics_t_small.json`, `cold_start_t_small.json` (all four cold/warm x metadata
  cells), `longtail_t_small.json`, `efficiency_t_small.json` (every FR-7 field, including
  `timed_metadata_source == "item_metadata"`, which confirms REV-110 on a real run for the
  first time), and the regenerated gradient diagnostics. AC-08 is met for t-small.
Worth recording: BUILD corrected three factual errors in this finding's own stated
  procedure -- the T6.3 pretrain gate does NOT regenerate gradient diagnostics (the hook
  fires at `grad_diag_interval_steps = 500` and the gate runs 200 steps); the
  `gradient_diagnostics` and `routing_diagnostics` stages were dead on a `NameError` and
  needed fixing first; and `--stage evaluate` writes only `test_metrics_*.json`, not the
  other three reports. Correcting the finding rather than working around it is the right
  response, and the corrections check out against the code.
```

```
REV-119
Severity: LOW | Status: RESOLVED | Requirement: FR-1 | AC: AC-01
File: data/sequence_generator.py (load_eval_candidates, load_or_build_eval_candidates);
      training/local_train.py
Resolution: `mmap: bool = False` added to both loaders, so every existing caller keeps the
  eager path unchanged, and only the row-sliced consumer (`run_local_personalization`)
  passes `mmap=True`. The block-wise fingerprint reads a memmap without materialising it,
  so the mapped path runs the same integrity check -- and the extended test asserts that
  verification still RAISES on a corrupted meta file through the mapped path, which is the
  assertion that matters: a fast path that silently skips integrity checking would be
  worse than the copy it saves.
  The consequence BUILD handled is the kind that is easy to miss: a read-only memmap makes
  `torch.as_tensor` alias a non-writable buffer, so the per-row conversion now copies 101
  elements explicitly. `federated.py` and `longtail.py` were deliberately left eager
  because both consume the whole matrix -- correct reasoning, not an oversight.
```

```
REV-093 / REV-094 / REV-095 / REV-109 / REV-110
Severity: MEDIUM / LOW / LOW / LOW / LOW | Status: RESOLVED
Re-confirmed. Candidate-set routing goes through the single loader everywhere; the
  long-tail report carries its bucket populations and partition block; latency timing
  synchronises around every timed run; and REV-110's metadata-provenance fields are
  present in the real efficiency artifact (`timed_metadata_source == "item_metadata"`),
  which closes it on evidence rather than on the diff alone.
```

```
REV-133
Severity: LOW | Status: RESOLVED | Requirement: FR-5 | AC: —
File: specs/evaluation-diagnostics/tasks.md (T8.1)
Resolution: T8.1's superseded figures are now collected under an explicit *"Superseded values
  (2026-09-17/18), retained so the supersession is checkable rather than merely asserted"*
  block -- cold 0.05774/0.05485, warm 0.09286/0.08093, buckets 3,356 + 3,357 + 14,007 with
  head NDCG 0.09937 -- sitting below the current values. Retaining them beats deleting them:
  a reader can verify the change rather than take it on trust, which is the same standard
  REV-130 asked of T6.3.
  The note also records the second reason they are superseded -- REV-137 changed the training
  data underneath -- so the supersession is attributed to both causes, not just the bucketing.
```

```
REV-134
Severity: INFO | Status: OPEN | Requirement: — | AC: —
File: outputs/logs/moe_parameter_report_t_base_A_arv_mmoe_gradnorm*.json
Note: running `pytest` writes into the project's real `outputs/logs/`. Both MoE parameter
  reports were rewritten during this review's test runs (mtime 2026-09-18 13:34, matching
  the suite invocation). `outputs/` is gitignored so nothing is corrupted in version
  control, and these two files are pure functions of the config so their contents do not
  change -- but a test suite that mutates the deliverable directory can in principle
  overwrite a real run's artifact with a fixture's. Worth a `tmp_path` log_dir in the
  ablation tests at some point. No action required, and not a defect in this spec's code.
```

## Design Compliance

Compliant. The constraints this spec places on itself are all respected:

- **Fixed candidate sets generated once and reused, never regenerated per model** — held
  by routing every consumer through the shared loaders, and now held through the
  memory-mapped path as well.
- **Diagnostics use declared diagnostic data, never optimizer-step batches** — the
  gradient hook draws from a separate `diag_loader`, and the caller re-zeros gradients
  after the excursion so the diagnostic pass cannot contaminate the training step.
- **Report without editorializing** — the artifacts carry measured numbers. The routing
  utilization `[0.065, 0.311, 0.395, 0.229]` against a uniform 0.25 is recorded as an
  observation from a capped run and not interpreted, which is exactly what the constraint
  asks for.
- **No silent substitution** — the new `ValueError` in `compute_popularity_buckets` is the
  clearest instance of this rule being applied deliberately rather than by habit.

## Task Completion

| Task | Marked | Verdict |
|---|---|---|
| T3.3 record bucket populations and partition | `[x]` | Complete, and the fields earned their keep immediately — they are what makes T3.4's 0.20/0.20/0.60 checkable from the artifact rather than by re-deriving from the pickle |
| T3.4 catalogue-wide split | `[x]` | Complete. Every measured claim re-verified here, and the non-vacuity claim reproduced independently. The cache hazard BUILD names is real and was handled: `run_longtail_evaluation` loads `popularity_buckets.pkl` whenever it exists, so a stale pickle would have survived the code change silently |
| T6.5 fp32 gradient measurement | `[x]` | Complete and confirmed on the artifact, not only in the unit test |
| T6.6 memory-mapped candidates | `[x]` | Complete, with the integrity check preserved through the mapped path and the non-writable-buffer consequence handled |
| T8.1 stage execution | `[x]` | Complete as to execution and artifacts. Its recorded long-tail figures are stale relative to T3.4 — REV-133 |
| T1.x–T7.x | `[x]` | Complete. Verified against the code, the artifacts and 38 passing tests across the three relevant test modules |

No false-complete tasks.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **275 passed**, 0 failed, 55.74 s |
| `uv run python -m pytest tests/test_diagnostics.py tests/test_metrics.py tests/test_ann_retrieval.py -q` | **38 passed**, 3.45 s |
| `uv run python -m pytest tests/test_diagnostics.py -q` **with the superseded bucketing reinstated** | **1 failed, 21 passed** — `test_disjoint_complete_correct_percentiles`. Source restored |
| Read `data/processed/popularity_buckets.pkl` (direct) | head 4,144 / mid 4,144 / tail 12,432, sum 20,720 |
| Read `outputs/logs/longtail_t_small.json` (direct) | `bucket_populations` as above; `bucket_partition` `disjoint: true`, `complete_union_size: 20720`, nominal == realized at 0.20/0.20/0.60; head NDCG 0.09744 > mid 0.04760 > tail 0.04623 |
| Read `outputs/logs/gradient_diagnostics_t_small.json` (direct) | diagonal `[1.0000011, 1.0000010, 1.0000010, 1.0000018]`; max deviation 1.79e-06; no entry > 1.0 |
| Read `outputs/logs/efficiency_t_small.json` (direct) | all FR-7 fields present, including `timed_metadata_source`, `routing_overhead`, `flops_per_inference` |
| Read `outputs/logs/cold_start_t_small.json` (direct) | all four cells: `cold_meta_on/off`, `warm_meta_on/off` |
| `grep -n synchronize evaluation/flops.py` | `_synchronize` defined with both MPS and CUDA branches; called at three points in the timing loop |
| `ls outputs/plots/` | all eight plots present |

## Final Assessment

**Nothing for BUILD in this spec. It is accepted.**

Every finding this spec owns is closed, and the two that mattered were closed in the way
that leaves the decision auditable: the long-tail supersession is recorded with both the old
and new values, and T8.1's stale figures are retained-and-labelled rather than quietly
overwritten.

Worth noting that the long-tail work survived a change it did not anticipate. REV-137
regenerated `popularity_buckets.pkl` from different training data, and the catalogue-wide
split still lands at exactly 4,144 / 4,144 / 12,432 — because the rule is expressed as a
property of the catalogue rather than of a particular popularity distribution. That is the
difference between a fix and a fix that holds.

Two items carried, neither actionable here:

1. **`training-pipeline` REV-138 — RESOLVED.** The provenance enumeration omitted
   `evaluation/`, so the six artifacts this spec produces carried a revision that did not move
   when their producing code changed. The owner widened the rule to an exclusion rule and BUILD
   implemented it; verified here, editing `evaluation/longtail.py` now moves `source_hash`, and
   all 7 `evaluation/` modules are covered. This spec was the strongest argument for the change
   and is the main beneficiary of it. Note existing artifacts keep their pre-widening hash and
   are **not** to be re-stamped (REV-140).
2. **REV-134 (INFO).** `pytest` still writes `moe_parameter_report_*.json` into the real
   `outputs/logs/` — timestamps moved again during this review's test runs. Harmless today
   (the contents are a pure function of config, and `outputs/` is gitignored), but a suite
   that mutates the deliverable directory can in principle overwrite a real run's artifact.
   A `tmp_path` log dir in the ablation tests closes it whenever it is convenient.
