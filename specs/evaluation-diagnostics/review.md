# Review — Evaluation & Diagnostics

## Review Status

**CHANGES_REQUIRED**

(Down from `PASS_WITH_WARNINGS`, for an evidence reason rather than a regression. All five
prior defects remain fixed. What changed is that a real run finally exists, which (a) makes
the missing `--stage evaluate` artifacts actionable rather than blocked — REV-129, MEDIUM —
and (b) exposed a defect in a real diagnostics artifact that no synthetic test could show —
REV-128.)

## Feature

`specs/evaluation-diagnostics` — `evaluation/metrics.py`, `cold_start.py`, `longtail.py`,
`flops.py`, `routing_diagnostics.py`, `gradient_diagnostics.py`, and
`models/ann_retrieval.py`.

## Review Scope

- Spec directory: `specs/evaluation-diagnostics/` (requirements.md, design.md, tasks.md) —
  none modified since the last review.
- Steering: `steering/code-quality.md` (device synchronization before latency timing),
  `conventions.md` (`OBJECTIVE_ORDER`, plot standards), `architecture.md` (test data
  untouched until frozen).
- Baseline: working tree vs `HEAD` = `d438950` on `master`; `origin/master` at the same
  commit. Of this spec's files only `evaluation/flops.py` is modified (`+30/-7`, REV-110);
  the rest are at their committed state. `tests/test_diagnostics.py` is modified (`+106`).
- Files inspected: `evaluation/flops.py`, `longtail.py`, `cold_start.py`,
  `gradient_diagnostics.py`, `routing_diagnostics.py`, `metrics.py`,
  `models/ann_retrieval.py`, `tests/test_diagnostics.py`, `tests/test_metrics.py`,
  `tests/test_ann_retrieval.py`.
- **New this pass:** the real diagnostics artifacts produced by the 09-17 T6.3 run
  (`outputs/logs/routing_diagnostics_t_small.json`,
  `outputs/logs/gradient_diagnostics_t_small.json` and their five plots) were loaded and
  checked numerically. Previous reviews had no such artifacts to check.
- Re-review. Prior findings carried in: REV-093, REV-094, REV-095, REV-109, REV-110 (all
  RESOLVED), REV-119 (LOW, OPEN).
- Date: 2026-09-17

## Summary

All five prior defects stay fixed and were re-verified, not inherited. REV-093's structural
fix holds: `grep` for `filter_negatives(` across `data models training evaluation main.py`
still returns only the definition's own use inside `generate_eval_candidates` and the one
site in `train_user_lora` that passes an explicit run-scoped `rng` — the per-call rebuild
is still gone from `longtail.py`, `cold_start.py` and `local_train.py`. REV-110's fix is
confirmed in the diff: `benchmark_latency` gathers real genome rows and records
`timed_metadata_source`, falling back to seeded noise but never to zeros.

The new information this pass is that **real artifacts exist for the first time**. The
T6.3 pretraining run wrote `routing_diagnostics_t_small.json` and
`gradient_diagnostics_t_small.json` plus five plots, so FR-8 and FR-9 now have artifact
evidence rather than test-only evidence. I checked them numerically and both are
structurally correct: the objective×expert heatmap is `(4, 4)` matching
`heatmap_shape`, with rows summing to 1.0 to within 3e-6 (AC-06); the gradient cosine
matrix is `(4, 4)` and exactly symmetric (AC-07's first half).

Checking them also turned up something no synthetic test could have: **the cosine matrix's
diagonal is not 1.0 — it reads 1.00078, 1.00391, 0.99844, 1.00078.** Gradients are gathered
in the model's bf16 and the cosine is computed in bf16, so a vector's similarity with
itself lands off unity, and three of four entries are **above 1**, which is outside the
range a cosine similarity can take. The AC-07 test passes only because its tolerance is
`< 0.01`. This is a one-line fix and is filed as REV-128.

Finally, `--stage evaluate` has still never been run, so FR-4's cold-start report and plot,
FR-5's long-tail report and plot, and FR-7's efficiency report do not exist — AC-08
explicitly requires them. This was previously blocked on the absence of a pretrained
checkpoint; `outputs/checkpoints/t_small/best.pt` now exists, so it is actionable
(REV-129).

## Git Changes

| File | Change | Attribution |
|---|---|---|
| `evaluation/flops.py` | `+30/-7`: real metadata gathered for the timed path, `timed_metadata_source` recorded, seeded-noise fallback | REV-110, this spec |
| `tests/test_diagnostics.py` | `+106`: `TestCandidateSetConsistency` and the REV-110 metadata-source assertions | this spec |

**Scope assessment:** in scope and proportionate. No new dependency beyond an intra-project
import (`load_metadata_matrix`), no committed artifact.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Fixed candidate protocol | PASS | Single `load_or_build_eval_candidates` entry point with a row-count guard; verified at the boundary by the `filter_negatives(` grep, and by `TestCandidateSetConsistency` asserting the rows each evaluator *actually scores* are the stored rows |
| FR-2 ANN retrieval | PASS | Force-inclusion implemented with lowest-scoring replacement; rate reported; `tests/test_ann_retrieval.py`. Documented as exact brute force, which is honest about what it measures |
| FR-3 Metrics | PASS | NDCG/MRR/Recall/Precision per the stated formulas, per-example values plus mean and std; toy tests |
| FR-4 Cold-start evaluation | **PARTIAL** | Implemented, loads stored candidates, asserts cold∖train disjointness. **`cold_start_{model}.json` and `cold_vs_warm_{model}.png` have never been produced** (REV-129) |
| FR-5 Long-tail evaluation | **PARTIAL** | Implemented with head/mid/tail from training popularity only, zero-training items as tail, and metadata supplied (REV-094). **`longtail_{model}.json` and its plot have never been produced** (REV-129) |
| FR-6 Latency benchmarking | PASS | Warmup discarded; per-run device synchronization; full serving path; mean/std/min/p95/ms-per-rec; timed input is now the serving input (REV-110) |
| FR-7 Efficiency report | **PARTIAL** | The writer covers every listed field. **`efficiency_{model}.json` has never been produced** (REV-129) |
| FR-8 Routing diagnostics | PASS | **Now artifact-backed.** `routing_diagnostics_t_small.json` carries entropy, utilization, per-objective, pairwise cosine and KL, collapse fractions, heatmap and `heatmap_shape`; three plots present |
| FR-9 Gradient-interference diagnostics | PASS (see REV-128) | **Now artifact-backed.** `(4, 4)` cosine matrix, per-objective norms, two plots present. Computed on diagnostic batches, not optimizer-step batches. The diagonal's bf16 precision is REV-128 |
| FR-10 Calibration | PASS | `use_logit_calibration=False`, `eval_temperature=1.0` by default; no test-set fitting path exists |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 same candidate sets for all models | PASS | Single loader + `TestCandidateSetConsistency`, which spies on `score_candidates_batch` inside both `evaluation.longtail` and `evaluation.cold_start` and compares against the stored rows. The fixture deliberately gives half its examples a shared target — the exact case the old per-target RNG fallback collapsed |
| AC-02 metrics correct on toy inputs | PASS | `tests/test_metrics.py` |
| AC-03 cold items disjoint from training | PASS | Asserted in `cold_start.py`, and independently re-measured against the real artifacts in the data-pipeline review (warm ∩ cold = ∅, 18,648 / 2,072) |
| AC-04 long-tail buckets partition all IDs | PASS | Disjoint-and-complete assertion; zero-training items classified tail |
| AC-05 latency includes device synchronization | PASS | Verified in code: `torch.mps.synchronize()` / `torch.cuda.synchronize(device)` around each timed run |
| AC-06 heatmap shape `(num_objectives, num_experts)` | PASS | **Verified against the real artifact**, not only the test: `(4, 4)`, matching the recorded `heatmap_shape`, rows summing to 1.0 within 3e-6 |
| AC-07 cosine matrix symmetric with 1s on diagonal | **PARTIAL** | Symmetry: PASS, exactly, on the real artifact. Diagonal: the real artifact reads `[1.00078, 1.00391, 0.99844, 1.00078]` — not 1, and three values are impossibly above 1. REV-128 |
| AC-08 all JSON reports and PNG plots in correct locations | **PARTIAL** | FR-8 and FR-9 artifacts: present and verified. FR-4, FR-5 and FR-7 artifacts: absent (REV-129) |
| AC-09 ANN force-inclusion rate reported | PASS | Rate computed and included in the evaluation output when `use_ann=True`; toy test |

## Findings

```
REV-128
Severity: LOW | Status: OPEN | Requirement: FR-9 | AC: AC-07
File: evaluation/gradient_diagnostics.py:17-35 (`gradient_cosine_similarity`);
      tests/test_diagnostics.py:94-99 (the AC-07 test)
Problem: the per-objective gradients are flattened and compared in whatever dtype the
  model holds, which is bf16 everywhere in this project. `F.cosine_similarity` on bf16
  vectors of ~790k elements does not return 1.0 for a vector against itself.
Evidence (measured against the real artifact, not a fixture) --
  `outputs/logs/gradient_diagnostics_t_small.json`, `cosine_similarity` diagonal:
      [1.0007813, 1.00390625, 0.99843752, 1.0007813]
  Three of the four are ABOVE 1, which is outside the range a cosine similarity can take.
  The matrix is exactly symmetric, so only the diagonal half of AC-07 is affected.
Impact: Diagnostic reporting only -- nothing reads this matrix back into training, and the
  off-diagonal interference numbers the Ablation D claim rests on are affected only at the
  same ~4e-3 magnitude, which is far below the effect being measured. The real cost is
  presentational and reputational: `gradient_cosine_similarity_t_small.png` is a paper
  figure, and a heatmap whose diagonal reads 1.004 invites exactly the wrong question from
  a reviewer.
  Note why this survived: the AC-07 test asserts `abs(cos_sim[i, i] - 1.0) < 0.01`, a
  tolerance wide enough to accept 1.0039. The test is not wrong to be tolerant of fp
  noise; it is just not tight enough to distinguish fp noise from a dtype choice.
Correction: cast the flattened gradients to fp32 before the comparison --
      grads[name] = flat.float()
  in `gradient_cosine_similarity`, which costs one temporary copy on a diagnostic path
  that already runs rarely. `per_objective_grad_norms` is worth the same treatment for
  consistency. Then tighten the AC-07 test to `< 1e-4` and add an assertion that no entry
  exceeds 1.0, so the range violation cannot come back. This is the same fp32-at-the-
  measurement-site pattern `routing_diagnostics` and `local_train` already use
  (`normalize(..., .float())`).
```

```
REV-129
Severity: MEDIUM | Status: OPEN | Requirement: FR-4, FR-5, FR-7 | AC: AC-08
File: outputs/logs/ (no cold_start_*.json, longtail_*.json, efficiency_*.json);
      outputs/plots/ (no cold_vs_warm_*.png, longtail_*.png)
Problem: `--stage evaluate` has never been run, so three of this spec's five required
  report artifacts and two of its required plots do not exist. Listing `outputs/logs/` and
  `outputs/plots/` directly this pass, the only artifacts belonging to this spec are the
  routing and gradient diagnostics the T6.3 pretraining run produced as a side effect.
  AC-08 ("All JSON reports and PNG plots generated in correct locations") is therefore
  half-met at best.
Impact: FR-4 and FR-5 carry two of the paper's four research questions -- cold-start
  performance with metadata ON vs OFF, and head/mid/tail behaviour. Neither has a single
  measured number behind it. FR-7's efficiency report is the source for the
  parameter/latency/FLOPs columns of the comparison tables. The code for all three is
  implemented and unit-tested, so this is an execution gap, not a correctness gap -- but
  unit tests on synthetic fixtures cannot produce the numbers these requirements exist to
  produce.
  Raised to a finding this pass (it was prose in the previous review) because the reason
  to defer it is gone: `outputs/checkpoints/t_small/best.pt` now exists and was verified
  in the training-pipeline review, so the stage has an input.
Correction: Run
      uv run python main.py --stage evaluate --model t-small
  and verify against the files rather than the console: `cold_start_t_small.json` reports
  metrics for cold and warm separately under metadata ON and OFF (all four cells);
  `longtail_t_small.json` reports head/mid/tail with the bucket populations, and the
  buckets partition the catalogue; `efficiency_t_small.json` carries every field FR-7
  lists, including `timed_metadata_source == "item_metadata"` (the REV-110 fix, confirmed
  on a real run for the first time); and both plots render.
  Sequencing, and it is load-bearing:
    - Fix REV-128 BEFORE this run, so the gradient diagnostics that end up in the paper
      carry fp32 cosines.
    - `loss-functions` REV-105 changes the multi-step objective, which makes the current
      `outputs/checkpoints/t_small/best.pt` a model trained under a superseded objective.
      This stage evaluates that checkpoint. Re-run the ~190 s T6.3 pretrain gate after
      REV-105 lands and evaluate the REGENERATED checkpoint; otherwise every metric here
      describes an experiment the code no longer implements.
    - REV-122 (transformer-backbone, CRITICAL) does NOT affect this run -- it is MPS/SDPA,
      which is the correct path. No artifact on disk is contaminated by it.
```

```
REV-119
Severity: LOW | Status: OPEN | Requirement: FR-1 | AC: AC-01
File: data/sequence_generator.py:679-692 (`load_eval_candidates`), reached from
      training/local_train.py:250 and evaluation/longtail.py:100
Problem: unchanged. `load_eval_candidates` does a plain `np.load(path)` and then
  fingerprints the whole matrix. Re-measured this pass: `test_candidates.npy` is
  (1,059,494 x 101) int64 = 856 MB. For `longtail.py` that is fine -- it iterates the
  whole test split anyway. For `run_local_personalization` it is not proportionate: that
  stage scores 1,000 users, i.e. a few thousand rows, and pays a ~1 GB resident copy and a
  full-matrix hash to reach them.
Impact: Operational, not a correctness defect -- and strictly better than the behaviour it
  replaced, which was wrong. Its practical weight is about to increase: REV-127 asks for
  the first real personalization run, which is exactly the path that pays this cost, and
  `--stage all` would hold this alongside the 11 GB preprocessing peak.
Correction: Load with `mmap_mode='r'` and slice the needed rows, materialising only what
  is scored; the fingerprint pass is already block-wise (REV-087) and works on a memmap.
  Keep the eager path for `--stage evaluate`, which consumes the whole matrix. Worth doing
  before REV-127's personalization run if that run hits memory pressure; otherwise still
  not urgent.
```

```
REV-093
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-1, FR-5 | AC: AC-01
Resolution: re-verified at the boundary rather than at the call sites. `grep` for
  `filter_negatives(` over `data models training evaluation main.py` returns only the
  definition's own use inside `generate_eval_candidates` and one site in `train_user_lora`
  that passes an explicit run-scoped `rng`. No evaluation entry point rebuilds candidates.
```

```
REV-094
Severity: LOW | Status: RESOLVED | Requirement: FR-5 | AC: —
Resolution: re-confirmed. `run_longtail_evaluation` supplies the metadata matrix, so it no
  longer scores a metadata-enabled model on a metadata-free path.
```

```
REV-095
Severity: LOW | Status: RESOLVED | Requirement: FR-1 | AC: AC-01
Resolution: re-confirmed. `TestCandidateSetConsistency` spies on `score_candidates_batch`
  in both evaluators and asserts the rows scored are the stored rows, with a fixture that
  deliberately reproduces the shared-target case the old fallback collapsed. This is a
  test that can fail on the defect it guards.
```

```
REV-109
Severity: LOW | Status: RESOLVED | Requirement: FR-9 | AC: —
Resolution: re-confirmed. `run_gradient_diagnostics` saves and restores both
  `model.training` and `moe_layer.training`.
```

```
REV-110
Severity: LOW | Status: RESOLVED | Requirement: FR-6 | AC: AC-05
Resolution: confirmed in the diff. `benchmark_latency` gathers `metadata_matrix[dummy_seq]`
  when a matrix is available, falls back to seeded `randn` when it is not, never to zeros,
  and records which in `timed_metadata_source`. The timed input is now unambiguously the
  serving input.
  Not yet confirmed on a real `--stage evaluate` run, because none has happened --
  REV-129 folds that check in.
```

## Design Compliance

Compliant. The Constraints line this spec turns on — "Fixed evaluation candidate sets
generated once and reused — never regenerated per model" — is enforced by a single loader
with a row-count guard rather than by each call site remembering to do the right thing.
That is a strengthening of the design.

`OBJECTIVE_ORDER` is respected by the routing and gradient diagnostics and by their plots;
I confirmed this against the real artifact, where `objective_names` is recorded alongside
the matrix. Latency timing synchronizes the device per `code-quality.md`. Plot output goes
to `outputs/plots/` at `config.plot_dpi`.

One design point is worth recording now that real numbers exist: the Constraints line "Do
not interpret low entropy as proof of usefulness — report without editorializing" is
respected by the artifact, which reports `overall_utilization`
`[0.065, 0.311, 0.395, 0.229]` and the two collapse fractions as bare numbers with no
verdict attached. That utilization is notably non-uniform for a 200-step run, which is
information for the owner rather than a defect — it is exactly the kind of thing the
diagnostic exists to surface.

## Task Completion

| Task | Verdict |
|---|---|
| T1.1–T1.5 metrics, eval loop, ANN, calibration | Checked and verified |
| T2.x cold-start, T3.x long-tail | Checked; both load stored candidates and T3.x supplies metadata. Implementation complete; **never executed** (REV-129) |
| T4.1–T4.3 latency + efficiency | Checked; the timed path is now the serving path. Report never produced (REV-129) |
| T5.x routing diagnostics | Checked, and **now artifact-verified** — heatmap shape and row sums measured on the real JSON |
| T6.x gradient diagnostics | Checked, and now artifact-verified — which is how REV-128 surfaced |
| T7.1 candidate-set consistency test | Checked and justified — this was the false-complete REV-095 flagged, and its replacement is sound |

No false-complete tasks: every task claims implementation and every implementation exists.
As in `personalization-federated`, no task in this spec asks anyone to *run* the stage.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **267 passed**, 0 failed, 49.20 s |
| Load `outputs/logs/routing_diagnostics_t_small.json` (direct) | heatmap `(4, 4)` == recorded `heatmap_shape`; row sums 1.0 ± 3e-6; collapse fractions and utilization present — AC-06 |
| Load `outputs/logs/gradient_diagnostics_t_small.json` (direct) | cosine matrix `(4, 4)`, **exactly symmetric**; diagonal `[1.00078, 1.00391, 0.99844, 1.00078]` — AC-07 half-met, REV-128 |
| `grep -rn "filter_negatives(" data models training evaluation main.py` | definition's own use + one `rng`-passing site — REV-093 still closed |
| `git diff evaluation/flops.py` | REV-110 fix confirmed; `timed_metadata_source` recorded |
| `ls outputs/logs outputs/plots` | no `cold_start_*`, `longtail_*`, `efficiency_*`, `cold_vs_warm_*` — REV-129 |
| `uv run python -m compileall -q evaluation models` | clean |

## Final Assessment

For BUILD, in order:

1. **REV-128 (LOW, but do it first)** — cast the flattened gradients to fp32 in
   `gradient_cosine_similarity` (and `per_objective_grad_norms`), then tighten the AC-07
   test to `< 1e-4` and assert no entry exceeds 1.0. One line plus a test. It comes first
   only because REV-129's run should regenerate these artifacts afterwards.
2. **REV-129 (MEDIUM, blocking)** — run `--stage evaluate --model t-small` and verify
   the five artifacts against the files. AC-08 cannot be met any other way, and FR-4/FR-5
   carry two of the paper's research questions.
3. **REV-119 (LOW)** — `mmap_mode='r'` in `load_eval_candidates` for the row-sliced
   consumers. Worth doing before the personalization run (`personalization-federated`
   REV-127) if that run hits memory pressure.

For the **spec owner** (no action required, one thing to look at):

4. The first real routing artifact shows expert utilization of
   `[0.065, 0.311, 0.395, 0.229]` against a uniform `1/K = 0.25`. From a 200-step capped
   run this is not evidence of anything, but it is the number the Ablation G/collapse
   discussion will turn on, so it is worth having seen early.
