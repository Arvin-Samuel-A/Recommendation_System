# Evaluation Integrity — Review

## Review Status

**PASS** (re-review, 2026-10-17). All requirements and acceptance criteria are met. The four LOW findings of the first pass (REV-001 to REV-004) are RESOLVED, each verified independently. Three INFO observations remain; none is a defect.

Previous status: PASS_WITH_WARNINGS (first pass, same day).

## Feature

`specs/eval-integrity` covers:
- fp32 scoring;
- tie-averaged metrics;
- per-forward autocast with the weight cache off;
- the training-semantics fingerprint bump;
- the full ablation test split;
- the popularity floor;
- `--stage ablation_rescore` and its launch scripts.

## Review Scope

- **Spec directory:** `specs/eval-integrity/` (requirements.md, design.md, tasks.md)
- **Worktree:** `.claude/worktrees/zealous-jennings-8fe234`, branch `claude/sharp-faraday-2dc2f0`
- **Baseline:** `HEAD` = `c094b2c`. Every change is uncommitted: `git diff` plus three untracked paths.
- **Date:** 2026-10-17 (first pass and re-review)
- **Re-review scope:** BUILD's fixes for REV-001 to REV-004:
  - `main.py::run_multi_seed_stage`;
  - `training/ablation_rescore.py::_load_identity`;
  - three new tests in `tests/test_eval_integrity.py`;
  - BUILD notes in `tasks.md`.
- **Steering read:** `steering/{architecture,code-quality,conventions,testing}.md`

## Summary

Everything was checked against the repository rather than BUILD's claims:
- Every forward site now opens its own `autocast_context` with `cache_enabled=False`.
- The stage-wide region in `main.py` is gone.
- Scoring is fp32 outside autocast, both at the choke point and in ANN.
- The tie rule is the McSherry–Najork expectation. It is bit-identical to the old rule when there are no ties.
- The popularity floor is built from `train_user_history.pkl` only, which holds training items and no val/test items.

The rescore stage was exercised beyond BUILD's tests:
- Study B was trained with the **unmodified `c094b2c` code**, including its stage-wide region. The new code then rescored that run. This is the real production scenario: a legacy fingerprint, a foreign `source_hash`, and old checkpoint configs.
- It ran on CPU and again on MPS. It worked, left the originals byte-identical, and a second run was idempotent.

Further checks:
- Seven code mutations, each reverting one fix, were each caught by the new tests.
- The AC-28 Mac smoke was reproduced independently. Logged and post-load val NDCG differ by 0.0 on MPS.

Re-review: all four first-pass items are fixed.
- REV-001: `final_results_*.json` now carries `eval_protocol` and `popularity_reference`.
- REV-002: an automated legacy-run rescore test exists.
- REV-003: the BUILD notes are recorded in tasks.md.
- REV-004: the identity check memory-maps checkpoints.

Each new test was mutation-checked: reverting its fix makes it fail. `test_eval_integrity.py` (52 passed) and `test_cli.py` (63 passed) are green.

## Git Changes

- **Status:**
  - 26 modified tracked files: 1,034 insertions, 269 deletions.
  - Untracked: `specs/eval-integrity/`, `tests/test_eval_integrity.py` (1,397 lines) and `training/ablation_rescore.py` (655 lines).
- **Relevant changes:**
  - `utils.py`, `evaluation/{metrics,cold_start,longtail,flops,gradient_diagnostics}.py`
  - `models/ann_retrieval.py`
  - `training/{pretrain,local_train,ablation_runner}.py`, new `training/ablation_rescore.py`
  - `main.py`, `configs/config.py`, `scripts/package_ablation_run.py`, `README.md`
  - Tests: `tests/test_{metrics,training,cli,scripts}.py`, new `tests/test_eval_integrity.py`
  - Documentation: superseded-clause notes in `specs/ablation-run`, `specs/evaluation-diagnostics`, `specs/BUILD_ORDER.md` and `steering/*`
- **Scope assessment:**
  - Every change maps to an FR or task.
  - There are no new dependencies.
  - `tests/test_determinism.py` and `tests/test_ablation_robustness.py` are unmodified (empty `git diff`).
  - Existing tests were edited in two places:
    - the planned `TestMixedPrecision` replacement;
    - an extra `load_item_popularity` patch in two `test_cli.py` tests. This is judged acceptable under BUILD deviations below.
  - No artifacts are committed or left in the worktree. My temporary `data/` symlinks were removed, and the untracked set is unchanged.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1.1 fp32 choke point | PASS | `evaluation/metrics.py::score_candidates_batch`: the forward and MoE run under `autocast_context(device_type=param device)`. The upcast, then normalisation, einsum and temperature run under `autocast_disabled`. The function asserts float32. AC-05 covers mixed with no region, mixed inside an enclosing region, and pure bf16. |
| FR-1.2 all paths through choke point | PASS | grep: `evaluate_model`, cold_start, longtail, `evaluate_user`, federated (`evaluate_model`) and the estimator (`evaluate_model`) all reach `score_candidates_batch`. No other einsum, `@` or matmul scoring exists outside the losses and LoRA. |
| FR-1.3 ANN fp32 | PASS | `models/ann_retrieval.py`: `_user_embeddings` returns the un-normalised embedding from its own region. The item table and user vectors are upcast before `l2_normalize` under `autocast_disabled`, with an fp32 assert. AC-06 checks this with and without an enclosing region. |
| FR-1.4 scoring-site guard | PASS | `test_no_unaudited_scoring_sites` uses AST and fails on stale allow-list entries. Adding an einsum to `evaluation/longtail.py` made it fail (verified). |
| FR-1.5 protocol stamp | PASS | Stamped in arm results, `test_metrics_*`, `cold_start_*`, `longtail_*` and rescored records. Since the REV-001 fix it is also in the multi-seed `final_results_*` file (`main.py` ~1071-1087). |
| FR-2.1–2.5 tie-averaged metrics | PASS | `compute_metrics_batch` uses a cumulative-table gather, float64 (float32 on MPS), no sort, and one NaN sync per batch. `TestTieAveraged` covers AC-01 to AC-04. |
| FR-3.1 no stage-wide region | PASS | `main.py:1199-1204`. AC-11 covers it, and a mutation that re-adds the region makes AC-11 fail. |
| FR-3.2 per-site regions | PASS | All seven sites were inspected. The regions match the design D3 table: backward, GradNorm, clip and step are outside, and `autograd.grad` is outside in the diagnostics. AC-10 covers all seven sites. |
| FR-3.3 cache off | PASS | `utils.autocast_context(..., cache_enabled=False)`. Removing it makes AC-09, AC-12 and the T1.2(b) test fail (mutation). |
| FR-3.4 no assertion; nested OK | PASS | The assertion is removed. The AC-09 enclosing-region test passes. |
| FR-3.5 golden unchanged | PASS | `test_determinism.py`: 6 passed, file unmodified. |
| FR-3.6 cast audit | PASS | design D5 records the audit. The code agrees (metadata casts unchanged). |
| FR-4.1/4.2 fingerprint version | PASS | `TRAINING_SEMANTICS_VERSION = 2` and `_training_fields`. The resume check (`_training_fingerprint_of`) and the packager plan use the same function. AC-14 passes, and the invariance suites pass unmodified. |
| FR-5.1–5.3 selection / val floor | PASS | AC-15 (tiny fixture) passes. The real-data MPS smoke shows logged = post-load with diff 0.0. The header line is written once, has no `epoch=`, and is returned as `val_popularity_reference`. |
| FR-6.1–6.4 full test split | PASS | `select_eval_rows`. Only the selected rows are gathered from the mmap. Row-count check. The `test_eval` block is recorded. `ablation_eval_seed` is in `POST_TRAINING_ONLY_FIELDS`. AC-16 and AC-17 pass. |
| FR-7.1 popularity ranker | PASS | `item_popularity` reads `train_user_history.pkl`. `data/sequence_generator.py:242` shows this holds only the train items, so nothing leaks from val/test. Pad and MASK are zeroed. AC-18 passes. |
| FR-7.2 popularity in every result | PASS | Present in the arm result, `test_metrics` (ANN candidates when ANN is on), `cold_popularity`/`warm_popularity` and `popularity_head`/`mid`/`tail`. Since REV-001 it is also in `final_results_*`. |
| FR-7.3 study JSON / summary | PASS | Present per arm. `_save_summary` writes `best_ndcg`, `popularity_reference_ndcg`, `best_minus_popularity_ndcg` and the `extra` dict. |
| FR-7.4 plot line | PASS | `_plot_study` `axhline` (spy test). |
| FR-8.1 rescore stage | PASS | `training/ablation_rescore.py` and `main.run_ablation_rescore_stage`. The backbone is located from the recorded fingerprint, never recomputed. Verified on a real legacy run (see Testing). |
| FR-8.2 originals never modified | PASS | AC-20 compares SHA-256 hashes. In the legacy-run check, every original arm JSON, `.pt` file and `backbone.json` was byte-identical afterwards. Loads are read-only (`map_location="cpu"`). |
| FR-8.3 runs where the checkpoints are | PASS (CUDA not verifiable here) | `config_from_checkpoint` copies only the runtime fields and calls `check_processed_dir`. Backbone paths are relative (`outputs/checkpoints/...`), so the stage works from the run directory and from a copied `outputs/`. Verified on CPU and MPS. CUDA was inspected only: autocast `cuda` is supported and the state dict copies onto the device. |
| FR-8.4 resumable / idempotent / lock | PASS | Per-epoch atomic writes. DEC-8 validity covers code, protocol, file size and mtime. `rescore_key`. `flock` per fingerprint. AC-22, AC-23 and AC-24 pass. In the legacy run, the second invocation made 0 evaluations. |
| FR-8.5 rescored record | PASS | `build_rescored_record` has every listed field. `code_revision` is current, so `assess_study` accepts the record. `runtime`, `source_revision` and `arm_key` are kept. |
| FR-8.6 extras quarantined | PASS | AC-26 passes. Named plots degrade gracefully when an extra is missing. |
| FR-8.7 caveat | PASS | Verbatim in each arm, in `summary.rescore_caveat`, and in WARNING lines at start and end (AC-20 counts 2). |
| FR-8.8 collection / exit 1 | PASS | Uses `collect_study` on the rescored config. `--study all` writes the summary and exits 1 when studies are absent (test). |
| FR-8.9 epoch identity | PASS | `_verify_identity` runs before any evaluation. AC-25 passes. Since the REV-004 fix it loads each file with `mmap=True` (`_load_identity`). |
| FR-9.1 rescore launch script | PASS (accepted deviation) | `render_rescore_script` uses the same `study_command` base flags and goes through `run_with_retry`. The script is generated only with `--rescore-scripts` (see deviations). |
| FR-9.2 README runbook | PASS | The "Rescoring a finished run" section covers tarball-over-code, the command, the outputs, local MPS/CPU use, the caveat word for word, and retraining on re-run. |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 ties | PASS | `TestTieAveraged::test_all_scores_equal` (a, e), `test_target_tied_with_one_at_top` (b), `test_ties_among_others_do_not_matter` (c), `test_batched_equals_single` (d). |
| AC-02 Monte Carlo | PASS | `test_monte_carlo_random_tie_breaking`: 2,000 jitter tie-breaks, atol 0.01. |
| AC-03 no-tie identity | PASS | Existing tests are unmodified. `test_no_ties_identical_to_former_rule` uses `torch.equal`. Mutating to the old rule fails 3 tests. |
| AC-04 NaN | PASS | `test_nan_raises`. |
| AC-05 fp32 scoring | PASS | `test_score_candidates_batch_is_fp32`: 6 parametrisations, hook-captured fp32 reference at atol 1e-6. The bf16-scoring mutation fails 4 of them. |
| AC-06 ANN fp32 | PASS | `test_ann_retrieval_ranks_by_fp32_scores` (spies `topk`). The ANN mutation fails both cases. |
| AC-07 scoring-site guard | PASS | Verified by an injected einsum. |
| AC-08 stale-cache regression | PASS | `test_train_one_step_forwards_current_weights`. Every Linear in the model, MoE and multi-step loss is `torch.equal` to a fresh-region `F.linear` with the post-step-1 weights. Removing the step region makes it fail. |
| AC-09 defence in depth | PASS | `..._inside_cached_region` plus the characterisation test `test_raw_autocast_region_forwards_stale_weights`. Removing `cache_enabled=False` fails it. |
| AC-10 forward-site dtype | PASS | `test_forward_site_owns_its_autocast_region` ×7. Removing the step or LoRA regions makes it fail. |
| AC-11 no stage region | PASS | `test_main_runs_stages_without_an_autocast_region`. |
| AC-12 LoRA current adapters | PASS | `test_evaluate_user_sees_loaded_lora_adapters`, mixed and pure, with a control re-evaluation. |
| AC-13 golden | PASS | `test_determinism.py`: 6 passed, unmodified. |
| AC-14 fingerprint version | PASS | `test_fingerprint_changes_with_semantics_version` and `test_legacy_backbone_is_not_reused`. The robustness, CLI and training suites pass. |
| AC-15 selection consistency | PASS | `test_logged_val_ndcg_equals_post_load_evaluation`. |
| AC-16 full test split | PASS | `test_arm_scores_the_full_test_split`. |
| AC-17 seeded sample | PASS | `test_select_eval_rows_is_seeded_and_not_a_prefix` and `test_arm_records_a_seeded_random_sample`. |
| AC-18 popularity helper | PASS | Four tests, including hand-computed tie-averaged NDCG and MRR. |
| AC-19 popularity everywhere | PASS | Study JSON, summary, plot spy, `test_metrics`, cold_start and longtail tests, each also checking `eval_protocol`. `final_results` is now covered too, by `test_multi_seed_final_results_carry_protocol_and_popularity`. |
| AC-20 rescore end to end | PASS | `test_rescore_stage_end_to_end` and the new `test_rescore_of_a_legacy_run`. Also verified independently on a legacy (`c094b2c`-trained) run. |
| AC-21 post-fix no-op | PASS | `test_rescore_of_a_post_fix_run_reproduces_logged_values` (1e-6, same best epoch). |
| AC-22 idempotent | PASS | Second run: `calls == []`, byte-identical outputs. Legacy run: 0 evaluations. |
| AC-23 resumable | PASS | `test_rescore_resumes_after_a_failed_epoch`. |
| AC-24 shared fingerprint | PASS | `test_arms_sharing_a_fingerprint_share_one_evaluation` (3+1 calls). |
| AC-25 config identity | PASS | `test_epoch_with_foreign_config_fails_only_its_arm`. |
| AC-26 extras quarantined | PASS | `test_rescore_quarantines_weight_dependent_extras`. |
| AC-27 packager | PASS | `TestRescoreScripts`: `bash -n`, the studies, base flags equal to the run scripts, the tarball member, and opt-in. The existing packager tests are unmodified and pass. |
| AC-28 Mac smoke | PASS (independently reproduced) | MPS, `t-small`, 10% data, `--mixed_precision --max_steps 200 --epochs 2`, run in a scratch copy of the worktree. Epoch 0: stored 0.278484 vs post-load 0.278484. Epoch 1: 0.313711 vs 0.313711. Both diffs are 0.0. The header reads `popularity_ndcg@10=0.407031 rows=100591 eval_protocol_version=2`. All numbers were read from the artifacts. |

## Findings

```
REV-001
Severity: LOW | Status: RESOLVED | Requirement: G6, FR-1.5 (general rule) | AC: —
File: main.py — run_multi_seed_stage(), `out = {...}` at ~line 1071 (final_results_{model}.json)
Problem: The multi-seed headline file aggregates only per-seed entries that carry "mean".
  So `popularity_reference` and `eval_protocol`, both present in each per-seed
  test_metrics file, are silently dropped. final_results_*.json reports ranking metrics but
  carries neither the protocol stamp nor the popularity floor. FR-1.5's enumerated file list
  omits it, but its general rule ("every result file that reports ranking metrics") and G6
  ("every ranking result file") cover it.
Impact: The final multi-seed numbers cannot be told apart as pre-fix or post-fix from the
  file alone, and cannot be read against the floor without opening the per-seed logs.
Correction: Add `"eval_protocol": dict(EVAL_PROTOCOL)` and a `popularity_reference` to
  `out`. Take the reference from the first per-seed payload that has one; the test
  candidates are fixed, so it is the same across seeds. Assert it in an existing multi-seed
  CLI test.
Resolution (re-review, verified):
  - `main.py` ~1071-1087 now writes `"popularity_reference"` (from the first seed that has
    one) and `"eval_protocol": dict(EVAL_PROTOCOL)` into `out`. Aggregation is unaffected,
    because the reference has no "mean" key.
  - New test `test_multi_seed_final_results_carry_protocol_and_popularity` passes.
  - Removing the two keys on a scratch copy makes the test fail.
```

```
REV-002
Severity: LOW | Status: RESOLVED | Requirement: FR-8.1, FR-8.3 | AC: AC-20 (coverage)
File: tests/test_eval_integrity.py — rescore tests (T4.1/T4.2)
Problem: Every automated rescore test trains study B with the CURRENT code, so the arm
  records carry a version-2 fingerprint and the current source_hash. The production
  scenario is a legacy fingerprint, a foreign `code_revision.source_hash`, and checkpoints
  written by other code. It is not under test. I verified it manually: trained with
  c094b2c, rescored with this code, and it worked. But a future change that, for example,
  recomputed the fingerprint or compared `arm_key` in collection would pass every test and
  break the real rescore.
Impact: A regression in the one path the stage exists for would go undetected.
Correction: Add a test that, on a `_copy_run` copy:
  - renames one store directory to a legacy-style fingerprint (for example
    `_legacy_fingerprint(...)`) and rewrites `backbone.json` to match;
  - rewrites that arm's `training_fingerprint` and sets `code_revision.source_hash` to a
    foreign value;
  - asserts the rescore finds it, `not_written == []`, and the rescored record's
    `code_revision` is current.
Resolution (re-review, verified):
  - New test `test_rescore_of_a_legacy_run` sets up the legacy scenario:
    - moves a store entry to a version-less legacy fingerprint and relinks `backbone.json`;
    - gives every checkpoint a foreign `source_revision`, drops `ablation_eval_seed` from
      the checkpoint config and adds an unknown field;
    - rewrites the arm record's fingerprint and `code_revision.source_hash`.
  - It asserts:
    - `not_written == []`;
    - the record's `code_revision` is current;
    - the record's `source_revision` and `original_code_revision` are kept;
    - `epochs_scored == [0, 1, 2]`;
    - the original arm file is unchanged.
  - The test passes. A mutation that makes `rescore_arm` RECOMPUTE the fingerprint from
    `resolved_config` instead of using the recorded one makes it fail.
```

```
REV-003
Severity: LOW | Status: RESOLVED | Requirement: tasks.md T1.5, T2.5, T5.1, T5.2 ("record ... in the BUILD note") | AC: —
File: specs/eval-integrity/tasks.md (all tasks [x]); no BUILD note in the repository
Problem: Four tasks make "Done when" depend on a recorded BUILD note:
  - T1.5: the guard fails on an injected einsum;
  - T2.5: AC-15 fails with T2.1 reverted;
  - T5.1: per-file pass/skip counts and test_eval_integrity time;
  - T5.2: AC-28 numbers read from the artifacts.
  No such note exists in the worktree, and the T5.2 artifacts are not present.
Impact: The tasks are checked [x] without the evidence they require being reviewable.
  (I re-did the T1.5, T5.1 and T5.2 checks independently and they hold; see Testing.)
Correction: Record the BUILD note, either in tasks.md under each task or in the hand-off,
  with the actual numbers. No code change is needed.
Resolution (re-review, verified):
  - tasks.md now carries a dated BUILD note under each of the four tasks:
    - T1.5: the injected einsum fails the guard, and the test passes after revert.
    - T2.5: a literal revert aborts AC-15. A simulated pre-fix pattern gives logged vs
      post-load NDCG differences of up to 0.0078; the fixed code shows 0.0 difference.
    - T5.1: per-file counts, and test_eval_integrity takes 56.6 s.
    - T5.2: 0.275768 and 0.313153, both with a 0.0 difference.
  - These are consistent with my independent runs:
    - T1.5 and T5.1 match exactly.
    - T5.2: my run gave 0.278484 and 0.313711, also with a 0.0 difference. The absolute
      values differ because separate MPS training runs are not bitwise reproducible; the
      0.0 logged-vs-post-load difference is the criterion.
  - No T5.2 artifacts remain in the worktree (no `outputs/`, no symlinks), which matches
    the note.
```

```
REV-004
Severity: LOW | Status: RESOLVED | Requirement: FR-8.9 / design D9 cost | AC: —
File: training/ablation_rescore.py — _verify_identity() (line 329), called at line 410
Problem: Identity verification does a full `torch.load` of every unscored epoch
  checkpoint, including optimizer, scheduler and loss state. The evaluation loop then loads
  every file again. At t-base, 20 epochs, this roughly doubles checkpoint reads per
  fingerprint. It is not a correctness defect.
Impact: Extra disk I/O and wall time per fingerprint on the VM and H100, and noticeably
  more on a Mac reading copied outputs.
Correction (original): Keep the check before any evaluation, as FR-8.9 requires, but make it cheap:
  `torch.load(path, map_location="cpu", weights_only=False, mmap=True)` in
  `_verify_identity`, so only `config` and `source_revision` are materialised.
  Alternatively, read those two keys once and cache them for the evaluation loop.
Resolution (re-review, verified):
  - `training/ablation_rescore.py:302-314` adds `_load_identity`, which uses
    `torch.load(path, map_location="cpu", weights_only=False, mmap=True)` and falls back to
    a full load on `RuntimeError` (a non-zip legacy file). `_verify_identity` uses it
    (line 350).
  - The check still runs before any evaluation, so FR-8.9 semantics are unchanged.
  - New test `test_identity_check_does_not_read_checkpoints_twice` asserts:
    - each epoch file is mapped exactly once;
    - full reads total 2×3 + 2 (one per val pass, plus the selected epoch for the test
      pass).
  - Reverting to `_load` on a scratch copy makes it fail. AC-25 (the foreign config still
    fails its arm) still passes.
```

INFO (not defects):

- **INFO-1.** `rescore.json` validity ignores device and backend. A record scored on CPU or MPS is reused on CUDA. This matches DEC-8, and `eval_runtime` records where it was computed. MPS and CPU per-epoch values differed by about 5e-4 on the tiny legacy run, with the same best epoch.
- **INFO-2.** `rescore_<machine>.sh` runs one process, one GPU and the studies in sequence. On the multi-GPU AWS VM, extra manual processes for other studies are safe because of the per-fingerprint `flock`. The README could say so.
- **INFO-3.** The memory note says the AWS run finishes around 22:10 UTC. The README tells the owner to extract the new code over the old code only after the run has finished. That ordering matters, because the code tarball overwrites `launch/` and the sources the running process imports lazily.

## BUILD deviations (judged)

| Deviation | Verdict |
|---|---|
| Rescore scripts opt-in behind `--rescore-scripts` | **Accepted.** `tests/test_scripts.py:612` asserts the exact launch file set `{"LAUNCH.md","run_aws.sh","run_h100.sh"}`, so writing the scripts unconditionally would have broken an existing test that AC-27 requires to stay unmodified. The README command includes the flag, and `test_rescore_scripts_are_opt_in` pins the behaviour. This is a technical deviation from FR-9.1's wording, not a functional gap. |
| Extra `load_item_popularity` patch in two `test_cli.py` tests | **Accepted.** Those tests fake `load_tensorised` and `load_eval_candidates` with no real processed dir. The new popularity read of `train_user_history.pkl` needs the same faking. No assertion was changed. |
| `map_location="cpu"` in the rescore | **Accepted, and required.** Without it, CUDA-trained checkpoints could not be rescored on MPS or CPU (FR-8.3). `load_state_dict` copies onto the model's device. |
| Epoch files located next to `backbone.json`'s `checkpoint_path` | **Accepted.** `register_backbone` writes `checkpoint_path = model_checkpoint_dir()/best.pt`, so `dirname` equals `join(checkpoint_dir, model_dir_name)`, and `find_backbone` has already checked that the file exists. It is also correct for multi-seed registrations outside the store. |
| Per-epoch dicts keyed by epoch string | **Accepted.** JSON object keys are strings. `epochs_scored` is an int list as AC-20 requires, and selection converts back to `int` (`_earliest_argmax`). |

## Design Compliance

- **D1 (ties):** as designed. MPS uses a float32 cumulative table because MPS has no float64. That is accurate to well under 1e-6.
- **D2 (fp32 scoring), D3/D4 (regions and cache off), D6 (fingerprint), D7 (rows), D8 (popularity):** as designed.
- **D9 (rescore):** as designed, with two harmless additions:
  - the record validity also includes `commit` and `dirty`, which is stricter;
  - `EvalData` is keyed by (width, `use_metadata`, vocab).
- **D10 (guard):** as designed. It also covers `mm` and `bmm`.
- **D11 (packager):** as designed, plus the accepted opt-in flag.
- No unapproved dependencies and no silent redesign.

## Task Completion

- **T1.1–T4.3:** implemented and verified, as above.
- **T5.1:** every listed file passes (table below). `test_eval_integrity.py` runs in 57 s, under 3 min.
- **T5.2:** reproduced independently. BUILD's notes are now recorded (REV-003 resolved).
- No false-complete task remains.

## Testing and Verification

Commands run in the worktree unless noted:

| Command | Result |
|---|---|
| `uv run python -m pytest tests/test_eval_integrity.py -m "not slow" -q` | 49 passed (57 s) |
| `... tests/test_metrics.py` | 14 passed |
| `... tests/test_determinism.py` | 6 passed (file unmodified) |
| `... tests/test_training.py` (`data/processed*` symlinked read-only, then removed) | 26 passed, 2 deselected. Without the symlink, one pre-existing data-dependent test errors with FileNotFoundError, which is environmental. |
| `... tests/test_ann_retrieval.py` | 9 passed |
| `... tests/test_diagnostics.py` | 22 passed |
| `... tests/test_moe.py` | 19 passed |
| `... tests/test_lora.py` | 7 passed |
| `... tests/test_federated.py` | 13 passed, 6 deselected |
| `... tests/test_resume.py` | 15 passed |
| `... tests/test_ablation_robustness.py` | 31 passed |
| `... tests/test_ablation_reseed.py` | 2 passed |
| `... tests/test_ablation_parallel.py` | 33 passed |
| `... tests/test_cli.py` | 63 passed |
| `... tests/test_scripts.py` | 31 passed |
| `... tests/test_smoke.py` | 2 passed |
| Mutation run on a scratch copy: 7 single-fix reverts, each with its targeted tests | All caught: no cache-off → 3 fail; stage region → AC-11 fails; bf16 scoring → 4 fail; ANN bf16 → 2 fail; old tie rule → 3 fail; no step region → 3 fail; no LoRA region → 1 fails |
| Injected `torch.einsum` in `evaluation/longtail.py` (scratch copy) | `test_no_unaudited_scoring_sites` fails, then passes after revert |
| Legacy-run rescore, scratch: `git archive HEAD` (`c094b2c`) trains study B (mixed bf16, 3 epochs, inside the old stage-wide region), then this code runs `run_ablation_rescore(..., "B")` on CPU | `not_written == []`. 8 evaluations (2 fingerprints × (3 val + 1 test)). Legacy fingerprints `6992e288d584`/`7cdf26fcf70c` located. Originals byte-identical. Best epochs re-selected (1→0, 1→2). Test rows full (152). Second run: 0 evaluations, both arms "current". |
| The same legacy run, fresh copy, `device="auto"` (MPS) | Rescored on `mps`, same best epochs as CPU |
| AC-28 smoke (scratch copy): `main.py --stage pretrain --model t-small --user_sample_fraction 0.1 --dtype bf16 --mixed_precision --attention_backend sdpa --epochs 2 --max_steps 200 --no_resume`, then a post-load re-evaluation script | Exit 0. Stored and post-load val NDCG identical for both epochs (diff 0.0). Popularity header present. |

### Re-review commands (2026-10-17)

| Command | Result |
|---|---|
| `uv run python -m pytest tests/test_eval_integrity.py -m "not slow" -q` | 52 passed (57 s), including the 3 new tests |
| `uv run python -m pytest tests/test_cli.py -m "not slow" -q` | 63 passed |
| Mutation, scratch copy: `_verify_identity` back to `_load` | `test_identity_check_does_not_read_checkpoints_twice` fails |
| Mutation, scratch copy: drop `popularity_reference`/`eval_protocol` from `final_results` | `test_multi_seed_final_results_carry_protocol_and_popularity` fails |
| Mutation, scratch copy: `rescore_arm` recomputes the fingerprint from `resolved_config` | `test_rescore_of_a_legacy_run` fails |

The scratch copy was deleted afterwards. The worktree has no symlinks or `outputs/`, and its untracked set is unchanged.

## Final Assessment

The feature does what the spec requires:
- Training stays bf16 mixed precision with fp32 masters. Only scoring is fp32.
- Every forward sees current weights.
- Ties are honest and reduce exactly to the old values when there are no ties.
- The popularity floor has no leakage.
- A post-fix re-run retrains every arm.
- `--stage ablation_rescore` works in place on a genuinely legacy run, on CPU and MPS.

CUDA was verified by code inspection only. After the re-review, all four first-pass findings are RESOLVED with verified evidence, and the fixes introduce no regressions (test_eval_integrity and test_cli are green). Only the three INFO observations remain. BUILD has no further action; the feature is ready for the orchestrator to commit.
