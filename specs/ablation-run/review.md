# Ablation Run — Review

## Review Status

**PASS_WITH_WARNINGS** — verdict for the orchestrator: **APPROVE** (round 2, 2026-10-16).

**Round 2:** REV-001..REV-006 are all RESOLVED and verified (see "Round 2" at the end). No new
BLOCKER, MAJOR or MINOR finding. One new NIT (REV-011). The INFO observations REV-007..REV-010
stand. The status stays PASS_WITH_WARNINGS only because of those INFO items and REV-011.

The packager now refuses the uncommitted tree by design, so the feature must be committed
before packaging.

Round 1 (superseded): 0 BLOCKER, 0 MAJOR, 2 MINOR, 4 NIT, 4 INFO. Every requirement and every
Mac-verifiable acceptance criterion was met.

Severity mapping used in this file: BLOCKER = CRITICAL, MAJOR = HIGH, MINOR = LOW, NIT = LOW
(cosmetic or test gap), INFO = observation (not a defect).

## Feature

`specs/ablation-run`: get the repository ready to run ablation studies A–H at t-base with one
seed (42), on a seeded 10% user sample, in mixed bf16 at batch size 256. The work is split
across H100 (FA3: A, D, E, H) and AWS (SDPA: B, C, F, G). It also covers FA3 tooling, the LR
floor, stored width with narrowing, exact resume, OOM retry, the backbone-store lock, arm
persistence and packaging.

## Review Scope

- Spec directory: `specs/ablation-run/` (`requirements.md`, `design.md`, `tasks.md`); steering
  `steering/*.md`.
- Baseline: `master` at `0a91094`. All feature changes are uncommitted: 31 modified tracked files
  (+2377/−415) plus the untracked `scripts/{fetch_fa3_source.sh,install_fa3.sh,
  package_ablation_run.py,run_with_retry.py,verify_attention_backend.py}` and
  `tests/{conftest.py,test_ablation_robustness.py,test_determinism.py,test_fixture_tiny.py,
  test_resume.py,test_scripts.py}`.
- Date: 2026-10-16. Reviewer: REVIEW agent. This is the first review; no prior `review.md`
  existed.

## Summary

The implementation matches the design closely, and BUILD's claims hold up when checked
independently:

- **Golden re-capture is legitimate.** I rebuilt a "pre-FR-4/FR-8" tree in the scratchpad. It
  uses HEAD's `training/pretrain.py` with only the T2.5 `check_processed_dir` call and the T3.1
  LR floor applied, plus HEAD's `training/gradnorm.py` and `evaluation/gradient_diagnostics.py`,
  and today's RNG-neutral probe. The golden test on that tree produced exactly
  `a0cef99f…bfbda18b5b`, the recorded constant. Swapping only the probe back to HEAD's
  global-RNG `randn` at D=16 produced `3e5e596f…`, which is the superseded first capture BUILD
  reported. So the probe really was the only cause, and the current constants describe code with
  no FR-4/FR-8 changes. The live tree (FR-4 + FR-8) reproduces them bit for bit, with 0 and with
  2 workers.
- **The probe draws from its own CPU generator.** `models/transformer.py` `_probe` creates
  `torch.Generator(device="cpu").manual_seed(0)` and never touches the global CPU or CUDA RNG.
- **Resume is exact on CPU.** The repo tests cover LR trace, data order, RNG, GradNorm, a
  completed run and refusals. Two ad-hoc probes also passed (see Testing):
  - A crash between `log_epoch` and the `last.pt` write: on re-run, the epoch-log and GradNorm
    JSONL truncation removes the redone epoch's line and its step-100 record.
  - An end-to-end OOM inside study B's second arm: `run_ablation` propagates it and the lock is
    released. On re-run, arm 1 is reused (no training, no evaluation) and arm 2 resumes at
    epoch 1, running only the 2 remaining steps.
- **Varlen is correct for arbitrary masks.** An ad-hoc probe compared output and q/k/v
  gradients against masked SDPA (fp32, atol 1e-5) for right-padded and holed masks as well as
  left-padded ones. Pads come back as exact zeros. In-order compaction makes it correct for any
  mask, not just left padding.
- **Every reader narrows.** No width-dependent reader was missed, including all
  `*_examples.pkl` and `load_tensorised` sites, `scripts/estimate_training_time.py` and
  `evaluation/flops.py`. `user_sequences.pkl` has no reader outside tests.
- **Packaging output is correct.** I ran `package_ablation_run.py --skip-data` against the real
  `data/processed_users10pct`. The emitted commands match FR-12: mixed bf16, batch 256, FA3 gate
  then A D E H on H100, `/opt/pytorch` then B C F G on AWS, t-base, seed 42, 10% users, 8
  workers, retry wrapper, and the `setsid nohup … < /dev/null &` one-liners.

The two MINOR findings are operational:

- The packager currently produces a code tarball without the three new launch-time scripts,
  because they are untracked. It warns but exits 0 (REV-001).
- Ablation arms share one per-model epoch/GradNorm log, so resume-time truncation can splice
  records from two arms in an edge case (REV-002).

## Git Changes

- **Status:** branch `master`, HEAD `0a91094`, every change uncommitted (see Review Scope).
  `.gitignore` gains `data/processed_users*/` and `/logs/`.
- **Relevant changes:** `configs/config.py`, `main.py`, `utils.py`,
  `data/{preprocessing,sequence_generator}.py`, `models/transformer.py`,
  `training/{pretrain,gradnorm,ablation_runner,federated,local_train}.py`,
  `evaluation/{cold_start,longtail,gradient_diagnostics}.py`,
  `scripts/estimate_training_time.py`, the five new scripts, the new and modified tests, and
  `README.md`.
- **Scope assessment:** in scope.
  - The edits to `steering/*.md`, `specs/BUILD_ORDER.md` and the superseded clauses in other
    specs' `requirements.md` are PLAN's steering-conflict pointers named in requirements
    "Dependencies" and design "Steering Conflicts Resolved". They are not BUILD scope creep.
  - `tests/test_training.py::_build_subset_processed_dir` (T0.2) is an owner-approved extra.
  - `.claude/` and `HANDOFF-ablation-run.md` are untracked tooling and input. They are not part
    of the code tarball.
  - No new third-party dependency: `fcntl` and `tarfile` are stdlib, and FA3 is not in
    `requirements.txt`.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Precision | PASS | `Config` defaults unchanged (`TestAblationRunConfig`). Generated `run_h100.sh`, `run_aws.sh` and `LAUNCH.md` carry `--dtype bf16 --mixed_precision` (I inspected the packager output). |
| FR-2 FA3 backend | PASS | `VALID_BACKENDS`/`validate()` hold exactly three values. `preferred_backend` (9.x → FA3). Lazy `_import_fa3`. Own `varlen_meta`/`unpad`/`repad` with a per-mask cache. `causal=` on both calls. dtype gate before import. Probe at `hidden_dim // num_heads`, own RNG. `_runtime_record` per arm. Real kernel: target machine. |
| FR-3 FA3 tooling | PASS (Mac part) | `fetch_fa3_source.sh`, `install_fa3.sh` (preflight, `--dry-run`, skip-if-installed, flag check, ends in verify) and `verify_attention_backend.py` (exit 0/1/2, JSON report). The build itself is a target-machine step. NITs REV-006 and REV-007 apply. |
| FR-4 Bit-identical utilisation | PASS | Golden reproduced independently (Summary). Workers equivalence, CPU-side negatives, batched checks A/B, per-epoch materialisation and branch-free renorm are all tested. |
| FR-5 LR floor | PASS | `build_scheduler` follows the D5 formula with clamp. Default 0.1, validated to [0,1], in the fingerprint. No scheduler in LoRA/federated (grep: the only `LambdaLR` is in `pretrain.py`). |
| FR-6 User sampling | PASS | `_read_ratings` samples after both filters, sorted users, `RandomState(user_sample_seed)`, full histories. Five fields in both JSONs. Derived dir `data/processed_users10pct`. Real manifest: 0.1 / 42 / 200 / 136677 → 13668. |
| FR-7 Stored width + narrowing | PASS | `stored_seq_length`, `narrow_examples`/`load_examples`, width guard in `load_tensorised`, model position guard, `check_processed_dir` in `run_pretrain` and `run_ablation`. Reader grep is complete. |
| FR-8 Resume | PASS | Atomic `atomic_torch_save` for every checkpoint. `last.pt` holds the D8 state plus the global Python/NumPy RNG. Refusal on fingerprint, `total_steps` or revision. Completed-run short-circuit. Log/JSONL hygiene (MINOR REV-002 in the ablation context). |
| FR-9 OOM / retry | PASS | `is_oom_error`, `main` → `SystemExit(75)`, `run_ablation` re-raises OOM only, `run_with_retry.py` retries only 75 and handles SIGTERM/SIGINT. |
| FR-10 Store lock | PASS | `backbone_lock` uses `flock`, re-checks the store after acquiring, logs the waiting holder pid once, has an optional timeout, and releases in `finally`. Kernel release on death is tested (SIGKILL). |
| FR-11 Arm persistence | PASS | `_arm_key` (config minus path/runtime fields, study, label, model, revision) with atomic arm files. `--no_resume` disables reuse. Results go under `seed_{seed}/`. |
| FR-12 Packaging | PASS | Manifest validation, one command builder, `bash -n`, tracked-files tarball plus `launch/`, data tarball, placeholders only. MINOR REV-001 and NIT REV-005 apply. |
| FR-13 Fingerprint hygiene | PASS | `_RUNTIME_ONLY_FIELDS = {num_workers, pin_memory, resume_training}`. `persistent_workers` stays in. |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 | PASS | `TestAblationRunConfig::test_project_default_precision_is_unchanged`, `TestPackageAblationRun::test_builds_tarballs_and_launch_files`. Re-checked on my own packager output. |
| AC-02 | PASS | `TestFlashAttention3Backend::test_valid_backends_are_exactly_three` and `test_config_rejects_removed_backends_naming_the_three`. Grep finds no `flash_attention_4` or `flash_attn` import in code. |
| AC-03 | PASS | `test_preferred_backend` (9.0, 8.0, 8.9, 7.5, mps, cpu), `test_auto_on_hopper_without_fa3_*`, `test_explicit_fa3_without_the_package_raises`. |
| AC-04 | PASS | `test_import_and_sdpa_build_need_no_fa3`: a fresh interpreter with `sys.modules["flash_attn_interface"]=None`. Test collection succeeds on the Mac. |
| AC-05 | PASS | `TestVarlenCore`: roundtrip, `cu_seqlens [0,8,13,14]`, out+grad parity at atol 1e-5, `causal=True`, meta computed once per 3-layer forward. Arbitrary-mask parity confirmed by my probe. |
| AC-06 | PASS | `test_padded_batch_matches_masked_sdpa[returns_lse]`, `test_unpadded_batch_takes_the_dense_call[...]`, `test_fp32_input_raises_naming_the_dtype`. |
| AC-07 | PASS (Mac) / target | `TestRealFlashAttention3Kernel` is reported SKIPPED with the expected reason. |
| AC-08 | PASS (Mac) / target | `TestVerifyAttentionBackend`: sdpa exits 0 with a full report; FA3 exits 2 and prints "unavailable". |
| AC-09 | PASS (Mac) / target | `TestFA3InstallScripts`: `bash -n`, executable bit, dry-run exits 2 with "PREFLIGHT FAILED: CUDA" and no build dir, script ends in verify. Launch scripts pass `bash -n` (packager and test). |
| AC-10 | PASS | `test_workers_yield_identical_batches`: 2 epochs, 0 vs 2 workers with `pin_memory`, every key `torch.equal`. |
| AC-11 | PASS | `test_golden_tiny_run[0,1]` and `test_golden_hash_with_two_workers` pass. The constants are independently re-derived from a reconstructed pre-FR-4 tree (Summary). |
| AC-12 | PASS | `test_cpu_side_negatives_identical` (CPU, copy, MPS) and `test_step_loop_samples_negatives_before_the_device_copy` (on MPS the sampler always sees CPU tensors). |
| AC-13 | PASS | `TestBatchedFiniteChecks` (NaN loss names the objective and moves no parameter; inf grad norm raises; NaN lambdas raise before the step) and `TestVectorisedRenormalisation`. |
| AC-14 | PASS | `TestLRScheduler::test_cosine_decays_to_floor`, `test_zero_floor_matches_previous_schedule` (exact `==`), `test_floor_is_part_of_the_training_fingerprint`. Validation cases are in `TestAblationRunConfig`. The old near-zero test is removed. |
| AC-15 | PASS | `TestUserSampling` (pre-written, unweakened) plus `TestUserSamplingOnTheTinyFixture` (five fields in both JSONs, every user sampled, `num_items` = distinct sampled items, no `RandomState` at 1.0). |
| AC-16 | PASS | `TestStoredWidth`: width 16 everywhere, manifest 16, narrowed-to-8 equal to generated-at-8 for every split (tensorised and examples). |
| AC-17 | PASS | `TestEveryReaderNarrows` (static AST audit plus a defect-shape self-test) and `TestNarrowedReadersRunEndToEnd` (evaluate, routing, gradient diagnostics, personalize, federated, cold-start, long-tail at width 8 over width 16). |
| AC-18 | PASS | `TestPositionGuard`. |
| AC-19 | PASS | `TestStoredWidth`: requesting 17 raises naming `--stage preprocess` and `sequence_length_grid`. |
| AC-20 | PASS | `TestProcessedDirCheck`: both entry points raise with 0 `train_one_step` calls. |
| AC-21 | PASS | `TestResume::test_interrupted_run_resumes_bit_identically[0,2]`. The truncation branch is not exercised by repo tests but I verified it with a probe (NIT REV-003). |
| AC-22 | PASS | `test_resume_is_refused_for_another_run[learning_rate,pretrain_epochs,source_revision]`, `test_total_steps_mismatch_is_refused`, `test_no_resume_trains_from_scratch`, `test_completed_run_trains_zero_steps`, `test_unchanged_copy_is_accepted`. |
| AC-23 | PASS | `TestLastCheckpoint::test_a_crash_mid_write_leaves_the_previous_checkpoints_intact`. |
| AC-24 | PASS | `TestOOMHandling`: predicate table, `SystemExit(75)`, re-raise of a non-OOM error, propagation versus recording. |
| AC-25 | PASS | `TestRunWithRetry`: 75,75,0 → 0 after 3 attempts; max-retries → 75; 3 → 3 after 1 attempt; SIGTERM → 143 with the child gone. |
| AC-26 | PASS | `TestBackboneLock` (timeout raises `TimeoutError`; waiter reuses the backbone, never calls `run_pretrain`; holder pid in the file; SIGKILL leaves no stale lock). |
| AC-27 | PASS | `TestArmPersistence` (reuse, identical JSON, post-training change re-evaluates without retraining, `--no_resume`, `seed_42/` path, revision changes the key). The single-arm selectivity is untested (NIT REV-004). |
| AC-28 | PASS | `TestPackageAblationRun` (exact member set, data tarball, order, flag set, one-liners, refusals). |
| AC-29 | PASS | `TestFingerprint` (3 unchanged-by, 5 changed-by). |
| AC-30 | PASS | `TestRuntimeRecord`. |
| AC-31 | PARTIAL (reviewer) / PASS (BUILD evidence) | (a) I read the manifest: fraction 0.1, seed 42, `stored_seq_length` 200. (d) I re-ran the packager and the commands match FR-12. (b) and (c) are real-data runs that would overwrite `outputs/`, so I did not repeat them. BUILD's T9.2 note quotes the resume line and the arm-reuse lines, and my tiny-fixture probes reproduce the same mechanisms. Not a defect. |

## Findings

```
REV-001
Severity: MINOR (LOW) | Status: RESOLVED (round 2) | Requirement: FR-12 | AC: AC-28
File: scripts/package_ablation_run.py — main() line 363 / _warn_untracked_requirements() line 312
Problem: When scripts/run_with_retry.py, scripts/verify_attention_backend.py or
  scripts/install_fa3.sh are not git-tracked, the packager prints a WARNING and still exits 0,
  writing a code.tar.gz that lacks the files every launch command executes. All three are
  untracked right now: my run produced an 87-member tarball without them, exit 0.
Impact: If the owner packages before the commit lands, the H100's FA3 gate fails
  ("can't open file scripts/verify_attention_backend.py") and run_h100.sh exits 1 with no
  study run. On AWS every study exits 2 within seconds. The failure is loud but costs a
  copy/extract round trip on both machines.
Correction: Treat a missing REQUIRED_TRACKED file as a refusal (non-zero exit naming the
  files), or add them to the tarball explicitly. Add a test that packages with one of them
  untracked/missing and expects a non-zero exit.
```

```
REV-002
Severity: MINOR (LOW) | Status: RESOLVED (round 2) | Requirement: FR-8 (log hygiene), FR-11
File: training/pretrain.py — run_pretrain() lines 1053-1054 (_truncate_epoch_log /
  _truncate_gradnorm_log); training/ablation_runner.py — _run_single_config() line 448 (only
  checkpoint_dir is namespaced per fingerprint; log_dir stays the base config's)
Problem: Every ablation arm of a model writes the same `pretrain_{model}.log` and
  `gradnorm_{model}.jsonl`. Resume truncation keeps whatever lines are in those files with
  epoch < start_epoch or step < global_step, without checking which arm wrote them.
Impact: Scenario: in process 1, arm W fails with a non-OOM error mid-training, then arm X
  trains several epochs and OOMs. In process 2, W is re-run and resumes from its last.pt. W
  keeps X's epoch lines below its own start_epoch and appends its own. X then resumes and
  keeps a mix of both. The per-epoch log and the GradNorm trajectory plot for these arms show
  spliced data. Results JSON, checkpoints and training are unaffected. The plain OOM-retry
  path is fine: the resumed arm is always the last writer, and my probe confirmed it.
Correction: Namespace the training logs per fingerprint in _run_single_config (for example
  set cfg.log_dir under the store directory, or put the fingerprint in the log file names).
  Alternatively, stamp each record with the training fingerprint and filter on it when
  truncating.
```

```
REV-003
Severity: NIT (LOW, test gap) | Status: RESOLVED (round 2) | Requirement: FR-8 | AC: AC-21
File: tests/test_resume.py — TestResume::test_interrupted_run_resumes_bit_identically
Problem: In this test nothing needs truncating. The crash comes before epoch 2's log line,
  and GradNorm logs only every 100 steps, so the 12-step run has a single record (step 0).
  Neither _truncate_epoch_log nor _truncate_gradnorm_log ever removes a line in the repo's
  tests. If either function were a no-op, every test would still pass.
Impact: A regression in the duplicate-removal path would go unnoticed. (I verified the
  current behaviour with an ad-hoc probe: a crash after log_epoch and before last.pt in
  epoch 2 at 40 steps/epoch leaves epochs [0,1,2] and steps [0,100], no duplicates.)
Correction: Add a test that patches save_checkpoint to raise for last.pt at the final
  epoch, after log_epoch, and spans more than 100 steps (or makes the GradNorm log interval
  patchable). Assert one line per epoch and unique steps after the resume.
```

```
REV-004
Severity: NIT (LOW, test gap) | Status: RESOLVED (round 2) | Requirement: FR-11 | AC: AC-27
File: tests/test_ablation_robustness.py — TestArmPersistence::test_rerun_reuses_completed_arms
Problem: AC-27 says that changing a post-training field of ONE arm re-runs only that arm.
  The test changes ablation_eval_examples for every arm and asserts both are re-evaluated.
Impact: Per-arm selectivity holds by construction (the key is per arm), but nothing tests it.
Correction: Change a field for one label only (for example monkeypatch
  resolve_study_configs to alter arm 2), then assert evaluate_model ×1 and that arm 1 is
  marked reused_from_previous_run.
```

```
REV-005
Severity: NIT (LOW) | Status: RESOLVED (round 2) | Requirement: FR-12
File: scripts/package_ablation_run.py — render_script() line 152 (generated launch/run_aws.sh)
Problem: run_aws.sh runs under `set -uo pipefail` and then does `source /opt/pytorch/bin/activate`.
  An activate script that expands an unset variable (some conda and older venv templates do)
  aborts the non-interactive shell before the first study. If activation silently fails
  instead, PY=python may be an interpreter without torch, and every study fails with a
  non-75 status.
Impact: The AWS run does not start. This is visible in logs/run_aws.log, not silent.
Correction: Emit `set +u; source …/bin/activate; set -u`, then a one-line
  `"$PY" -c 'import torch'` check that exits with a clear message.
```

```
REV-006
Severity: NIT (LOW) | Status: RESOLVED (round 2) | Requirement: FR-3
File: scripts/install_fa3.sh — line 148 (build pipeline under set -euo pipefail)
Problem: A failed `pip install` makes the script exit with pip's status (usually 1). The
  script header documents 1 as "parity failure" from the verification step.
Impact: Someone reading only the exit status of a detached run cannot tell a build failure
  from a parity failure. The log does show which it was.
Correction: Run the build in an `if ! …; then log "build FAILED"; exit 3; fi` block (or
  similar) and document the code.
```

INFO (observations, not defects):

- **REV-007 (INFO, FR-3/D3):** `verify_attention_backend.py` uses atol and rtol of 2e-2 (lines
  36-37). BUILD measured a max grad diff of 2.2e-2 for bf16 SDPA on MPS, which passes only
  through the rtol term. If FA3 on the H100 lands slightly worse, the gate exits 1 and the
  whole H100 run aborts. That is fail-safe, not silent. Look at the `max_abs_*_diff` values in
  `logs/verify_fa3.json` before loosening anything.
- **REV-008 (INFO, FR-2/FR-4):** BUILD deviated from the T3.2 ordering: the golden was
  re-captured at T5.1 start. I verified it was legitimate (see Summary). Because the probe now
  uses its own RNG, initial weights for a given seed differ from HEAD's. This is an intended
  FR-2 consequence, not an FR-4 regression, and backbones registered before this feature are
  invalidated anyway by the new fingerprint fields (D13).
- **REV-009 (INFO, pre-existing):** `raw_data_dir` is not in `_PATH_FIELDS`, so it is part of
  the training fingerprint (BUILD flagged this in T6.2). It does not matter for this run,
  because the default `data/raw` is the same on every machine.
- **REV-010 (INFO, FR-10):** On Linux, DataLoader workers are forked inside the locked region
  and inherit the `flock` descriptor. After a SIGKILL of the trainer the lock lasts until the
  workers' parent-death watchdog ends them (seconds), so it is not stale. Separately, the
  GradScaler resume path is untested because only fp16 uses it and this run is bf16.

## Design Compliance

The implementation matches D1–D14, with these documented and acceptable deviations:

- `_probe` draws from its own CPU generator. This is not in D2, but it is a strict
  improvement and is the root cause behind the golden re-capture.
- `VarlenMeta.indices` is `None` when `all_real`, which avoids a second sync.
- The dtype check runs before the FA3 import.
- `last.pt` also saves the global Python and NumPy RNG states.
- The data tarball is named by the derived directory rather than the source basename
  (T8.1 deviation 2). This is correct, because the launched commands resolve the derived name.
- Narrowing helpers were added in T2.2 rather than T2.3.

There is no silent redesign, no new dependency and no unapproved architectural change.
`has_padding`, MIM/multi-step and contrastive syncs are kept, as D4 decided. The FA3 call
signature (keyword `cu_seqlens_q/k`, `max_seqlen_q/k`, `causal`) and the handling of both
return forms (`out` and `(out, lse)`) match D2.

## Task Completion

Every task marked `[x]` (T0.1–T9.2, including the owner-approved T0.2) is implemented. Its
Done-when checks that can run on the Mac were re-run, and each passed:

- **T0.1, T1.x, T2.x, T3.1, T4.x, T5.x, T6.x, T7.x, T8.x:** I re-ran the tests and inspected
  the code.
- **T3.2:** I re-derived the constants independently.
- **T9.1:** I re-ran the per-file sweep for the eleven feature files. The counts match BUILD's
  note. I did not re-run `test_federated`, `test_diagnostics`, `test_lora`, `test_moe` or
  `test_smoke`.
- **T9.2:** (a) and (d) are re-verified. (b) and (c) are accepted on BUILD's quoted evidence
  plus my equivalent tiny-fixture probes.

No task is falsely marked complete.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run pytest tests/test_fixture_tiny.py -m "not slow" -q -rs` | 4 passed |
| `uv run pytest tests/test_cli.py -m "not slow" -q -rs` | 63 passed |
| `uv run pytest tests/test_preprocessing.py -m "not slow" -q -rs` | 45 passed |
| `uv run pytest tests/test_sequence_generator.py -m "not slow" -q -rs` | 37 passed |
| `uv run pytest tests/test_transformer.py -m "not slow" -q -rs` | 53 passed, 1 skipped (real FA3 kernel: "needs CUDA capability 9.x and flash_attn_interface") |
| `uv run pytest tests/test_training.py -m "not slow" -q -rs` | 26 passed, 2 deselected |
| `uv run pytest tests/test_losses.py -m "not slow" -q -rs` | 40 passed |
| `uv run pytest tests/test_determinism.py -m "not slow" -q -rs` | 6 passed (golden ×2, golden with 2 workers) |
| `uv run pytest tests/test_resume.py -m "not slow" -q -rs` | 12 passed (96 s) |
| `uv run pytest tests/test_ablation_robustness.py -m "not slow" -q -rs` | 29 passed |
| `uv run pytest tests/test_scripts.py -m "not slow" -q -rs` | 14 passed |
| Golden test on a scratch copy with HEAD `pretrain.py` (+T2.5, +T3.1), HEAD `gradnorm.py` and `gradient_diagnostics.py` (`GOLDEN_CAPTURE=1`) | 1 passed, hash `a0cef99f…18b5b` (equals the recorded constant) |
| Same, with the probe reverted to HEAD's global-RNG `randn` at D=16 | hash `3e5e596f…` (the superseded first capture); fails the current constants, as expected |
| Scratch probe: varlen vs masked SDPA for left, right and holed masks (fp32) | output and q/k/v grads match at atol 1e-5; pads exactly 0 |
| Scratch probe: crash after `log_epoch` and before `last.pt` (3 epochs × 40 steps), then re-run | epoch log `[0,1,2]`, GradNorm steps `[0,100]`, no duplicates |
| Scratch probe: `run_ablation` study B, OOM at arm 2's epoch-1 first step, then re-run | OOM propagates; re-run makes 2 train steps and 1 evaluation; arm 1 reused, arm 2 ok |
| `uv run python scripts/package_ablation_run.py --skip-data --out <scratchpad>/dist` | exit 0; commands match FR-12; warns that the three new scripts are untracked (REV-001) |
| Read `data/processed_users10pct/preprocessing_manifest.json` | fraction 0.1, seed 42, stored width 200, 136677 → 13668 users |
| `grep` for `flash_attention_4`, `from flash_attn import` and `flash_attn.bert` in code dirs | no matches |
| `df -h /` before and after | 59 GiB free throughout; repo `git status` unchanged; no `logs/` created |

All scratch copies and probes lived in the session scratchpad. No application code, test or
spec file was modified.

## Final Assessment

**APPROVE (PASS_WITH_WARNINGS).** The feature satisfies FR-1–FR-13 and every Mac-verifiable
acceptance criterion. Training semantics under FR-4 and FR-8 are bit-identical, and that is
proven independently of BUILD's own test. Resume, OOM retry, lock and arm persistence work
together end to end.

Before launch:

1. **Commit before packaging** (orchestrator). Otherwise `dist/code.tar.gz` lacks the three new
   scripts (REV-001). BUILD should ideally make the packager refuse in that case.
2. Optional hardening (BUILD, non-blocking):
   - per-fingerprint `log_dir` for ablation arms (REV-002);
   - a truncation test (REV-003);
   - a single-arm selectivity test (REV-004);
   - `set +u` around the AWS venv activation (REV-005);
   - a distinct exit code for FA3 build failures (REV-006).
3. On the target machines, the owner checks AC-07, AC-08 and AC-09 with the FA3 build and
   verification, and checks `logs/verify_fa3.json` margins (REV-007).

---

## Round 2 (re-review of the "Review fixes (round 2)" section of tasks.md)

**Date:** 2026-10-16.

**Scope:**
- the fixes for REV-001..REV-006;
- the `steering/testing.md` and `requirements.md` text about the subset helper;
- regressions those fixes could cause.

The baseline is unchanged: `master` at `0a91094`, everything uncommitted.

**Method.** I did not rely on BUILD's claims:
- I read every changed code path.
- I re-ran the affected test files.
- I ran the packager myself, both without and with `--allow-dirty`.
- I mutation-tested the new tests in a scratch copy, reverting each fix to confirm the test
  then fails.

### Findings re-evaluated

| ID | Round-2 status | Evidence |
|---|---|---|
| REV-001 | RESOLVED | See "REV-001" below. |
| REV-002 | RESOLVED | See "REV-002" below. |
| REV-003 | RESOLVED | See "REV-003" below. |
| REV-004 | RESOLVED | See "REV-004" below. |
| REV-005 | RESOLVED | See "REV-005" below. |
| REV-006 | RESOLVED | See "REV-006" below. |
| REV-007..REV-010 | Unchanged (INFO) | REV-007 is left as is. BUILD measured relative Frobenius error ≤ 2.4e-3 and recommends a relative-error gate. That recommendation is reasonable but optional and does not block. |

#### REV-001: packager refuses uncommitted code

- **What changed.** `launch_readiness_problems()` (`scripts/package_ablation_run.py:353`) runs
  before anything is written. It refuses when:
  - a launch-time file is untracked;
  - any covered `.py` (same `_is_covered_source` rule as the provenance hash) is modified or
    untracked.
- **Today's tree is refused.** On the real tree, the packager exits 1 and lists 19 problems:
  the 3 untracked scripts plus the modified and untracked covered sources. It writes no `dist/`.
- **`--allow-dirty` works.** It exits 0, and the tarball has 90 members, which now include
  `scripts/run_with_retry.py`, `verify_attention_backend.py` and `install_fa3.sh`.
- **It will pass once committed.**
  - `TestPackagerRefusesUncommittedCode::test_committed_tree_packages_without_override` copies
    the tracked and untracked-not-ignored files into a throwaway repo, commits them, and
    packages without the flag: exit 0, all four required files present.
  - The only untracked files left out of that copy are `.claude/*.md`, which are not covered
    sources.
  - So once the orchestrator commits the feature files (the five scripts, the tests and the
    modified sources), the readiness check finds nothing.
- **The refusal test is non-vacuous.** It asserts that each named problem appears and that
  nothing is written.
- **No regression in the other refusals.** `validate_manifest` still runs first, so the
  fraction and stored-width refusal tests still exercise their own messages.

#### REV-002: per-arm training logs

- **What changed.** `_run_single_config` sets `cfg.log_dir = training_log_dir(root, fp)`
  (`<store>/<fp>/logs`) only around `run_pretrain`, and restores it in a `finally` block.
  - `log_dir` is in `_PATH_FIELDS`, so neither the fingerprint nor `_arm_key` changes.
  - `resolved_config` is captured before the swap.
  - Post-training reports (`moe_parameter_report`, gradient diagnostics extras) still go to the
    arm's `log_dir`.
- **Nothing that collects results reads the moved files.** I grepped every non-test module:
  the epoch log, the GradNorm JSONL and the routing JSON are read by no code outside
  `run_pretrain` itself. `plot_gradnorm_trajectory` reads the same per-fingerprint directory
  it wrote.
- **Plots** stay in `plot_dir`, as before.
- **The reuse path is unaffected.**
  - Store reuse and persisted-arm reuse do not touch training logs.
  - A resumed arm finds its logs in the same per-fingerprint directory in every process.
  - `TestArmPersistence`, `TestSingleArmRerun` and `TestBackboneLock` are green.
- **Collection.** `LAUNCH.md`'s collect command and the README now include
  `outputs/checkpoints/ablations/by_fingerprint/*/logs`.
- **The test catches the bug.** Mutation: dropping the `cfg.log_dir` swap makes
  `TestPerArmTrainingLogs` fail (the per-fingerprint `pretrain_t_small.log` is missing).

#### REV-003: resume log-cleanup test

- **What changed.** `GRADNORM_LOG_INTERVAL_STEPS` is a module constant, still 100 and read at
  call time, so tests can patch it. The golden test is still green.
- **The new test is non-vacuous.** `TestResumeLogCleanup` asserts that before the resume the
  logs already hold epochs `0,1,2` and steps `0..10`.
- **It catches both failure modes.** Mutations:
  - making `_truncate_epoch_log` a no-op → the test fails with `['0','1','2','2']`;
  - making `_truncate_gradnorm_log` a no-op → the test fails with duplicate steps.

#### REV-004: single-arm selectivity test

- `TestSingleArmRerun` changes `ablation_eval_examples` only for `input_only`, by wrapping
  `resolve_study_configs`.
- It asserts `run_pretrain` ×0 and `evaluate_model` ×1.
- The other arm is `reused_from_previous_run` with 32 examples; the changed arm is
  re-evaluated with 16.
- It would fail if the arm key were not per arm, or ignored config.

#### REV-005: safe venv activation

- The generated `run_aws.sh` now runs `set +u` / `source /opt/pytorch/bin/activate` /
  `set -u`, then `"$PY" -c "import torch" || { …; exit 1; }` before the study loop.
- I inspected my own generated copy, and it passes `bash -n`.
- The packaging test asserts the content and the order.

#### REV-006: distinct build-failure exit code

- **What changed.** `install_fa3.sh` wraps the build as
  `if ! (cd … && $BUILD_CMD) 2>&1 | tee …; then build_failed …`.
  - Under `pipefail` the pipeline status is the build's, and `set -e` does not fire inside an
    `if`.
  - Extraction failure also maps to 3.
  - The exit codes 0, 1, 2 and 3 are documented in the script header, the README and
    `LAUNCH.md`.
- **The test runs the real script.** `TestFA3InstallExitCodes` uses stub
  `python`/`nvcc`/`ninja` on `PATH` and gets exit 3, with the build output in
  `fa3_build.log` and no verification run.
- **It would fail against round-1 code,** which exited with pip's status (1).

#### Spec and steering text

`requirements.md` (Non-Goals and "Out-of-scope findings") and `steering/testing.md` now record
that the subset-helper disk hog was fixed by owner-approved T0.2. This is accurate. I checked
the T0.2 implementation in round 1.

### New finding

```
REV-011
Severity: NIT (LOW, test robustness) | Status: OPEN | Requirement: FR-3
File: tests/test_scripts.py — TestFA3InstallExitCodes (both tests)
Problem: The stubbed run still goes through install_fa3.sh's real disk preflight, which needs
  20 GB or more free under the tmp build dir.
Impact: On a host with less than 20 GB free (the Mac has hit a full disk before), both tests
  fail with "PREFLIGHT FAILED: disk", not with the behaviour under test. This is a false
  failure, not a false pass.
Correction: Make MIN_FREE_GB overridable (for example FA3_MIN_FREE_GB) and set it to 0 in the
  test env.
```

### Round 2 testing

| Command | Result |
|---|---|
| `uv run pytest tests/test_scripts.py -m "not slow" -q -rs` | 18 passed |
| `uv run pytest tests/test_resume.py -m "not slow" -q -rs` | 13 passed (112 s) |
| `uv run pytest tests/test_ablation_robustness.py -m "not slow" -q -rs` | 31 passed |
| `uv run pytest tests/test_determinism.py -m "not slow" -q -rs` | 6 passed (golden unchanged) |
| `uv run pytest tests/test_cli.py -m "not slow" -q -rs` | 63 passed |
| `uv run pytest tests/test_training.py -m "not slow" -q -rs` | 26 passed, 2 deselected |
| `uv run pytest tests/test_sequence_generator.py -m "not slow" -q` | 37 passed (end-to-end readers unaffected) |
| `uv run python scripts/package_ablation_run.py --skip-data --out <scratchpad>/dist_r2` | exit 1; 19 problems listed; nothing written |
| same command with `--allow-dirty --out <scratchpad>/dist_r2b` | exit 0; 90 members including the 3 launch scripts; `run_aws.sh` passes `bash -n` |
| Scratch-copy mutation: drop the `cfg.log_dir` swap → `TestPerArmTrainingLogs` | FAILED (as required) |
| Scratch-copy mutation: no-op `_truncate_epoch_log` → `TestResumeLogCleanup` | FAILED (as required) |
| Scratch-copy mutation: no-op `_truncate_gradnorm_log` → `TestResumeLogCleanup` | FAILED (as required) |

Free disk stayed at 57–58 GiB. No repo file other than this `review.md` was modified, and no
`logs/` directory was created.

### Round 2 final assessment

**APPROVE (PASS_WITH_WARNINGS).** All six round-1 findings are fixed, tested, and their tests
would fail without the fixes. Nothing regressed: results collection, plots, arm reuse, store
reuse and resume behave as before.

Next steps:
1. The orchestrator commits the feature, including the five new scripts and the new tests. The
   packager then runs without `--allow-dirty`; this is shown by the throwaway-repo test.
2. REV-011 is an optional NIT.
