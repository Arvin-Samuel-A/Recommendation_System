# Review — Ablations & CLI

## Review Status

**CHANGES_REQUIRED**

(All four prior actionable findings — REV-099, REV-100, REV-101, REV-115 — are fixed and
verified, and REV-116 was closed by the owner. What reopens the spec is REV-131: the
owner's 2026-09-17 smoke-gate decision needs `personalization_eval_users`,
`clients_per_round` and `local_epochs` at the CLI, and none of the three has a flag. New
work created by a deliberate decision, not a regression.)

## Feature

`specs/ablations-cli` — `main.py` (argument parsing, override logging, stage dispatch,
`--stage all`, `--stage plots`, multi-seed aggregation, startup routine) and
`training/ablation_runner.py` (`STUDY_CONFIGS`, `STUDY_EXTRAS`, `resolve_study_configs`,
`_run_single_config`, per-study plotting).

## Review Scope

- Spec directory: `specs/ablations-cli/` — `requirements.md` and `design.md` unmodified;
  `tasks.md` edited (T1.1 `--no_train`, T2.2 REV-100 note, T3.1 REV-101 note).
- Steering: `conventions.md` (Config-Driven Design, plot standards), `code-quality.md`
  (CLI override logging), `architecture.md`.
- Baseline: working tree vs `HEAD` = `d438950` on `master`; `origin/master` at the same
  commit. Of this spec's files, `main.py` (`+52`) and `training/ablation_runner.py`
  (`+34`) are modified, plus `tests/test_cli.py` (`+274`).
- Files inspected: `main.py`, `training/ablation_runner.py`, `tests/test_cli.py`,
  `evaluation/flops.py` (`count_parameters` key set), `outputs/ablations/`.
- CLI surfaces were exercised in a live process, not only read.
- Re-review. Prior findings carried in: REV-099 (MEDIUM), REV-100 (MEDIUM), REV-101 (LOW),
  REV-115 (LOW), REV-116 (INFO).
- Date: 2026-09-17

## Summary

All four actionable findings are fixed, and each was fixed at the level the correction
asked for rather than at the symptom.

**REV-099** (Ablation C's cost panel plotting zeros) — the plot now reads
`total_parameters`, `expert_parameters` and `router_parameters`, the three quantities FR-8
actually names, instead of the invented `total` key. More importantly the *test* was
fixed: `_real_parameters(moe_variant)` now builds its fixture by calling the production
`evaluation.flops.count_parameters` on a real t-small build, so a future key rename fails
the test instead of silently reintroducing zero-height bars. The assertions check the drawn
bar heights against the expected values, not merely that a PNG exists. That is precisely
the class of fix that prevents recurrence, and it is worth crediting: the original defect
existed *because* the fixture invented its own key.

**REV-100** (study A's arm 6 labelled federated but running single-machine LoRA) — the arm
now carries `system_eval="communication"`, routing it into `run_federated`.
`personalization_mode` stays `"lora"` because T2.2 names both, which is correct. The second
test here is the one that shows care: `test_federated_arm_reuses_the_gradnorm_arm_backbone`
asserts `_training_fingerprint(arms[4]) == _training_fingerprint(arms[5])`, so the fix
cannot silently add a training run to `--study all` — `system_eval` is post-training only.

**REV-101** (two FR-14 figures plotting a different quantity from the one named) — size now
plots `model_size_bf16_mb` under an "MB" axis, and communication plots the per-round
`total_round_kb` rather than the cumulative sum. The tests capture the real figures and
read the scatter offsets back, asserting `xs == [22.5]` and `xs == [64.0]` with an explicit
message naming `256.0` as the cumulative value that would indicate regression. They also
assert `ys == [0.17]`, i.e. the REV-117 full-split headline — so this test now guards the
`personalization-federated` invariant too.

**REV-115** (`train=False` reachable only by editing source) — `--no_train` is a real flag,
defaulting to `False`, plumbed to `run_ablation(..., train=not args.no_train)`.

Verified live: `STAGE_MAP` exposes all 13 FR-1 stages plus `lora_rank_sweep` and
`multi_seed`; `_all_pipeline_stages()` returns the 11 stages FR-2 lists in the specified
order and **does not** include `ablation` (AC-03).

What remains is that no ablation study has ever been run — `outputs/ablations/` is empty.
Unlike the equivalent gaps in the evaluation and federated specs, this one is correctly
sequenced behind full training and is not actionable now (REV-130).

## Git Changes

| File | Change | Attribution |
|---|---|---|
| `main.py` | `+52`: `--max_steps` (training-pipeline T6.3), `--no_train` (REV-115), FR-14 axis corrections (REV-101), `final_val_ndcg` use (REV-117) |
| `training/ablation_runner.py` | `+34`: study A arm 6 `system_eval="communication"` (REV-100), Ablation C three-series parameter panel (REV-099) |
| `tests/test_cli.py` | `+274`: `test_max_steps_override`, `test_study_A_final_arm_is_federated`, `test_federated_arm_reuses_the_gradnorm_arm_backbone`, `TestCoreTradeoffPlotAxes`, `TestAblationNoTrainFlag`, production-driven `_real_parameters` |
| `specs/ablations-cli/tasks.md` | T1.1/T2.2/T3.1 notes recording the three fixes |

**Scope assessment:** in scope and proportionate. No new dependency, no committed artifact.
The `tasks.md` edits record fixes and add a flag to T1.1's list — no task was weakened.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 CLI design | **PARTIAL** | Verified live: `STAGE_MAP` carries all 13 named stages (plus `lora_rank_sweep`, `multi_seed`); `--model`, `--moe_variant`, `--num_experts`, `--use_gradnorm`/`--no_gradnorm`, `--study`, `--epochs`, `--batch_size`, `--max_steps`, `--no_train`, `--seed` all present and logged original→new. Missing: `personalization_eval_users`, `clients_per_round`, `local_epochs` — required by the owner's smoke-gate decision (REV-131) |
| FR-2 `--stage all` | PASS | `_all_pipeline_stages()` returns the 11 stages in the specified order, `ablation` absent — checked by calling it, not by reading it |
| FR-3 Startup routine | PASS | All four random sources seeded, `cudnn.deterministic = True`, `seed_worker` defined, `outputs/logs/run_config.json` written (present on disk from the 09-17 run) |
| FR-4 Evaluate skip | PASS | Missing checkpoint → skip message naming the remedy command, returns; no exception |
| FR-5 Ablation runner | PASS | `STUDY_CONFIGS`/`STUDY_EXTRAS` are data; `resolve_study_configs` returns fully resolved Configs; per-study and summary JSON paths implemented. **Never executed** (REV-130) |
| FR-6 Ablation A | PASS | Six arms, and arm 6 now genuinely routes into `run_federated` (REV-100) while sharing arm 5's trained backbone |
| FR-7 Ablation B | PASS | `objective_conditioned` vs `input_only`, same K and seed |
| FR-8 Ablation C | PASS | `match_mmoe_expert_count=True`; the cost panel now plots total, expert and router/gate parameters — the three FR-8 names — plus latency (REV-099) |
| FR-9 Ablation D | PASS | Four arms with gradient-interference extras |
| FR-10 Ablation E | PASS | Three personalization arms |
| FR-11 Ablation F | PASS | Reference plus three single-auxiliary-removed arms; next-item always enabled; disabled objectives excluded from GradNorm and active λ renormalized (verified in `loss-functions`) |
| FR-12 Ablation G | PASS | K ∈ {2,4,6,8}; NDCG left axis / latency right axis asserted by test |
| FR-13 Ablation H | PASS | H1–H6 implemented and selectable without source edits. H3's wording was corrected by the owner on 2026-09-17 to describe the metadata ON/OFF comparison it actually runs, with the ratio sweep explicitly scoped out (REV-116) |
| FR-14 Core tradeoff plots | PASS | All four implemented; the two mislabelled axes corrected and now asserted against the drawn data (REV-101) |
| FR-15 Multi-seed | PASS | Iterates `final_seeds`, aggregates mean ± std across independent runs, records the exact seed list |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 experiments run without source edits | **PARTIAL** | True for every ablation: arg parsing and dispatch verified live, and `--no_train` closed the last ablation-side gap (REV-115). **Not** true for the personalization/federated smoke gate the owner authorised on 2026-09-17, which needs three parameters that have no flag (REV-131) |
| AC-02 `--study all` runs A–H and produces summary JSON | PASS (code) | Sequential dispatch and `summary_{model}.json` writer implemented; never executed (REV-130) |
| AC-03 `--stage all` excludes ablations | PASS | Verified by calling `_all_pipeline_stages()`: 11 stages, `ablation` not among them |
| AC-04 missing checkpoint → skip + remedy, no exception | PASS | Skip path returns with a remedy command; covered in `tests/test_cli.py` |
| AC-05 CLI overrides logged original vs new | PASS | Override tuples logged; `test_max_steps_override` and the other override tests assert the applied values |
| AC-06 resolved config saved with every result | PASS | `_run_single_config` stores `resolved_config` per arm |
| AC-07 all 8 studies produce JSON + plots | **NOT VERIFIABLE** | No study has been run; `outputs/ablations/` is empty (REV-130) |
| AC-08 multi-seed mean ± std across independent seeds | PASS (code) | Aggregation is across independent runs and labelled as such; no artifact yet (REV-130) |
| AC-09 core tradeoff plots correct axes and data | PASS (axes) / NOT VERIFIABLE (artifacts) | Axis quantities now asserted against the drawn offsets (REV-101). The PNGs themselves need `--stage evaluate` and `--stage federated` outputs, which do not exist — see evaluation-diagnostics REV-129 and personalization-federated REV-127 |
| AC-10 startup seeds everything + saves config | PASS | Verified in code; `outputs/logs/run_config.json` present from the 09-17 run |

## Findings

```
REV-099
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-8 | AC: AC-07, AC-09
File: training/ablation_runner.py:642-675; tests/test_cli.py:432-450
Resolution: the plot reads `total_parameters`, `expert_parameters` and
  `router_parameters` -- FR-8's three named quantities -- instead of the `total` key
  `count_parameters` never returned. Confirmed against the production function: its key
  set contains `total_parameters`, and `grep` finds no remaining `.get("total"` in the
  runner.
  The test fix is the part that matters and it was done right: `_real_parameters` now
  calls `evaluation.flops.count_parameters` on a real t-small build, so the fixture cannot
  drift from production. The assertions read the drawn bar heights back and compare them
  to `r["parameters"][key] / 1e6`, with a separate assertion that no bar is zero-height --
  the exact failure the finding described. A PNG-exists assertion could not have caught
  this; these can.
```

```
REV-100
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-6 | AC: AC-07
File: training/ablation_runner.py:25-35; tests/test_cli.py:140-166
Resolution: study A's arm 6 now carries `system_eval="communication"`, which routes
  `_run_single_config` into `run_federated` -- so the progression's final point measures
  the federated system FR-6 names rather than single-machine LoRA under a federated label.
  `personalization_mode` remains "lora", correctly, since T2.2 names both and federated
  training here is LoRA-based.
  `test_study_A_final_arm_is_federated` asserts the arm's fields.
  `test_federated_arm_reuses_the_gradnorm_arm_backbone` asserts
  `_training_fingerprint(arms[4]) == _training_fingerprint(arms[5])`, which is the check
  that keeps the fix from quietly adding a training run to `--study all`. Good instinct:
  the cheap version of this fix would have done exactly that.
```

```
REV-101
Severity: LOW | Status: RESOLVED | Requirement: FR-14 | AC: AC-09
File: main.py (accuracy_vs_model_size, accuracy_vs_communication);
      tests/test_cli.py:180-260
Resolution: (a) size plots `model_size_bf16_mb` under an axis containing "MB", matching
  FR-14's "Size MB vs NDCG@10". (b) communication plots the per-round `total_round_kb`
  under "Communication per round (KB)", not the cumulative sum -- so the ratio between two
  models no longer depends on how many rounds each happened to run.
  Both tests capture the real figures and read the scatter offsets back: `xs == [22.5]` and
  `xs == [64.0]`, the latter with a message naming 256.0 as the cumulative value that would
  signal regression. The communication test additionally asserts `ys == [0.17]`, the
  full-split headline, so it now also guards personalization-federated's REV-117.
```

```
REV-115
Severity: LOW | Status: RESOLVED | Requirement: FR-5 | AC: AC-02, AC-07
File: main.py:49, :533-535; tests/test_cli.py (TestAblationNoTrainFlag)
Resolution: `--no_train` exists as a real flag defaulting to False and is plumbed through
  as `train=not getattr(args, "no_train", False)`. Two tests cover the default and the
  plumbing. `conventions.md`'s "no ablation requires editing Python source" now holds
  without exception.
```

```
REV-116
Severity: INFO | Status: RESOLVED | Requirement: FR-13 | AC: AC-07
File: specs/ablations-cli/requirements.md (FR-13, H3)
Resolution: closed by the spec owner on 2026-09-17. H3 now reads "Cold-start evaluation,
  metadata ON vs OFF, on the held-out cold-item split" -- which is what the arm implements
  -- with a note recording why the ratio sweep is excluded: `cold_start_ratio` is a
  preprocessing-time parameter baked into `data/processed/`, so sweeping it means
  re-running `--stage preprocess` per arm (measured this pass at 108 s and 11 GB peak RSS
  per run), not a config change at training time. A ratio sweep is scoped out as a separate
  multi-preprocessing-run experiment.
```

```
REV-131
Severity: MEDIUM | Status: OPEN | Requirement: FR-1 | AC: AC-01, AC-05
File: main.py:32-63 (`parse_args`), :66-104 (`apply_cli_overrides`)
Problem: `personalization_eval_users`, `clients_per_round` and `local_epochs` have no CLI
  surface. `parse_args` exposes `--epochs`, `--batch_size`, `--moe_variant`,
  `--num_experts`, `--use_gradnorm`/`--no_gradnorm`, `--study`, `--no_train`, `--seed` and
  `--max_steps`; `--epochs` maps to `federated_rounds` for the federated stage, so rounds
  ARE reachable, but the three parameters above are not.
  This became a live blocker on 2026-09-17, when the spec owner reduced AC-05 and AC-06 of
  `specs/personalization-federated` to smoke gates specified at
  `personalization_eval_users = 10` and `clients_per_round = 10`. Running that gate today
  requires editing `configs/config.py` -- which `conventions.md` ("Config-Driven Design")
  forbids, and which would silently change the configured values for every later run.
Impact: The gate the owner just authorised cannot be run the way the project requires
  experiments to be run. Worse than an inconvenience: a source edit to make a smoke run
  cheap is exactly the failure mode REV-066 caught once already, where the recorded
  configuration drifted from the one the gate actually exercised. It also blocks
  `personalization-federated` REV-127 from being closed reproducibly.
  This is the same shape as REV-115 (`--no_train` had no CLI surface) and the `--max_steps`
  addition that Phase 17 needed -- the third instance of the same gap, which suggests the
  rule is worth applying prospectively rather than one flag per finding.
Correction: Add `--eval_users` (-> `personalization_eval_users`), `--clients_per_round`
  and `--local_epochs` to `parse_args`, applied and logged through `apply_cli_overrides`
  like every other override (original -> new). Default them to `None` so the configured
  values stay authoritative when the flags are absent, and add a `tests/test_cli.py` case
  in the shape of `test_max_steps_override`: assert the default is untouched, then assert
  the override applies. Do NOT change the defaults in `configs/config.py` -- the configured
  values are what Phase 14 must run, and the smoke gate is a CLI-time reduction of them.
```

```
REV-130
Severity: INFO | Status: OPEN | Requirement: FR-5 | AC: AC-01, AC-02, AC-07, AC-08
File: outputs/ablations/ (empty)
Problem: no ablation study has ever been run. `outputs/ablations/` contains no
  `{study}_{model}.json` and no `summary_{model}.json`, and none of the per-study plots
  exist. AC-07 ("All 8 ablation studies produce their respective JSON and plot files") has
  no evidence.
Impact: None that is actionable today, which is why this is INFO rather than MEDIUM --
  and it is the distinction from the sibling findings in the other two specs. Running the
  studies requires a trained checkpoint per distinct training fingerprint across A-H, which
  is the Phase 14 full-training job; `--no_train` can evaluate existing checkpoints, but
  only t-small's exists. The gap is correctly sequenced rather than neglected.
Correction: None now. After Phase 14, run `--stage ablation --study all --model t-base`
  and verify each `{study}_{model}.json` records the configs compared, the config diffs,
  metrics with mean ± std, and plot pointers (the Constraints section's per-study
  requirement), and that `summary_{model}.json` is written. Recorded so AC-07 is not
  mistaken for satisfied on the strength of a green suite -- every test in this spec
  exercises config resolution and plotting on fixtures, not a real study.
```

## Design Compliance

Compliant. The config-driven principle now holds without exception: `STUDY_CONFIGS` and
`STUDY_EXTRAS` are data, arms are resolved programmatically from `--study`, every arm saves
its `resolved_config`, and the last behaviour that required a source edit (`train=False`)
has a CLI surface (REV-115).

`_all_pipeline_stages()` being exposed as data rather than inspected as source text is a
genuinely better way to hold AC-03's property, and I verified it by calling it. Multi-seed
aggregation computes std across independent runs and says so in the output, which is the
honest form and directly satisfies FR-15's "don't manufacture std from repeated evaluation
of one seed".

Both design deviations flagged last pass are gone: arm 6 now implements the progression
`design.md` describes, and `train` is no longer source-only.

## Task Completion

| Task | Verdict |
|---|---|
| T1.1 argument parsing | Checked; the `--no_train` addition to the task text records a real new flag |
| T1.2–T1.4 overrides, dispatch, startup | Checked and verified live |
| T2.1 study config resolution | Checked and verified |
| T2.2 Ablation A | Checked, and **now justified** — this was the false-complete REV-100 flagged. The appended note is accurate in every clause I verified, including the backbone-sharing claim |
| T2.3–T2.9 studies B–H | Checked; config resolution verified. End-to-end runs pending (REV-130), which the notes say |
| T3.1 core tradeoff plots | Checked, and now justified — the axis quantities are asserted against drawn data |
| T3.2 multi-seed aggregation | Checked and verified |

No false-complete tasks remain in this spec. The two that were false-complete last pass
(T2.2 and, via REV-099, T3.x's plotting claim) are now backed by tests that can fail.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest tests/test_cli.py -q` | **35 passed**, 1.69 s |
| `uv run python -m pytest -q` (whole suite) | **267 passed**, 0 failed, 49.20 s |
| `STAGE_MAP.keys()` (live) | all 13 FR-1 stages present, plus `lora_rank_sweep`, `multi_seed` |
| `_all_pipeline_stages()` (live) | 11 stages in FR-2's order; `'ablation' in ...` → **False** (AC-03) |
| `grep -n "no_train" main.py` | flag at `:49`, plumbed at `:533-535` (REV-115) |
| `grep -n '"total"' training/ablation_runner.py` | only in the REV-099 explanatory comment; no live read |
| `ls -la outputs/ablations/` | **empty** — REV-130 |
| `uv run python -m compileall -q training main.py` | clean |

## Final Assessment

For BUILD:

1. **REV-131 (MEDIUM, blocking)** — add `--eval_users`, `--clients_per_round` and
   `--local_epochs`, applied and logged through `apply_cli_overrides` like every other
   override, with a `test_max_steps_override`-shaped test. This is the third instance of
   the same gap (after `--no_train` and `--max_steps`), and it blocks
   `personalization-federated` REV-127 from being run reproducibly. Leave the defaults in
   `configs/config.py` alone — the configured values are what Phase 14 runs.

Nothing else. All four prior actionable findings are resolved, and in each case the
regression test was strengthened to the point where the original defect would now fail CI
— `_real_parameters` driving its fixture from the production `count_parameters` is the
clearest example, since the original REV-099 defect existed precisely because the fixture
invented its own key.

Nothing outstanding for the spec owner: REV-116 was closed on 2026-09-17.

**REV-130 (INFO)** remains as a marker, not a task: no ablation study has been run and
`outputs/ablations/` is empty, so AC-07 and AC-08 have no evidence. That is correctly
sequenced behind Phase 14 full training and is not actionable now. It is recorded so a
green suite is not mistaken for study evidence — every test in this spec exercises config
resolution and plotting on fixtures, never a real study.
