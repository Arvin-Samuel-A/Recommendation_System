# Review — Ablations & CLI

## Review Status

**PASS_WITH_WARNINGS**

(Up from `CHANGES_REQUIRED`. REV-131, the blocking finding, is implemented and verified —
and it unblocked `personalization-federated` REV-127, which is now closed. One INFO
remains: no ablation study has ever been run, which is correct sequencing rather than
neglect, but it leaves AC-02, AC-07 and AC-08 without real-run evidence.)

## Feature

`specs/ablations-cli` — `main.py` (argument parsing, override logging, stage dispatch,
`--stage all`, `--stage plots`, multi-seed aggregation, startup routine) and
`training/ablation_runner.py` (`STUDY_CONFIGS`, `STUDY_EXTRAS`, `resolve_study_configs`,
`_run_single_config`, per-study plotting).

## Review Scope

- Spec directory: `specs/ablations-cli/` (requirements.md, design.md, tasks.md). None
  modified in the working tree. FR-13's H3 carries the owner's 2026-09-17 REV-116
  amendment.
- Steering: `conventions.md` (Config-Driven Design, Output Organization, Plot Standards),
  `code-quality.md` (override logging, error handling), `testing.md`.
- Baseline: `master` at `0bc67ee`, `origin/master` at the same commit. No file owned by
  this spec is modified in the working tree.
- Files inspected: `main.py` (935 lines), `training/ablation_runner.py` (810 lines),
  `tests/test_cli.py` (922 lines), and `outputs/ablations/` (empty),
  `outputs/logs/run_config.json`.
- Re-review (third pass). Prior findings carried in: REV-130 (INFO, OPEN), REV-131,
  REV-135, REV-099, REV-100, REV-101, REV-115, REV-116 (RESOLVED).
- Date: 2026-09-18 (post-BUILD-fix)

## Summary

REV-131 is implemented exactly as the finding specified, and I verified the parts that
matter rather than reading the note.

`--eval_users`, `--clients_per_round` and `--local_epochs` exist, all default to `None`,
and `configs/config.py` is untouched — measured: a fresh `Config()` still carries
`personalization_eval_users=1000`, `clients_per_round=100`, `local_epochs=3`. Each override
emits the literal `CLI override: {field} {original} -> {new}` line `code-quality.md`
requires; I saw all four of them emitted. The flags are orthogonal: a run passing
`--clients_per_round` leaves `personalization_eval_users` at 1000, and vice versa. That is
the property that makes a CLI-time reduction safe, and it is the one a careless
implementation would break.

This is the third instance of the same gap (`--no_train`, then `--max_steps`, now these
three), and BUILD's note says so. The observation is worth acting on prospectively — but
that is a suggestion, not a finding.

T1.6 is the more interesting item, because **it is not in any review**. BUILD found it
while carrying out REV-129's verification: `_load_trained_model` referenced
`load_metadata_matrix` without importing it, so `--stage gradient_diagnostics` and
`--stage routing_diagnostics` had **never been runnable**, raising `NameError` before
constructing a model. A 267-test green suite coexisted with two dead stages because every
CLI test stopped at argument parsing or returned at the checkpoint guard above the faulty
line. I reproduced the non-vacuity claim: removing the import again makes
`TestLoadTrainedModelResolvesItsNames` fail with the original `NameError`, and restoring it
passes. That is a real defect found by running the thing, and it is exactly what the
"artifacts, not tests" discipline is for.

What remains is REV-130: `outputs/ablations/` is empty. Every test in this spec exercises
config resolution and plotting on fixtures, never a real study.

## Git Changes

`main.py` (+62) and `training/ablation_runner.py` (+8) this round: `source_revision` into
the run manifest, and `artifact_source_revision(ckpt)` into each of the five
checkpoint-consuming stages plus each ablation arm result. `tests/test_cli.py` (+80).

The REV-131 flags and the T1.6 import fix were committed in `daffb41` / `aa04dee`.

**Scope assessment:** proportionate; provenance plumbing only, no change to argument
parsing, dispatch or study resolution — confirmed by diff and by the 38 CLI tests passing
unchanged.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 CLI design | PASS | All 13 stages dispatchable; `--model`, `--moe_variant`, `--num_experts`, `--use_gradnorm`/`--no_gradnorm`, `--study`, `--epochs`, `--batch_size`, plus `--no_train`, `--max_steps`, `--seed` and the three new smoke-gate flags. Every override logged original → new |
| FR-2 `--stage all` | PASS | `_all_pipeline_stages()` returns the eleven stages in the specified order and, measured directly, **does not contain `ablation`** — the property is held by data rather than by a source-text check, which is the better way to hold it |
| FR-3 Startup routine | PASS | All three RNGs seeded, `cudnn.deterministic` set, and the resolved config written to `outputs/logs/run_config.json` — which also carries the runtime block (torch version, device, dtype, requested and resolved attention backend, fallback reason) |
| FR-4 Evaluate skip | PASS | `_load_trained_model` and the evaluate stage log a warning naming the missing path plus the remedy command and return; no exception. `TestMissingCheckpoint` covers it |
| FR-5 Ablation runner | PASS (execution NOT VERIFIABLE) | `STUDY_CONFIGS` is data; `resolve_study_configs` returns fully resolved `Config` instances — measured arm counts A=6, B=2, C=2, D=4, E=3, F=4, G=4, H=9. Each arm records `config_diff` and `resolved_config`. `--study all` writes `summary_{model}.json`. No study has been executed — REV-130 |
| FR-6–FR-13 Ablations A–H | PASS (definitions) | Every study's arm count and shape matches its requirement, including A's six-step progression with the final arm carrying `system_eval="communication"` so it routes into `run_federated`, and H's six sub-studies. H3 follows the owner's REV-116 amendment (cold-start on the held-out split, not a `cold_start_ratio` sweep) |
| FR-14 Core tradeoff plots | PASS (code) | `accuracy_vs_model_size` plots `model_size_bf16_mb`, `accuracy_vs_communication` plots per-round `total_round_kb` — both axes now carry the quantity FR-14 names (REV-101). Not yet rendered, since they need multi-model results |
| FR-15 Multi-seed | PASS (code) | Iterates `config.final_seeds`, runs each in an independent output directory, aggregates mean ± std **across independent runs**, and records the exact seed list. This directly satisfies "don't manufacture std from repeated evaluation of one seed" |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 `--stage ablation --study A` runs without source edits | PASS | Config resolution verified directly for all eight studies; every behaviour that once needed a source edit now has a flag. `tests/test_cli.py` covers parsing and resolution |
| AC-02 `--study all` runs A–H and produces summary JSON | **NOT VERIFIABLE** | The code path exists and is unit-tested on fixtures, but no run has occurred — `outputs/ablations/` is empty. REV-130 |
| AC-03 `--stage all` excludes ablations | PASS | Measured: `"ablation" not in _all_pipeline_stages()`. `TestStageAllExcludesAblations` asserts the same |
| AC-04 missing checkpoint → skip + remedy, no exception | PASS | Verified in code and by `TestMissingCheckpoint`. Note this guard sits **above** the line T1.6 fixed, which is precisely why the suite stayed green over two dead stages |
| AC-05 CLI overrides logged with original vs new | PASS | Measured live: `CLI override: personalization_eval_users 1000 -> 10`, `local_epochs 3 -> 1`, `federated_rounds 20 -> 3`, `clients_per_round 100 -> 10`. `test_smoke_gate_overrides_are_logged` pins the literal format |
| AC-06 resolved config saved with every result | PASS | Each arm record carries `resolved_config` (full `asdict(cfg)`) and `config_diff`; the run-level snapshot is `outputs/logs/run_config.json` |
| AC-07 all 8 studies produce JSON and plots | **NOT VERIFIABLE** | No study JSON and no per-study plot exists. REV-130 |
| AC-08 multi-seed mean ± std across independent seeds | **NOT VERIFIABLE** | Implemented correctly and honestly in code; no multi-seed run has been executed |
| AC-09 core tradeoff plots with correct axes | PARTIAL | The axis defect REV-101 identified is fixed in code and the labels match FR-14, but the four plots are not on disk — they need results from more than one model |
| AC-10 startup seeds everything, saves snapshot | PASS | Verified in code and against the real `run_config.json` |

## Findings

```
REV-131
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-1 | AC: AC-01, AC-05
File: main.py:63-84 (parse_args), :120-140 (apply_cli_overrides)
Resolution verified by running the code, not by reading the note. All three flags exist,
  all default to `None`, and `configs/config.py` is untouched -- a fresh `Config()` still
  reads 1000 / 100 / 3. Each override emits the required `CLI override: {field}
  {original} -> {new}` line. Orthogonality confirmed: `--eval_users 10 --local_epochs 1`
  leaves `clients_per_round` at 100, and `--clients_per_round 10 --epochs 3` leaves
  `personalization_eval_users` at 1000 while mapping `--epochs` to `federated_rounds` for
  the federated stage.
  This closed `personalization-federated` REV-127 as a side effect, which was the point.
Observation, not a requirement: this is the third flag-shaped gap found one finding at a
  time. A cheap prospective guard would be a test asserting that every Config field a
  documented gate reduces has a CLI surface.
```

```
REV-135
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-1 | AC: AC-04
File: main.py:437-441 (_load_trained_model)
This defect appears in no prior review. BUILD found it by RUNNING the stage during
  REV-129's verification, and filed it against T1.6 with that provenance stated plainly.
Problem (as was): `_load_trained_model` called `load_metadata_matrix(processed, config)`
  while importing only `build_model`, `load_checkpoint`, `_build_loss_fns` and
  `get_device`. Both call sites import that name into their own function scope, which does
  not reach the helper, so `--stage gradient_diagnostics` and `--stage routing_diagnostics`
  raised `NameError` before constructing a model -- neither had ever been runnable.
  The reason a green suite tolerated it: `TestMissingCheckpoint` returns at the checkpoint
  guard ABOVE the faulty line, and every other CLI test stops at argument parsing.
  `routing_diagnostics_t_small.json` exists only because `run_pretrain` emits it through
  `_emit_routing_diagnostics`, not through the stage of the same name -- so the artifact's
  presence actively concealed the dead stage.
Resolution: one-line import added; `TestLoadTrainedModelResolvesItsNames` mocks the four
  heavy collaborators and asserts the helper reaches model construction.
Verification (independent): removing the import again makes that test FAIL with the
  original `NameError`, and restoring it passes. `main.py` restored; `git diff` on it is
  empty.
```

```
REV-130
Severity: INFO | Status: OPEN | Requirement: FR-5 | AC: AC-01, AC-02, AC-07, AC-08
File: outputs/ablations/ (still empty)
Problem: unchanged. No ablation study has ever been run -- no `{study}_{model}.json`, no
  `summary_{model}.json`, no per-study plot. Confirmed again this pass by listing the
  directory.
Impact: none actionable today, which is why this stays INFO. Running the studies needs a
  trained checkpoint per distinct training fingerprint across A-H, which is the Phase 14
  full-training job; `--no_train` can evaluate existing checkpoints but only t-small's
  exists. The gap is correctly sequenced, not neglected.
Correction: none now. After Phase 14, run `--stage ablation --study all --model t-base`
  and verify each `{study}_{model}.json` records the configs compared, the config diffs,
  metrics with mean ± std and plot pointers (the Constraints section's per-study
  requirement), and that `summary_{model}.json` is written.
Re-recorded because AC-07 must not be read as satisfied on the strength of a green suite:
  every test here exercises config resolution and plotting on fixtures, not a real study.
  T1.6 is the concrete demonstration that this distinction has teeth -- two stages were
  dead for the entire life of a passing test suite.
```

```
REV-099 / REV-100 / REV-101 / REV-115
Severity: MEDIUM / MEDIUM / LOW / LOW | Status: RESOLVED
Re-confirmed. Study C reports the parameter breakdown FR-8 requires and its comparison
  plot exists in code; Study A's final arm carries `system_eval="communication"` so it
  routes into `run_federated` rather than running single-machine LoRA under a federated
  label, while still sharing arm 5's backbone fingerprint; the two tradeoff plots carry
  the quantities FR-14 names on their axes; and `--no_train` has a CLI surface.
```

```
REV-116
Severity: INFO | Status: RESOLVED | Requirement: FR-13 | AC: AC-07
Re-confirmed against the amended FR-13. H3 is now "cold-start evaluation, metadata ON vs
  OFF, on the held-out cold-item split", which is a training-time configuration, and the
  rationale records why a `cold_start_ratio` sweep is a separate multi-preprocessing-run
  experiment (108 s and 11 GB peak RSS per arm).
```

## Design Compliance

Compliant. The config-driven principle holds without exception: `STUDY_CONFIGS` and
`STUDY_EXTRAS` are data, arms resolve programmatically from `--study`, every arm saves its
`resolved_config` and `config_diff`, and every behaviour that once required a source edit
now has a CLI surface.

`conventions.md` compliance checked specifically:

- **Output Organization** — per-study JSON to `outputs/ablations/`, plots to
  `outputs/plots/`, logs to `outputs/logs/`, checkpoints under `{model_dir_name}/`.
- **Config-Driven Design** — no ablation requires commenting or uncommenting code, and the
  smoke-gate reduction was achieved entirely at the CLI with `configs/config.py` untouched.
  This is the requirement REV-131 existed to protect and it is now genuinely enforceable.
- **Plot Standards** — `dpi=config.plot_dpi`, labelled axes and legends throughout.

One structural point worth naming as a strength: exposing the pipeline stage list as data
(`_all_pipeline_stages()`) rather than asserting over source text means AC-03 is checkable
by calling a function, which is how I verified it.

## Task Completion

| Task | Marked | Verdict |
|---|---|---|
| T1.5 smoke-gate CLI overrides | `[x]` | Complete. Every claim in the note re-measured: the three flags, the untouched defaults, the logged override lines, and orthogonality between the flags |
| T1.6 `_load_trained_model` import | `[x]` | Complete, and its non-vacuity claim reproduced independently. The honesty of this entry is worth noting — it is filed with "**this is not in any review.md**" and records the consequence (two stages never runnable) rather than quietly fixing a one-line bug |
| T1.1–T1.4 | `[x]` | Complete. Verified against the code and by exercising `parse_args` / `apply_cli_overrides` / `_all_pipeline_stages` directly |
| T2.1–T2.10 ablation definitions | `[x]` | Complete **as definitions**, which is what the task text asks. Arm counts verified for all eight studies. Execution is REV-130 |
| T3.1–T3.2 plots and multi-seed | `[x]` | Complete in code; FR-14's axis fix confirmed, and multi-seed std is computed across independent runs |
| T4.1–T4.5 tests | `[x]` | Complete; 38 tests pass |

No false-complete tasks. The T2.x tasks say "implement", and implementation is what is
checked off — AC-07's execution evidence is tracked separately as REV-130, which is the
right separation.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **275 passed**, 0 failed, 55.74 s |
| `uv run python -m pytest tests/test_cli.py -q` | **38 passed**, 1.52 s |
| `uv run python -m pytest tests/test_cli.py::TestLoadTrainedModelResolvesItsNames -q` **with the import removed** | **1 failed** — `NameError: name 'load_metadata_matrix' is not defined`. `main.py` restored, `git diff` empty |
| `Config()` defaults (direct) | `personalization_eval_users=1000`, `clients_per_round=100`, `local_epochs=3` — unchanged |
| `parse_args` + `apply_cli_overrides` with the three flags (direct) | all four `CLI override: … -> …` lines emitted; flags orthogonal; `--epochs` maps to `federated_rounds` for the federated stage |
| `_all_pipeline_stages()` (direct) | eleven stages in FR-2's order; `"ablation"` absent |
| `resolve_study_configs` for A–H (direct) | 6 / 2 / 2 / 4 / 3 / 4 / 4 / 9 arms |
| `ls -la outputs/ablations/` | **empty** — REV-130 |
| Read `outputs/logs/run_config.json` (direct) | full config snapshot plus the runtime block |

## Final Assessment

**Nothing for BUILD today.** This spec is accepted for everything reachable before Phase 14.

The blocking finding is closed and, more importantly, closed in the way that protects the
rule behind it: the gate scale is reachable from the command line, so no one has to edit
`configs/config.py` to run it, and the configured values Phase 14 depends on stay
authoritative.

Two things to carry forward:

1. **REV-130 stays open until Phase 14.** AC-02, AC-07 and AC-08 have no real-run evidence
   and must not be read as satisfied because the suite is green. T1.6 is the proof that
   this distinction is not pedantic — two stages were dead for the entire life of a
   passing 267-test suite, and only running them found it.
2. **The flag-shaped gap has now occurred three times** (`--no_train`, `--max_steps`, and
   this trio). A test asserting that each Config field a documented gate reduces has a CLI
   surface would turn a recurring finding into a build failure. That is a suggestion; the
   spec does not require it and nothing is wrong without it.

**Spec amendment — implemented.** `training/ablation_runner.py:316` now sets
`result["source_revision"] = artifact_source_revision(arm_ckpt)`, so each per-arm result
carries the revision of the checkpoint that arm was evaluated from, alongside
`resolved_config` and `config_diff`. That is the right granularity: an ablation JSON compares
arms, and per-arm stamping is what lets a reader confirm the arms came from the same code.
Unverifiable against a real file today — `outputs/ablations/` is still empty (REV-130) — so
this is confirmed in code only, and the first Phase 14 study run is where it gets tested.

The caveat that was inherited from `training-pipeline` REV-138 is now closed: the coverage rule
was widened to an exclusion rule and `evaluation/metrics.py` — through which ablation arms are
scored — is among the 27 covered files. The per-arm stamping and the identifier it stamps are
both sound now; what remains untested is the artifact itself, since `outputs/ablations/` is
still empty (REV-130).
