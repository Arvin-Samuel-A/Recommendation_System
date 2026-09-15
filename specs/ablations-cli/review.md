# Review — Ablations & CLI

## Review Status

**CHANGES_REQUIRED**

## Feature

`specs/ablations-cli` — `main.py` CLI entrypoint and stage dispatch, `training/ablation_runner.py` studies A–H, core tradeoff plots, LoRA-rank sweep, multi-seed aggregation.

## Review Scope

- Spec directory: `specs/ablations-cli/` (requirements.md, design.md, tasks.md)
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`, `specs/BUILD_ORDER.md`
- Baseline: working tree vs `HEAD` = `32b0904` on `master`. All feature work is uncommitted.
- Files inspected: `main.py` (856 lines, untracked), `training/ablation_runner.py` (786 lines, untracked), `tests/test_cli.py` (547 lines, untracked), `configs/config.py`, `evaluation/flops.py`, `training/local_train.py`
- Re-review of the prior `review.md`. `ablation_runner.py` (21:46), `main.py` (21:38) and `tests/test_cli.py` (21:35) were all modified after it.
- Date: 2026-09-15

## Summary

Four of the five prior findings are fixed and verify. REV-065 is closed — all four specifically named figures now exist as dispatched functions (`architecture_progression.png`, `mmoe_vs_arvmmoe_comparison.png`, `expert_count_ablation.png`, `global_vs_personalized_ablation_E.png`), and `_plot_expert_count` puts NDCG on the left axis and latency on the right as FR-12 states. REV-067 is closed — `expert_count_grid` and `sequence_length_grid` are now read through `_GRID_SPECS` at resolution time, so a CLI or config override changes the arms without a source edit. REV-083 is closed — `main.setup_seed` delegates to `utils.set_seed`, with three tests covering the delegation, the cuDNN flag and cross-generator reproducibility. REV-084 is closed — `_training_fingerprint` hashes everything except the three verified post-training-only fields, so Ablation A's arms 5/6 and E's `global_only`/`global_lora` share one backbone instead of training two bit-identical models. REV-070 is closed: `use_lora` and `use_federated` are gone from `configs/config.py`.

Two new defects. The parameter bars in `mmoe_vs_arvmmoe_comparison.png` are always zero because the plot reads a dictionary key that `count_parameters` never produces — and the test passes because it feeds that fabricated key in by hand. And Ablation A's sixth arm is specified as "+federated" but runs LoRA personalization instead, so the architecture-progression figure's final point does not show the federated system.

## Git Changes

| File | State |
|---|---|
| `main.py` | untracked (new, 856 lines) |
| `training/ablation_runner.py` | untracked (new, 786 lines) |
| `tests/test_cli.py` | untracked (new, 547 lines) |
| `configs/config.py` | modified — `system_eval`, `personalization_mode`, `ablation_eval_examples`, `max_steps_per_epoch` added; `use_lora`/`use_federated` removed |
| `specs/ablations-cli/tasks.md` | modified |
| `specs/ablations-cli/review.md` | untracked (this file) |

**Scope assessment:** in scope. No new dependencies — `matplotlib` was already declared. `outputs/logs/` holds two stray `moe_parameter_report_t_base_A_*.json` files from an aborted ablation attempt, with no matching `outputs/ablations/*.json`; harmless residue, not committed.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 CLI design | PASS | `parse_args` ([main.py:32](main.py:32)) accepts every stage in FR-1's list plus `lora_rank_sweep` and `multi_seed`, all three models, and every named override. `apply_cli_overrides` logs `name old -> new` for each; `TestCLIParsing` and `TestOverrideLogging` cover both. |
| FR-2 `--stage all` | PASS | `_all_pipeline_stages()` ([main.py:768](main.py:768)) returns exactly the eleven stages FR-2 lists in order, and `ablation` is not among them. `TestStageAllExcludesAblations` checks the resolved stage list rather than grepping source (REV-033's fix holds). Per-seed isolation for final runs is `run_multi_seed_stage`, which gives each seed its own `output_dir`/`checkpoint_dir`/`log_dir`/`plot_dir`. |
| FR-3 Startup routine | PASS | `setup_seed` delegates to `utils.set_seed`, which seeds torch, numpy, random, `cuda.manual_seed_all`, and sets `cudnn.deterministic = True` / `benchmark = False`. `save_config_snapshot` writes the resolved config plus a `runtime` block recording torch version, device, dtype, the requested backend and the backend that actually ran. |
| FR-4 Evaluate skip | PASS | All five checkpoint-consuming stages (`evaluate`, `cold_start`, `longtail`, `latency`, `_load_trained_model`) log a warning naming the remedy command and `return` — none raise. `TestMissingCheckpoint::test_evaluate_skips_missing_checkpoint`. |
| FR-5 Ablation runner | PASS | `run_ablation(config, model, study)` with `study="all"` iterating A–H; each study writes `outputs/ablations/{study}_{model}.json`, and `--study all` additionally writes `summary_{model}.json`. Every result carries `resolved_config` (full `asdict`), `config_diff` computed per-config against the base (REV-019's fix), and `study_overrides`. A failing arm is caught and recorded rather than losing the rest of the study. |
| FR-6 Ablation A | PARTIAL | Six arms, correctly ordered, with `architecture_progression.png` drawn as connected series over the ordered arms with a baseline reference line. Arm 6 is LoRA personalization, not federated — REV-100. |
| FR-7 Ablation B | PASS | Two arms differing only in `router_conditioning`, both `arv_mmoe`, same K and seed by construction (`copy.deepcopy(base_config)`). `STUDY_EXTRAS["B"] = ("routing",)` gathers the routing diagnostics FR-7 requires for both. |
| FR-8 Ablation C | PARTIAL | Two arms with `match_mmoe_expert_count=True`; `count_router_parameters` is called per arm with a distinct `run_tag`; latency and metrics gathered. The figure's parameter axis is broken — REV-099. |
| FR-9 Ablation D | PASS | The full 2×2 grid of `{none, arv_mmoe} × {gradnorm off, on}`; `STUDY_EXTRAS["D"] = ("gradient",)` runs `run_gradient_diagnostics` for all four arms and stores the report inline. |
| FR-10 Ablation E | PASS | Three arms — `global_only`, `global_lora`, `global_lora_arv` — genuinely distinguished by `personalization_mode` and `moe_variant`. `_plot_personalization_study` overlays every arm that reported `per_user_scores` as its own scatter series. |
| FR-11 Ablation F | PASS | Reference plus three single-auxiliary-removed arms; `next_item` is never disabled (no arm touches it). Disabled objectives are excluded from GradNorm through `has_valid_signal` and the active lambdas renormalized in `GradNormController.update`. |
| FR-12 Ablation G | PASS | `{"_grid": "expert_count"}` expands against `config.expert_count_grid` at resolution time; `STUDY_EXTRAS["G"] = ("routing", "efficiency")` so latency is gathered; `expert_count_ablation.png` has NDCG left and latency right, verified by a test that inspects the axis labels. |
| FR-13 Ablation H | PASS | H1 via `use_metadata`, H2 via the `sequence_length` grid (four arms, not one), H3–H6 via `system_eval`, which `_run_single_config` dispatches to cold-start / long-tail / latency / federated. Every arm is a config flag; no source edit selects a sub-study. |
| FR-14 Core tradeoff plots | PARTIAL | All four figures are implemented with graceful skip-and-warn when their inputs are absent. Two have axes that do not match the spec's wording — REV-101. |
| FR-15 Multi-seed runs | PASS | `run_multi_seed_stage` iterates `config.final_seeds`, gives each seed an isolated output tree, runs pretrain + evaluate per seed, and aggregates with `std(ddof=1)` **across** seeds, recording `n_seeds`, the exact seed list and a note stating the std is across independent runs. This is precisely what FR-15's "don't manufacture std from repeated evaluation of one seed" asks for. |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 `--stage ablation --study A` runs without source edits | PASS | `TestAblationConfigResolution::test_study_A_produces_6_configs`, `test_study_A_correct_fields`, `test_grid_studies_track_config_not_source`. |
| AC-02 `--study all` runs A–H and produces the summary | PASS | `run_ablation` iterates `STUDY_CONFIGS.keys()` and calls `_save_summary`; `test_all_studies_produce_expected_count` covers the arm counts. Never executed end-to-end — no checkpoints exist. |
| AC-03 `--stage all` excludes ablations | PASS | `TestStageAllExcludesAblations::test_all_does_not_dispatch_ablation` and `test_all_stages_are_real_registered_stages`. |
| AC-04 missing checkpoint → skip, no exception | PASS | `TestMissingCheckpoint`. |
| AC-05 CLI overrides logged with old and new | PASS | `TestOverrideLogging::test_override_logged`. |
| AC-06 resolved config saved with every result | PASS | `result["resolved_config"] = asdict(cfg)` plus `config_diff` and `study_overrides` on every arm, including failed ones. |
| AC-07 all 8 studies produce JSON and plots | PARTIAL | Every study writes its JSON and generic bar chart, and the four named figures dispatch correctly. Ablation C's figure renders with a zeroed parameter axis (REV-099), and Ablation A's final arm plots the wrong experiment (REV-100). Not demonstrated end-to-end — no run has produced `outputs/ablations/*.json`. |
| AC-08 multi-seed mean ± std across independent seeds | PASS | `run_multi_seed_stage`'s aggregation uses `ddof=1` across the per-seed test metrics and records `per_seed` values so the claim is auditable. |
| AC-09 core tradeoff plots with correct axes and data | PARTIAL | All four are produced; two axes deviate from FR-14's wording (REV-101). |
| AC-10 startup seeds everything and saves the snapshot | PASS | `TestSeeding` (3 tests), including `test_setup_seed_is_reproducible_across_all_generators`. |

## Findings

```
REV-099
Severity: MEDIUM | Status: OPEN | Requirement: FR-8 | AC: AC-07, AC-09
File: training/ablation_runner.py:641 (`_plot_mmoe_comparison`);
       tests/test_cli.py:270 (the test that hides it)
Problem: The cost panel reads
  `r.get("parameters", {}).get("total", 0.0) / 1e6`.
  `evaluation/flops.py::count_parameters` never returns a `"total"` key. Its keys are
  exactly: expert_parameters, lora_parameters, model_size_bf16_mb, model_size_fp32_mb,
  model_size_mb, router_parameters, total_parameters, trainable_parameters -- confirmed by
  calling it on a real t-small build, where `parameters.get("total", 0.0)` returns 0.0.
  `_run_single_config:348` stores the dict verbatim as `result["parameters"]`, so every
  Ablation C arm plots a zero-height "Parameters (M)" bar.
  The test does not catch this because `test_every_named_figure_is_written` hand-feeds
  `"parameters": {"total": 4.2e6 + i * 1e6}` -- a key the production path never writes --
  and then asserts only that the PNG file exists and is larger than 1000 bytes.
Impact: FR-8 requires the Ablation C figure to report "total params, expert params,
  router/gate params, latency, metrics", and FR-6 of moe-routing closes with "Never claim
  'parameter-matched' without the report backing it". The figure that is supposed to
  visualise the parameter match shows both arms at zero, so it silently supports any claim
  put next to it.
Correction: Read `total_parameters`. While there, the panel currently shows only the
  total -- FR-8 also names expert params and router/gate params, both of which are already
  in the same dict. Fix the test too: drive it from a real `count_parameters(...)` result
  (or at minimum assert the plotted values are non-zero) rather than a fabricated key, so
  the next rename cannot pass unnoticed.
```

```
REV-100
Severity: MEDIUM | Status: OPEN | Requirement: FR-6 | AC: AC-07
File: training/ablation_runner.py:28 (study A, arm 6);
       specs/ablations-cli/tasks.md:22 (T2.2, marked [x])
Problem: FR-6 defines Ablation A as "single-objective baseline -> +all objectives -> +MMoE
  -> +ARV-MMoE -> +GradNorm -> +federated", and T2.2 spells arm 6 out as "(6) federated
  ARV-MMoE+GradNorm+LoRA". The implemented arm is
    {"label": "arv_mmoe_gradnorm_lora", ..., "personalization_mode": "lora"}
  and `STUDY_EXTRAS["A"] = ("personalization",)`, so `_run_single_config` calls
  `run_local_personalization`. Nothing in study A ever calls `run_federated`; the arm's
  `system_eval` stays "none".
Impact: `architecture_progression.png` is the figure for the paper's central "each
  component earns its place" argument, and its final point is labelled as the federated
  system while measuring single-machine LoRA personalization. The two are different
  experiments -- federated adds FedAvg aggregation across 100 clients per round and the
  communication cost that RQ4 is about. T2.2 is marked [x] on a claim the code does not
  meet.
Correction: The mechanism already exists and needs no new code: add
  `"system_eval": "communication"` to arm 6, which routes `_run_single_config` into
  `run_federated` and records rounds, final val NDCG and cumulative KB. Decide with the
  spec owner whether arm 6 is federated-only or federated-plus-LoRA (T2.2's wording
  suggests both, which the two flags together express). Note the cost interaction: study A
  arm 6 would then inherit personalization-federated REV-112's full-split per-round
  evaluation, so fix that first. Until the arm runs what it is named for, T2.2 should be
  unchecked.
```

```
REV-101
Severity: LOW | Status: OPEN | Requirement: FR-14 | AC: AC-09
File: main.py:536-548 (`accuracy_vs_model_size`), main.py:554-581
       (`accuracy_vs_communication`)
Problem: Two of FR-14's four figures plot a different quantity from the one specified.
  (a) FR-14: "`accuracy_vs_model_size.png`: Size MB vs NDCG@10". The implementation plots
      `results[m].get("total_parameters", 0)` with the axis labelled "Total Parameters".
      `model_size_bf16_mb` and `model_size_fp32_mb` are both already in the same
      efficiency report.
  (b) FR-14: "`accuracy_vs_communication.png`: Comm cost/round vs NDCG@10". The
      implementation plots `sum(r["total_round_kb"] for r in rounds)` -- the CUMULATIVE
      cost across all rounds -- labelled "Cumulative communication (KB)".
      `total_round_kb` is already the per-round figure.
Impact: Minor but it is a paper figure: a reader taking the x-axis at face value against
  the requirement's wording would read a per-round cost off a cumulative axis, and the
  ratio between models changes if they ran different numbers of rounds. The FLOPs and
  LoRA-rank figures are correct.
Correction: (a) plot `model_size_bf16_mb` (or plot both sizes as two series) and relabel.
  (b) plot `rounds[-1]["total_round_kb"]`, or the mean across rounds, and relabel -- or
  take "cumulative" to the spec owner as a deliberate improvement and have FR-14 reworded.
  Either way the axis label and the requirement should agree.
```

```
REV-115
Severity: LOW | Status: OPEN | Requirement: FR-5 | AC: AC-02, AC-07
File: training/ablation_runner.py:470-516 (`run_ablation`), :283-293 (`_run_single_config`)
Problem: `run_ablation` defaults to `train=True`, so `--stage ablation --study all` on a
  fresh checkout attempts a full `run_pretrain` per distinct training fingerprint -- 34 arms across A-H,
  deduplicated by REV-084's fingerprint to 28 distinct training runs (measured by calling
  `_training_fingerprint` over every resolved arm), at `pretrain_epochs=20` over 7.8M
  examples. There is no
  `--no_train` flag and no resume: FR-5's Edge Cases say "`--study all` with partial prior
  results: re-run all (no incremental resume)", which is honoured, but combined with
  unconditional training it makes the study suite effectively un-runnable in one
  invocation and un-resumable after any interruption.
Impact: Operational, not a correctness defect. The per-arm try/except means one arm's
  failure does not lose the others, and `train=False` with a missing checkpoint records
  `status: "skipped_no_checkpoint"` cleanly -- but `train=False` is not reachable from the
  CLI, so an operator cannot evaluate previously trained arms without editing source,
  which brushes against the "No ablation requires editing Python source" constraint.
Correction: Expose `train` as a CLI flag (`--no_train` / `--train/--eval-only`) plumbed
  through `run_ablation_stage`. The plumbing already exists in the function signature; only
  `parse_args` and the one call site need changing.
```

```
REV-116
Severity: INFO | Status: OPEN | Requirement: FR-13 | AC: AC-07
File: training/ablation_runner.py:71 (H3 arm); specs/ablations-cli/requirements.md:93
Problem: FR-13 defines H3 as "Cold-start ratio / metadata ON vs OFF". The implemented arm
  sets `system_eval: "cold_start"`, which runs the metadata ON/OFF comparison in
  `evaluation/cold_start.py`. The cold-start *ratio* half is not swept --
  `cold_start_ratio` is fixed at 0.10 and, because it is a preprocessing-time parameter,
  sweeping it would require re-running `--stage preprocess` per arm, not just a config
  change at training time.
Impact: None on what is produced; the metadata ON/OFF comparison is the substantive half
  and it runs. Recorded because FR-13's wording promises more than the arm delivers, and
  because the reason (the ratio is baked into `data/processed/`) is a real structural
  constraint the spec does not acknowledge.
Correction: None for BUILD. Raise with the spec owner: either reword H3 to "metadata ON vs
  OFF on the cold-item split", or accept that a ratio sweep is a multi-preprocessing-run
  experiment and scope it separately.
```

## Design Compliance

Matches `design.md` and the steering constraints.

- **`conventions.md` config-driven design** — "Every experiment variation must be expressible as a config flag" now holds for every study. `system_eval` and `personalization_mode` were added precisely to make Ablation E's and H's arms distinguishable (REV-019), the two grids are read from config rather than frozen at import (REV-067), and `resolve_study_configs` sets values programmatically from `--study`. `_config_diff` plus `resolved_config` means every result is reproducible from its own JSON.
- **`conventions.md` naming** — `{study}_{model}.json` with the underscore spelling; plots at `config.plot_dpi`; the four spec-named figures written without a model suffix, exactly as the requirements spell them.
- **`code-quality.md` error handling** — every checkpoint-consuming stage skips with a remedy rather than raising, as FR-4 and the steering doc both require.
- **`code-quality.md` manifest** — `save_config_snapshot` records torch version, device, dtype and the *resolved* attention backend including any fallback reason, which is the steering requirement rather than just the requested value.
- **REV-020's mutation bug stays fixed** — `overrides = dict(entry)` copies before popping, so `STUDY_CONFIGS` is never mutated in place.

Two additions beyond `design.md`, both justified: the `lora_rank_sweep` and `multi_seed` stages (FR-14's fourth figure and FR-15 had no entry point at all), and `POST_TRAINING_ONLY_FIELDS` with its conservative default — anything not explicitly listed is treated as training-relevant, so a missing entry costs a redundant run rather than silently sharing a backbone between genuinely different arms. `test_only_verified_post_training_fields_are_excluded` guards that list.

## Task Completion

All 21 tasks are `[x]`. Twenty verify; one does not.

| Task | Verdict |
|---|---|
| T1.1–T1.4 CLI | Complete. Stage dispatch, overrides, startup routine and missing-checkpoint handling all tested. |
| T2.1 study config resolution | Complete, with grid expansion at resolve time. |
| **T2.2 Ablation A** | **False-complete.** Arm 6 is specified as federated and runs LoRA personalization (REV-100). The other five arms and the named figure are correct. |
| T2.3 Ablation B | Complete. |
| T2.4 Ablation C | Complete as an experiment; the figure's parameter axis is broken (REV-099). |
| T2.5 Ablation D | Complete — gradient diagnostics gathered for all four arms. |
| T2.6 Ablation E | Complete, with the third series genuinely overlaid. |
| T2.7 Ablation F | Complete. |
| T2.8 Ablation G | Complete, including the twin-axis figure. |
| T2.9 Ablation H | Complete; H3's ratio-sweep wording is REV-116 (INFO). |
| T2.10 `--study all` aggregation | Complete. |
| T3.1 `--stage plots` | Complete; two axis deviations (REV-101). |
| T3.2 multi-seed aggregation | Complete, and the std is genuinely across independent runs. |
| T4.1–T4.5 tests | Complete. `TestSharedBackboneReuse` (3 tests) and `TestNamedAblationFigures` (5 tests) are real regression coverage for REV-084 and REV-065 — with the one gap noted in REV-099, where the fixture supplies a key production never writes. |

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` (full suite) | 231 passed, 2 failed, 135 s. Neither failure is in this spec — both are `transformer-backbone` REV-062's armed MPS backend gate. |
| `uv run python -m pytest tests/test_cli.py -q` | All pass (included in the run above). |
| Scratch: call `evaluation.flops.count_parameters(model, moe)` on a real t-small build and apply `_plot_mmoe_comparison`'s accessor | Keys returned: `expert_parameters, lora_parameters, model_size_bf16_mb, model_size_fp32_mb, model_size_mb, router_parameters, total_parameters, trainable_parameters`. `parameters.get("total", 0.0)` → **0.0** — REV-099 confirmed. |
| Source read: `STUDY_CONFIGS["A"]` arm 6 and `STUDY_EXTRAS["A"]` | `personalization_mode: "lora"`, `system_eval` unset; `run_federated` is unreachable from study A — REV-100 confirmed. |
| Source read: `main.py:536-548`, `:554-581` | x-axes are `total_parameters` and cumulative KB — REV-101 confirmed. |
| `grep -rn "use_lora\|use_federated" configs/config.py` | No matches — REV-070 confirmed resolved. |
| `ls outputs/ablations outputs/logs` | `ablations/` empty; `logs/` holds `run_config.json` and two `moe_parameter_report_t_base_A_*.json` from an aborted attempt. No study has completed, so AC-02 and AC-07 are code-verified only. |

## Final Assessment

The CLI and ablation runner are substantially complete and the prior findings are properly closed — the grid plumbing, the named figures, the backbone-reuse fingerprinting and the multi-seed aggregation are all real implementations rather than placeholders.

BUILD should, in order:

1. **REV-099** — read `total_parameters` in `_plot_mmoe_comparison`, add the expert and router/gate bars FR-8 also names, and rewrite the fixture in `test_every_named_figure_is_written` to use a real `count_parameters` result so the key can never drift unnoticed again.
2. **REV-100** — add `"system_eval": "communication"` to Ablation A arm 6 (or settle the arm's definition with the spec owner first), and leave T2.2 unchecked until it runs what it is named for. Sequence this after personalization-federated REV-112, since the arm would otherwise inherit a full-split per-round evaluation.
3. **REV-101** and **REV-115** — small, independent; fix the two plot axes and expose `train` as a CLI flag.
4. **REV-116** goes to the spec owner, not to BUILD.

AC-07 and AC-09 stay PARTIAL until REV-099/REV-100/REV-101 land. AC-02 and AC-07 additionally need one completed `--study all` run for evidence, which is blocked behind `specs/transformer-backbone` REV-062 and tracked in `specs/training-pipeline/review.md` REV-106.
