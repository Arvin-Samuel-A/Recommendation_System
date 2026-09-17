# Ablations & CLI — Tasks

Ordered tasks. Depends on all other specs. This is the final integration layer.

## Phase 1: CLI Entrypoint

- [x] **T1.1: Implement argument parsing in `main.py`** — All stages: preprocess, pretrain, personalize, federated, evaluate, cold_start, longtail, latency, routing_diagnostics, gradient_diagnostics, ablation, plots, all. Model: `--model {t-small|t-base|t-large}`. Overrides: `--epochs`, `--batch_size`, `--moe_variant`, `--num_experts`, `--use_gradnorm`/`--no_gradnorm`, `--study`, `--no_train` (ablations: evaluate existing per-arm checkpoints instead of training each one — REV-115). (FR-1)

- [x] **T1.2: Implement CLI override logic** — Apply `--epochs`, `--batch_size`, `--moe_variant`, `--num_experts`, `--use_gradnorm` to Config. Log original vs new values for each override. Map `--epochs` to `pretrain_epochs` for pretrain stage and `federated_rounds` for federated stage. (FR-1, AC-05)

- [x] **T1.3: Implement startup routine** — Seed all RNGs (`torch.manual_seed`, `np.random.seed`, `random.seed`). Set `torch.backends.cudnn.deterministic = True`. Save resolved config to `outputs/logs/run_config.json`. (FR-3, AC-10)

- [x] **T1.4: Implement stage dispatch** — Map stage name to function. Handle `--stage all` (full pipeline minus ablations). Handle missing checkpoint gracefully (print skip + remedy, no exception). (FR-2, FR-4, AC-03, AC-04)

## Phase 2: Ablation Runner

- [x] **T2.1: Implement study config resolution in `training/ablation_runner.py`** — Define `STUDY_CONFIGS` dict mapping study letter to list of config override dicts. Each study's configs are complete and explicit. `resolve_study_configs(study, base_config)` returns list of fully resolved Config instances. (FR-5)

- [x] **T2.2: Implement Ablation A — Architecture Progression** — 6 configs: (1) single-objective baseline, (2) 4-objective no routing, (3) conventional MMoE, (4) ARV-MMoE, (5) ARV-MMoE+GradNorm, (6) federated ARV-MMoE+GradNorm+LoRA. Train and evaluate each. Save `outputs/ablations/A_{model}.json`. Generate `outputs/plots/architecture_progression.png`. (FR-6, AC-07) **REV-100 fixed:** arm 6 carries `system_eval="communication"`, so it routes into `run_federated` rather than running single-machine LoRA under a federated label; verified by `tests/test_cli.py::TestAblationStudyConfigs::test_study_A_final_arm_is_federated` and by the fingerprint test that keeps it sharing arm 5's backbone. Like every other T2.x, the end-to-end study run is still pending behind training-pipeline REV-106.

- [x] **T2.3: Implement Ablation B — Routing Formulation** — 2 configs: `objective_conditioned` vs `input_only`. Both `arv_mmoe`, same K and seed. Save JSON + routing diagnostics for both. (FR-7, AC-07)

- [x] **T2.4: Implement Ablation C — MMoE vs ARV-MMoE** — 2 configs: `mmoe` vs `arv_mmoe`, `match_mmoe_expert_count=True`. Report params + metrics + latency. Generate `outputs/plots/mmoe_vs_arvmmoe_comparison.png`. (FR-8, AC-07)

- [x] **T2.5: Implement Ablation D — ARV vs GradNorm** — 4 configs: 2×2 grid of {none, arv_mmoe} × {gradnorm off, on}. Report metrics + full gradient diagnostics for all four. (FR-9, AC-07)

- [x] **T2.6: Implement Ablation E — Personalization** — 3 configs: global-only, global+LoRA, global+LoRA+ARV. Extend personalization scatter plot with third series. (FR-10, AC-07)

- [x] **T2.7: Implement Ablation F — Objective Ablation** — 4 configs: full reference + 3 single-auxiliary-removed. Next-item always on. Disabled objective excluded from GradNorm. Active lambdas renormalized. (FR-11, AC-07)

- [x] **T2.8: Implement Ablation G — Expert Count** — 4 configs: K ∈ {2, 4, 6, 8}. Report metrics + latency + routing diagnostics. Generate `outputs/plots/expert_count_ablation.png`. (FR-12, AC-07)

- [x] **T2.9: Implement Ablation H — System-Level** — 6 sub-studies: H1 metadata, H2 seq length, H3 cold-start, H4 long-tail, H5 latency, H6 communication. Each resolved from config flags without source edits. (FR-13, AC-07)

- [x] **T2.10: Implement `--study all` aggregation** — Run A–H sequentially. Write `outputs/ablations/summary_{model}.json` aggregating headline numbers from all studies. (FR-5, AC-02)

## Phase 3: Plot Generation

- [x] **T3.1: Implement `--stage plots` in `main.py`** — Collect results from completed evaluations. Generate all core tradeoff plots: accuracy_vs_model_size, accuracy_vs_communication, accuracy_vs_flops, personalization_gain_vs_lora_rank. (FR-14, AC-09) **REV-101 fixed:** `accuracy_vs_model_size` plots `model_size_bf16_mb` against "Model size (MB, BF16)" and `accuracy_vs_communication` plots the per-round `total_round_kb` against "Communication per round (KB)", so both axes now carry the quantity FR-14 names.

- [x] **T3.2: Implement multi-seed aggregation** — For final paper runs: iterate `config.final_seeds`, execute pipeline per seed in independent output directory, aggregate results (mean ± std across seeds). Include exact seed list in final results. (FR-15, AC-08)

## Phase 4: Testing

- [x] **T4.1: Test CLI argument parsing** — Verify all `--stage` values accepted. Verify `--model` choices. Verify overrides applied. (AC-01)

- [x] **T4.2: Test ablation config resolution** — Verify Ablation A produces 6 configs with correct fields. Verify all studies produce expected number of configs. (AC-01, AC-07)

- [x] **T4.3: Test missing checkpoint handling** — Mock missing `best.pt` → verify skip message + remedy printed, no exception. (AC-04)

- [x] **T4.4: Test override logging** — Apply `--epochs 5 --batch_size 32` → verify log shows original and new values. (AC-05)

- [x] **T4.5: Test `--stage all` exclusion of ablations** — Verify `run_all_pipeline` does NOT call `run_ablation`. (AC-03)
