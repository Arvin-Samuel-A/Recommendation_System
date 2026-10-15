# Ablations & CLI — Requirements

## Overview

Implement the main.py CLI entrypoint dispatching all pipeline stages, the ablation runner executing studies A–H without source-code edits, and all final tradeoff/comparison plots.

## Problem Statement

The research project requires 8 ablation studies (A–H), each comparing multiple configurations. Every ablation must be config-driven and executable via CLI without editing Python source. The main entrypoint must support independent stage execution, CLI overrides, and a full pipeline mode.

## Goals

- G1: `main.py` CLI dispatching all stages with model/config overrides
- G2: Config-driven ablation runner for studies A–H (no source edits)
- G3: `--stage all` for full pipeline; `--stage ablation --study all` for all ablations
- G4: Multi-seed final runs with proper aggregation
- G5: All tradeoff and comparison plots generated

## Non-Goals

- GUI or web interface
- Experiment tracking platform integration (MLflow, W&B)
- Parallel ablation execution

## Functional Requirements

### FR-1: CLI Design (Step 22.1)
- Stages: preprocess, pretrain, personalize, federated, evaluate, cold_start, longtail, latency, routing_diagnostics, gradient_diagnostics, ablation, plots, all
- Model selection: `--model {t-small|t-base|t-large}`
- MoE overrides: `--moe_variant {none|mmoe|arv_mmoe}`, `--num_experts K`
- GradNorm: `--use_gradnorm` / `--no_gradnorm`
- Ablation: `--study {A|B|C|D|E|F|G|H|all}`
- General: `--epochs N`, `--batch_size B`
- Log which config values were overridden

### FR-2: `--stage all` (Step 22.1)
- Runs: preprocess → pretrain → personalize → federated → evaluate → cold_start → longtail → latency → routing_diagnostics → gradient_diagnostics → plots
- Does NOT run ablation suite (always explicit)
- For final paper: execute independently for every seed in `final_seeds`; don't overwrite one seed's outputs

### FR-3: Startup Routine (Step 22.5)
- Seed everything: `torch.manual_seed`, `np.random.seed`, `random.seed`
- `torch.backends.cudnn.deterministic = True`
- Worker seed function for DataLoader
- Save resolved config to `outputs/logs/run_config.json`

### FR-4: Evaluate Skip (Step 22.4)
- Missing checkpoint → print skip message + remedy command → return
- Never raise exception on missing checkpoint

### FR-5: Ablation Runner (Step 21.7)
- `python main.py --stage ablation --study {A..H|all} --model {model}`
- Runner resolves study-specific sub-configurations internally
- Every resolved configuration saved with result
- `--study all` runs A–H sequentially; writes `outputs/ablations/summary_{model}.json`
- Each study writes `outputs/ablations/{study}_{model}.json` *(superseded by `ablation-run` FR-11: `outputs/ablations/seed_{seed}/{study}_{model}.json`, and the summary likewise)*

### FR-6: Ablation A — Architecture Progression
- 6 configurations: single-objective baseline → +all objectives → +MMoE → +ARV-MMoE → +GradNorm → +federated
- Plot: `architecture_progression.png`

### FR-7: Ablation B — Routing Formulation
- `objective_conditioned` vs `input_only` router conditioning
- Both with `moe_variant="arv_mmoe"`, identical K and seed
- Report metrics delta and routing diagnostics for both

### FR-8: Ablation C — MMoE vs ARV-MMoE
- `moe_variant="mmoe"` vs `"arv_mmoe"` with `match_mmoe_expert_count=True`
- Report: total params, expert params, router/gate params, latency, metrics
- Plot: `mmoe_vs_arvmmoe_comparison.png`

### FR-9: Ablation D — ARV vs GradNorm
- 4 configs: neither, ARV-MMoE only, GradNorm only, ARV-MMoE + GradNorm
- Report metrics + full gradient-interference diagnostics for all four

### FR-10: Ablation E — Personalization
- 3 configs: global only, global+LoRA, global+LoRA+ARV
- Extend personalization scatter plot with third series

### FR-11: Ablation F — Objective Ablation
- Full 4-objective reference + 3 single-auxiliary-removed runs
- Next-item always enabled (serving embedding must be trained)
- Disabled objective excluded from GradNorm; active lambdas renormalized

### FR-12: Ablation G — Expert Count
- Sweep `K ∈ {2, 4, 6, 8}` with `moe_variant="arv_mmoe"`
- Report metrics, latency, routing diagnostics per K
- Plot: `expert_count_ablation.png` (NDCG left axis, latency right axis)

### FR-13: Ablation H — System-Level
- H1: Metadata ON/OFF
- H2: Sequence length sweep over `sequence_length_grid`
- H3: Cold-start evaluation, metadata ON vs OFF, on the held-out cold-item split
  - *(REV-116, owner 2026-09-17: this previously read "Cold-start ratio / metadata ON vs OFF". `cold_start_ratio` is a preprocessing-time parameter baked into `data/processed/`, so sweeping it means re-running `--stage preprocess` per arm — measured at 108 s and 11 GB peak RSS per run — not a config change at training time. A ratio sweep is a separate, multi-preprocessing-run experiment and is out of scope for H3.)*
- H4: Long-tail with training-derived popularity buckets
- H5: Latency with Step 21 benchmark
- H6: Communication cost with Step 17 config
- Each selectable without source edits

### FR-14: Core Tradeoff Plots (Step 23.1)
- `accuracy_vs_model_size.png`: Size MB vs NDCG@10
- `accuracy_vs_communication.png`: Comm cost/round vs NDCG@10
- `accuracy_vs_flops.png`: FLOPs vs NDCG@10
- `personalization_gain_vs_lora_rank.png`: LoRA rank [2,4,8,16] vs mean NDCG gain

### FR-15: Multi-Seed Final Runs
- Independent seeds from `config.final_seeds = [42, 123, 456]`
- Report mean ± std across seeds
- Don't manufacture std from repeated evaluation of one seed
- Include exact seed list in final results

## Edge Cases

- Model checkpoint missing for an ablation config: skip with message
- Ablation study requires model not yet pretrained: fail with clear remedy
- `--study all` with partial prior results: re-run all (no incremental resume) *(superseded by `ablation-run` FR-11: completed arms are persisted and reused unless `--no_resume`)*

## Constraints

- No ablation requires editing Python source
- All choices expressible as config flags
- Each ablation study JSON includes: configs compared, config diffs, metrics with mean ± std, plot pointers
- All plots: PNG, 150 DPI, labeled axes, legend, consistent palette

## Dependencies

- All other specs (data-pipeline, transformer-backbone, moe-routing, loss-functions, training-pipeline, personalization-federated, evaluation-diagnostics)
- `configs/config.py` for all parameters and CLI override logic

## Acceptance Criteria

- **AC-01:** `python main.py --stage ablation --study A --model t-base` executes without source edits
- **AC-02:** `--study all` runs A–H sequentially and produces summary JSON
- **AC-03:** `--stage all` runs full pipeline but NOT ablations
- **AC-04:** Missing checkpoint → skip message + remedy command (no exception)
- **AC-05:** CLI overrides are logged with original vs new values
- **AC-06:** Resolved config saved with every experiment result
- **AC-07:** All 8 ablation studies produce their respective JSON and plot files
- **AC-08:** Multi-seed results report mean ± std across independent seeds
- **AC-09:** All core tradeoff plots generated with correct axes and data points
- **AC-10:** Startup routine seeds all random sources and saves config snapshot
