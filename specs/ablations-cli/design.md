# Ablations & CLI — Design

## Relevant Existing Architecture

Greenfield. Design follows Prompt_v6.2 Steps 21.7, 22–23.

## Files Created / Modified

| File | Action | Role |
|---|---|---|
| `main.py` | Create | CLI entrypoint, argument parsing, stage dispatch, startup routine |
| `training/ablation_runner.py` | Create | Study-specific config resolution, execution orchestration, result aggregation |

## Proposed Architecture

### main.py Structure

```python
def main():
    args = parse_args()
    config = Config()
    apply_cli_overrides(config, args)  # log overrides
    setup_seed(config)
    save_config_snapshot(config)

    stage_map = {
        'preprocess': run_preprocess,
        'pretrain': run_pretrain,
        'personalize': run_local_personalization,
        'federated': run_federated,
        'evaluate': run_evaluate,
        'cold_start': run_cold_start,
        'longtail': run_longtail,
        'latency': run_latency,
        'routing_diagnostics': run_routing_diagnostics,
        'gradient_diagnostics': run_gradient_diagnostics,
        'ablation': run_ablation,
        'plots': run_plots,
        'all': run_all_pipeline,
    }
    stage_map[args.stage](config, args)
```

### Ablation Runner Architecture

```python
STUDY_CONFIGS = {
    'A': [  # 6 architecture progression configs
        {'use_masked_lm': False, 'multi_step_k': 0, 'use_contrastive': False,
         'moe_variant': 'none'},
        # ... 5 more
    ],
    'B': [  # 2 router conditioning configs
        {'router_conditioning': 'objective_conditioned'},
        {'router_conditioning': 'input_only'},
    ],
    # ... C through H
}

def run_ablation(config, args):
    studies = ['A','B','C','D','E','F','G','H'] if args.study == 'all' else [args.study]
    for study in studies:
        configs = resolve_study_configs(study, config)
        results = []
        for study_config in configs:
            result = train_and_evaluate(study_config)
            results.append(result)
        save_study_results(study, results)
        generate_study_plots(study, results)
    if args.study == 'all':
        generate_summary(all_results)
```

### Config Resolution per Study

Each ablation study overrides specific config fields while keeping everything else identical:

| Study | Fields Modified |
|---|---|
| A | use_masked_lm, multi_step_k, use_contrastive, moe_variant, use_gradnorm (6 configs) |
| B | router_conditioning (2 configs) |
| C | moe_variant (2 configs: mmoe vs arv_mmoe) |
| D | moe_variant, use_gradnorm (4 configs: 2×2 grid) |
| E | (3 configs: global-only, +LoRA, +LoRA+ARV) |
| F | use_masked_lm, multi_step_k, use_contrastive (4 configs: reference + 3 removed) |
| G | num_experts (4 configs: [2, 4, 6, 8]) |
| H | use_metadata, max_seq_length, cold_start_ratio, etc. (6 sub-studies) |

### `--stage all` Pipeline

```python
def run_all_pipeline(config, args):
    run_preprocess(config, args)
    run_pretrain(config, args)
    run_local_personalization(config, args)
    run_federated(config, args)
    run_evaluate(config, args)
    run_cold_start(config, args)
    run_longtail(config, args)
    run_latency(config, args)
    run_routing_diagnostics(config, args)
    run_gradient_diagnostics(config, args)
    run_plots(config, args)
    # Note: ablations NOT included — always explicit
```

### Multi-Seed Final Runs

```python
for seed in config.final_seeds:
    seed_config = replace(config, seed=seed)
    seed_output_dir = f"outputs/seed_{seed}/"
    run_all_pipeline(seed_config, args, output_dir=seed_output_dir)

aggregate_seeds(config.final_seeds, "outputs/")
```

Each seed gets independent output directory. Aggregation computes mean ± std across seeds.

### Plot Generation

The `--stage plots` collects results from all completed evaluations and generates:
- Core tradeoff plots (accuracy vs model size, vs communication, vs FLOPs, vs LoRA rank)
- Ablation-specific plots (architecture progression, MMoE comparison, expert count, etc.)
- All diagnostic plots (routing, gradient, cold-start, long-tail)

## Key Design Decisions

### D1: Config-Driven Ablation Runner
Every ablation variation is expressed as a Config override dict. The runner applies overrides to a base config, saves the resolved config with results, and generates the study-specific outputs. No source code is edited.

### D2: Ablations NOT in `--stage all`
The ablation suite trains many model variants — running it as part of the default pipeline would be extremely expensive and unexpected. It requires explicit `--stage ablation --study {X}`.

### D3: CLI Override Logging
When `--epochs`, `--batch_size`, `--moe_variant`, etc. are passed, the original config value and the new value are both logged. This ensures experiment reproducibility.

### D4: Graceful Checkpoint Absence
`run_evaluate` and other downstream stages check for `best.pt` before proceeding. Missing → skip with message + remedy command. This allows running evaluation for only the models that have been trained so far.

### D5: Study Result Contract
Every study writes JSON with: configurations compared, config diffs, NDCG@10/MRR@10/Recall@10/Precision@10 (mean ± std), and pointers to generated plots. `--study all` also writes a summary JSON aggregating headline numbers.

## Affected Components

- Depends on: all other specs (data-pipeline through evaluation-diagnostics)
- `main.py` is the user-facing entrypoint for everything
- `training/ablation_runner.py` orchestrates multi-config training and evaluation
- Plot generation depends on completed evaluation results

## Testing Strategy

- CLI parsing: verify all arguments accepted and override config correctly
- Ablation config resolution: verify study A produces exactly 6 configs with correct fields
- Missing checkpoint: verify skip message printed, no exception raised
- Stage dispatch: verify each `--stage` value routes to correct function
- Seed isolation: verify two seeds produce independent output directories
- Override logging: verify original and new values appear in log
