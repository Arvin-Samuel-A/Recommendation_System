# Training Pipeline — Design

## Relevant Existing Architecture

Greenfield. Design follows Prompt_v6.2 Steps 14–15.

## Files Created / Modified

| File | Action | Role |
|---|---|---|
| `training/pretrain.py` | Create | `build_scheduler()`, `run_pretrain()`, training loop, validation, checkpointing |
| `main.py` | Create (partial) | `--stage pretrain` dispatch |

## Proposed Architecture

### Training Loop Pseudocode

```python
def run_pretrain(config, model_name):
    # Setup
    dataset, val_dataset = load_tensorised_datasets(config)
    model = build_model(config, model_name)  # Transformer + MoE
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = build_scheduler(optimizer, config, total_steps)
    gradnorm = GradNormController(...) if config.use_gradnorm else None
    neg_pool = load_global_negative_pool(config)  # loaded once, never regenerated
    best_val_ndcg = -1.0

    for epoch in range(config.pretrain_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # 1. Masking (if enabled)
            # 2. Forward pass (return_all_hidden=True)
            # 3. Per-objective MoE routing + loss computation
            # 4. GradNorm or static weighted sum
            # 5. Backward + GradNorm update + clip + step + schedule
            # 6. Accumulate routing snapshots

        # Validation
        model.eval()
        val_metrics = evaluate(model, val_dataset, config)

        # Checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, gradnorm, val_metrics)
        if val_metrics['ndcg'] > best_val_ndcg:
            best_val_ndcg = val_metrics['ndcg']
            save_best_checkpoint(...)

        # Logging
        log_epoch(epoch, train_loss, val_metrics, lr, gradnorm_lambdas)
```

### LR Scheduler

```python
def build_scheduler(optimizer, config, total_steps):
    warmup_steps = int(total_steps * config.warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)
```

Called per-batch, not per-epoch.

### Checkpoint Save/Load

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),      # includes MoE + embeddings
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'gradnorm_state_dict': gradnorm.state_dict() if gradnorm else None,
    'val_ndcg': val_ndcg,
    'val_mrr': val_mrr,
    'val_recall': val_recall,
    'val_precision': val_precision,
    'config': asdict(config),
}
```

### Validation Protocol

- Uses fixed evaluation candidate sets (loaded from disk)
- Uses `eval_user_history` for negative filtering (NOT `train_user_history`)
- Reports NDCG@10, MRR@10, Recall@10, Precision@10
- Full serving path with MoE routing (e_next objective) when applicable

## Key Design Decisions

### D1: Per-Batch LR Scheduling
The scheduler steps after every batch, not every epoch. With 10 epochs on MovieLens-20M, per-epoch scheduling would be too coarse for the warmup/cosine curve.

### D2: Retain Graph for GradNorm
`L_total.backward(retain_graph=use_gradnorm)` — when GradNorm is active, the graph must be retained so GradNorm can compute per-objective gradients w.r.t. the shared anchor after the main backward pass. Without GradNorm, the graph is freed immediately.

### D3: Routing Snapshot Accumulation
Every routing distribution r_ut computed for an active objective is detached and appended to a per-epoch buffer. This buffer feeds the routing diagnostics in Step 21.5. The buffer is reset each diagnostic interval. This is done inside the training loop, not as a separate post-hoc pass (though a comprehensive final pass on validation data also runs at end of training).

### D4: Independent Per-Model Training
Each model (T-Small, T-Base, T-Large) is trained independently via `--model` flag. No state is shared between models. This ensures clean experimental comparisons.

### D5: Best Checkpoint by Val NDCG Only
Only validation NDCG@10 determines the best checkpoint. This prevents overfitting to multiple metrics and ensures a single, clear model selection criterion. Test data is not consulted.

## Affected Components

- Depends on: `models/transformer.py`, `models/moe.py`, `training/losses.py`, `training/gradnorm.py`, `data/sequence_generator.py`, `evaluation/metrics.py`
- Produces: checkpoints, logs, GradNorm trajectory
- Consumed by: `training/local_train.py`, `training/federated.py`, `evaluation/*`, `training/ablation_runner.py`

## Testing Strategy

- LR schedule: verify warmup increases linearly, cosine decreases to 0
- Checkpoint roundtrip: save → load → verify model output is identical
- Best checkpoint: simulate improving/plateauing NDCG → verify correct epochs selected
- Training loop: one-batch test verifying loss is finite and decreasing
- Routing buffer: verify buffer collects expected number of snapshots per epoch
