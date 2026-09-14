# Training Pipeline — Tasks

Ordered tasks. Depends on all model specs (transformer-backbone, moe-routing, loss-functions) and data-pipeline.

## Phase 1: LR Scheduler

- [ ] **T1.1: Implement `build_scheduler` in `training/pretrain.py`** — Linear warmup for `warmup_ratio` fraction of `total_steps`, then cosine decay to 0. Use `LambdaLR`. `total_steps = num_batches_per_epoch * pretrain_epochs`. (FR-1, AC-01)

- [ ] **T1.2: Test LR scheduler** — Verify warmup increases linearly over warmup steps. Verify cosine decay reaches near-zero at end. Verify per-batch stepping (not per-epoch). (AC-01)

## Phase 2: Training Loop

- [ ] **T2.1: Implement model construction function** — Build Transformer + MoELayer based on model name and config. Initialize all parameters in BF16. Move to device via `get_device(config)`. (FR-7)

- [ ] **T2.2: Implement training loop in `training/pretrain.py`** — Per epoch: iterate batches → apply masking (if enabled) → transformer forward (return_all_hidden=True) → extract pooled_hidden → per-objective MoE routing + loss → GradNorm or static weighted sum → backward(retain_graph=use_gradnorm) → gradnorm.update (if enabled) → clip_grad_norm(1.0) → optimizer.step → scheduler.step. Accumulate routing snapshots for diagnostics buffer. Load global negative pool once (never per-epoch). (FR-5, FR-6, AC-04, AC-05, AC-06)

- [ ] **T2.3: Implement contrastive loss integration** — On contrastive-interval batches: generate augmented sequence via `augment_sequence()`, run second forward pass through same model, L2-normalize embeddings, compute InfoNCE. On non-interval batches: zero tensor, exclude from GradNorm active set. (FR-5)

- [ ] **T2.4: Implement validation step** — After each epoch: model.eval(), run evaluation on validation dataset using fixed candidate sets and `eval_user_history`. Report NDCG@10, MRR@10, Recall@10, Precision@10. (FR-8, AC-02)

## Phase 3: Checkpointing

- [ ] **T3.1: Implement checkpoint save** — Save dict with: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, gradnorm_state_dict (if enabled), val metrics, config. Per-epoch: `outputs/checkpoints/{model}/epoch_{N:03d}.pt`. (FR-3)

- [ ] **T3.2: Implement best checkpoint selection** — Track `best_val_ndcg`. Update `best.pt` only on strict NDCG improvement. Store best_epoch and all best_val metrics in checkpoint metadata. Never use MRR, Recall, Precision, training loss, or test metrics for selection. (FR-4, AC-02)

- [ ] **T3.3: Implement checkpoint loading** — Load full state: model, optimizer, scheduler, gradnorm. Support resuming training from a checkpoint. (FR-3)

## Phase 4: Logging

- [ ] **T4.1: Implement per-epoch logging** — Log to `outputs/logs/pretrain_{model_name}.log`: epoch, train loss (total + per-objective), val NDCG@10, MRR@10, Recall@10, Precision@10, current LR, GradNorm lambda values (per objective). (FR-5, AC-07)

- [ ] **T4.2: Implement GradNorm lambda trajectory logging** — Log per-objective lambda values at configured intervals to `outputs/logs/gradnorm_{model}.jsonl`. At end of training, generate `outputs/plots/gradnorm_lambda_trajectory_{model}.png`. (FR-9)

- [ ] **T4.3: Implement routing snapshot accumulation** — Per batch, for each active objective: detach r_ut and append to per-epoch buffer. Reset buffer at each diagnostic interval. Feed buffer to routing diagnostics (evaluation/routing_diagnostics.py). (FR-6, AC-06)

## Phase 5: Integration

- [ ] **T5.1: Implement `run_pretrain` function** — Orchestrate: load datasets → build model → build optimizer/scheduler/gradnorm → training loop → validation → checkpointing → logging. Accept config and model_name. (FR-all)

- [ ] **T5.2: BF16 assertions** — Add assertions verifying all model parameters, outputs, and losses are bfloat16. Add finite-value assertions on losses, lambdas, and gradients on every step. (FR-7, AC-08)

## Phase 6: Testing

- [ ] **T6.1: Checkpoint roundtrip test** — Save checkpoint → load → verify model produces identical output on same input. Verify gradnorm state preserved. (AC-03)

- [ ] **T6.2: One-batch end-to-end smoke test** — Run one complete training iteration on real data shapes. Verify: loss is finite, gradients exist, no NaN/Inf, optimizer step completes, scheduler steps. (Smoke test gate per steering/testing.md Phase 16)

- [ ] **T6.3: One-epoch T-Small smoke test** — Run one full epoch of T-Small on real MovieLens data. Verify: training loss decreases, validation metrics computed, checkpoint saved, log written. (Smoke test gate per steering/testing.md Phase 17)

- [ ] **T6.4: Negative pool reuse test** — Verify the negative pool is loaded from disk (not regenerated) on second run with same fingerprint. (AC-05)
