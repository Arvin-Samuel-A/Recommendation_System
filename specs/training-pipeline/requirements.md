# Training Pipeline — Requirements

## Overview

Implement central pretraining with LR scheduling, per-model checkpointing, the full ARV-MMoE + GradNorm forward path, and the training loop integrating all four objectives. Each model is trained independently.

## Problem Statement

The pretraining loop must correctly orchestrate: masked input preparation, transformer forward pass, MoE routing for each active objective, loss computation, GradNorm balancing, gradient clipping, optimizer/scheduler steps, validation, checkpointing, and diagnostic logging — all in BF16 without data leakage.

## Goals

- G1: Independent per-model training via `--model` CLI flag
- G2: Linear warmup + cosine decay LR scheduling (per-batch stepping)
- G3: Correct ARV-MMoE + GradNorm forward/loss path
- G4: Best checkpoint selected by validation NDCG@10 only
- G5: Complete per-epoch logging (losses, metrics, LR, GradNorm lambdas)
- G6: Routing distribution accumulation for diagnostics

## Non-Goals

- Distributed/multi-GPU training
- Model parallelism
- Automatic hyperparameter search

## Functional Requirements

### FR-1: LR Scheduler (Step 14)
- Linear warmup for first `warmup_ratio` fraction of total steps, then cosine decay to 0
- `total_steps = num_batches_per_epoch * pretrain_epochs`
- Call `scheduler.step()` after every batch (not every epoch)
- Log current LR at end of each epoch

### FR-2: Per-Model Checkpoint Directories (Step 15.2)
- T-Small → `outputs/checkpoints/t_small/`
- T-Base → `outputs/checkpoints/t_base/`
- T-Large → `outputs/checkpoints/t_large/`
- Per epoch: `epoch_{N:03d}.pt`; best: `best.pt`

### FR-3: Checkpoint Format (Step 15.3)
- Contains: epoch, model_state_dict, moe_state_dict, loss_fns_state_dict, optimizer_state_dict, scheduler_state_dict, gradnorm_state_dict (holding log_lambdas, its optimizer state and initial_losses), val_ndcg, val_mrr, val_recall, val_precision, config
  - *(REV-126, owner 2026-09-17: MoE and GradNorm state are sibling top-level keys, not nested inside `model_state_dict` as this line previously implied. The MoE is a separate module and GradNorm's Adam state has no place in a model state dict, so the three-key layout is correct; the wording was wrong.)*
- Best checkpoint metadata: best_epoch, best_val_ndcg, best_val_mrr, best_val_recall, best_val_precision, seed, model_name, moe_variant, use_gradnorm, num_experts
- **`source_revision`** — required on every checkpoint. The git commit when the tree is clean;
  otherwise the commit plus a content hash over the covered sources, with a `dirty: true` marker.
  Every artifact derived from a checkpoint echoes this value verbatim, so "which code produced
  this number?" is answerable from the files. `config` alone is insufficient: a change to a loss
  function or an attention kernel leaves it identical.
  - **Coverage is defined by `steering/code-quality.md` ("Coverage: a property, not a list",
    amended 2026-09-18) and is deliberately NOT restated here.** The governing rule is the
    property — the revision must change whenever any code that can change a published artifact
    changes — realised as an exclusion rule rather than an enumeration. This requirement
    previously restated the enumeration `models/`, `training/`, `data/`, `configs/`; that list
    omitted `evaluation/`, where every derived artifact is produced, and a second copy of a list
    is a second thing to drift. One definition, in steering, referenced from here.

### FR-4: Best Checkpoint Selection (Step 15.3.1)
- Selected ONLY by validation NDCG@10 — strict improvement over best seen so far
- Never use MRR, Recall, Precision, training loss, routing diagnostics, or any test metric
- Full test set untouched until model configuration and best checkpoint are frozen

### FR-5: Training Loop (Step 15.4)
- Never refresh global negative pool per epoch; load and reuse prebuilt pool
- For `use_masked_lm=True`: apply masking before forward pass, collect all hidden states, compute MaskedItemLoss
- For `use_contrastive=True`: generate augmented view on interval batches, second forward pass, InfoNCE
- Combine all active losses per §7 formulas
- Log per epoch: train loss, val NDCG@10, MRR@10, Recall@10, Precision@10, current LR, GradNorm λ values
- Log to `outputs/logs/pretrain_{model_name}.log`

### FR-6: ARV-MMoE + GradNorm Forward Path (Step 15.4.1)
- Transformer forward with `return_all_hidden=True` → all_hidden (B,S,D)
- Extract pooled_hidden from last non-pad position → (B,D)
- For each active objective: MoE forward → Z_{u,t}, r_{u,t}
- Compute per-objective loss from Z_{u,t}
- Accumulate routing snapshots r_{u,t} for diagnostics buffer
- Weighted sum via GradNorm or static weights → L_total
- backward(retain_graph=use_gradnorm) → gradnorm.update → clip_grad_norm → optimizer.step → scheduler.step

### FR-7: BF16 Training (Step 15.5)
- All model computation in BF16
- `optimizer.zero_grad(set_to_none=True)`
- No FP16 autocast or GradScaler
- Gradient/finite-value assertions enabled
- No silent dtype change on numerical issues

### FR-8: Validation
- Run validation after each epoch using fixed evaluation candidate sets
- Report NDCG@10, MRR@10, Recall@10, Precision@10
- Use `eval_user_history` for evaluation negatives (not `train_user_history`)

### FR-9: GradNorm Lambda Logging
- Log per-objective λ values at configured intervals to `outputs/logs/gradnorm_{model}.jsonl`
- Generate `outputs/plots/gradnorm_lambda_trajectory_{model}.png` at end of training

## Edge Cases

- Zero valid masked positions in a batch: MIM loss returns 0.0, excluded from GradNorm active set
- Contrastive on non-interval batch: excluded from active set
- All multi-step targets invalid (zero): multi-step loss returns 0.0
- Validation NDCG does not improve for many epochs: best checkpoint unchanged

## Constraints

- No cross-model state sharing between independently trained models
- Single seed for development; multiple independent seeds for final paper runs
- Test data never seen during training or model selection

## Dependencies

- `models/transformer.py`, `models/moe.py` for model
- `training/losses.py`, `training/gradnorm.py` for loss computation
- `data/sequence_generator.py` for datasets and DataLoaders
- `evaluation/metrics.py` for validation evaluation
- `configs/config.py` for all parameters

## Acceptance Criteria

- **AC-01:** LR follows warmup then cosine decay; stepping happens per-batch
- **AC-02:** Best checkpoint is selected by val NDCG@10 only; test data not used
- **AC-03:** Checkpoint save/load roundtrip preserves full model state including GradNorm
- **AC-04:** Training loss decreases over epochs on real data
- **AC-05:** Global negative pool is loaded once and never regenerated per epoch
- **AC-06:** Routing distributions are accumulated in per-epoch buffer (not discarded)
- **AC-07:** Per-epoch log contains all required fields (losses, metrics, LR, lambdas)
- **AC-08:** All assertions (finite losses, finite lambdas, finite gradients) hold throughout training
- **AC-09:** Every saved checkpoint carries a `source_revision`, and every artifact derived from a
  checkpoint carries the same value. Verified by reading a real checkpoint and at least one derived
  artifact and comparing the two fields — not by the presence of the code that writes them. A run
  from a dirty working tree is marked `dirty: true` rather than attributed to its nearest commit.
  **Coverage is complete against the rule in `steering/code-quality.md`, and completeness is
  tested rather than assumed:** the set of files actually hashed equals the set that rule defines,
  and the dirtiness check uses the same rule as the hash. A test showing that *some* covered change
  moves the revision proves the mechanism works, not that it covers what it must — that distinction
  is what let the superseded enumeration pass a green suite while omitting `evaluation/`
