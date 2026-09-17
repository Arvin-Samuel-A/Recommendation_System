# Training Pipeline — Tasks

Ordered tasks. Depends on all model specs (transformer-backbone, moe-routing, loss-functions) and data-pipeline.

## Phase 1: LR Scheduler

- [x] **T1.1: Implement `build_scheduler` in `training/pretrain.py`** — Linear warmup for `warmup_ratio` fraction of `total_steps`, then cosine decay to 0. Use `LambdaLR`. `total_steps = num_batches_per_epoch * pretrain_epochs`. (FR-1, AC-01)

- [x] **T1.2: Test LR scheduler** — Verify warmup increases linearly over warmup steps. Verify cosine decay reaches near-zero at end. Verify per-batch stepping (not per-epoch). (AC-01)

## Phase 2: Training Loop

- [x] **T2.1: Implement model construction function** — Build Transformer + MoELayer based on model name and config. Initialize all parameters in BF16. Move to device via `get_device(config)`. (FR-7)

- [x] **T2.2: Implement training loop in `training/pretrain.py`** — Per epoch: iterate batches → apply masking (if enabled) → transformer forward (return_all_hidden=True) → extract pooled_hidden → per-objective MoE routing + loss → GradNorm or static weighted sum → backward(retain_graph=use_gradnorm) → gradnorm.update (if enabled) → clip_grad_norm(1.0) → optimizer.step → scheduler.step. Accumulate routing snapshots for diagnostics buffer. Load global negative pool once (never per-epoch). (FR-5, FR-6, AC-04, AC-05, AC-06)

- [x] **T2.3: Implement contrastive loss integration** — On contrastive-interval batches: generate augmented sequence via `augment_sequence()`, run second forward pass through same model, L2-normalize embeddings, compute InfoNCE. On non-interval batches: zero tensor, exclude from GradNorm active set. (FR-5)

- [x] **T2.4: Implement validation step** — After each epoch: model.eval(), run evaluation on validation dataset using fixed candidate sets and `eval_user_history`. Report NDCG@10, MRR@10, Recall@10, Precision@10. (FR-8, AC-02)

## Phase 3: Checkpointing

- [x] **T3.1: Implement checkpoint save** — Save dict with: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, gradnorm_state_dict (if enabled), val metrics, config. Per-epoch: `outputs/checkpoints/{model}/epoch_{N:03d}.pt`. (FR-3)

- [x] **T3.2: Implement best checkpoint selection** — Track `best_val_ndcg`. Update `best.pt` only on strict NDCG improvement. Store best_epoch and all best_val metrics in checkpoint metadata. Never use MRR, Recall, Precision, training loss, or test metrics for selection. (FR-4, AC-02)

- [x] **T3.3: Implement checkpoint loading** — Load full state: model, optimizer, scheduler, gradnorm. Support resuming training from a checkpoint. (FR-3)

## Phase 4: Logging

- [x] **T4.1: Implement per-epoch logging** — Log to `outputs/logs/pretrain_{model_name}.log`: epoch, train loss (total + per-objective), val NDCG@10, MRR@10, Recall@10, Precision@10, current LR, GradNorm lambda values (per objective). (FR-5, AC-07)

- [x] **T4.2: Implement GradNorm lambda trajectory logging** — Log per-objective lambda values at configured intervals to `outputs/logs/gradnorm_{model}.jsonl`. At end of training, generate `outputs/plots/gradnorm_lambda_trajectory_{model}.png`. (FR-9)

- [x] **T4.3: Implement routing snapshot accumulation** — Per batch, for each active objective: detach r_ut and append to per-epoch buffer. Reset buffer at each diagnostic interval. Feed buffer to routing diagnostics (evaluation/routing_diagnostics.py). (FR-6, AC-06)

## Phase 5: Integration

- [x] **T5.1: Implement `run_pretrain` function** — Orchestrate: load datasets → build model → build optimizer/scheduler/gradnorm → training loop → validation → checkpointing → logging. Accept config and model_name. (FR-all)

- [x] **T5.2: BF16 assertions** — Add assertions verifying all model parameters, outputs, and losses are bfloat16. Add finite-value assertions on losses, lambdas, and gradients on every step. (FR-7, AC-08)

## Phase 6: Testing

- [x] **T6.1: Checkpoint roundtrip test** — Save checkpoint → load → verify model produces identical output on same input. Verify gradnorm state preserved. (AC-03)

- [x] **T6.2: One-batch end-to-end smoke test** — Run one complete training iteration on real data shapes. Verify: loss is finite, gradients exist, no NaN/Inf, optimizer step completes, scheduler steps. (Smoke test gate per steering/testing.md Phase 16)
  - BUILD note (REV-062): the test exists and its training assertions pass. The gate is
    armed (`require_attention_backend=True`) and was failing because the then-sanctioned
    MPS kernel, FlexAttention, has no backward pass on MPS in PyTorch 2.14.0.
  - PLAN note (REV-062, RESOLVED -- owner decision): resolution (a) was taken. SDPA is
    now the sanctioned MPS/CPU kernel (`steering/code-quality.md` "Attention Backend
    Discipline"; `specs/transformer-backbone` FR-4 / D2b), and FlexAttention is removed
    from the backend surface by `transformer-backbone T1.3`. **Keep
    `require_attention_backend = True` here** -- the arming was never the problem, and
    the gate now passes because the preferred backend genuinely executes. Check this
    task off only after re-running the gate unmodified against the narrowed backend
    surface.
  - BUILD note (gate re-run): done. `tests/test_smoke.py` was changed on comment lines
    ONLY (`git diff` shows no non-comment edit); `require_attention_backend = True` is
    still set and no assertion was touched. The gate passes on MPS with the preferred
    backend resolving to `sdpa` and no `fallback_reason` recorded. Full suite: 256
    passed, 0 failed.

- [x] **T6.3: Few-step T-Small smoke test on real data** — Run `run_pretrain` on the real
  MovieLens `data/processed` with a capped step budget
  (`--stage pretrain --model t-small --epochs 1 --max_steps N`, a few hundred steps).
  Verify: validation metrics computed, checkpoint saved, log written.
  (Smoke test gate per steering/testing.md Phase 17)
  - **Owner decision (2026-09-17):** this task previously read "Run one full epoch of
    T-Small on real MovieLens data ... Verify: training loss decreases, validation
    metrics computed, checkpoint saved, log written." The spec owner has reduced the gate
    to a few-step run on real data; `steering/testing.md` Phase 17 and `BUILD_ORDER`
    step 11.2 were updated to match, and both record the decision.
    Context for the next review: REV-066 flagged an earlier, unauthorised version of this
    same relaxation, where BUILD rewrote the gate to fit what it had run. This one is the
    owner's, made explicitly. `--max_steps` was added to `parse_args` so the gate is
    runnable without editing source (`conventions.md`, "Config-Driven Design").
  - **AC-04 is NOT covered by this gate.** "Training loss decreases over epochs on real
    data" cannot be observed over a few hundred steps: under LR warmup both dev-host
    probes showed loss *rising* (100 steps 26.23 -> 28.51; 400 steps 27.32 -> 29.15).
    AC-04 is unchanged and stays outstanding until full training (BUILD_ORDER Phase 14).
    The reduced gate no longer claims it.
  - Dev-host measurement for the record (2026-09-17, after loss-functions REV-120): the
    uncapped epoch is 30,447 steps at ~0.235 s/step ≈ 2.0 h, down from the 0.641 s/step /
    ~5.5 h measured before the factored-`G_t` change.
  - `tests/test_training.py::TestFewStepSmokeViaRunPretrain` is the in-suite form of this
    gate: the same capped `run_pretrain` path against a subset of the real
    `data/processed`. It runs on every `pytest` invocation and is kept.
  - **RUN AND VERIFIED (2026-09-17):**
        uv run python main.py --stage pretrain --model t-small --epochs 1 --max_steps 200
    against the freshly re-preprocessed `data/processed` (REV-090a). 190.5 s wall, 7.2 GB
    peak RSS, exit 0. All three criteria of the reduced gate hold, checked against the
    artifacts rather than the console:
    1. **Validation metrics computed** over the FULL 993,372-example validation split
       against the stored `val_candidates.npy`: `val_ndcg@10=0.090022`,
       `val_mrr@10=0.064059`, `val_recall@10=0.176741`, `val_precision@10=0.017674`.
    2. **Checkpoint saved** — `outputs/checkpoints/t_small/best.pt` and `epoch_000.pt`,
       both carrying the full FR-4 metadata set (best_epoch, best_val_{ndcg,mrr,recall,
       precision}, seed, model_name, moe_variant, use_gradnorm, num_experts, config) plus
       the GradNorm state.
    3. **Log written** — `outputs/logs/pretrain_t_small.log` with every AC-07 field:
       per-objective losses, all four validation metrics, LR and the four λ values.
    Also produced, from a real run rather than a fixture:
    `outputs/plots/gradnorm_lambda_trajectory_t_small.png` (rendered from the real
    `gradnorm_t_small.jsonl`), the routing diagnostics JSON and its three plots.
  - Two things a reader of that log should not misread:
    - `lr=0.00000000` is correct, not a defect. `max_steps_per_epoch` also shrinks
      `total_steps` (`training/pretrain.py:682-684`), so the capped run executes the
      WHOLE schedule: warmup 0 -> 1e-3 over 20 steps, then cosine decay back to 0 at
      step 200. The logged value is the schedule's endpoint.
    - The saved λ values sum to 3.93, not 4. Contrastive was inactive on the final step
      (`contrastive_interval`), and the three active λ sum to exactly 3.000000 — which is
      AC-05's `sum(active) == num_active` holding, not violating.

- [x] **T6.4: Negative pool reuse test** — Verify the negative pool is loaded from disk (not regenerated) on second run with same fingerprint. (AC-05)
