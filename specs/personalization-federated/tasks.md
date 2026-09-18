# Personalization & Federated — Tasks

Ordered tasks. Depends on training-pipeline (pretrained checkpoints) and all model specs.

## Phase 1: LoRA Implementation

- [x] **T1.1: Implement `LoRALinear` in `models/lora.py`** — Wraps existing `nn.Linear`. Stores frozen `weight` and `bias` from original. Adds `A = nn.Parameter(randn(d_in, rank) * 0.02)`, `B = nn.Parameter(zeros(rank, d_out))`. Forward: `F.linear(x, weight, bias) + x @ A @ B`. (FR-1)

- [x] **T1.2: Implement `inject_lora`** — Traverse model attention layers. Replace Q and V projections with `LoRALinear`. Freeze all params (`requires_grad=False`). Unfreeze LoRA A and B only (`requires_grad=True`). (FR-1)

- [x] **T1.3: Implement `get_lora_state_dict`** — Filter model state dict by name patterns (`lora_`, `.A`, `.B`). Explicitly exclude keys starting with `moe.router`, `moe.experts`, `moe.objective_embeddings`, `moe.gates`. (FR-1, FR-2)

- [x] **T1.4: Implement `load_lora_state_dict` and `count_lora_parameters`** — Load: restore LoRA params from dict. Count: sum of all LoRA parameter elements. (FR-1)

## Phase 2: LoRA Testing

- [x] **T2.1: Test LoRA injection** — Verify Q and V wrapped, other layers unchanged. Verify forward output changes after LoRA training step. (AC-01)

- [x] **T2.2: Test LoRA state dict filtering** — Verify no MoE/router/expert/embedding keys. Verify only LoRA-pattern keys present. (AC-02)

- [x] **T2.3: Test parameter count** — Verify count matches expected: `2 * num_attention_layers * (d_in * rank + rank * d_out)`. (AC-03)

- [x] **T2.4: Test freeze enforcement** — After injection, assert zero trainable backbone parameters. Verify only LoRA params have `requires_grad=True`. (AC-04)

## Phase 3: Local Personalization

- [x] **T3.1: Implement `run_local_personalization` in `training/local_train.py`** — Function signature per Step 16.2. Sample 1000 users (seeded). Per user: load frozen backbone → inject fresh LoRA → train for local_epochs=3 → evaluate with LoRA → evaluate without LoRA → record per-user metrics including cold/warm classification. Every user starts from same frozen checkpoint with fresh LoRA (never reuse). (FR-3)

- [x] **T3.2: Implement per-user training function** — Construct optimizer with LoRA params only (AdamW, lr=lora_lr, weight_decay=0). Train using `train_user_history` for negatives. Assert trainable backbone params == 0 before training. (FR-2, FR-3)

- [x] **T3.3: Implement personalization report** — Save `outputs/logs/personalization_gain_{model}.json` with: num_users_evaluated, num_users_skipped, with_lora stats, without_lora stats, gain stats, per_user_gains array. (FR-4)

- [x] **T3.4: Implement personalization plots** — Histogram: `personalization_gain_histogram_{model}.png` with vertical lines at x=0 and mean. Scatter: `global_vs_personalized_{model}.png` with y=x diagonal. (FR-4)

## Phase 4: Federated Simulation

- [x] **T4.1: Implement `weighted_fedavg` in `training/federated.py`** — Weight each client's LoRA state dict by sample count / total. Simple mean when `use_weighted_fedavg=False`. (FR-5)

- [x] **T4.2: Implement federated simulation loop** — Initialize `global_lora_state`. Per round: sample clients → each loads global LoRA → local training → collect updates → assert no MoE keys in updates → aggregate via weighted_fedavg → update global_lora_state → evaluate → log. (FR-5, FR-6)

- [x] **T4.3: Implement frozen-state checksum** — Compute hash over all non-LoRA parameters before round 1. Assert unchanged before and after every round. (FR-6, AC-08)

- [x] **T4.4: Implement communication cost logging** — Per round: `upload_kb = lora_params * 2 / 1024`, `download_kb` same, `total_round_kb`. Log to `outputs/logs/federated_rounds_{model}.json`. (FR-7, AC-07)

- [x] **T4.5: Implement federated convergence plot** — `outputs/plots/federated_convergence_{model}.png`: val NDCG@10 vs round, dashed baseline. (FR-8) **REV-117 fixed:** every round — the final one included — is now scored on the same bounded `federated_eval_examples` prefix, so the plotted series is one population end to end; the final round additionally runs the full split into `val_ndcg_full`, which is what `final_val_ndcg` and `accuracy_vs_communication` quote. `test_scores_stored_candidates_and_bounds_each_round` asserts every plotted point came from the same number of examples.

## Phase 5: Testing

- [x] **T5.1: Test weighted FedAvg** — Two clients with known state dicts and sample counts → verify weighted average matches manual computation. (AC-09)

- [x] **T5.2: Test checkpoint integrity** — Run personalization for 2 users → verify frozen checksum identical before and after. (AC-04, AC-08)

- [x] **T5.3: Test federated convergence** — Run 2 rounds with small user set → verify NDCG logged, communication cost matches formula, global_lora_state persists. (AC-06, AC-07, AC-09)

- [x] **T5.4: Test MoE key exclusion assertion** — Inject a model with MoE → verify `get_lora_state_dict` contains no MoE keys. Attempt to add MoE key → verify assertion catches it. (AC-02)

## Phase 6: Gate Execution (REV-127)

- [x] **T6.1: Run both stages at the owner's gate scale and verify against the files (REV-127)** — Neither `--stage personalize` nor `--stage federated` had ever been run, so FR-4's report, FR-7's round log and FR-8's convergence plot had no existence evidence. Blocked on `ablations-cli` REV-131 (no CLI surface for the reduced scale) and on `loss-functions` REV-105 (the checkpoint was trained under a superseded objective). Both preconditions cleared, then run both stages and verify against the artifacts rather than the console. Do NOT reduce the defaults in `configs/config.py`, and do NOT record these numbers as evidence for AC-05's distribution or AC-06's convergence. (FR-4, FR-7, FR-8, AC-05, AC-06)
  - **RE-RUN AND VERIFIED (2026-09-18), superseding the 2026-09-17 record below (REV-131).**
    REV-131 found that this note and `specs/loss-functions` T3.7 gave two contradictory
    accounts of which checkpoint these artifacts came from, both written into commit
    `0bc67ee`, and that neither could be settled from the repository. It is settled by
    re-running off a single freshly written checkpoint, in one unbroken sequence, and by
    making the artifacts say so themselves.

        uv run python main.py --stage preprocess
        uv run python main.py --stage pretrain     --model t-small --epochs 1 --max_steps 200
        uv run python main.py --stage personalize  --model t-small --eval_users 10 --local_epochs 1
        uv run python main.py --stage federated    --model t-small --clients_per_round 10 --epochs 3

    All four exit 0. `configs/config.py` untouched; all four overrides logged original ->
    new per `code-quality.md`: `personalization_eval_users 1000 -> 10`, `local_epochs 3 -> 1`,
    `federated_rounds 20 -> 3`, `clients_per_round 100 -> 10`.

    **Provenance is now a lookup, not an argument.** Both artifacts carry
    `source_revision` `{"commit": "0bc67ee...", "dirty": true, "source_hash": "3a59e7cb70a8be69"}`,
    read back from the files and compared field-for-field against
    `outputs/checkpoints/t_small/best.pt` — they MATCH. `federated_rounds_*.json` is a list
    read by index at `main.py`'s plots stage, so the stamp is on each round record rather
    than on a wrapper that would break that reader; all 3 rounds match. The contradicting
    claim in `specs/loss-functions` T3.7 has been rewritten to point at
    `specs/training-pipeline` T6.3 instead of standing as a third account.

    Every value below was read back out of the artifact after the run, never from console
    output (REV-130).

    **FR-4 / personalization** — `personalization_gain_t_small.json` carries all six fields
    including `num_users_skipped` (REV-113's rename): `num_users_evaluated=10`,
    `num_users_skipped=0`, `with_lora.mean=0.080702`, `without_lora.mean=0.090499`,
    `gain.mean=-0.009797` (std 0.027134, min -0.052734, max 0.053809). `per_user_gains` has
    10 entries with **both signs — 1 positive, 5 negative, 4 zero** — so the gains are not
    all-constant, which is AC-05's smoke-gate condition. Both plots rendered:
    `personalization_gain_histogram_t_small.png`, `global_vs_personalized_t_small.png`.
    OBSERVATION, not a finding: the mean gain is NEGATIVE at this scale, as it was before
    the re-run. The amended AC-05 is a pipeline-integrity gate and explicitly disclaims the
    distribution claim (10 users, 1 local epoch, off a 200-step checkpoint). RQ3 belongs to
    Phase 14 and is NOT supported by this artifact.

    **FR-7 / FR-8 / federated** — `federated_rounds_t_small.json`, 3 rounds, and
    `federated_convergence_t_small.png` rendered. REV-127's three invariants, re-checked
    against the new file:
      * `val_examples_scored` is UNIFORM at 5000 across all three rounds -> the plotted
        series is one population (the REV-117 invariant).
      * `val_ndcg_full` present on the FINAL round ONLY: `[None, None, 0.094607]` over
        993,372 examples, alongside that round's own 5,000-example `val_ndcg=0.08990`.
      * `upload_kb == lora_params * 2 / 1024` re-derived INDEPENDENTLY, since the round log
        does not record `lora_params`: `count_lora_parameters` on a t-small build with
        `lora_rank=8` gives 24,576, and `24576 * 2 / 1024 = 48.0`, matching the recorded
        `upload_kb` exactly. `download_kb` 48.0; `total_round_kb` 960.0 == (48.0 + 48.0) × 10
        clients. Unchanged by the re-run, as it must be — this is a function of the
        architecture, not of the data.
      * The frozen-state checksum assertion did not fire in any round (run exited 0).
    Per-round `val_ndcg` 0.08663 -> 0.08693 -> 0.08990. NOT recorded as convergence
    evidence: AC-06 is a pipeline gate at this scale and RQ4 belongs to Phase 14.

    *Superseded record (2026-09-17), retained for history:* reported
    `with_lora.mean=0.07002 / without_lora.mean=0.07476 / gain.mean=-0.00474`, 4 positive /
    3 negative / 3 zero gains, and per-round `val_ndcg` 0.0956 -> 0.0969 -> 0.1054 with
    `val_ndcg_full=0.11288`. Those artifacts were produced from a checkpoint whose
    provenance REV-131 could not establish, and they additionally predate `data-pipeline`
    REV-137, which changed the training data. Superseded on both counts.

    This run also exercised the REV-119 memory-mapped candidate path for the first time
    on real data — `run_local_personalization` is its only consumer.
