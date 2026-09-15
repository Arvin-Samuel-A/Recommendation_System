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

- [x] **T3.3: Implement personalization report** — Save `outputs/logs/personalization_gain_{model}.json` with: num_users_evaluated, num_skipped, with_lora stats, without_lora stats, gain stats, per_user_gains array. (FR-4)

- [x] **T3.4: Implement personalization plots** — Histogram: `personalization_gain_histogram_{model}.png` with vertical lines at x=0 and mean. Scatter: `global_vs_personalized_{model}.png` with y=x diagonal. (FR-4)

## Phase 4: Federated Simulation

- [x] **T4.1: Implement `weighted_fedavg` in `training/federated.py`** — Weight each client's LoRA state dict by sample count / total. Simple mean when `use_weighted_fedavg=False`. (FR-5)

- [x] **T4.2: Implement federated simulation loop** — Initialize `global_lora_state`. Per round: sample clients → each loads global LoRA → local training → collect updates → assert no MoE keys in updates → aggregate via weighted_fedavg → update global_lora_state → evaluate → log. (FR-5, FR-6)

- [x] **T4.3: Implement frozen-state checksum** — Compute hash over all non-LoRA parameters before round 1. Assert unchanged before and after every round. (FR-6, AC-08)

- [x] **T4.4: Implement communication cost logging** — Per round: `upload_kb = lora_params * 2 / 1024`, `download_kb` same, `total_round_kb`. Log to `outputs/logs/federated_rounds_{model}.json`. (FR-7, AC-07)

- [x] **T4.5: Implement federated convergence plot** — `outputs/plots/federated_convergence_{model}.png`: val NDCG@10 vs round, dashed baseline. (FR-8)

## Phase 5: Testing

- [x] **T5.1: Test weighted FedAvg** — Two clients with known state dicts and sample counts → verify weighted average matches manual computation. (AC-09)

- [x] **T5.2: Test checkpoint integrity** — Run personalization for 2 users → verify frozen checksum identical before and after. (AC-04, AC-08)

- [x] **T5.3: Test federated convergence** — Run 2 rounds with small user set → verify NDCG logged, communication cost matches formula, global_lora_state persists. (AC-06, AC-07, AC-09)

- [x] **T5.4: Test MoE key exclusion assertion** — Inject a model with MoE → verify `get_lora_state_dict` contains no MoE keys. Attempt to add MoE key → verify assertion catches it. (AC-02)
