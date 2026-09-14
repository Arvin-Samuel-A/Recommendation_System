# Personalization & Federated — Requirements

## Overview

Implement LoRA adapter injection, local personalization evaluation (1000-user sample), and federated simulation with weighted FedAvg, persistent client state, communication accounting, and strict parameter isolation guarantees.

## Problem Statement

Federated learning requires strict separation between global parameters (frozen backbone, MoE, router, experts, objective embeddings) and local parameters (LoRA adapters only). The communication-efficiency claim (RQ4) depends on verifying that only LoRA state crosses the wire. Personalization gains must be measured against the same frozen global checkpoint.

## Goals

- G1: LoRA injection on Q and V attention layers
- G2: 1000-user local personalization with gain measurement and plotting
- G3: Federated simulation with weighted FedAvg and persistent global LoRA state
- G4: Strict parameter isolation: LoRA-only trainable, LoRA-only communication
- G5: Communication cost accounting per federated round
- G6: Federated convergence tracking and plotting

## Non-Goals

- Real distributed federated infrastructure
- Differential privacy
- Client dropout simulation beyond sampling

## Functional Requirements

### FR-1: LoRA Implementation (Step 10)
- `LoRALinear`: wraps existing `nn.Linear`, adds rank-r A and B matrices
- `A = nn.Parameter(randn(d_in, rank) * 0.02)`, `B = nn.Parameter(zeros(rank, d_out))`
- Forward: `linear(x, weight, bias) + x @ A @ B`
- `inject_lora(model, rank)`: replace Q and V in all attention layers
- `get_lora_state_dict(model)`: LoRA params only, filtered by name pattern (`lora_`, `.A`, `.B`)
- `load_lora_state_dict(model, sd)`: restore LoRA params
- `count_lora_parameters(model)`: trainable count
- After injection: freeze all params, then unfreeze only LoRA A and B

### FR-2: Parameter Isolation (Step 16.5)
- Global/frozen during local training: Transformer backbone, embeddings, SharedExperts, ARV router (W_r, b_r), conventional MMoE gates, objective embeddings, shared task heads, GradNorm log_lambdas
- Local/trainable: LoRA adapters only
- `requires_grad = False` for all global params before constructing optimizer
- Optimizer constructed with LoRA parameters only — backbone params NEVER passed
- Runtime assertion: verify zero trainable backbone parameters before training
- LoRA state dict must NEVER contain keys starting with `moe.router`, `moe.experts`, `moe.objective_embeddings`, `moe.gates`

### FR-3: Local Personalization (Step 16)
- Sample 1000 users (seeded) from `personalization_eval_users`
- Per user: load frozen backbone → inject fresh LoRA → train for `local_epochs=3` → evaluate on test split
- Also evaluate backbone without LoRA on same test split
- Record per-user: NDCG@10 with/without LoRA, gain, whether test item is cold/warm
- Every user starts from same frozen global checkpoint with fresh LoRA (never reuse another user's LoRA)
- Frozen checkpoint checksum identical for every user

### FR-4: Personalization Report & Plots (Step 16.4)
- Save to `outputs/logs/personalization_gain_{model}.json`: num_users_evaluated, num_users_skipped, with_lora stats, without_lora stats, gain stats, per_user_gains array
- Plot 1: `personalization_gain_histogram_{model}.png` — histogram of per-user NDCG gains with vertical lines at x=0 and mean
- Plot 2: `global_vs_personalized_{model}.png` — scatter (NDCG without LoRA vs with LoRA) with y=x diagonal

### FR-5: Federated Simulation (Step 17)
- Persistent `global_lora_state` across all rounds
- Per round: sample `clients_per_round=100` clients, each loads global LoRA state, trains locally, uploads LoRA update
- Weighted FedAvg (default): weight each client's update by its local training sample count
- Simple mean when `use_weighted_fedavg=False`
- Non-participating clients unaffected; receive latest state when next sampled

### FR-6: Client State Integrity (Step 17.2)
- Before round 1: compute checksum over all frozen global parameters (backbone, embeddings, experts, router, gates, objective embeddings, heads, non-LoRA state)
- Before and after every round: assert checksum unchanged
- Only aggregated LoRA state may change on server
- Assert LoRA state dict contains no MoE/router/expert/objective-embedding keys

### FR-7: Communication Cost (Step 17.3)
- Per round: `upload_kb = lora_params * 2 / 1024` (BF16), `download_kb` same
- `total_round_kb = (upload + download) * clients_per_round`
- MoE parameters must NOT appear in communication cost
- Log per round to `outputs/logs/federated_rounds_{model}.json`

### FR-8: Federated Convergence Plot (Step 17.4)
- `outputs/plots/federated_convergence_{model}.png`
- X: FL round number, Y: global validation NDCG@10
- Horizontal dashed line: non-federated personalized baseline

### FR-9: LoRA Training Constraints
- LoRA LR tuned independently of backbone LR (typical range: 1e-3 to 1e-2)
- LoRA weight decay: 0
- Optimizer: AdamW recommended
- LoRA parameters identified by name pattern or explicit parameter group

## Edge Cases

- User with very few test items: evaluate on available items
- User where LoRA hurts (negative gain): valid and expected; included in report
- Client with zero training samples: skip, do not include in aggregation

## Constraints

- Any implementation that updates backbone parameters during personalization is invalid
- Any federated payload including router/expert/objective-embedding parameters is invalid
- LoRA LR must be independent of backbone pretraining LR

## Dependencies

- Pretrained backbone checkpoint from `training/pretrain.py`
- `models/lora.py` for LoRA utilities
- `evaluation/metrics.py` for NDCG/MRR/Recall/Precision evaluation
- `configs/config.py` for all parameters

## Acceptance Criteria

- **AC-01:** `inject_lora` correctly wraps Q and V in all attention layers; forward output changes after LoRA training
- **AC-02:** `get_lora_state_dict` returns only LoRA keys; no MoE/router/expert/embedding keys
- **AC-03:** `count_lora_parameters` matches sum of A and B parameter sizes
- **AC-04:** Frozen backbone checksum identical before and after personalization for every user
- **AC-05:** Personalization gain histogram shows expected distribution (some negative gains)
- **AC-06:** Federated convergence NDCG improves over rounds (compared to initial state)
- **AC-07:** Communication cost per round matches formula: `2 * lora_params * 2 / 1024 * clients_per_round`
- **AC-08:** Global frozen-state checksum unchanged through all federated rounds
- **AC-09:** `global_lora_state` persists across all rounds; clients resume from it
