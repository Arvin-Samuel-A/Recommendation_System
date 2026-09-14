# Loss Functions & GradNorm — Requirements

## Overview

Implement all four training objectives (sampled softmax, masked item modeling, multi-step prediction, contrastive InfoNCE) and the GradNorm dynamic loss balancing controller. All losses reside in `training/losses.py`; GradNorm in `training/gradnorm.py`.

## Problem Statement

Multi-objective training requires careful loss balancing — static weights are suboptimal as objectives train at different rates. GradNorm dynamically balances gradient magnitudes across objectives. Each loss has specific masking, negative-sampling, and causal-position requirements that must be correct to avoid data leakage.

## Goals

- G1: Four loss functions implementing sampled softmax, MIM, multi-step, and InfoNCE
- G2: All losses consume objective-specific MoE output Z_{u,t} (or raw hidden states when moe_variant="none")
- G3: GradNorm dynamically learns per-objective weights λ_t
- G4: Correct handling of active/inactive objectives per batch
- G5: Static weight fallback when GradNorm is disabled

## Non-Goals

- Implementing custom gradient computation (use standard autograd)
- Alternative loss functions not specified in the paper

## Functional Requirements

### FR-1: Sampled Softmax Loss (Step 7.1)
- Tied weights: uses item embedding weight as output projection
- Positive: dot product of user embedding and positive item embedding
- Negatives: dot product of user embedding and K sampled negative embeddings
- Logits: `[pos_score, neg_scores]` → cross-entropy with label 0
- Input: `user_emb (B, D)`, `pos_ids (B,)`, `neg_ids (B, K)`, `item_emb_weight (V, D)`

### FR-2: Masked Item Modeling Loss (Step 7.2)
- BERT4Rec-style: compute cross-entropy at positions where input was replaced with MASK_ID
- Before forward pass: randomly mask `mask_prob=15%` of non-padding items with MASK_ID = num_items + 1
- Store original item IDs as `true_item_ids` for the loss
- Pass masked input to transformer; collect all hidden states
- If zero masked positions in a batch: return tensor 0.0
- Full vocabulary logits via tied weights: `masked_hidden @ item_emb_weight.T`

### FR-3: Multi-Step Prediction Loss (Step 7.3)
- Predict next k items using hidden states at causal positions
- For step k (predicting item t+k): use hidden state at position `S-1-step` (left-padding convention)
- Never use `seq_lens-1-step` — positions are fixed under left-padding
- Per-step linear projection: `nn.ModuleList` of k `nn.Linear(D, D, bias=False)`
- Only valid targets (non-zero) contribute; targets that cross split boundaries are zero
- Return mean loss across valid steps
- Requires `return_all_hidden=True` from transformer

### FR-4: Contrastive InfoNCE Loss (Step 7.4)
- L2-normalize both original and augmented sequence embeddings
- In-batch negatives: each sample's augmented view is negative for all others
- Similarity matrix: `(2B, 2B)` with temperature scaling, self-similarity masked to -1e9
- Positive pairs: (i, i+B) and (i+B, i)
- Computed only every `contrastive_interval` batches (default 4) — skip second forward pass otherwise
- `contrastive_interval=0` is invalid; use `use_contrastive=False` to disable
- Requires second full forward pass on augmented sequence through same model (shared weights)

### FR-5: GradNorm Controller (Step 7.5)
- Maintains `log_lambdas` for fixed OBJECTIVE_ORDER; operates only on active indices
- Separate optimizer over `log_lambdas` (not the main model optimizer)
- `lambda_t = exp(log_lambda_t)`
- At first valid step: store `L_t(0) = detach(L_t)` for each active objective
- Per update: compute G_t (gradient norm of λ_t * L_t w.r.t. shared anchor params), Lhat_t (ratio to initial loss), relative training rate r_t, target gradient G*_t
- GradNorm loss: `sum |G_t - detach(G*_t)|` over active objectives
- After every update: renormalize active lambdas so `sum(lambda_t) == num_active_objectives`
- Anchor: parameters of last shared Transformer block (before optional MoE) — exists in every variant
- Inactive objectives: effective weight zero, excluded from GradNorm entirely (no zero-valued placeholder losses to autograd.grad)
- Contrastive is inactive on batches where its interval does not fire

### FR-6: Loss Combination
- Training loop contract: compute active_losses dict → weighted_sum via GradNorm or static weights → backward → gradnorm.update → clip_grad_norm → optimizer.step → scheduler.step
- `L_total.backward(retain_graph=config.use_gradnorm)` — retain graph when GradNorm needs per-objective gradients
- Static fallback: `L_next + masked_lm_weight * L_mim + multi_step_weight * L_ms + contrastive_weight * L_cl`
- Assert finite losses, finite GradNorm loss, finite lambdas on every step

## Edge Cases

- All masked positions are padding (empty batch for MIM): return 0.0 loss
- Multi-step target all zeros (no valid targets): return 0.0 loss
- Contrastive on non-interval batch: 0.0 tensor, not included in GradNorm active set
- Only one active objective: GradNorm still operates but trivially (lambda = 1.0)
- GradNorm initial loss is near-zero: handle division safely

## Constraints

- `mask_token = num_items + 1` (never `num_items`)
- GradNorm `log_lambdas` frozen during local/federated training
- GradNorm anchor must be a parameter set that exists in every model variant
- All loss computation in BF16

## Dependencies

- `models/transformer.py` for hidden states
- `models/moe.py` for objective-specific representations Z_{u,t}
- `data/augmentation.py` for contrastive augmentation
- `configs/config.py` for all loss parameters

## Acceptance Criteria

- **AC-01:** Sampled softmax loss decreases on toy data with known positive/negative items
- **AC-02:** MIM loss correctly targets only masked positions (non-masked positions contribute zero)
- **AC-03:** Multi-step loss reads hidden state at `S-1-step` for each step (not seq_lens-based)
- **AC-04:** InfoNCE produces correct similarity matrix with masked self-similarity
- **AC-05:** GradNorm lambdas sum to num_active_objectives after every update (within tolerance)
- **AC-06:** GradNorm excludes inactive objectives from gradient computation
- **AC-07:** GradNorm anchor resolves to last Transformer block parameters in all variants including `moe_variant="none"`
- **AC-08:** Static weight fallback produces same result as manual weighted sum
- **AC-09:** All losses are finite on every training step (asserted)
