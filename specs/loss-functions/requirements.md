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
- Predict multiple future items from **one** context representation, at uniform horizons
- **All steps read the same hidden state, at position `S-1`** — the last position, which is
  the end of the context under left-padding. Never `S-1-step`; never `seq_lens-1-step`
- Horizons are **2, 3, …, k+1** interactions ahead of the last context item. Horizon 1 is
  the next-item objective's target and is deliberately not duplicated here: for a context
  `items[:i]`, next-item predicts `items[i]` and `targets_k[s] = items[i+1+s]`
- Per-step linear projection: `nn.ModuleList` of k `nn.Linear(D, D, bias=False)` — each
  projection is a pure horizon head, since all of them read the same input vector
- Only valid targets (non-zero) contribute; targets that cross split boundaries are zero
- Return mean loss across valid steps
- Requires `return_all_hidden=True` from transformer

**Rationale (REV-105, owner decision 2026-09-17):** this requirement previously said
"Predict next k items" while mandating position `S-1-step`. Those two clauses are not
compatible. Step `s` reading position `S-1-s` sees context `items[:i-s]` and predicts
`items[i+1+s]`, so the horizons were **2, 4, 6** — each step simultaneously moved the
target further away *and* shortened the context, conflating horizon with context length and
making "step" mean two things at once. Nothing in the spec ever motivated that; it was a
consequence of the position rule, not a design choice.

Reading every step from `S-1` gives uniform horizons **2, 3, 4** from one fixed context,
which is what "multi-step prediction" denotes in the literature and is how a reader will
interpret the paper. It also makes the per-step projections interpretable: they differ only
by horizon, because their input is identical.

Neither form leaks — every target is strictly in the future of the position it is predicted
from — so this is a change of experiment, not a correctness fix. It is made now because the
project has not yet run full training, which is the last moment it is free.

Note the consequence the old wording hid: the multi-step objective starts at horizon **2**,
not 1, under either scheme. Horizon 1 belongs to the next-item objective, and FR-11 keeps
that objective always enabled.

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
- **G_t must be computed in its factored form** `G_t = λ_t * ‖∇_W L_t‖`, where `‖∇_W L_t‖` is taken with `create_graph=False` and **detached** before being scaled by λ_t. λ_t is a scalar that does not depend on the anchor parameters W, so `∇_W(λ_t · L_t) = λ_t · ∇_W L_t` exactly and the two forms are algebraically identical — this is a change of expression, not of algorithm
- **No second-order differentiation anywhere in the GradNorm path.** `create_graph=True` must not appear in the controller. A conforming implementation must run on any host whose attention kernel supports only a first-order backward
- GradNorm loss: `sum |G_t - detach(G*_t)|` over active objectives
- The λ gradient may be taken either by `torch.autograd.grad(gradnorm_loss, [log_lambdas])` over the (now graph-free except through λ) expression, or in closed form as `∂L_grad/∂log λ_t = sign(G_t - G*_t) · G_t`. The two are equivalent and both are conforming; whichever is used, it must write into `log_lambdas.grad` only and must never leave an autograd graph attached to that tensor
- After every update: renormalize active lambdas so `sum(lambda_t) == num_active_objectives`
- Anchor: parameters of last shared Transformer block (before optional MoE) — exists in every variant
- Inactive objectives: effective weight zero, excluded from GradNorm entirely (no zero-valued placeholder losses to autograd.grad)
- Contrastive is inactive on batches where its interval does not fire

**Rationale (REV-120, owner decision):** the earlier wording was implemented literally as
`‖∇_W(λ_t · L_t)‖` built with `create_graph=True`, which made the λ gradient a second-order
derivative through the anchor block's attention kernel. That is a property of the
expression, not of GradNorm: because λ_t is a scalar independent of W, the factored form
yields bit-identical G_t and bit-identical `∂L_grad/∂log λ`. Requiring the factored form
costs nothing numerically and removes three problems at once — the double-backward is the
single largest cost in a training step (measured 276 ms of a 605 ms t-small step on MPS),
it cannot execute at all on CPU-only hosts, and its support on the H100 FlashAttention-4
path was never verified. See design D7.

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
- **AC-03:** Multi-step loss reads the hidden state at position `S-1` for **every** step (never `S-1-step`, never seq_lens-based), so all k per-step projections receive the same input vector and the horizons are uniform at 2..k+1
- **AC-04:** InfoNCE produces correct similarity matrix with masked self-similarity
- **AC-05:** GradNorm lambdas sum to num_active_objectives after every update (within tolerance)
- **AC-06:** GradNorm excludes inactive objectives from gradient computation
- **AC-07:** GradNorm anchor resolves to last Transformer block parameters in all variants including `moe_variant="none"`
- **AC-08:** Static weight fallback produces same result as manual weighted sum
- **AC-09:** All losses are finite on every training step (asserted)
- **AC-10:** The GradNorm path performs no second-order differentiation: `create_graph=True` appears nowhere in `training/gradnorm.py`, and a full `update()` completes on a device/backend whose attention kernel implements only a first-order backward (CPU + SDPA is the reference case)
- **AC-11:** The λ gradient matches the value `torch.autograd.grad` produces for the same `gradnorm_loss` expression, verified on all-active, partially-active, and `G_t == G*_t` inputs; and `log_lambdas.grad` carries no autograd graph (`requires_grad is False`) after `update()`
