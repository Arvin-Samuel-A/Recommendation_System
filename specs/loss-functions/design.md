# Loss Functions & GradNorm — Design

## Relevant Existing Architecture

Greenfield. Design follows Prompt_v6.2 Steps 5, 7.

## Files Created / Modified

| File | Action | Role |
|---|---|---|
| `training/losses.py` | Create | SampledSoftmaxLoss, MaskedItemLoss, MultiStepLoss, InfoNCELoss |
| `training/gradnorm.py` | Create | GradNormController |
| `data/augmentation.py` | Create | random_crop, random_mask, subsequence_sample, augment_sequence |

## Proposed Architecture

### Loss Module Layout (all in `training/losses.py`)

```python
class SampledSoftmaxLoss(nn.Module):
    # (user_emb, pos_ids, neg_ids, item_emb_weight) → scalar loss

class MaskedItemLoss(nn.Module):
    # (hidden_states, mask_positions, true_item_ids, item_emb_weight) → scalar loss

class MultiStepLoss(nn.Module):
    # (all_hidden_states, targets_k, item_emb_weight) → scalar loss
    # Contains: nn.ModuleList of k step_projections

class InfoNCELoss(nn.Module):
    # (z_orig, z_aug) → scalar loss
    # Both L2-normalized before calling
```

### GradNorm Controller (`training/gradnorm.py`)

```python
class GradNormController:
    def __init__(self, num_objectives, alpha, lr):
        self.log_lambdas = nn.Parameter(torch.zeros(num_objectives))
        self.optimizer = torch.optim.Adam([self.log_lambdas], lr=lr)
        self.initial_losses = {}  # stored on first valid step

    def weighted_sum(self, active_losses: dict) → Tensor
    def update(self, active_losses: dict, shared_layer_params) → None
    def state_dict() / load_state_dict()
```

### Training Loop Integration

```python
# 1. Prepare masked input (if use_masked_lm)
masked_seq, mask_positions, true_ids = apply_masking(seq, mask_prob, MASK_ID)

# 2. Transformer forward
all_hidden = model.transformer(masked_seq, metadata, return_all_hidden=True)
pooled = extract_last_nonpad(all_hidden)

# 3. Per-objective MoE routing + loss
active_losses = {}
for obj_idx, name in enumerate(OBJECTIVE_ORDER):
    if not objective_active(name, config, batch_idx):
        continue
    Z_ut, r_ut = model.moe(all_hidden, pooled, obj_idx)
    active_losses[name] = compute_loss(name, Z_ut, pooled, batch, model, config)

# 4. Weighted combination
L_total = gradnorm.weighted_sum(active_losses) if use_gradnorm else static_sum(...)

# 5. Backward + GradNorm update
L_total.backward(retain_graph=use_gradnorm)
if use_gradnorm:   # every step -- see note below
    gradnorm.update(active_losses, anchor_params)
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()

# REV-121 (owner 2026-09-17): this pseudocode previously gated the controller on
# `step % gradnorm_update_interval == 0`. No such field has ever existed in `Config`
# and no such gating has ever existed in `training/pretrain.py` -- lambda is updated on
# every step, which is what FR-5's "Per update" wording implies. The clause is removed
# rather than implemented: updating lambda every N steps would change the algorithm, and
# nothing has asked for that. Since REV-120 removed ~275 ms/step from the controller, an
# update interval is also no longer an attractive performance lever.
```

### Per-Objective Loss Computation

```
next_item:   SampledSoftmaxLoss(Z_ut[:, -1, :], target, neg_ids, item_emb_weight)
masked_item: MaskedItemLoss(Z_ut, mask_positions, true_ids, item_emb_weight)
multi_step:  MultiStepLoss(Z_ut, targets_k, item_emb_weight)
contrastive: InfoNCELoss(normalize(Z_ut[:, -1, :]), normalize(Z_aug[:, -1, :]))
```

Where Z_ut is the MoE-routed representation for each objective (or raw hidden states when moe_variant="none").

## Key Design Decisions

### D1: All Losses in One File
Centralizes the contract between the model's output format and the loss computation. Easier to verify consistency (all use the same tied weights, same ID conventions, same masking logic).

### D2: Multi-Step Causal Position Rule
Step k reads `all_hidden_states[:, S-1-k, :]` — fixed positions under left-padding. This prevents information leakage: position S-1 has seen items up to t, so predicting t+2 from it would be cheating. Position S-1-k has only seen up to t-k, which is correct.

### D3: Contrastive Interval Gating
InfoNCE requires a second full forward pass. Computing it every batch doubles training time. The interval (default 4) amortizes this cost. On non-interval batches, contrastive is a zero tensor NOT passed to GradNorm — it's completely excluded from the active set, not included as a zero-valued loss.

### D4: GradNorm Active-Only Operation
GradNorm only operates on losses that are genuinely active in the current step. This means the active set can change per-batch (contrastive enters/exits). The lambda renormalization constraint (`sum == num_active`) applies only to active lambdas. Inactive lambdas are not zeroed — they retain their last value but contribute nothing.

### D5: GradNorm Anchor Selection
The anchor is the last shared Transformer block's parameters (before MoE). This guarantees the anchor exists for all variants including `moe_variant="none"`. Using expert parameters as anchor would break the "none" variant.

### D6: Separate GradNorm Optimizer
GradNorm's `log_lambdas` have their own Adam optimizer, separate from the main model optimizer. This prevents the lambda learning dynamics from interfering with model parameter updates (different learning rate, no weight decay on lambdas).

### D7: Factored G_t — no second-order differentiation (REV-120, owner decision)

**Decision:** `G_t` is computed as `λ_t · ‖∇_W L_t‖` with the norm detached, not as
`‖∇_W(λ_t · L_t)‖` built with `create_graph=True`. `create_graph=True` is forbidden in the
GradNorm path.

**Why it is the same algorithm.** λ_t is a scalar held in `log_lambdas`, a parameter with
no dependence on the anchor parameters W. Gradients are linear in the objective, so

```
∇_W(λ_t · L_t) = λ_t · ∇_W L_t        ⟹        G_t = λ_t · ‖∇_W L_t‖
```

exactly — not an approximation. `G*_t` is already fully detached (`G_mean` comes from
`G_t.detach().mean()`, `Lhat` from detached losses), so nothing else in `gradnorm_loss`
carries a λ dependence. Verified numerically before adopting: G_t and `∂L_grad/∂log λ` are
**bit-identical** between the two forms (max difference 0.0) across all-active,
partially-active, and `G_t == G*_t` inputs.

**Why it matters.** Differentiating `‖∇_W(λ_t L_t)‖` w.r.t. λ means differentiating a graph
whose nodes are themselves derivative computations, i.e. second-order kernels over the
anchor block, once per active objective. Consequences, all measured on t-small at B=256:

| | |
|---|---|
| λ-gradient pass, `create_graph=True` | 276.4 ms |
| λ-gradient pass, factored form | 0.45 ms |
| share of a 605 ms training step recovered | ~275 ms (45%) |

Building the graph is nearly free (160.1 ms vs 159.3 ms for the four anchor-local
backwards); the entire cost is paid when that graph is traversed. Beyond speed, the
second-order form **cannot run at all** where the attention kernel lacks a second
derivative — `aten::_scaled_dot_product_flash_attention_for_cpu_backward` is not
implemented, so every CPU-only host failed on the first optimizer step — and its support
on the CUDA/FlashAttention-4 path was never verified on real hardware.

**Two admissible implementations of the λ gradient**, both conforming:

1. `torch.autograd.grad(gradnorm_loss, [log_lambdas])`. The graph is now a scalar multiply
   on a `num_active`-element tensor; costs ~0.45 ms. Preferred for maintainability: it
   stays correct if this document ever changes the `gradnorm_loss` expression.
2. Closed form, `∂L_grad/∂log λ_t = sign(G_t - G*_t) · G_t`, scattered into a
   `zeros_like(log_lambdas)` buffer at the active indices. Costs ~0. Preferred if the goal
   is a structural guarantee that no graph exists in the controller at all.

**Trap to avoid in either case.** `self.lambdas[idx] * grad_norm` still builds a graph
through `log_lambdas`. Assigning a tensor derived from it straight to `log_lambdas.grad`
is silently accepted by PyTorch (`requires_grad=True`, `grad_fn=CopySlices`) and keeps
that graph alive across steps. Detach before assigning, or build `G_t` detached and
compute `gradnorm_loss` for logging only.

**Unchanged by this decision:** the post-step renormalization (it runs under `no_grad()`
after `optimizer.step()` and never saw the λ gradient), the active-only semantics of D4,
the anchor choice of D5, and the separate optimizer of D6.

**Consequence for the attention backend.** Double-backward support stops being a GradNorm
prerequisite. `probe_double_backward` and `DoubleBackwardUnsupportedError` in
`training/gradnorm.py`, and the `probe_double_backward(model, config)` call in
`run_pretrain`, become dead code and should be removed rather than left as a check of an
invariant the code no longer relies on.

## Affected Components

- Consumed by `training/pretrain.py` (training loop integration)
- `data/augmentation.py` provides augmented sequences for contrastive loss
- `models/moe.py` provides Z_ut for each objective
- `models/transformer.py` provides `item_embedding.weight` for tied projections
- GradNorm state saved/loaded with checkpoints

## Testing Strategy

- Sampled softmax: verify loss decreases when positive score increases
- MIM: verify only masked positions contribute (mask all → all contribute, mask none → loss 0.0)
- Multi-step: verify positions S-1-step are used (inject known values at those positions)
- InfoNCE: verify symmetric similarity matrix, self-similarity masked, correct labels
- GradNorm lambda sum: verify constraint holds after update (within tolerance 1e-4)
- GradNorm with single active objective: lambda = 1.0 trivially
- GradNorm λ-gradient equivalence (AC-11): assert the shipped λ gradient equals
  `torch.autograd.grad(gradnorm_loss, [log_lambdas])` on (a) all objectives active,
  (b) contrastive inactive, (c) `G_t == G*_t` exactly — the subgradient boundary, where
  `abs()` backward and `torch.sign` both return 0. Also assert
  `log_lambdas.grad.requires_grad is False` after `update()`
- GradNorm first-order-only (AC-10): assert `create_graph` appears nowhere in
  `training/gradnorm.py`, and run one full `update()` with `config.device = "cpu"` — the
  configuration that previously raised on the missing SDPA CPU double-backward
- Augmentation: verify random_crop returns valid subsequence, random_mask uses MASK_ID not num_items
