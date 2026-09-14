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
if use_gradnorm and step % gradnorm_update_interval == 0:
    gradnorm.update(active_losses, anchor_params)
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
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
- Augmentation: verify random_crop returns valid subsequence, random_mask uses MASK_ID not num_items
