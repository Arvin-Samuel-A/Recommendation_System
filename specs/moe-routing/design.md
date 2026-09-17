# MoE Routing — Design

## Relevant Existing Architecture

Greenfield. Design follows Prompt_v6.2 Steps 8.5–8.6.

## Files Created / Modified

| File | Action | Role |
|---|---|---|
| `models/moe.py` | Create | Expert, SharedExperts, ARVRouter, ConventionalMMoEGates, MoELayer, route_objective, count_router_parameters |

## Proposed Architecture

### Component Hierarchy

```
MoELayer
├── SharedExperts (K Expert FFNs)
├── [ARVRouter] (when variant="arv_mmoe")
│   └── nn.Linear(hidden_dim + emb_dim, num_experts)
├── [ConventionalMMoEGates] (when variant="mmoe")
│   └── nn.ModuleList of T nn.Linear(hidden_dim, num_experts)
└── [ObjectiveEmbeddings] (when variant="arv_mmoe")
    └── nn.Embedding(4, objective_embedding_dim)
```

### Forward Flow

```
MoELayer.forward(all_hidden_states, pooled_hidden, objective_idx):

  if variant == "none":
      return (all_hidden_states, None)

  expert_outputs = SharedExperts(all_hidden_states)  # (B, S, K, D) — computed ONCE

  if variant == "arv_mmoe":
      e_t = objective_embeddings.weight[objective_idx]
      r_ut = ARVRouter(pooled_hidden, e_t)                  # (B, K)
  elif variant == "mmoe":
      r_ut = ConventionalMMoEGates(pooled_hidden, objective_idx)  # (B, K)

  Z_ut = (r_ut[:, None, :, None] * expert_outputs).sum(dim=2)  # (B, S, D)
  return (Z_ut, r_ut)
```

### Expert Sharing Pattern

The training loop calls `MoELayer.forward` once per active objective per batch:

```python
for obj_idx, name in enumerate(OBJECTIVE_ORDER):
    if not active(name):
        continue
    Z_ut, r_ut = moe(all_hidden, pooled, obj_idx)
    losses[name] = compute_loss(name, Z_ut, ...)
    routing_buffer[name].append(r_ut.detach())
```

**Critical:** `SharedExperts.forward()` is called inside `MoELayer.forward`. To avoid recomputing experts for each objective, the design caches expert outputs within a single batch's forward pass:

```python
# Inside MoELayer, expert outputs are computed on first call and cached
if self._cached_expert_outputs is None:
    self._cached_expert_outputs = self.experts(all_hidden_states)
expert_outputs = self._cached_expert_outputs
```

The cache is cleared at the start of each new batch (or by an explicit `clear_cache()` call).

### ARV Router Detail

```
Input: [pooled_hidden (B, D_h) ; objective_emb (D_e)] → concat → (B, D_h + D_e)
  → Linear(D_h + D_e, K) → Softmax → r_ut (B, K)

Broadcast: r_ut (B, 1, K, 1) * expert_outputs (B, S, K, D) → sum(dim=2) → Z_ut (B, S, D)
```

For `router_conditioning = "input_only"` (Ablation B): the Linear input dimension is `D_h` only, objective embedding is not concatenated.

### Conventional MMoE Detail

```
Per-objective gate t: pooled_hidden (B, D_h) → Linear_t(D_h, K) → Softmax → r_ut (B, K)
```

T independent gates, no shared parameters between them, no objective embedding.

## Key Design Decisions

### D1: Cache Expert Outputs
Computing K FFN passes per forward is expensive. Since experts are shared and position-wise, their outputs are identical across objectives for the same input. Caching avoids redundant computation. The cache lives within a single forward pass (one batch) and is invalidated between batches.

### D2: Broadcast Routing
Routing is computed from the pooled representation (one distribution per user), then broadcast across positions. This is the paper's formulation: routing is user-adaptive but not position-adaptive. It keeps the routing computation cheap and avoids a quadratic routing matrix.

### D3: e_rec ≡ e_next Identity
At inference, `objective_embeddings.weight[0]` is used directly. No separate `e_rec` parameter exists. This is a deliberate non-design: the paper defines e_rec as identical to e_next (Eq. 5–7), and any separate parameter would be untrained.

### D4: MoELayer as a Single Module
All three variants (`none`, `mmoe`, `arv_mmoe`) are behind the same `MoELayer.forward` interface. The training loop doesn't branch on variant — it calls MoELayer uniformly. This keeps the training loop clean and makes ablation switching purely config-driven.

## Affected Components

- Receives `all_hidden_states` and `pooled_hidden` from `models/transformer.py`
- Produces `Z_ut` consumed by each loss in `training/losses.py`
- Produces `r_ut` accumulated in routing diagnostics buffer (`evaluation/routing_diagnostics.py`)
- Parameter counts reported by `count_router_parameters()` consumed by efficiency report and Ablation C
- MoE parameters frozen during local/federated training (`training/local_train.py`, `training/federated.py`)

## Testing Strategy

- Shape tests: verify Z_ut is (B, S, D) and r_ut is (B, K) for all variants
- Routing differentiation: same input, different objectives → different r_ut (ARV-MMoE)
- User differentiation: different inputs, same objective → different r_ut (ARV-MMoE)
- Expert sharing: verify expert outputs computed once (cache hit on second objective call)
- Passthrough: `variant="none"` returns input unchanged, routing=None
- Parameter accounting: `count_router_parameters()` matches manual calculation
- Conventional MMoE: verify gates are independent (no shared params between objectives)
- Ablation B: `input_only` router has no objective embedding input dimension

### D-dropout: one dropout realisation is shared across all four objectives (REV-079)

`Expert` contains `nn.Dropout`, and FR-1 requires the K expert outputs to be computed
**once** per forward pass and reused by every objective. A direct consequence is that all
four objectives in a training step see the *same* dropout mask on the expert outputs; four
independent forwards would each draw their own.

This is correct and intended — the alternative contradicts FR-1 and G1 — but it must be
stated when the Ablation D gradient-interference numbers are reported. Part of the measured
inter-objective cosine similarity is attributable to the shared noise realisation rather
than to genuine representational conflict between the objectives.
