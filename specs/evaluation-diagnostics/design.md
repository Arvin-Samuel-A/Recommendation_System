# Evaluation & Diagnostics — Design

## Relevant Existing Architecture

Greenfield. Design follows Prompt_v6.2 Steps 9, 18–21.6.

## Files Created / Modified

| File | Action | Role |
|---|---|---|
| `evaluation/metrics.py` | Create | NDCG@10, MRR@10, Recall@10, Precision@10, candidate scoring, calibration |
| `evaluation/cold_start.py` | Create | Cold-start evaluation protocol |
| `evaluation/longtail.py` | Create | Long-tail popularity bucket evaluation |
| `evaluation/flops.py` | Create | Latency benchmarking, parameter counting, efficiency report |
| `evaluation/routing_diagnostics.py` | Create | Entropy, utilization, collapse, heatmap |
| `evaluation/gradient_diagnostics.py` | Create | Gradient cosine similarity, gradient norms |

## Proposed Architecture

### Metric Computation

```python
def compute_metrics(scores, target_idx, k=10):
    """
    scores: (num_candidates,) — model scores for candidate items
    target_idx: int — index of the ground-truth item in the candidate list
    Returns: {ndcg, mrr, recall, precision}
    """
    ranked = scores.argsort(descending=True)
    rank = (ranked == target_idx).nonzero().item() + 1  # 1-indexed
    return {
        'ndcg': 1.0 / math.log2(rank + 1) if rank <= k else 0.0,
        'mrr': 1.0 / rank if rank <= k else 0.0,
        'recall': 1.0 if rank <= k else 0.0,
        'precision': 1.0 / k if rank <= k else 0.0,
    }
```

### Evaluation Loop

```python
def evaluate_model(model, eval_dataset, candidate_sets, config):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for example in eval_dataset:
            candidates = candidate_sets[example.id]  # pre-stored
            scores = model.score(example.seq, candidates, ...)  # full serving path
            metrics = compute_metrics(scores, target_idx=0)  # target is first in set
            all_metrics.append(metrics)
    return aggregate(all_metrics)  # mean ± std
```

### Routing Diagnostics Architecture

```python
# Entropy: -(r * log(r)).sum(dim=-1) → per-user/per-example
# Utilization: r.mean(dim=0) → per-expert mass
# Objective diff: pairwise cosine_sim and KL on mean routing per objective
# Collapse: CV < threshold → uniform; max(r) > threshold → single-expert
# Heatmap: (T, K) matrix of mean routing weights
```

Consumes routing buffers accumulated during training (Step 15.4.1) and/or recomputed in a dedicated evaluation pass on the validation set.

### Gradient Diagnostics Architecture

```python
def gradient_diagnostics(model, diagnostic_loader, shared_params, config):
    cos_sim_matrices = []
    norm_records = []
    for batch in diagnostic_loader:  # grad_diag_num_batches fresh batches
        losses = compute_per_objective_losses(model, batch)
        C = gradient_cosine_similarity(losses, shared_params)
        norms = per_objective_grad_norms(losses, shared_params)
        cos_sim_matrices.append(C)
        norm_records.append(norms)
    return aggregate(cos_sim_matrices, norm_records)
```

Uses `torch.autograd.grad` with `retain_graph=True` and `allow_unused=True` to compute per-objective gradients without interfering with training.

### Latency Benchmarking

```python
def benchmark_latency(model, config, device, feature_dim):
    model.eval()
    dummy_seq = torch.zeros(1, config.max_seq_length, dtype=torch.long, device=device)
    dummy_meta = torch.zeros(1, config.max_seq_length, feature_dim,
                             dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(config.latency_warmup_runs):
        with torch.no_grad():
            model(dummy_seq, metadata=dummy_meta)

    # Timed runs with mandatory synchronization
    synchronize(device)
    times = []
    for _ in range(config.latency_timed_runs):
        synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy_seq, metadata=dummy_meta)
        synchronize(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return stats(times)
```

### Routing Overhead Measurement

Run `benchmark_latency` twice:
1. Full model (with MoE routing)
2. Same backbone with `moe_variant="none"`

Overhead = full_latency - none_latency.

## Key Design Decisions

### D1: Pre-Stored Candidate Sets
Candidate sets are generated once (in the data pipeline) and loaded by every model evaluation. This guarantees identical comparison conditions — the metric comparison between models is never confounded by different candidate sampling.

### D2: Diagnostic Batches Separate from Training
Gradient diagnostics use freshly-drawn batches, never the same batch used for the optimizer step. This prevents diagnostic computation from perturbing training gradients (retain_graph interactions).

### D3: Report, Don't Editorialise
Routing diagnostics measure whether the router differentiates — they don't prove the differentiation is useful. Low entropy or high expert utilization imbalance are reported as-is. Similarly, gradient cosine similarity measures direction, not magnitude — negative entries are evidence of conflicting gradients, but non-negative entries don't prove absence of interference.

### D4: Device Synchronization is Non-Negotiable
Both MPS and CUDA execute kernels asynchronously. Without `synchronize()` before/after timing, the measured latency is the time to enqueue the kernel, not to execute it. This would undercount latency by 10-100x.

## Affected Components

- Depends on: pretrained models, fixed candidate sets, cold/warm item sets, popularity buckets
- Used by: `training/pretrain.py` (validation), `training/ablation_runner.py` (all ablations), `main.py` (standalone evaluation stages)
- Routing diagnostics consume buffers from `training/pretrain.py`

## Testing Strategy

- Metric correctness: known ranking input → verify NDCG/MRR/Recall/Precision values
- Candidate set consistency: verify same sets loaded for different model evaluations
- Cold-start: verify cold items disjoint from training
- Long-tail: verify bucket partition (disjoint, complete, correct percentiles)
- Latency: verify synchronize calls present (code inspection); verify timing > 0
- Routing entropy: uniform distribution → max entropy; delta distribution → zero entropy
- Gradient cosine: identical losses → similarity 1.0; orthogonal gradients → similarity 0.0
