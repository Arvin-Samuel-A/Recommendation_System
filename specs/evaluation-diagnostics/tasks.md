# Evaluation & Diagnostics — Tasks

Ordered tasks. Depends on training-pipeline (for pretrained models and routing buffers) and data-pipeline (for candidate sets, cold/warm sets, popularity buckets).

## Phase 1: Core Metrics

- [x] **T1.1: Implement metric functions in `evaluation/metrics.py`** — `compute_metrics(scores, target_idx, k=10)` returning dict with ndcg, mrr, recall, precision per the single-target formulas. `aggregate_metrics(per_example_metrics)` returning mean ± std. (FR-3, AC-02)

- [x] **T1.2: Implement evaluation loop in `evaluation/metrics.py`** — `evaluate_model(model, dataset, candidate_sets, config)`: for each example, load pre-stored candidate set, score via full model serving path (including MoE routing with e_next when applicable), compute metrics, aggregate. Use `eval_user_history` for filtering. (FR-1)

- [x] **T1.3: Implement ANN retrieval path** — When `use_ann=True`: retrieve `ann_candidates` items from `models/ann_retrieval.py`, filter invalid/seen, force-include target (replace lowest if needed), rerank with full model. Report force-inclusion fraction. (FR-2)

- [x] **T1.4: Implement calibration support** — When `use_logit_calibration=True` and temperature != 1.0: divide logits by temperature before scoring. Temperature fit on validation only. Default: disabled. (FR-10)

- [x] **T1.5: Test metrics** — Unit test: known ranking → verify exact NDCG/MRR/Recall/Precision values. Edge cases: target at rank 1, rank 10, rank 11 (miss). (AC-02)

## Phase 2: Cold-Start Evaluation

- [x] **T2.1: Implement `evaluation/cold_start.py`** — Load `cold_item_set` and `warm_item_set`. Assert cold items disjoint from training. For each test example: classify target as cold/warm. Run evaluation with metadata ON and metadata OFF. Report metrics separately for cold/warm × ON/OFF. (FR-4, AC-03)

- [x] **T2.2: Implement cold-start report and plot** — Save `outputs/logs/cold_start_{model}.json`. Generate `outputs/plots/cold_vs_warm_{model}.png` — grouped bar chart comparing cold/warm × metadata ON/OFF for 4 metrics. (FR-4)

## Phase 3: Long-Tail Evaluation

- [x] **T3.1: Implement `evaluation/longtail.py`** — Compute popularity from training set. Assign buckets: head (top 20%), tail (bottom 60%), mid (remaining). Zero-training items → tail. Assert partition: disjoint, complete over all item IDs. Save `data/processed/popularity_buckets.pkl`. (FR-5, AC-04)

- [x] **T3.2: Implement long-tail evaluation and plotting** — For each test example: classify target by bucket. Report metrics per bucket. Save `outputs/logs/longtail_{model}.json`. Generate `outputs/plots/longtail_{model}.png` — bar chart (3 groups × 4 metrics). (FR-5)

## Phase 4: Latency Benchmarking

- [x] **T4.1: Implement `benchmark_latency` in `evaluation/flops.py`** — Warmup runs (config.warmup_runs), timed runs (config.timed_runs) with mandatory device sync before and after each. Full serving path including MoE routing. Return: mean_ms, std_ms, min_ms, p95_ms, ms_per_recommendation. (FR-6, AC-05)

- [x] **T4.2: Implement routing overhead measurement** — Benchmark full model and `moe_variant="none"` copy with same backbone. Overhead = difference. (FR-6)

- [x] **T4.3: Implement efficiency report** — Compute total_parameters, trainable_parameters, lora_parameters, router_parameters, expert_parameters, model sizes. Save `outputs/logs/efficiency_{model}.json`. (FR-7)

## Phase 5: Routing Diagnostics

- [x] **T5.1: Implement `routing_entropy` in `evaluation/routing_diagnostics.py`** — `-(r * log(r.clamp(1e-12))).sum(dim=-1)`. Report mean overall, per-objective, per-moe-variant. (FR-8)

- [x] **T5.2: Implement `expert_utilization`** — `r.mean(dim=0)` → per-expert mass fraction. (FR-8)

- [x] **T5.3: Implement `objective_routing_diff`** — Pairwise cosine similarity and KL divergence on mean routing distributions per objective. (FR-8)

- [x] **T5.4: Implement collapse tests** — Fraction near-uniform (CV < threshold), fraction single-expert dominated (max > threshold). Per-objective utilization mean/std. (FR-8)

- [x] **T5.5: Implement routing diagnostics report and plots** — Save `outputs/logs/routing_diagnostics_{model}.json`. Generate: routing_entropy bar chart, expert_utilization bar chart with 1/K line, objective_expert_heatmap (T × K matrix). (FR-8, AC-06)

## Phase 6: Gradient Diagnostics

- [x] **T6.1: Implement `gradient_cosine_similarity` in `evaluation/gradient_diagnostics.py`** — `(T, T)` matrix using `torch.autograd.grad` with `retain_graph=True, allow_unused=True`. Flatten per-objective gradients, compute pairwise cosine sim. (FR-9, AC-07)

- [x] **T6.2: Implement `per_objective_grad_norms`** — Norm of flattened gradients per objective + coefficient of variation. (FR-9)

- [x] **T6.3: Implement gradient diagnostics execution** — Run every `grad_diag_interval_steps` on `grad_diag_num_batches` freshly-drawn batches (not optimizer-step batches). For each moe_variant under comparison. (FR-9)

- [x] **T6.4: Implement gradient diagnostics report and plots** — Save `outputs/logs/gradient_diagnostics_{model}.json`. Generate: gradient_cosine_similarity heatmap, gradient_norms bar chart with CV in title. (FR-9)

## Phase 7: Testing

- [x] **T7.1: Test candidate set consistency** — Load candidate sets from two "different model" evaluation calls → verify sets are identical. (AC-01)

- [x] **T7.2: Test routing diagnostics correctness** — Uniform routing → max entropy. Delta routing → zero entropy. Verify heatmap shape `(T, K)`. (AC-06)

- [x] **T7.3: Test gradient cosine symmetry** — Verify matrix is symmetric with 1s on diagonal. (AC-07)

- [x] **T7.4: Test long-tail bucket partition** — Verify head ∪ mid ∪ tail = all_items, all disjoint, correct percentiles. (AC-04)
