# Evaluation & Diagnostics — Requirements

## Overview

Implement the evaluation protocol (fixed candidate sets, four metrics), cold-start evaluation, long-tail evaluation, latency benchmarking, routing diagnostics, and gradient-interference diagnostics.

## Problem Statement

Fair model comparison requires identical evaluation infrastructure. Routing and gradient diagnostics must measure — not assume — whether ARV-MMoE and GradNorm actually differentiate expert allocation and reduce gradient interference. All diagnostics must use declared diagnostic data, not optimizer-step batches.

## Goals

- G1: Fixed candidate protocol ensuring identical evaluation across all models
- G2: Four standard metrics: NDCG@10, MRR@10, Recall@10, Precision@10
- G3: Cold-start evaluation with metadata ON/OFF comparison
- G4: Long-tail evaluation across head/mid/tail popularity buckets
- G5: Latency benchmarking with mandatory device synchronization
- G6: Routing diagnostics: entropy, utilization, objective differentiation, collapse tests
- G7: Gradient-interference diagnostics: cosine similarity, gradient norms, convergence behavior

## Non-Goals

- Online A/B testing infrastructure
- User-facing recommendation serving system
- Real-time metric dashboards

## Functional Requirements

### FR-1: Fixed Candidate Protocol (Step 18.1)
- Pre-stored candidate sets generated before model comparison
- Each set: exactly one ground-truth target + configured negatives after filtering
- Negatives exclude: target, padding ID, mask ID, items prohibited by eval-history rule
- Same candidate IDs for Transformer, MMoE, and ARV-MMoE
- Score candidates with full model serving path; rerank by model score

### FR-2: ANN Retrieval (Optional, Step 18.1)
- When `use_ann=True`: retrieve `ann_candidates` items, filter invalid/seen, force-include target
- Replace lowest-scoring candidate if target not in initial retrieval
- Report fraction of examples requiring target force-inclusion

### FR-3: Metrics (Step 18.2)
- NDCG@10: `1/log2(rank+1)` if rank ≤ 10, else 0
- MRR@10: `1/rank` if rank ≤ 10, else 0
- Recall@10: 1 if rank ≤ 10, else 0 (= HR@10 for single-target protocol)
- Precision@10: `1/10` if rank ≤ 10, else 0
- Return per-example values plus mean and std
- *Amended 2026-10-16 by `specs/eval-integrity`:*
  - Ties are scored as the expectation under random tie-breaking. The REV-076 rule, which gave
    ties to the target, is superseded (eval-integrity FR-2).
  - Candidate scores are computed in fp32 outside autocast (FR-1).
  - Every reported result carries a popularity reference (FR-7).
- Multi-seed final results: mean ± std across independent seeds (not repeated evaluation of one seed)

### FR-4: Cold-Start Evaluation (Step 19)
- Cold items from Step 4.3 protocol (first observation after training cutoff)
- Assert no cold item in training
- Evaluate with metadata ON and metadata OFF
- Classify each test target as cold or warm
- Report metrics separately for cold/warm × metadata ON/OFF
- Save to `outputs/logs/cold_start_{model}.json`
- Plot: `outputs/plots/cold_vs_warm_{model}.png` — grouped bar chart

### FR-5: Long-Tail Evaluation (Step 20)
- Popularity from training set only
- Head: top 20%, Mid: middle 20%, Tail: bottom 60%
- Zero-training items classified as tail
- All item IDs partitioned (disjoint, complete)
- Report metrics per bucket
- Save to `outputs/logs/longtail_{model}.json`
- Plot: `outputs/plots/longtail_{model}.png` — bar chart (3 groups × 4 metrics)

### FR-6: Latency Benchmarking (Step 21)
- Warmup runs (default 20) discarded
- Timed runs (default 100) with mandatory device synchronization before and after each
- MPS: `torch.mps.synchronize()`; CUDA: `torch.cuda.synchronize(device)`
- Synthetic single-sample input; full serving path including ARV routing when applicable
- Report: mean_ms, std_ms, min_ms, p95_ms, ms_per_recommendation
- Routing overhead: difference between full model and `moe_variant="none"` with same backbone

### FR-7: Efficiency Report (Step 21.2)
- Save to `outputs/logs/efficiency_{model}.json`
- Contents: total_parameters, trainable_parameters, lora_parameters, router_parameters, expert_parameters, moe_variant, num_experts, model_size_fp32_mb, model_size_bf16_mb, flops_per_inference, latency stats, routing overhead

### FR-8: Routing Diagnostics (Step 21.5)
- Routing entropy: per-user and aggregate, per-objective, per-moe-variant
- Expert utilization: mean routing mass per expert
- Objective-specific routing differences: pairwise cosine similarity and KL divergence
- Objective × expert heatmap: `(num_objectives, num_experts)` matrix of mean routing weights
- Collapse tests: fraction near-uniform (CV threshold), fraction single-expert dominated
- Run every `routing_diag_interval_epochs` during training + comprehensive pass on full validation set at end
- Save to `outputs/logs/routing_diagnostics_{model}.json`
- Plots: routing_entropy, expert_utilization (with 1/K reference line), objective_expert_heatmap

### FR-9: Gradient-Interference Diagnostics (Step 21.6)
- Gradient cosine similarity: `(T, T)` matrix of cos(grad L_i, grad L_j) w.r.t. shared params
- Per-objective gradient norms with coefficient of variation
- Computed on diagnostic batches (not optimizer-step batches)
- Computed every `grad_diag_interval_steps` on `grad_diag_num_batches` fresh batches
- For each moe_variant under comparison
- Save to `outputs/logs/gradient_diagnostics_{model}.json`
- Plots: gradient_cosine_similarity heatmap, gradient_norms bar chart with CV in title

### FR-10: Calibration (Step 9)
- Temperature scaling optional, NOT a ranking correction
- `use_logit_calibration=False` and `eval_temperature=1.0` by default
- If temperature learned: fit on validation only, freeze before test
- Never fit on test data; never claim temperature changes ranking

## Edge Cases

- Test sample with target not in candidate set (ANN): force-include target
- Cold item with zero metadata: zero vector, not a learned missing signal
- Zero-training items in long-tail: classified as tail
- moe_variant="none": no routing entropy/utilization (skip routing diagnostics)
- Single objective active: gradient cosine matrix is 1×1 (trivial)

## Constraints

- Fixed evaluation candidate sets generated once and reused — never regenerated per model
- Diagnostics use declared diagnostic data, never optimizer-step batches
- Do not interpret low entropy as proof of usefulness — report without editorializing
- Report measured numbers; let variant comparison speak for itself

## Dependencies

- Pretrained models from `training/pretrain.py`
- Fixed candidate sets from `data/sequence_generator.py`
- Cold/warm item sets and popularity buckets from preprocessing
- `configs/config.py` for all parameters

## Acceptance Criteria

- **AC-01:** Same candidate sets used for all model evaluations (verified by set equality)
- **AC-02:** NDCG/MRR/Recall/Precision compute correctly on known toy inputs
- **AC-03:** Cold items verified disjoint from training data in cold-start eval
- **AC-04:** Long-tail buckets partition all item IDs (disjoint and complete)
- **AC-05:** Latency measurement includes device synchronization (verified in code)
- **AC-06:** Routing diagnostics produce correct heatmap shape `(num_objectives, num_experts)`
- **AC-07:** Gradient cosine similarity matrix is symmetric with 1s on diagonal
- **AC-08:** All JSON reports and PNG plots generated in correct locations
- **AC-09:** ANN force-inclusion rate reported when `use_ann=True`
