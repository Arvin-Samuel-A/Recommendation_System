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

- [x] **T3.3: Record bucket populations and the partition in the long-tail artifact (REV-129 verification)** — REV-129 asks a verifier to confirm, against `longtail_{model}.json`, that it "reports head/mid/tail with the bucket populations, and the buckets partition the catalogue". Only the per-bucket metrics reached the file, so both facts had to be re-derived from `popularity_buckets.pkl` by hand. Add `bucket_populations` and `bucket_partition` to the saved report, leaving the returned/plotted `report` dict unchanged so `_plot_longtail` is unaffected. (FR-5, AC-04)
  - BUILD note: done. `_save_report` takes the buckets and writes `bucket_populations`
    plus a `bucket_partition` block carrying `disjoint`, `complete_union_size`, the
    nominal fractions and the realized fractions. The `report` dict that
    `run_longtail_evaluation` returns and plots is untouched — the new keys exist only in
    the JSON, so no plotting code sees a non-metric entry.
    Measured against the real artifacts: head=3,356 mid=3,357 tail=14,007, total 20,720 =
    `num_items`; pairwise disjoint TRUE; union == {1..20720}; pad and mask absent. **AC-04
    verified directly, not inferred.**

  - **OPEN QUESTION FOR THE SPEC OWNER — not a defect, deliberately not changed by BUILD.**
    The realized catalogue fractions are head 0.162 / mid 0.162 / tail 0.676, not FR-5's
    nominal 0.20 / 0.20 / 0.60. Cause: `compute_popularity_buckets` sets
    `n = len(sorted_items)`, which counts only the 16,781 items that HAVE training
    interactions; the 20/20/60 split is applied to those, and the 3,939 zero-training
    items are then appended to tail under FR-5's second clause.
    FR-5 carries three clauses — (1) "Head: top 20%, Mid: middle 20%, Tail: bottom 60%",
    (2) "Zero-training items classified as tail", (3) "All item IDs partitioned". The
    implementation satisfies (2) and (3) exactly and (1) relative to RANKED items, which
    is what the code comment at `longtail.py:22` states. The alternative reading — rank the
    whole catalogue, with zero-popularity items tying at the bottom — satisfies all three
    exactly and yields 20/20/60 of the catalogue.
    Both readings are defensible and no review has flagged this; the reviews mark AC-04
    PASS on the partition property alone, which holds either way. BUILD is not changing it
    because switching moves the bucket boundaries and changes the long-tail study's
    published numbers, which is a research decision for the owner (§9). The realized
    fractions are now recorded in the artifact so the choice is visible rather than
    implicit.

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

- [x] **T6.5: Measure gradient cosines and norms in fp32 (REV-128)** — Gradients are flattened and compared in the model's dtype, which is bf16 everywhere. A bf16 cosine of a ~790k-element vector against itself does not return 1.0: the real artifact's diagonal read `[1.00078, 1.00391, 0.99844, 1.00078]`, three entries **above 1**, outside the range a cosine similarity can take. Cast the flattened gradients to fp32 in `gradient_cosine_similarity`, and give `per_objective_grad_norms` the same treatment for consistency. Tighten the AC-07 test from `< 0.01` (wide enough to accept 1.0039) to `< 1e-4` and assert no entry exceeds 1.0, so the range violation cannot come back. Same fp32-at-the-measurement-site pattern `routing_diagnostics` and `local_train` already use. (FR-9, AC-07)
  - BUILD note: done and verified. `grads[name] = flat.float()` and
    `flat.float().norm(p=2)`; one temporary copy on a path that runs rarely.
    The AC-07 test now asserts each diagonal entry is within `1e-4` of 1.0 (measured
    error after the fix: 5e-7), asserts the matrix range with a documented `FP32_ULP =
    1e-5` tolerance — four orders of magnitude tighter than the bf16 excursion it guards
    against, while absorbing genuine fp32 round-off — and tightens the symmetry assertion
    from `< 0.01` to `< 1e-6`.
    Non-vacuity verified, not assumed: reverting `flat.float()` to `flat` makes the test
    FAIL with `diagonal entry 0 is 1.007812`. `tests/test_diagnostics.py` 22 passed.
    NOTE: `outputs/logs/gradient_diagnostics_t_small.json` and
    `gradient_cosine_similarity_t_small.png` on disk still carry the bf16 cosines. They are
    regenerated by the T6.3 pretrain re-run (`training-pipeline` T6.3), which is why this
    task lands BEFORE that run.

- [x] **T6.6: Memory-map the candidate matrix for row-sliced consumers (REV-119)** — `load_eval_candidates` did a plain `np.load` and fingerprinted the whole matrix. `test_candidates.npy` is (1,059,494 x 101) int64 = 856 MB. That is proportionate for `--stage evaluate` and long-tail, which iterate the whole split, but not for `run_local_personalization`, which scores a few thousand rows and paid a ~1 GB resident copy to reach them. Add `mmap` to `load_eval_candidates` / `load_or_build_eval_candidates` and use it at the row-sliced consumer; keep the eager path everywhere the whole matrix is consumed. (FR-1, AC-01)
  - BUILD note: done and verified. `mmap: bool = False` on both loaders (default unchanged,
    so every existing caller keeps the eager path); `training/local_train.py` passes
    `mmap=True`. The block-wise fingerprint (REV-087) reads a memmap without materialising
    it, so the mapped path still runs the same integrity check.
    `training/federated.py` was deliberately left eager: it does
    `torch.as_tensor(np.asarray(val_candidates))` over the whole matrix and the REV-117
    final round scores the full split, so it is a whole-matrix consumer, not a row-sliced
    one. `evaluation/longtail.py` likewise iterates the whole test split.
    One consequence handled: a read-only memmap makes `torch.as_tensor` alias a
    non-writable buffer (undefined behaviour on write, plus a UserWarning every run), so
    the per-row conversion in `local_train.py` now copies with `np.array` — 101 elements.
    Test: `test_stored_candidates_are_a_contiguous_matrix` extended to assert the mapped
    read returns a `np.memmap`, that its contents and row slices equal the eager read, and
    that fingerprint verification still RAISES on a corrupted meta file through the mapped
    path — so the mapped path cannot silently skip the integrity check.
    `tests/test_sequence_generator.py` 30 passed.

- [x] **T6.3: Implement gradient diagnostics execution** — Run every `grad_diag_interval_steps` on `grad_diag_num_batches` freshly-drawn batches (not optimizer-step batches). For each moe_variant under comparison. (FR-9)

- [x] **T6.4: Implement gradient diagnostics report and plots** — Save `outputs/logs/gradient_diagnostics_{model}.json`. Generate: gradient_cosine_similarity heatmap, gradient_norms bar chart with CV in title. (FR-9)

## Phase 7: Testing

- [x] **T7.1: Test candidate set consistency** — Load candidate sets from two "different model" evaluation calls → verify sets are identical. (AC-01)

- [x] **T7.2: Test routing diagnostics correctness** — Uniform routing → max entropy. Delta routing → zero entropy. Verify heatmap shape `(T, K)`. (AC-06)

- [x] **T7.3: Test gradient cosine symmetry** — Verify matrix is symmetric with 1s on diagonal. (AC-07)

- [x] **T7.4: Test long-tail bucket partition** — Verify head ∪ mid ∪ tail = all_items, all disjoint, correct percentiles. (AC-04)

## Phase 8: Stage Execution (REV-129)

- [x] **T8.1: Produce the missing evaluation artifacts and verify them against the files (REV-129)** — FR-4's cold-start report and plot, FR-5's long-tail report and plot, and FR-7's efficiency report had never been produced, leaving AC-08 half-met and two of the paper's research questions with no measured number behind them. Run the stages against the REGENERATED checkpoint (after `loss-functions` REV-105) with REV-128 already landed, and verify against the artifacts. (FR-4, FR-5, FR-7, AC-08)
  - BUILD note: all produced and verified; every stage exited 0.

    **Correction to REV-129's stated procedure — it could not be followed as written:**
    1. It says the T6.3 re-run "regenerates `best.pt` AND the gradient diagnostics with
       fp32 cosines, in one pass". It does not. `grad_diag_interval_steps = 500` and the
       gate runs 200 steps, so the in-training hook at `training/pretrain.py:810-814`
       never fires. The gradient artifact needs `--stage gradient_diagnostics`.
    2. That stage — and `--stage routing_diagnostics` — were DEAD: `_load_trained_model`
       referenced an unimported `load_metadata_matrix` and raised `NameError` before
       constructing a model. Fixed under `ablations-cli` T1.6.
    3. It says to run `--stage evaluate` and then verify `cold_start_*`, `longtail_*` and
       `efficiency_*`. `--stage evaluate` writes only `test_metrics_*.json`; the other
       three come from the separate `cold_start`, `longtail` and `latency` stages, which
       were run individually.

    **Artifacts verified against the files:**
      * `test_metrics_t_small.json` — NDCG@10 0.09307, MRR 0.06675, Recall 0.18102,
        Precision 0.01810 over the full test split.
      * `cold_start_t_small.json` + `cold_vs_warm_t_small.png` — all FOUR cells present:
        cold/warm x metadata ON/OFF. Cold 0.05774 (on) vs 0.05485 (off); warm 0.09286 (on)
        vs 0.08093 (off).
      * `longtail_t_small.json` + `longtail_t_small.png` — head/mid/tail metrics plus the
        bucket populations added by T3.3. **AC-04 verified directly against the buckets:**
        3,356 + 3,357 + 14,007 = 20,720 = `num_items`, pairwise disjoint, union == {1..N},
        pad and mask absent. head NDCG 0.09937 > mid 0.04793 > tail 0.04597.
      * `efficiency_t_small.json` — every FR-7 field, including
        **`timed_metadata_source == "item_metadata"`**, which confirms REV-110's fix on a
        real run for the first time (REV-110 was closed on the diff alone). 1.824 ms/rec,
        9,800,964 params, 18.69 MB bf16, 463,936,512 FLOPs/inference, routing overhead
        0.438 ms.
      * `gradient_diagnostics_t_small.json` + 2 plots — regenerated with the T6.5 fp32 fix:
        diagonal `[1.00000107, 1.00000095, 1.00000095, 1.00000179]`, max deviation from
        1.0 now **1.79e-06** against the previous bf16 **3.91e-03**, matrix exactly
        symmetric, no entry outside range. **REV-128 closed on the artifact, not only in
        unit test.** Off-diagonal interference: max 0.0612 (next_item/masked_item).
    AC-08 is now met for t-small. Note the routing artifact reports expert utilization
    `[0.065, 0.311, 0.395, 0.229]` against uniform 0.25 — the owner observation carried
    from the prior review, unchanged and still from a capped run.
