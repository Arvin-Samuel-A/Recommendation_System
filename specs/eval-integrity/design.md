# Evaluation Integrity — Design

## Relevant Existing Architecture (inspected at `c094b2c`)

- **Autocast today.**
  - `utils.autocast_context(config)` returns `torch.autocast(device_type=get_device(config).type,
    dtype=resolve_dtype(config), enabled=config.mixed_precision)`, with the default
    `cache_enabled=True`.
  - `main.py::main` (~line 1166) enters it once around `STAGE_MAP[args.stage](config, args)`. This
    is the stage-wide region.
  - `training/pretrain.py::train_one_step` asserts `torch.is_autocast_enabled(device.type)` under
    mixed precision, so it relies on that region.
  - `scripts/estimate_training_time.py` wraps each `train_one_step` and its validation timing in
    its own `autocast_context`.
  - `training/losses.py:179` disables autocast locally for InfoNCE.
- **Scoring.**
  - `evaluation/metrics.py::score_candidates_batch` runs forward, then the optional MoE, then
    gathers candidate rows, `l2_normalize`s both sides and takes `einsum("bd,bcd->bc")`.
  - `models/transformer.py::l2_normalize` normalises in fp32 but **casts back to the input
    dtype**, so a bf16 user embedding stays bf16, and the `einsum` runs under the caller's
    autocast in bf16.
  - Every evaluation caller goes through `score_candidates_batch`, either via `evaluate_model`
    (`run_pretrain` validation, `--stage evaluate`, ablation test, `evaluate_user`, federated) or
    directly (`cold_start.py:138`, `longtail.py:169`).
  - The one other scoring site is `models/ann_retrieval.py::retrieve_candidates`. It already does
    `user_emb.float() @ item_embs.float().T`, but its `_user_embeddings` normalises a bf16 tensor
    back to bf16 first.
- **Metrics.** `compute_metrics_batch` sets `rank = 1 + (scores > target).sum(1)`, so ties go to
  the target (REV-076). `compute_metrics` wraps it.
- **Forward sites.** `grep "model(\|return_all_hidden"` outside tests finds:
  - `pretrain.train_one_step` (two forwards on contrastive steps);
  - `pretrain.collect_validation_routing`;
  - `metrics.score_candidates_batch`;
  - `gradient_diagnostics.run_gradient_diagnostics` (two);
  - `flops.benchmark_latency` (warm-up and timed);
  - `ann_retrieval._user_embeddings`;
  - `local_train.train_user_lora`.

  `main.py` has no direct forward.
- **Ablation test slice.** `ablation_runner._run_single_config` sets
  `n = min(cfg.ablation_eval_examples, len(test_dataset), cand_matrix.shape[0])` and scores
  `sequences[:n]` and `cand_matrix[:n]`. The default `ablation_eval_examples` is 5000 and the
  field is in `POST_TRAINING_ONLY_FIELDS`.
- **Backbone store.**
  - Each arm's checkpoints live at `{checkpoint_root}/ablations/by_fingerprint/<fp>/<model_dir>/`
    (`model_checkpoint_dir()` = `{checkpoint_dir}/{t_base}`). That directory holds
    `epoch_XXX.pt` for every epoch (`run_pretrain` deletes none), `best.pt` and `last.pt`.
  - `backbone.json` in the store directory records `checkpoint_dir` and `checkpoint_path`, which
    may point outside the store for multi-seed registrations.
  - Each epoch checkpoint stores `config` (`asdict` after dataset resolution),
    `source_revision`, `val_ndcg` and the other `val_*` values.
- **Arm persistence.**
  - `_arm_path(config, study, model, label)` is
    `{ablation_dir}/seed_{seed}/arms/{study}_{mn}/{label}.json`.
  - A record carries `arm_key`, `code_revision`, `training_fingerprint`, `source_revision`,
    `runtime`, `resolved_config`, `metrics` and the extras.
  - `assess_study` and `collect_study` rebuild study JSONs under a per-study lock. They treat a
    record whose `code_revision.source_hash` is not the current one as stale.
- **Popularity.** `evaluation/longtail.py::compute_popularity_buckets` counts training
  interactions from `train_user_history.pkl`. No popularity ranker exists.

## What Is Reused vs. Modified

| Item | Action |
|---|---|
| `utils.autocast_context` | **Modified**: `cache_enabled=False`, optional `device_type` argument; new `utils.autocast_disabled(device_type)` |
| `evaluation/metrics.py` | **Modified**: tie-averaged `compute_metrics_batch`; fp32 scoring in `score_candidates_batch`; new popularity helpers; `EVAL_PROTOCOL` constant |
| `models/ann_retrieval.py` | **Modified**: forward in its own region; fp32 normalisation and scoring |
| `training/pretrain.py` | **Modified**: `train_one_step` and `collect_validation_routing` own their regions; `run_pretrain` computes the val popularity reference |
| `training/local_train.py` | **Modified**: `train_user_lora` owns its region |
| `evaluation/gradient_diagnostics.py`, `evaluation/flops.py` | **Modified**: own regions |
| `evaluation/cold_start.py`, `evaluation/longtail.py` | **Modified**: popularity reference and protocol stamp in the reports |
| `main.py` | **Modified**: no stage-wide region; `--stage evaluate` adds popularity and stamp; new `ablation_rescore` stage |
| `configs/config.py` | **Modified**: `ablation_eval_examples` default 0 (= all), new `ablation_eval_seed`, validate both ≥ 0 |
| `training/ablation_runner.py` | **Modified**: fingerprint semantics version; `_training_fields` helper; test-row selection; popularity, protocol and `test_eval` in arm results; summary and plot popularity |
| `training/ablation_rescore.py` | **New**: the rescore stage. It is kept out of `ablation_runner.py`, which is already 1,419 lines |
| `scripts/package_ablation_run.py`, `README.md` | **Modified**: `rescore_{machine}.sh`, runbook section |
| `scripts/estimate_training_time.py` | **Unchanged** (audited, see D5) |
| `collect_study`, `assess_study`, `file_lock`, `atomic_write_json`, `load_checkpoint`, `build_model`, `load_tensorised`, `load_eval_candidates`, `resolve_study_configs`, `check_processed_dir` | Reused as-is |

## Proposed Architecture

### D1 — Tie-averaged metrics (FR-2)

The metrics are the expectation under uniformly random tie-breaking (McSherry & Najork, 2008). For
each row:

```
t_s  = scores[:, target_idx:target_idx+1]
h    = (scores >  t_s).sum(1)                 # strictly higher
ties = (scores == t_s).sum(1) - 1             # other candidates equal to the target
# rank is uniform on {h+1, ..., h+ties+1}
r     = arange(1, C+1)                        # possible ranks
table = {"ndcg": (r<=k)/log2(r+1), "mrr": (r<=k)/r, "recall": (r<=k), "precision": (r<=k)/k}
cum   = cat([0], cumsum(table[m]))            # (C+1,)
m_row = (cum[h+ties+1] - cum[h]) / (ties+1)   # gather on device; no sort, no host sync
```

- **Implementation.** Work in float64 for the cumulative table, then return float32 per-example
  tensors. Before ranking, `if torch.isnan(scores).any(): raise ValueError(...)`. That is one host
  sync per *batch*, not per example; REV-076 was about the per-example syncs. `±inf` is ordered
  and allowed: ANN masks pad and MASK with `-inf`.
- **Why not the owner's suggested expected rank** (`1 + h + ties/2` plugged into the metric
  formulas)? NDCG and MRR are convex in rank, so `f(E[rank]) ≠ E[f(rank)]` (Jensen).
  - Take a target tied with all 100 others: the plug-in rank 51 gives NDCG 0, while the true
    expectation over a random tie-break is 0.045.
  - The tie-averaged measure is exactly the mean of what a randomised evaluation would report,
    so a constant-score model scores the random-ranking value, and a popularity ranker over
    integer counts is scored fairly.
- **Why not pessimistic ranking?** It penalises genuinely tied scores, such as the popularity
  ranker's integer counts or cold items all scoring 0, below their expected value. That biases
  the popularity floor downward and would understate how far below it the arms are.
- **No ties.** `ties = 0` gives `m_row = table[h]`, identical to today (AC-03).

### D2 — fp32 scoring choke point (FR-1)

`score_candidates_batch` becomes:

```python
model.eval()
with torch.no_grad():
    device = next(model.parameters()).device
    ...move seqs/candidates/metadata as today...
    with autocast_context(config, device_type=device.type):        # bf16 under mixed precision
        all_hidden = model(seqs, metadata=metadata, return_all_hidden=True)
        pooled = all_hidden[:, -1, :]
        user_emb = MoE path as today (Z_ut[:, -1, :]) or pooled
    with autocast_disabled(device.type):                            # robust to a caller's region
        user_emb = user_emb.float()
        item_embs = model.item_emb_weight[candidate_ids].float()
        if config.normalize_embeddings:
            user_emb = l2_normalize(user_emb)       # input already fp32 -> stays fp32
            item_embs = l2_normalize(item_embs)
        scores = torch.einsum("bd,bcd->bc", user_emb, item_embs)
        if temperature != 1.0:
            scores = scores / temperature
assert scores.dtype == torch.float32
```

- **The region's device comes from the model's parameters**, not from `get_device(config)`. In
  tests, and on a Mac where `device="auto"` resolves to MPS, the model can be on CPU. An autocast
  region for the wrong device type would silently leave the CPU forward in fp32.
- **`l2_normalize` is unchanged** (it is shared with the losses). The fix is to feed it fp32.
- **ANN retrieval** (`models/ann_retrieval.py`). `_user_embeddings` opens
  `autocast_context(config, device_type=param_device.type)` around the forward and MoE and returns
  the **un-normalised** embedding. `retrieve_candidates` then, under `autocast_disabled`, computes
  `item_embs = l2_normalize(item_weight.float())` once and
  `user = l2_normalize(user_emb.float())` per batch, and does the existing fp32 matmul.

### D3 — Autocast scoping (FR-3)

`utils.py`:

```python
def autocast_context(config, device_type: Optional[str] = None):
    """Autocast for ONE forward region. cache_enabled=False: torch.autocast caches the
    low-precision cast of fp32 leaf params with requires_grad and clears it only when the
    OUTERMOST region exits; optimizer.step() updates params in place, so a cached cast goes
    stale. With the cache off, no region can ever forward stale weights (eval-integrity FR-3)."""
    return torch.autocast(
        device_type=device_type or get_device(config).type,
        dtype=resolve_dtype(config),
        enabled=getattr(config, "mixed_precision", False),
        cache_enabled=False,
    )

def autocast_disabled(device_type: str):
    return torch.autocast(device_type=device_type, enabled=False)
```

Region placement:

| Site | Region | Outside the region |
|---|---|---|
| `train_one_step` | from `model(masked_seq, …)` through `L_total` (both forwards, MoE routing for all objectives, all losses, `weighted_sum`/`static_weighted_sum`) | dtype asserts, finite check A, `zero_grad`, `backward`, `gradnorm.update`, clip, check B, `optimizer.step()`, `scheduler.step()` |
| `collect_validation_routing` | each batch's forward plus routing | host copies |
| `score_candidates_batch` | forward plus MoE | normalisation, `einsum` (D2) |
| `run_gradient_diagnostics` | each diagnostic batch's forward(s) plus per-objective losses | `torch.autograd.grad` and the cosine/norm arithmetic |
| `benchmark_latency` | each warm-up and timed forward (inside the timed window, so latency includes the casts, as in serving) | synchronisation and timing |
| `ann_retrieval._user_embeddings` | forward plus MoE | scoring (D2) |
| `train_user_lora` | per-example forward, MoE and `loss_fn` | `scaler.scale(loss).backward()`, unscale, clip, `scaler.step`, `update` |

- **`train_one_step`.** The `torch.is_autocast_enabled` assertion is deleted (FR-3.4). The
  existing dtype assertions stay. Under mixed precision the losses are fp32 (autocast picks fp32
  for the loss ops), which is what the current assert already checks; `L_total` comes out of the
  region unchanged.
- **Backward outside autocast** is PyTorch's documented recommendation and is numerically
  identical: backward ops take their dtypes from the recorded forward graph.
- **`main.py`.** The `with autocast_context(config):` wrapper and its comment are removed. The OOM
  handler stays around `STAGE_MAP[args.stage](config, args)`.

### D4 — Why both scoping and `cache_enabled=False` (decision for FR-3.1/3.3)

- **Scoping alone** clears the cache at every region exit. It is still fragile: any future caller
  that opens an outer region (a script, a test, a notebook) silently re-creates the bug, because
  inner regions do not clear the cache while an outer one is open.
- **`cache_enabled=False` alone** fixes the staleness even with the stage-wide region left in
  place, because `cached_cast` consults the thread-local cache flag that the innermost region
  sets. But it would leave backward under autocast, leave forward sites without an explicit
  compute-dtype contract, and leave scoring under the stage region.
- **Cost.** Each Linear weight is cast once per *use* instead of once per region. Uses per step:
  one per forward (two on contrastive steps), plus the router per objective. That is a few MB of
  casts per step at `t-base`, negligible next to `batch 256 × S × D` activations. The values are
  bit-identical to a cached cast of the same weights, so within a step nothing changes.
- **Decision: both.** The `estimate_training_time.py` wrappers (D5) and the AC-09
  enclosing-region test rely on the cache flag.

### D5 — Audit of autocast-dependent code and dtype casts (FR-3.6)

| Location | Finding | Action |
|---|---|---|
| `pretrain.py` ~1012 `metadata_matrix_t = metadata_matrix.to(dtype)` (`dtype = resolve_param_dtype`) | fp32 under mixed precision, bf16 under pure bf16. The matrix is a data tensor, not a leaf param, so it is never cached. `metadata_projection` (a Linear) casts it under the region. | Correct, no change |
| `score_candidates_batch` `metadata.to(dtype=model.item_emb_weight.dtype)` | Param dtype, so the same reasoning applies | Correct, no change |
| `transformer.forward` `metadata_projection(metadata.to(x.dtype))` | `x` comes from an embedding (fp32 under mixed, since autocast does not cast embeddings); the Linear output is bf16 and is added to fp32, which promotes to fp32 | Correct, no change |
| `flops.benchmark_latency` `dummy_meta … dtype=torch.bfloat16` | Under mixed precision the bf16 input goes to an autocast Linear (fine). Under pure bf16 it matches the weights. | Correct, no change |
| `estimate_training_time.py` `meta.to(param_dtype)`; `self._amp(cfg)` around `train_one_step` and `evaluate_model` | Nested regions are now harmless: an inner `autocast_context` has the cache off and enters the same dtype | No change (the script also supports older code via `HAS_MIXED`/`STEP_TAKES_SCALER`) |
| `losses.py:179` `torch.autocast(enabled=False)` for InfoNCE | Local disable, independent of the outer region | No change |
| `transformer._flash_attention_3` q/k/v dtype check | q/k/v come from Linears inside each forward region, so bf16 under mixed precision. A forward site missing a region now fails loudly on FA3 (good) and runs fp32 on SDPA; AC-10 covers every known site. | No change |
| `GradNormController` (`log_lambdas` leaf, `exp`, `mul`) | Not autocast-cast ops, so not cached | No change |
| `MultiStepLoss` step projections (`nn.Linear` leaf params) | Were stale under the stage region. Now covered by the `train_one_step` region (AC-08 hooks loss-fn Linears too). | Covered |
| LoRA `x @ lora_A @ lora_B` (leaf params) and `load_lora_state_dict` (in-place `load_state_dict`) | Were frozen at the stage's first forward. Now each `evaluate_user` / `train_user_lora` forward has a fresh, uncached region. | Covered (AC-12) |

### D6 — Training-semantics version in the fingerprint (FR-4)

```python
#: Bump when a code fix changes what training produces for an unchanged config, so backbones
#: trained before the fix are never reused (eval-integrity FR-4). 1 = implicit (<= c094b2c).
TRAINING_SEMANTICS_VERSION = 2

def _training_fields(cfg_dict: dict) -> dict:
    return {k: v for k, v in cfg_dict.items() if k not in POST_TRAINING_ONLY_FIELDS
            and k not in _PATH_FIELDS and k not in _RUN_SELECTION_FIELDS
            and k not in _RUNTIME_ONLY_FIELDS}

def _training_fingerprint(cfg: Config) -> str:
    fields = _training_fields(asdict(cfg))
    fields["_training_semantics"] = TRAINING_SEMANTICS_VERSION
    ...same hash...
```

- `_training_fields` is reused by the rescore identity check (FR-8.9).
- The version is bumped for every config, not just mixed-precision ones. Pure-bf16 backbones are
  bit-identical across the fix (golden test), but no experiment uses them, and a conditional bump
  is a rule that is easy to get wrong.
- `ablation_eval_seed` joins `POST_TRAINING_ONLY_FIELDS`.

### D7 — Ablation test rows (FR-6)

A new pure function in `ablation_runner.py`:

```python
def select_eval_rows(total: int, n: int, seed: int) -> Tuple[Optional[np.ndarray], dict]:
    if n <= 0 or n >= total:
        return None, {"method": "full", "num_examples": total, "total_rows": total, "seed": None}
    idx = np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))
    return idx, {"method": "random", "num_examples": n, "total_rows": total, "seed": seed,
                 "indices_sha256": hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()}
```

- `_run_single_config` uses `idx` to index `test_dataset.sequences` and the candidate rows. The
  candidate matrix is memory-mapped via `load_eval_candidates`; materialise only the selected rows
  with `np.array(cand_matrix[idx])`. With `None` it passes the whole tensors, as `--stage
  evaluate` does.
- The arm result records `test_eval`, `num_eval_examples`, `popularity_reference` and
  `eval_protocol`.

**Cost reasoning (measured 2026-10-16, dev Mac, `t-base`, `max_seq_length = 50`, `arv_mmoe`,
mixed bf16, current `evaluate_model`, random-init weights; throughput does not depend on the
weights):**

| Device | Throughput | One full val pass (100,591 rows) |
|---|---|---|
| MPS | 0.216 ms/row | ≈ 22 s |
| CPU | 5.29 ms/row | ≈ 9 min |

- The 10% test split has 107,282 rows, so one full test pass costs about 1.07 val passes.
- Each arm already runs 20 full validation passes during training. A full test split therefore
  adds about 5% of an arm's evaluation time and a negligible fraction of its training time (the
  L4 `t-base` arms are hours).
- **Decision: the full split by default**, with sampling kept only for tests and explicit opt-in.
  The tiny fixture's `tiny_config` sets `ablation_eval_examples = 32`, which exercises the random
  path.

### D8 — Popularity reference (FR-7)

New helpers in `evaluation/metrics.py`, which the conventions name as the metrics module:

```python
EVAL_PROTOCOL = {"version": 2, "scoring": "fp32", "ties": "tie_averaged"}

def item_popularity(train_user_history: Mapping[int, Sequence[int]], vocab_size: int) -> torch.Tensor:
    """(vocab,) float32 training-interaction counts; pad (0), MASK and unseen items are 0."""

def load_item_popularity(processed_dir: str, vocab_size: int) -> torch.Tensor:  # reads train_user_history.pkl

def popularity_scores(candidates: torch.Tensor, popularity: torch.Tensor) -> torch.Tensor:
    return popularity[candidates.long()]          # (N, C) float32

def evaluate_popularity(candidates, popularity, k, batch_size=None) -> Dict[str, Dict[str, float]]:
    """aggregate_metrics over compute_metrics_batch(popularity_scores(...)) in batches."""
```

- Counts are exact in float32: the largest count is at most the number of users (136,677) < 2^24.
- **Producers:**
  - `_run_single_config`: on the FR-6 rows.
  - `main.run_evaluate_stage`: on `candidate_tensors` as scored, so ANN candidates when ANN is on.
  - `run_cold_start_evaluation`: per bucket, by masking `compute_metrics_batch(popularity_scores
    (cands))` with the same `is_cold` selector. Stored as `cold_popularity` and
    `warm_popularity`, top-level keys with the metric shape, so `_save_report`'s serialisation
    and the plot are unaffected.
  - `run_longtail_evaluation`: per bucket, stored as `popularity_head`, `popularity_mid` and
    `popularity_tail`.
  - `run_pretrain`: once over the val candidates, logged in the epoch-log header and returned as
    `val_popularity_reference`.
- Each producer also adds `eval_protocol`. In the cold-start and long-tail `_save_report`s it is
  added beside `source_revision`, not inside the metric loop.
- **Summary** (`_save_summary`). Per study it adds `best_ndcg` (the existing `best` label is
  kept), `popularity_reference_ndcg` (from the first ok arm that has one; all ok arms share the
  test rows), and `best_minus_popularity_ndcg`. It takes an optional `extra: dict` merged at top
  level, which the rescore uses for `rescore_caveat`.
- **Plot** (`_plot_study`). `ax.axhline(pop_ndcg, linestyle="--", color="0.3",
  label="popularity (NDCG@10)")` when any usable arm has `popularity_reference`.

### D9 — `--stage ablation_rescore` (FR-8)

New module `training/ablation_rescore.py`; `main.py` gains `run_ablation_rescore_stage`, a
`VALID_STAGES` entry and a `STAGE_MAP` entry. `--allow_incomplete` applies as it does for
`ablation_collect`.

```text
run_ablation_rescore(config, model_name, study="all", allow_incomplete=False) -> dict
  log WARNING RESCORE_CAVEAT
  check_processed_dir(config)
  rescored = rescored_config(config)   # deepcopy; ablation_dir += "/rescored"; plot_dir += "/rescored"
  shared   = EvalData(config)          # lazily loads val/test tensors, candidates, popularity ONCE per process
  for s in studies:
    for cfg in resolve_study_configs(s, config):
      label = cfg._study_label
      orig  = load original arm JSON at _arm_path(config, s, model, label)
      if orig missing or orig.status != "ok": report skipped; continue
      try:
        rescore_arm(orig, s, label, config, rescored, shared)
      except Exception as exc:
        if is_oom_error(exc): raise            # exit 75, run_with_retry resumes
        log + report failed                     # nothing written: the arm shows as missing
    collect_study(rescored, model_name, s, allow_incomplete)
  if study == "all": _save_summary(rescored, ..., extra={"rescore_caveat": RESCORE_CAVEAT})
  log WARNING RESCORE_CAVEAT
  return report   # stage function exits 1 if any study not written
```

**`rescore_arm`:**

1. `fp = orig["training_fingerprint"]`.
2. `entry = find_backbone(config.checkpoint_dir, fp)`. If there is no entry, raise naming the
   store path.
3. `ckpt_dir = join(entry["checkpoint_dir"], <model_dir_name>)`.
4. `epochs = sorted(epoch_(\d{3}).pt in ckpt_dir)`. If empty, raise.
5. `key = sha256(json(orig.arm_key, fp, current source_hash, protocol))[:16]`. If a rescored arm
   file exists with `rescore.rescore_key == key`, skip.
6. `record = rescore_fingerprint(fp, entry, ckpt_dir, epochs, config, shared)`.
7. Build the rescored arm record (FR-8.5/8.6) and `atomic_write_json` it to
   `_arm_path(rescored, s, model, label)`.

**`rescore_fingerprint`:**

```text
store = backbone_store_dir(config.checkpoint_dir, fp)
with file_lock(join(store, ".rescore.lock"), poll=5.0, on_wait=log):
  rec = read join(store, "rescore.json") if valid for (current source_hash, protocol, epoch file
        list with size+mtime_ns) else fresh {format: 1, fingerprint, rescore_code, protocol, epochs: {}}
  if rec.complete: return rec
  best_ckpt   = torch.load(join(ckpt_dir, "best.pt"), weights_only=False)
  eval_cfg    = config_from_checkpoint(best_ckpt["config"], runtime=config)   # FR-8.3
  model, moe  = build_model(eval_cfg)        # built ONCE per fingerprint
  for e in epochs not in rec.epochs:
      ck = torch.load(epoch file, weights_only=False)
      verify _training_fields(ck["config"]) == _training_fields(best_ckpt["config"])
          and ck["source_revision"] == best_ckpt["source_revision"]          # FR-8.9
      load_checkpoint(path, model, moe, checkpoint=ck)
      val = evaluate_model(model, val_seqs, val_cands, eval_cfg, moe_layer=moe,
                           metadata_matrix=val_meta)
      rec.epochs[e] = {"val": val, "original_val_ndcg": ck["val_ndcg"], "checkpoint": path}
      atomic_write_json(rescore.json, rec)                                    # resumable
  best = earliest argmax of rec.epochs[e].val.ndcg.mean
  load epoch best; test = evaluate_model(... FR-6 rows ...)
  rec.update(best_epoch, best_val_ndcg, test={metrics, test_eval, popularity_reference},
             val_popularity_reference, complete=True)
  atomic_write_json(rescore.json, rec)
return rec
```

- **`config_from_checkpoint(ckpt_config, runtime)`**: start from `Config()` and `setattr` every
  key of `ckpt_config` that is a `Config` field. Log, at WARNING, the unknown keys it ignored.
  Then copy these runtime fields from `runtime`: `device`, `attention_backend`,
  `eval_batch_size`, `num_workers`, `pin_memory`, `processed_data_dir`, `output_dir`,
  `checkpoint_dir`, `log_dir`, `plot_dir`, `ablation_dir`. Then `check_processed_dir(eval_cfg)`
  and `eval_cfg.validate()`. `vocab_size`, `num_items` and `feature_dim` come from the
  checkpoint, as the trained model needs.
- **Shared evaluation data (`EvalData`).**
  - Loads `val_tensorised.pt` / `test_tensorised.pt` once per `max_seq_length`, narrowed (the
    H study's sequence-length arms use other widths). The `load_tensorised` metadata argument is
    built with the eval config.
  - Loads the val and test candidate matrices once, checking their row counts against the
    splits.
  - Computes item popularity once and the val and test popularity references once.
  - Selects test rows with `select_eval_rows(total, config.ablation_eval_examples,
    config.ablation_eval_seed)`.
- **The model is reused across epochs.** Loading weights in place into one model is only correct
  because no autocast region can cache casts (FR-3.3). AC-21 would catch a regression: every
  epoch would score the same.
- **Protocol dict.** `{**EVAL_PROTOCOL, "eval_k": k, "val_rows": N_val, "test_eval":
  descriptor-without-indices_sha256-or-with}`. It is part of both the fingerprint record's
  validity and the `rescore_key`.
- **Rescored arm record.** A deepcopy of `orig` with these changes:
  - `metrics` = the test metrics;
  - `best_epoch` and `best_val_ndcg` = the rescored values;
  - `popularity_reference`, `val_popularity_reference`, `test_eval`, `eval_protocol` and
    `num_eval_examples` set;
  - `code_revision` = `_current_revision()`;
  - `reused_from_previous_run` removed;
  - weight-dependent extras moved to `rescore.pre_rescore_extras`;
  - `rescore` = FR-8.5's block.

  `runtime`, `source_revision`, `resolved_config`, `study_overrides`, `config_diff`,
  `training_fingerprint` and `arm_key` are kept, so `_consistency_problems` sees the same values
  it saw for the originals.
- **`RESCORE_CAVEAT`** is the FR-8.7 text, as a module constant.
- **Cost.** Each fingerprint takes about `n_epochs` val passes plus one test pass:

  | Device | Per fingerprint (20 epochs) |
  |---|---|
  | MPS | ≈ 8 min |
  | CPU | ≈ 3.1 h |

  On the L4 and H100 it is expected to be at or below the MPS figure. Arms sharing a fingerprint
  pay once. For local rescoring, use MPS and `--study` to limit the work.

### D10 — Scoring-site guard (FR-1.4)

`tests/test_eval_integrity.py::test_no_unaudited_scoring_sites` reads every `.py` under
`evaluation/` plus `models/ann_retrieval.py`, and regex-matches `einsum(`, ` @ ` and `matmul(` on
non-comment lines. It asserts the `(file, function)` set equals this allow-list:

- `evaluation/metrics.py::score_candidates_batch` (fp32 `einsum`);
- `models/ann_retrieval.py::retrieve_candidates` (fp32 matmul);
- any audited non-scoring arithmetic BUILD finds, for example in the gradient diagnostics.
  Each such entry gets a one-line justification in the test.

The function name is found with `ast`. A new site fails the test until it is audited.

### D11 — Packager and runbook (FR-9)

`render_rescore_script(args, machine)` in `scripts/package_ablation_run.py` emits
`launch/rescore_{machine}.sh`:

- the same `_machine_setup` as the run scripts;
- then, for each of `machine_studies(args, machine)`, `retry_prefix + study_command(args,
  machine_backend(args, machine), py, study=s, stage="ablation_rescore")`;
- then a final `--stage ablation_rescore --study all` for the summary. That step only collects,
  because every arm already has its rescored record.

It is written next to the run scripts. `--gpus` handling is not involved: one process per machine,
and the per-fingerprint lock makes extra manual processes safe. The README section is described in
FR-9.2.

## Data Flow (after the change)

```
training step:  [autocast(cache off): forward + losses] -> backward -> GradNorm -> clip -> step
evaluation:     [autocast(cache off): forward + MoE] -> [autocast off: fp32 normalise + dot]
                -> tie-averaged metrics -> JSON {metrics, popularity_reference, eval_protocol}
rescore:        arm JSON -> fingerprint -> backbone.json -> epoch_XXX.pt (read-only)
                -> per-epoch val (fp32, tie-avg) -> best epoch -> test + popularity
                -> by_fingerprint/<fp>/rescore.json -> ablations/rescored/seed_S/arms/... -> collect_study
```

## API / Schema Changes

- `utils.autocast_context(config, device_type=None)`: new optional argument; `cache_enabled=False`.
  New `utils.autocast_disabled(device_type)`.
- `compute_metrics_batch` / `compute_metrics`: same signatures, new tie semantics, `ValueError`
  on NaN.
- `score_candidates_batch`: same signature; always returns fp32.
- New in `evaluation/metrics.py`: `EVAL_PROTOCOL`, `item_popularity`, `load_item_popularity`,
  `popularity_scores`, `evaluate_popularity`.
- `Config`:
  - `ablation_eval_examples` default 5000 becomes 0 (0 = all rows);
  - new `ablation_eval_seed: int = 0`;
  - `validate()` rejects negative values for either.
- `run_pretrain` result: adds `val_popularity_reference`.
- Arm result JSON: adds `test_eval`, `popularity_reference`, `eval_protocol`. Rescored records add
  the `rescore` block and `val_popularity_reference`.
- `summary_{model}.json`: per-study `best_ndcg`, `popularity_reference_ndcg`,
  `best_minus_popularity_ndcg`; the rescored summary also has `rescore_caveat`.
- `test_metrics_`, `cold_start_` and `longtail_{model}.json`: popularity keys (D8) and
  `eval_protocol`.
- CLI: `--stage ablation_rescore`.
- New files:
  - `by_fingerprint/<fp>/rescore.json` and `.rescore.lock`;
  - `{ablation_dir}/rescored/seed_{seed}/…`;
  - `{plot_dir}/rescored/…`;
  - `launch/rescore_{machine}.sh`.
- Training fingerprint: every value changes (D6).

## Error Handling

- NaN scores: `ValueError` (D1).
- Rescore:
  - a missing backbone entry, an empty epoch directory, or a config/revision mismatch fails that
    arm with a message naming the path;
  - OOM re-raises, the process exits 75 and the retry launcher resumes;
  - a corrupt or old-protocol `rescore.json` is discarded and recomputed, logged at WARNING;
  - collection refuses inconsistent studies, as `ablation_collect` does.
- `config_from_checkpoint`: `check_processed_dir` raises if the processed directory does not match
  the checkpoint's user sample.

## Security / Performance

- No new dependencies.
- Scoring in fp32 costs one `(B, C, D)` fp32 gather per batch: at B = 256, C = 101, D = 320 that is
  33 MB, transient.
- Casting with the cache off: see D4.
- `evaluate_popularity` is a gather plus metrics with no model, seconds per split.
- `torch.load(..., weights_only=False)` on the project's own checkpoints is the existing pattern.

## Testing Strategy

New file `tests/test_eval_integrity.py` (tiny fixture and in-memory tensors, CPU), plus edits to
existing tests:

| Area | Tests |
|---|---|
| Metrics (D1) | AC-01–AC-04 in `tests/test_metrics.py` (new class `TestTieAveraged`); existing tests untouched |
| Scoring (D2) | AC-05, AC-06, AC-07 |
| Autocast (D3/D4) | AC-08 (hooks on every Linear in model, MoE and loss fns; `lr = 0.05`; compare to fresh-region `F.linear` with post-step-1 and pre-step-1 snapshots); AC-09 (same inside an enclosing raw `torch.autocast` plus the characterisation test); AC-10 (parametrised forward-site dtype); AC-11 (`main()` with a stub stage); AC-12 (LoRA) |
| Existing mixed-precision tests | `tests/test_training.py::TestMixedPrecision::test_step_refuses_to_run_without_autocast` asserts the removed assertion; it is **replaced** by `test_step_runs_its_own_autocast_region` (no caller region; a hooked Linear sees bf16; losses fp32). `test_step_trains_fp32_master_weights_under_autocast` stays (nested regions are legal). |
| Golden | AC-13: `tests/test_determinism.py` unchanged |
| Fingerprint (D6) | AC-14 |
| Selection (FR-5) | AC-15: tiny `run_pretrain` with `mixed_precision=True`, `pretrain_epochs=3`, `max_steps_per_epoch=4` |
| Test rows / popularity (D7/D8) | AC-16–AC-19 |
| Rescore (D9) | AC-20–AC-26: tiny `run_ablation(study="B", ...)`, then `run_ablation_rescore`; spies on `evaluate_model` as `test_ablation_robustness.py` does |
| Packager (D11) | AC-27 in `tests/test_scripts.py` |
| Smoke | AC-28 on the Mac, capped |

Validation commands (targeted only, per `steering/testing.md`):
`uv run python -m pytest tests/<file>.py -m "not slow" -q`.

## Decisions

Decisions taken while writing this spec, without owner input; they can be changed.

- **DEC-1 (tie rule):** tie-averaged expected *metric* (McSherry & Najork), not the plug-in
  expected rank and not pessimistic (D1).
- **DEC-2 (stale cache):** per-forward scoping **and** `cache_enabled=False` in `autocast_context`;
  the stage-wide region is removed (D3/D4).
- **DEC-3 (backward placement):** backward, GradNorm and optimizer step run outside autocast
  (D3). This is numerically identical and the documented PyTorch pattern.
- **DEC-4 (fingerprint):** add `TRAINING_SEMANTICS_VERSION = 2` to the training fingerprint for
  all configs, so the A–H re-run on the same machines retrains instead of reusing pre-fix
  backbones (D6). The rescore therefore locates backbones by the fingerprint recorded in the arm
  JSON, never by recomputing it.
- **DEC-5 (test split):** full test split by default, based on the measured cost (≈ +5% of an
  arm's evaluation time). `ablation_eval_examples > 0` means a seeded uniform random sample with
  a dedicated `ablation_eval_seed` that does not depend on the training seed (D7). User
  stratification was rejected: with the full split as default, the sample exists only for tests
  and quick checks, where uniform sampling is enough and simpler to verify.
- **DEC-6 (popularity source):** training-interaction counts from `train_user_history.pkl`, the
  same source as the long-tail buckets, tie-averaged like the model (D8). Because of the tie rule,
  the popularity numbers will differ slightly from the owner's measurements (0.4076 val, 0.3742
  test).
- **DEC-7 (rescore outputs):** a separate results root, `{ablation_dir}/rescored/seed_{seed}/`
  (plots under `{plot_dir}/rescored/`). The per-fingerprint record sits in the store. Checkpoints
  and originals are never modified. "In place" means it runs on the run's machine and directory,
  reading checkpoints where they are.

  This also avoids an `arm_key` hazard. Rescored records keep the original `arm_key`, which
  embeds the old source hash, so a post-fix `--stage ablation` never reuses a rescored record as
  if it were a fresh training result.
- **DEC-8 (rescore record validity):** reused only for the same rescoring `source_hash`, protocol
  and epoch-file list (size + `mtime_ns`). Any change recomputes, which is safe and cheap enough.
- **DEC-9 (rescore config):** the model is rebuilt from each checkpoint's stored `config` with
  runtime fields from the CLI (D9). The arm's study config is not re-derived from CLI flags,
  because the backend and device legitimately differ between training and rescoring machines.
- **DEC-10 (extras):** the rescore recomputes only selection, test metrics and popularity.
  Weight-dependent extras move to `rescore.pre_rescore_extras`; weight-independent ones stay
  (FR-8.6).
- **DEC-11 (rescore code location):** a new module, `training/ablation_rescore.py`, reusing
  `ablation_runner` internals (`_arm_path`, `find_backbone`, `backbone_store_dir`,
  `collect_study`, `_save_summary`, `_training_fields`, `file_lock`, `select_eval_rows`).
- **DEC-12 (protocol stamp):** result files carry `eval_protocol` (FR-1.5), so post-fix numbers are
  distinguishable from the file alone. This follows the steering rule "provenance belongs in the
  artifact".
- **DEC-13 (packager):** `launch/rescore_{machine}.sh` is generated rather than leaving the owner
  to hand-copy flags, because a flag mismatch (sample fraction, backend) is the most likely
  operator error (FR-9).

## Alternatives Considered

- **`cache_enabled=False` only, keeping the stage region.** Rejected (D4): backward and scoring
  would stay under autocast, and forward sites would have no explicit dtype contract.
- **Keep the stage region and call `torch.clear_autocast_cache()` after every `optimizer.step()`.**
  Rejected: it fixes training only. Every other mutation site (`load_state_dict` for LoRA and
  federated, checkpoint loads in the rescore) would need the same call, and forgetting one
  silently re-creates the bug.
- **fp32 training.** Excluded by the owner directive.
- **Expected-rank plug-in or pessimistic ties.** Rejected (D1).
- **Rewrite the original arm files in place.** Rejected: it destroys the evidence and invites the
  `arm_key` reuse hazard (DEC-7).
- **Recompute arm configs from CLI flags in the rescore.** Rejected: the backend and device
  differ by machine and the fingerprint has changed (DEC-4/DEC-9).
- **User-stratified test sample by default.** Rejected in favour of the full split (DEC-5).
