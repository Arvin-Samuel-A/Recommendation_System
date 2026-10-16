# Evaluation Integrity — Requirements

## Overview

Make every reported ranking number honest and make mixed-precision training actually train. The
work covers five defects found in the 10% ablation run, which used code `c094b2c`, `t-base` and
mixed bf16:

1. Scores are computed in bf16, so the target often ties another candidate.
2. Ties are resolved in the target's favour.
3. A stage-wide autocast region freezes every `nn.Linear` at its first-step bf16 copy.
4. The ablation test slice covers only 637 of 13,582 users.
5. No popularity floor is reported, so a gain cannot be read against it.

The spec also adds a no-retraining rescore of the finished run's saved checkpoints.

This spec adds to `evaluation-diagnostics`, which owns the metric protocol, and to `ablation-run`,
which owns the ablation run. It does not duplicate either. The clauses it supersedes are listed
under Constraints.

## Problem Statement

These numbers were measured on 10% of the data with `t-base`, using checkpoints from `c094b2c`
re-scored on CPU with `evaluation/metrics.py::score_candidates_batch`.

- **bf16 scores tie.** `main.py:1166` runs every stage inside one `torch.autocast` region, so
  `score_candidates_batch` returns bf16 cosines in [-1, 1].
  - In 41–52% of examples the target ties another candidate. `compute_metrics_batch` ranks the
    target at `1 + #strictly higher`, which gives every tie to the target.
  - Baseline arm, epoch 12, full val, bf16 vs fp32 scoring:

    | Metric | bf16 | fp32 |
    |---|---|---|
    | NDCG@10 | 0.4639 | 0.3880 |
    | MRR@10 | 0.3966 | 0.2969 |
    | Recall@10 | 0.6834 | 0.6831 |

  - The inflation differs by arm: +0.08 NDCG for the baseline, +0.02 for
    `arv_mmoe_gradnorm`/`no_contrastive`. This biases the comparison between arms.
- **The stale autocast weight cache is a training-correctness bug.**
  - `torch.autocast` caches the low-precision cast of every fp32 leaf parameter that requires
    grad. That covers all `nn.Linear` weights and biases, the LoRA A/B matrices and the
    multi-step projections.
  - The cache is cleared only when the outermost region exits. `optimizer.step()` updates the
    master weights in place, so the cache never notices the change.
  - Under the stage-wide region, every Linear forwards with its first-step bf16 copy for the
    whole run. That affects training, in-training validation (and so `best.pt` selection), LoRA
    personalisation and federated rounds alike.
  - Gradients still reach the fp32 masters. The saved weights therefore drift under gradients
    evaluated at a stale point, and post-load evaluation scores weights that were never seen
    during training.
  - Probe on torch 2.14.0, CPU, Linear + AdamW lr = 0.5, 3 steps inside one region: the forward
    output changed by 0.0. A fresh region on the same weights differed by 16.6.
  - This explains why the baseline's logged val NDCG of 0.5357 is above every re-scoring of its
    checkpoint (0.4639 in bf16, 0.3880 in fp32).
- **The ablation test slice is biased.** `training/ablation_runner.py` (~line 630) scores the
  first `ablation_eval_examples = 5000` rows of `test_tensorised.pt`. Rows are sorted by user id,
  so those rows come from only 637 of 13,582 users.
- **There is no popularity floor.** A popularity-only ranker on the same candidates scores 0.4076
  val NDCG, 0.3742 test NDCG on all rows and 0.3705 on a random 5000 rows. In fp32, no trained
  arm beats it on test:

  | Arm | Test NDCG@10 (random 5000 rows) |
  |---|---|
  | baseline | 0.3554 |
  | `arv_mmoe_gradnorm` | 0.2890 |
  | `no_contrastive` | 0.2646 |

  Without the floor next to each table, a "gain" cannot be interpreted.

## Goals

- G1: Every evaluation path scores in fp32, outside autocast, and ties never favour the target.
- G2: Mixed-precision training, evaluation, LoRA and federated forwards always use the current
  weights. The owner directive stands: training stays bf16 mixed precision, with fp32 master
  weights and bf16 autocast compute.
- G3: A backbone trained before this fix can never be silently reused by a training run after it.
- G4: Checkpoint selection (`best.pt`) uses the fixed metric. The value logged during training
  equals the value a post-load evaluation of the same checkpoint produces.
- G5: Ablation test numbers come from the full test split. A sample is allowed only when
  explicitly requested, and its method and seed are recorded.
- G6: A popularity reference accompanies every ranking result file and every ablation study.
- G7: The finished 10% run can be re-scored honestly, on the machine that holds it, without
  retraining. The output says plainly that the rescore cannot repair the training bug.

## Non-Goals

- **Training precision.** Training does not switch to fp32. The losses' internal logit precision
  under autocast is unchanged (owner directive).
- **The popularity collapse and its remedies**, which are separate owner decisions recorded in the
  sibling note `study_a_gap_findings.md`:
  - a logit temperature;
  - the contrastive weight;
  - a clean forward for next-item (`next-item-clean-forward`).
- **Re-running the A–H studies.** That is an owner launch after this spec ships. This spec only
  makes the re-run possible and safe (G3).
- **Rescoring weight-dependent extras.** The rescore does not recompute routing, gradient,
  cold-start, long-tail, personalisation or communication results for old checkpoints. Those
  extras are quarantined (FR-8.6) and come back in the re-run.
- **Popularity references for the personalisation and federated outputs.** Those report gains
  relative to the global model, which is already their reference.
- **Federated validation prefix (out of scope).** The federated per-round validation pass also
  scores a prefix of a user-sorted split (`federated_eval_examples`). It is named under "Out-of-scope
  findings" and not fixed here.

## Functional Requirements

### FR-1: fp32 scoring in every evaluation path

- FR-1.1: `evaluation/metrics.py::score_candidates_batch` is the single choke point for scoring.
  - The model and MoE forward run under the configured autocast, which is bf16 under mixed
    precision.
  - L2 normalisation of the user and item embeddings and the dot products then run in fp32,
    outside autocast. This holds even if the caller has an autocast region open.
  - Optional temperature scaling also runs in fp32.
  - The returned scores are always `torch.float32`. This holds under mixed precision, under pure
    bf16 weights, and when the call is made inside an enclosing autocast region.
- FR-1.2: All of these score through FR-1.1, unchanged in behaviour otherwise:
  - validation (`run_pretrain`);
  - the test stage (`--stage evaluate`);
  - ablation test evaluation;
  - cold-start and long-tail evaluation;
  - personalisation (`evaluate_user`);
  - federated validation;
  - the training-time estimator's validation timing.
- FR-1.3: Paths that score outside `score_candidates_batch` obey the same rule. Today the only one
  is `models/ann_retrieval.py::retrieve_candidates`, which scores the full vocabulary: the user
  embedding is upcast to fp32 before normalisation, the item table is normalised in fp32, and the
  matmul runs outside autocast.
- FR-1.4: The audit of scoring sites is recorded in design.md. A test fails if a new
  `einsum`/`@`/`matmul` scoring site under `evaluation/` or `models/ann_retrieval.py` bypasses the
  choke point. The test is described in design D10.
- FR-1.5: Every result file that reports ranking metrics records
  `eval_protocol: {"version": 2, "scoring": "fp32", "ties": "tie_averaged"}`. That covers ablation
  arm results, `test_metrics_{model}.json`, `cold_start_{model}.json`, `longtail_{model}.json` and
  rescored arm records. A reader can then tell post-fix numbers from pre-fix ones from the file
  alone. Pre-fix files carry no stamp, which is version 1.

### FR-2: Honest tie handling in ranking metrics

- FR-2.1: `compute_metrics_batch` (and `compute_metrics`, which wraps it) reports the **expected
  value of each metric under a uniformly random ordering of tied scores**. This is the
  tie-averaged measure of McSherry & Najork (2008).
  - With `h` candidates strictly higher than the target and `t` other candidates exactly equal to
    it, the target's rank is uniform on `{h+1, …, h+t+1}`.
  - Each metric is the mean of its per-rank value over that range.
- FR-2.2: With no ties, every metric is identical to today's value. The existing
  `tests/test_metrics.py` tests pass unchanged.
- FR-2.3: Ties among non-target candidates do not change the target's metrics.
- FR-2.4: The implementation stays vectorised and free of per-example host syncs (REV-076). It
  needs no sort.
- FR-2.5: A score matrix containing NaN raises `ValueError` and names the batch shape. Ranking a
  NaN is undefined, and code-quality "Numerical Safety" says to fail loudly.

### FR-3: Autocast scoping (training-correctness fix)

- FR-3.1: No stage-wide autocast region. `main.py::main` no longer wraps `STAGE_MAP[stage]` in
  `autocast_context`. Every stage function runs with autocast disabled at entry.
- FR-3.2: Every function that runs a model forward opens its own autocast region around that
  forward, and only that forward. The forward sites are:

  | Site | Region covers |
  |---|---|
  | `training/pretrain.py::train_one_step` | forward and loss computation; backward, GradNorm update, clipping and `optimizer.step()` run outside |
  | `training/pretrain.py::collect_validation_routing` | forward |
  | `evaluation/metrics.py::score_candidates_batch` | forward only (FR-1) |
  | `evaluation/gradient_diagnostics.py::run_gradient_diagnostics` | per-diagnostic-batch forward and losses |
  | `evaluation/flops.py::benchmark_latency` | each warm-up and timed forward |
  | `models/ann_retrieval.py::_user_embeddings` | forward |
  | `training/local_train.py::train_user_lora` | per-example forward and loss; backward and step outside |

- FR-3.3: Cache defence in depth. `utils.autocast_context` disables the autocast weight cache
  (`cache_enabled=False`). A region opened through it can therefore never read or populate a
  stale cast, even when it is nested inside a caller's region with caching enabled.
- FR-3.4: `train_one_step` no longer requires the caller to open a region, so its
  `torch.is_autocast_enabled` assertion is removed. Under `mixed_precision=True` it runs its
  forward in the configured compute dtype by itself. It still works when called inside an
  enclosing region, as `scripts/estimate_training_time.py` does.
- FR-3.5: With `mixed_precision=False`, behaviour is bit-for-bit unchanged. The golden
  characterisation test in `tests/test_determinism.py` stays green without its constants being
  touched.
- FR-3.6: The dtype casts that run outside autocast were audited and are recorded in design.md:
  - `pretrain.py` `metadata_matrix.to(resolve_param_dtype)`;
  - `score_candidates_batch`'s metadata cast;
  - `benchmark_latency`'s bf16 dummy metadata;
  - the estimator's `meta.to(param_dtype)`.

  Any cast found incorrect is fixed in the task that touches that file.

### FR-4: Pre-fix backbones are never reused

- FR-4.1: The training fingerprint (`_training_fingerprint`) includes a training-semantics version
  constant. This spec bumps it, so every config's fingerprint changes.
  - A post-fix `--stage ablation`/`multi_seed` run on a machine that holds pre-fix backbones
    therefore trains new ones instead of reusing them.
  - The existing `source_revision` resume refusal stops a pre-fix `last.pt` from being continued.
- FR-4.2: The fingerprint keeps every existing invariance: runtime-only, path, post-training-only
  and run-selection fields still do not change it. The packager plan uses the same function, so
  plans and runs agree.

### FR-5: Best-epoch selection on the fixed metric

- FR-5.1: `run_pretrain` selects `best.pt` by validation NDCG@10 computed through FR-1, FR-2 and
  FR-3.
- FR-5.2: For every epoch `e` of a mixed-precision run, the `val_ndcg` stored in `epoch_{e}.pt`
  equals the NDCG@10 that `evaluate_model` produces after loading `epoch_{e}.pt` into a freshly
  built model, on the same validation data and device.
- FR-5.3: `run_pretrain` computes the validation popularity reference (FR-7) once per run. It logs
  it in the per-epoch log header and returns it as `val_popularity_reference`.

### FR-6: Ablation test evaluation on the full test split

- FR-6.1: By default each ablation arm is scored on **every** row of the stored test split, using
  the stored test candidate sets. `ablation_eval_examples = 0` means "all rows" and is the new
  default.
- FR-6.2: `ablation_eval_examples = N > 0` scores a seeded uniform random sample of `N` rows,
  drawn without replacement and in sorted order. The seed is `ablation_eval_seed`, a new
  `Config` field with default 0 that does not depend on the training seed. `N` greater than the
  split size means all rows.
- FR-6.3: Every arm result records a `test_eval` block:
  - `method`: `"full"` or `"random"`;
  - `num_examples`;
  - `total_rows`;
  - `seed`: `null` for a full split;
  - `indices_sha256`: the SHA-256 of the selected row indices, for a sample.

  `num_eval_examples` is kept.
- FR-6.4: `ablation_eval_examples` and `ablation_eval_seed` stay out of the training fingerprint
  because they are post-training only.

### FR-7: Popularity reference

- FR-7.1: The **popularity-only ranker** scores each candidate by the item's number of training
  interactions, counted over `train_user_history.pkl`. That is the same source the long-tail
  buckets use. It ranks the same candidate rows the model ranks and uses the FR-2 tie rule.
  Padding, MASK and items with no training interactions score 0.
- FR-7.2: These carry a `popularity_reference` with the same metric shape as the model's metrics
  (`{metric: {mean, std}}`), computed on exactly the rows and candidates they report:
  - every ablation arm result, on the test rows of FR-6;
  - `test_metrics_{model}.json`, on the candidates actually scored, including ANN candidates when
    `use_ann` is on;
  - `cold_start_{model}.json`, as `cold_popularity` and `warm_popularity`;
  - `longtail_{model}.json`, as `popularity_head`, `popularity_mid` and `popularity_tail`.
- FR-7.3: Each study JSON (the list of arm records) carries it per arm. `summary_{model}.json`
  carries per study:
  - `popularity_reference_ndcg`;
  - `best_ndcg`;
  - `best_minus_popularity_ndcg`.
- FR-7.4: The generic per-study bar chart (`ablation_{study}_{model}.png`) draws a dashed
  horizontal line at the popularity NDCG@10, labelled "popularity (NDCG@10)".

### FR-8: `--stage ablation_rescore` — re-evaluate a finished run without retraining

- FR-8.1: A new CLI stage, `--stage ablation_rescore`, takes `--study X|all` and `--model`. It is
  run with the same base flags as the run it rescores (`--seed`, `--user_sample_fraction`,
  `--dtype bf16 --mixed_precision`, an explicit `--attention_backend`).

  For every arm of the selected studies whose original arm file (`ablation_dir/seed_{seed}/arms/
  {study}_{model}/{label}.json`) has `status == "ok"`, it does the following:
  1. Locates the arm's backbone from the recorded `training_fingerprint` and that fingerprint's
     `backbone.json`. It does not recompute the fingerprint, because FR-4 changed it.
  2. Evaluates **every** `epoch_XXX.pt` present in the backbone's model checkpoint directory on
     the full validation split with the fixed metric.
  3. Re-selects the best epoch by the `run_pretrain` rule: highest val NDCG@10, earliest epoch on
     ties.
  4. Evaluates that epoch on the test rows of FR-6, with the popularity reference.
  5. Writes a rescored arm record.

  It trains nothing.
- FR-8.2: Checkpoints, original arm files and original study JSONs are never modified. The stage
  writes:
  - the per-fingerprint rescore record `by_fingerprint/<fp>/rescore.json`, written atomically after
    every epoch;
  - rescored arm records and study outputs under a separate results root,
    `{ablation_dir}/rescored/seed_{seed}/` (arms, study JSONs, summary);
  - rescored plots under `{plot_dir}/rescored/`.
- FR-8.3: It runs where the checkpoints are. That is the AWS VM for its studies and the H100 for
  its studies, in the run's own directory. It also runs locally on CPU or MPS against copied
  outputs. The device comes from the running config (`device="auto"`: CUDA, then MPS, then CPU;
  the CLI has no `--device` flag), and the attention backend from `--attention_backend`. The model
  is rebuilt from each
  checkpoint's stored `config`, with only runtime fields replaced from the CLI config:
  - device;
  - `attention_backend`;
  - `eval_batch_size`;
  - workers and pinned memory;
  - output, checkpoint, log, plot and ablation paths;
  - `processed_data_dir`.

  `check_processed_dir` must accept the result.
- FR-8.4: It is resumable and idempotent.
  - A fingerprint's per-epoch results are persisted as they complete, and a restarted process
    continues from the first unscored epoch.
  - A fingerprint whose record is complete for the current rescoring code and protocol is not
    re-evaluated.
  - An arm whose rescored record exists with the current `rescore_key` is skipped. The key covers
    the original `arm_key`, the fingerprint, the rescoring code's `source_hash` and the
    evaluation protocol.
  - Arms that share a fingerprint share one evaluation.
  - Concurrent processes on one machine (one per GPU, different studies) serialise per
    fingerprint through a file lock (`by_fingerprint/<fp>/.rescore.lock`).
- FR-8.5: Each rescored arm record is a copy of the original with these changes:
  - `metrics` is replaced by the rescored test metrics.
  - `best_epoch` and `best_val_ndcg` are replaced by the rescored values.
  - `popularity_reference` (test) and `val_popularity_reference` are added.
  - `test_eval` (FR-6.3) is added.
  - `code_revision` is set to the rescoring code's revision, so `--stage`-style collection
    accepts it as fresh.
  - `source_revision` (the training checkpoint's) and `runtime` (the training run's) are kept.
  - A `rescore` block is added with:
    - `rescored: true`;
    - `rescore_key`;
    - `rescored_by` (the rescoring code's full `source_revision`);
    - `original_code_revision`, `original_metrics`, `original_best_epoch`,
      `original_best_val_ndcg` and `original_num_eval_examples`;
    - `per_epoch_val_ndcg` (rescored), `per_epoch_original_val_ndcg` and `epochs_scored`;
    - `selected_checkpoint` (path);
    - `eval_runtime` (the device and backend the rescore ran on);
    - `caveat` (FR-8.7).
- FR-8.6: Extras that depend on the weights and were computed from the original `best.pt` with the
  defective pipeline move from the top level into `rescore.pre_rescore_extras`, so no plot or
  table reads them as current. These are `routing`, `gradient`, `cold_start`, `longtail`,
  `personalization` and `communication`. Weight-independent fields stay where they are:
  `router_parameters`, `latency`, `parameters` and `flops_per_inference`.
- FR-8.7: The rescore states its limit in three places: the `rescore.caveat` of every rescored arm
  record, the rescored `summary_{model}.json` (`rescore_caveat`), and a WARNING log line at the
  start and end of the stage. The statement is:

  > Rescoring gives honest numbers for these weights, but the weights were trained under the
  > stale-autocast-cache bug (eval-integrity FR-3). The A–H comparison requires a re-run with the
  > fixed training code.
- FR-8.8: After rescoring, the stage assembles each selected study from the rescored arm files
  with the existing `collect_study` machinery, pointed at the rescored root, plus the summary for
  `--study all`.
  - It exits 1 when any selected study could not be assembled, listing missing, failed or
    inconsistent arms as `ablation_collect` does.
  - An arm whose original file is missing or failed is reported and skipped, and its study is then
    incomplete.
- FR-8.9: Before any evaluation, every epoch checkpoint is checked against the arm. Its stored
  `config` must agree with the arm's backbone `best.pt` on every training-relevant field (the
  fingerprint's field set). Its `source_revision` must equal `best.pt`'s. On mismatch the arm
  fails and its own message names the offending file. The other arms continue.

### FR-9: Runbook

- FR-9.1: `scripts/package_ablation_run.py` writes `launch/rescore_{machine}.sh` per machine. It
  runs `--stage ablation_rescore --study <that machine's studies, one at a time>` with exactly
  the base flags of that machine's run commands, through `scripts/run_with_retry.py`.
- FR-9.2: The README ablation runbook gains a "Rescoring a finished run" section. It explains:
  - how to copy the new code tarball over the old code without touching `outputs/`;
  - how to run the script;
  - where the outputs land;
  - the FR-8.7 caveat;
  - that a re-run then retrains every arm (FR-4).

## Expected Behavior

- A mixed-precision forward after `optimizer.step()` always reflects the updated weights, inside
  `train_one_step` and in every other forward site.
- The val NDCG logged for an epoch can be reproduced from that epoch's checkpoint.
- Ablation arm JSONs, study JSONs, the summary and the generic study plot show the popularity
  floor next to the model numbers.
- After `--stage ablation_rescore`, `outputs/ablations/rescored/seed_42/` holds a complete,
  consistent set of study JSONs with honest metrics and an explicit caveat. The original results
  are untouched next to them.

## Edge Cases

- **All candidates tie** (for example a collapsed model, or a popularity ranker on cold items).
  The metric equals the random-ranking expectation: NDCG@10 = (Σ_{r=1..10} 1/log2(r+1)) / C.
- **Single candidate.** Rank 1, the same values as today.
- **NaN scores.** Raises `ValueError` (FR-2.5).
- **Pure-bf16 config** (`mixed_precision=False`). Autocast is a no-op, the forward runs in bf16
  weights, and scores are still fp32 (FR-1.1).
- **`train_one_step` called inside a caller's autocast region**, including a raw `torch.autocast`
  with the cache enabled. It still forwards with current weights (FR-3.3).
- **Rescore: a fingerprint registered by the multi-seed stage** has a `checkpoint_dir` outside the
  store. Epochs are read from `backbone.json`'s `checkpoint_dir`. `rescore.json` and the lock stay
  in the store directory.
- **Rescore: a backbone with fewer than `pretrain_epochs` epoch files** (for example a run cut
  short with an aborted epoch). Only the epochs present are scored, and `epochs_scored` lists
  them.
- **Rescore: a fingerprint shared by several arms or studies.** It is evaluated once.
- **Rescore: killed mid-fingerprint.** It resumes at the first unscored epoch.
- **Rescore: a process waiting on another's fingerprint lock** reuses the completed record.
- **Rescore: an arm file for a label the study no longer defines.** Ignored. The labels come from
  `resolve_study_configs`, as in `assess_study`.
- **Rescore: an original arm with `status != "ok"`.** Skipped and reported. The study is assembled
  only with `--allow_incomplete`.
- **`ablation_eval_examples` larger than the test split.** Treated as the full split, recorded
  with `method: "full"`.

## Constraints

- Owner directive: training precision is unchanged (fp32 master weights, bf16 autocast). Only the
  scoring arithmetic moves to fp32.
- **Deliberate exception to the code-quality rule "Training Hot Path and Reproducibility".** That
  rule says throughput changes must leave training bit-for-bit identical. FR-3 *intentionally*
  changes what mixed-precision training computes, because it is a correctness fix and not an
  optimisation. FR-4 is the provenance consequence. The pure-dtype golden test must stay green.
- Tests stay fast and use the tiny fixture (`tests/conftest.py`) or in-memory tensors. Run
  targeted files only, per `steering/testing.md`.
- Runs on the dev host (Mac, MPS/CPU) without CUDA. Nothing new imports FlashAttention.
- Superseded clauses:
  - The `evaluation-diagnostics` FR-3 tie behaviour, previously implicit in REV-076's
    "ties resolve in the target's favour".
  - The `ablation-run` design statement "one `autocast_context` around every stage".
  - The `ablation-run` "Out-of-scope findings" bullet on `ablation_eval_examples = 5000`, now
    addressed by FR-6.
  - The `Config` comment that justified capping ablation evaluation.

## Dependencies

- `evaluation/metrics.py`, `utils.py` (`autocast_context`), `training/pretrain.py`,
  `training/local_train.py`, `training/federated.py`, `evaluation/{cold_start,longtail,flops,
  gradient_diagnostics}.py`, `models/ann_retrieval.py`, `training/ablation_runner.py`, `main.py`,
  `configs/config.py`, `scripts/package_ablation_run.py`, `README.md`.
- The existing backbone store (`by_fingerprint/<fp>/`, `backbone.json`, `epoch_XXX.pt` per epoch;
  `run_pretrain` deletes none), arm persistence, `collect_study`, `file_lock` and
  `atomic_write_json`.
- Processed data artifacts: `val_/test_tensorised.pt`, `val_/test_candidates.npy`,
  `train_user_history.pkl`.

## Acceptance Criteria

- **AC-01 (ties, FR-2.1/2.3):** The tie test uses `C = 101` and `k = 10`.
  - (a) All scores equal: NDCG = Σ_{r=1..10} 1/log2(r+1) / 101, MRR = Σ_{r=1..10} (1/r) / 101,
    Recall = 10/101, Precision = 1/101, each within 1e-6.
  - (b) The target is tied with exactly one other candidate at the top: NDCG = (1 + 1/log2 3)/2
    and MRR = 0.75.
  - (c) The target is strictly highest but 50 other candidates tie each other: NDCG = 1.
  - (d) The batched values equal the single-example values row by row.
  - (e) Old behaviour now fails: for (a), the old rule (`1 + #higher`) would give NDCG 1.0, and
    the test asserts the new value is < 0.05.
- **AC-02 (ties, Monte Carlo, FR-2.1):** For random integer-valued score matrices (many ties), the
  tie-averaged metrics equal the average, over 2,000 random tie-breaking permutations, of the
  strict-rank metric, within 0.01.
- **AC-03 (no-tie identity, FR-2.2):** Every existing test in `tests/test_metrics.py` passes
  unmodified. On a random fp32 score matrix with no ties, the new metrics equal the old rule's
  exactly.
- **AC-04 (NaN, FR-2.5):** A score matrix containing NaN raises `ValueError`.
- **AC-05 (fp32 scoring, FR-1.1):** With a tiny model under `mixed_precision=True`, `dtype="bf16"`
  on CPU, `score_candidates_batch` returns `float32`. Its scores equal (atol 1e-6) an fp32
  reference: the user embedding captured from the same forward (via hook), upcast to fp32,
  normalised, and multiplied with the fp32-normalised candidate rows. This holds both with no
  enclosing region and inside an enclosing `torch.autocast(..., dtype=torch.bfloat16)`. With
  `mixed_precision=False`, the scores are also `float32`.
- **AC-06 (ANN fp32, FR-1.3):** Under the AC-05 setup, `retrieve_candidates` ranks by fp32 scores.
  The full-vocabulary score matrix it builds is `float32`, and its top-k equals the top-k of the
  fp32 reference.
- **AC-07 (scoring-site guard, FR-1.4):** A static test lists every `einsum(`/`@`/`matmul(`
  scoring expression under `evaluation/` and in `models/ann_retrieval.py`. It fails if any site
  outside the allow-list in design D10 appears.
- **AC-08 (stale-cache regression, FR-3.2/3.4):** In the real `train_one_step`, a tiny config with
  `mixed_precision=True` and `bf16` on CPU, AdamW with `lr = 0.05`, and no caller region:
  - At step 2, every `nn.Linear` in the model, the MoE and the loss functions that ran produces
    exactly the output of a fresh-region `F.linear` with its post-step-1 weights.
  - That output differs from the output with its pre-step-1 weights.
- **AC-09 (defence in depth, FR-3.3):** The AC-08 assertion also holds when both steps run inside
  one enclosing raw `torch.autocast(device_type="cpu", dtype=torch.bfloat16)` region with the
  cache enabled. A companion characterisation test shows that a bare Linear stepped inside such a
  region *does* forward stale weights, so AC-08 and AC-09 can detect the bug.
- **AC-10 (forward-site dtype, FR-3.2):** One parametrised test covers every forward site:
  `train_one_step`, `collect_validation_routing`, `score_candidates_batch`,
  `run_gradient_diagnostics`, `benchmark_latency`, `retrieve_candidates` and `train_user_lora`.
  Under `mixed_precision=True` with no caller region, a hooked `nn.Linear` inside the encoder
  produces `bfloat16` output. After the call returns, `torch.is_autocast_enabled("cpu")` is False.
- **AC-11 (no stage-wide region, FR-3.1):** `main(["--stage", <stub>, "--mixed_precision", ...])`
  calls the stage with `torch.is_autocast_enabled(<device>)` False. The stub is patched into
  `STAGE_MAP`.
- **AC-12 (LoRA uses current adapters, FR-3.2):** Under `mixed_precision=True`, the call sequence
  is: `evaluate_user`, then `load_lora_state_dict(non-zero B)`, then `evaluate_user`, all inside
  one enclosing raw autocast region. The two evaluations give different scores, as they do with
  `mixed_precision=False`.
- **AC-13 (golden unchanged, FR-3.5):** `tests/test_determinism.py` passes with its constants
  unmodified.
- **AC-14 (fingerprint version, FR-4):** `_training_fingerprint(Config())` changes when the
  semantics constant is changed (monkeypatched). The FR-4.2 invariance tests in
  `test_ablation_robustness.py`, `test_cli.py` and `test_training.py` pass unmodified. On the tiny
  fixture, a backbone registered under the legacy fingerprint (same payload without the version
  key) is not reused: `run_pretrain` is called.
- **AC-15 (selection consistency, FR-5.2):** Tiny fixture, `mixed_precision=True`, `bf16`, CPU,
  `pretrain_epochs = 3`. For every epoch, `epoch_{e}.pt["val_ndcg"]` equals post-load
  `evaluate_model` NDCG@10 within 1e-6. `best.pt["best_epoch"]` is the earliest argmax of those
  values. The `run_pretrain` result carries `val_popularity_reference`.
- **AC-16 (full test split, FR-6.1/6.3):** With `ablation_eval_examples = 0`, an arm on the tiny
  fixture records `num_eval_examples` equal to the number of test rows and `test_eval.method ==
  "full"`. Its metrics equal `evaluate_model` over all test rows of the loaded best checkpoint.
- **AC-17 (seeded sample, FR-6.2/6.3):** With `ablation_eval_examples = N` (smaller than the
  split), `test_eval` records `method: "random"`, `seed`, `num_examples = N` and
  `indices_sha256`.
  - Two runs give identical indices.
  - A different `ablation_eval_seed` gives different indices.
  - The selected rows are not the first `N`.
- **AC-18 (popularity helper, FR-7.1):** On a hand-built history, the popularity counts are exact,
  with pad and MASK scoring 0. On a hand-built candidate matrix with tied counts, the popularity
  metrics equal hand-computed tie-averaged values.
- **AC-19 (popularity everywhere, FR-7.2–7.4):** The files below carry their popularity fields
  with the `{metric: {mean, std}}` shape:
  - on the tiny fixture: an ablation arm result, the study JSON entries, the summary
    (`popularity_reference_ndcg`, `best_ndcg`, `best_minus_popularity_ndcg`), and
    `test_metrics_{model}.json`;
  - from the cold-start and long-tail runs: `cold_start_{model}.json` and
    `longtail_{model}.json`.

  The generic study plot function is called with a popularity line, verified by spying on `axhline`.
  Each of these files also carries the FR-1.5 `eval_protocol` stamp.
- **AC-20 (rescore end to end, FR-8.1/8.2/8.5/8.8):** On the tiny fixture, train study B (two
  arms, 3 epochs, `mixed_precision=True`, `bf16`, CPU) with the *current* code, then run
  `--stage ablation_rescore --study B`.
  - Each rescored arm record has `rescore.rescored: true`, `epochs_scored == [0, 1, 2]`,
    `best_epoch` equal to the argmax of `rescore.per_epoch_val_ndcg`, metrics equal to a direct
    `evaluate_model` of `epoch_{best}.pt` on all test rows, `popularity_reference`, `test_eval`,
    `rescore.original_metrics` equal to the original arm's `metrics`, and the caveat text.
  - The rescored study JSON and summary exist under `ablations/rescored/seed_{seed}/`.
  - Every original arm file, study JSON and checkpoint is byte-identical before and after (SHA-256).
- **AC-21 (rescore of a post-fix run is a no-op in value, FR-8.1):** For the AC-20 run, which used
  fixed code, `rescore.per_epoch_val_ndcg[e]` equals `epoch_{e}.pt["val_ndcg"]` within 1e-6 for
  every epoch. The rescored `best_epoch` equals the original.
- **AC-22 (rescore idempotent, FR-8.4):** Running the AC-20 stage a second time calls
  `evaluate_model` 0 times, leaves the rescored files byte-identical and exits 0.
- **AC-23 (rescore resumable, FR-8.4):** An exception injected into the second epoch evaluation of
  a fingerprint fails the arm. A re-run then evaluates only the remaining epochs: `evaluate_model`
  is called once per unscored epoch plus the test passes, never again for epoch 0.
- **AC-24 (rescore shares fingerprints, FR-8.4):** Two arms sharing one fingerprint trigger one
  per-epoch evaluation sequence, not two. The test may use, for example, study E's `global_only`
  and `global_lora`, or two original arm records that reference one trained fingerprint.
- **AC-25 (rescore config identity, FR-8.9):** Corrupting one epoch checkpoint's stored config on
  a training-relevant field (for example `num_experts`) fails that arm with a message naming the
  file. The other arm is still rescored.
- **AC-26 (extras quarantined, FR-8.6):** A rescored arm whose original carried `routing` and
  `latency` has `latency` at the top level and `routing` only under
  `rescore.pre_rescore_extras`.
- **AC-27 (packager, FR-9.1):** On the tiny fixture, the packager writes
  `launch/rescore_{machine}.sh` for each machine. It passes `bash -n`, contains
  `--stage ablation_rescore`, contains each of that machine's studies, and carries the same base
  flags as that machine's run commands, including `--dtype bf16 --mixed_precision` and the
  backend. The existing packager tests pass unmodified.
- **AC-28 (Mac smoke, capped):** A real-data capped run with `--mixed_precision` and `--max_steps`
  of a few hundred steps on MPS, 2 epochs, shows each epoch's logged val NDCG equal to a post-load
  `--stage evaluate`-style re-evaluation of that epoch on val within 1e-4. MPS kernels may differ
  run to run at that level. The quoted numbers are read back from the artifacts.

## Out-of-scope findings (named, not fixed)

- `training/federated.py` bounds each round's validation pass to the first
  `federated_eval_examples` rows of the user-sorted validation split (REV-112(b)). This is the
  same few-users bias as FR-6, but for the federated convergence curve.
- LoRA personalisation and federated results produced by mixed-precision runs before this fix
  were computed under the stale cache. The LoRA A/B casts were frozen at the first forward of the
  stage, so "with LoRA" could not differ from the first-seen adapter. These results must be
  regenerated by the re-run. The rescore quarantines them (FR-8.6) and does not repair them.
- The sibling note `study_a_gap_findings.md` gives the arm-file path as
  `outputs/ablations/arms/A_t_base/…`. The code writes `outputs/ablations/seed_{seed}/arms/
  A_t_base/…` (ablation-run FR-11).
