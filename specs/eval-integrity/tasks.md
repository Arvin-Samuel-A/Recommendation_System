# Evaluation Integrity — Tasks

Ordered. The metric and scoring fixes land first (Phase 1), then the training-correctness fix
(Phase 2), then reporting (Phase 3), then the rescore (Phase 4). Each task names the requirements,
design sections and ACs it implements, and ends with a **Done when** condition that can be checked
on the Mac (no CUDA).

## Rules for BUILD on this feature

- **Never run the whole test suite** (`steering/testing.md`). Run only the files a task names:
  `uv run python -m pytest tests/<file>.py -m "not slow" -q`, adding `-k` to narrow. Check
  `df -h .` before anything that writes data. The dev disk was at 92% on 2026-10-16.
- New tests use the tiny fixture (`tests/conftest.py`: `tiny_processed_dir`, `tiny_config`) or
  in-memory tensors, on CPU. Never copy `data/processed*/` or any real artifact into a test.
- Always `uv run python`.
- Training stays bf16 mixed precision (owner directive). Do not change what the training step
  computes beyond the autocast scoping in T2.1.
- `tests/test_determinism.py` (golden, pure bf16) must stay green **with its constants untouched**
  from T1.2 onward. If it goes red, the change is wrong; do not re-capture the constants.
- Do not commit or push; the orchestrator does that.

## Phase 1 — Metrics and scoring (land first)

- [x] **T1.1: Tie-averaged ranking metrics** *(FR-2; design D1; AC-01–AC-04)*
  - **Code.** Rewrite `evaluation/metrics.py::compute_metrics_batch` to the tie-averaged form in
    D1:
    - `h` = candidates strictly higher than the target, `ties` = other candidates equal to it;
    - a cumulative per-rank table for each metric, gathered on the device;
    - no sort and no per-example `.item()`;
    - `ValueError` when any score is NaN.
  - **Docstring.** Update it: drop REV-076's "ties resolve in the target's favour" and state the
    new rule. `compute_metrics` stays a thin wrapper.
  - **Tests.** Add class `TestTieAveraged` to `tests/test_metrics.py` covering AC-01(a)–(e),
    AC-02 (Monte Carlo over 2,000 random tie-breaks, tolerance 0.01), AC-03 (no-tie identity
    against the old formula, written inline in the test) and AC-04. Do not modify the existing
    tests.
  - **Done when:** `uv run python -m pytest tests/test_metrics.py -q` passes. The old tests
    are unchanged (`git diff` on them is empty apart from the new class).

- [x] **T1.2: Autocast helpers with the weight cache disabled** *(FR-3.3; design D3, D4; part of
  AC-09)*
  - **Code.** In `utils.py`:
    - `autocast_context(config, device_type=None)` gains `cache_enabled=False` and the optional
      `device_type` (default `get_device(config).type`);
    - add `autocast_disabled(device_type)`;
    - put the D3 docstring (the cache hazard) on `autocast_context`.
  - **Tests.** In the new `tests/test_eval_integrity.py`:
    - (a) the characterisation test: a bare `nn.Linear` stepped with AdamW (`lr = 0.5`) inside one
      raw `torch.autocast("cpu", torch.bfloat16)` region forwards a byte-identical output after
      the step. This documents the torch behaviour that caused the bug.
    - (b) the same loop with each forward in its own `autocast_context(cfg, "cpu")` nested inside
      that raw region forwards the updated weights, matching a fresh-region `F.linear`.
  - **Done when:** both tests pass, and `uv run python -m pytest tests/test_determinism.py -q`
    passes unmodified.

- [x] **T1.3: fp32 scoring at the choke point** *(FR-1.1, FR-1.2; design D2; AC-05)*
  - **Code.** Restructure `evaluation/metrics.py::score_candidates_batch` as in D2:
    - forward plus MoE under `autocast_context(config, device_type=<param device type>)`;
    - then, under `autocast_disabled`, upcast the user and candidate embeddings to fp32,
      `l2_normalize` them, take the `einsum` and apply the optional temperature;
    - assert the result is `float32`.

    Update the REV-044 comment. Callers stay unchanged.
  - **Tests.** Add AC-05 to `tests/test_eval_integrity.py` (tiny model built in memory, CPU):
    - mixed bf16 with no caller region;
    - mixed bf16 inside an enclosing `torch.autocast`;
    - `mixed_precision=False`.

    Capture the user embedding with a forward hook on the MoE layer (or on the model for
    `moe_variant="none"`).
  - **Done when:** AC-05 passes. `tests/test_metrics.py`, `tests/test_diagnostics.py`,
    `tests/test_moe.py` and `tests/test_ann_retrieval.py` pass unmodified.

- [x] **T1.4: ANN retrieval scores in fp32** *(FR-1.3; design D2; AC-06)*
  - **Code.** In `models/ann_retrieval.py`:
    - `_user_embeddings` runs its forward and MoE in its own `autocast_context(config,
      device_type=...)` and returns the un-normalised embedding;
    - `retrieve_candidates` normalises the fp32 item table once and the fp32 user embeddings per
      batch, under `autocast_disabled`, before the existing fp32 matmul.
  - **Tests.** Add AC-06 to `tests/test_eval_integrity.py`.
  - **Done when:** AC-06 passes and `tests/test_ann_retrieval.py` passes unmodified.

- [x] **T1.5: Scoring-site guard** *(FR-1.4; design D10; AC-07)*
  - **Code.** Add `test_no_unaudited_scoring_sites` to `tests/test_eval_integrity.py`. It uses
    `ast` to find the enclosing function of each `einsum(`, ` @ ` and `matmul(` occurrence under
    `evaluation/` and in `models/ann_retrieval.py`.
  - **Allow-list.** It holds `score_candidates_batch`, `retrieve_candidates`, and any non-scoring
    arithmetic found, each with a one-line justification.
  - **Done when:** the test passes. Temporarily adding an `einsum` scoring line to
    `evaluation/longtail.py` makes it fail. Record that check in the BUILD note, then revert it.
  - **BUILD note (2026-10-16).**
    - Appending `def _probe_scores(u, items): return torch.einsum("bd,bcd->bc", u, items)` to
      `evaluation/longtail.py` made `test_no_unaudited_scoring_sites` fail, naming the
      unaudited site.
    - The file was then restored with `git checkout evaluation/longtail.py`, and the test
      passed again (1 passed).

- [x] **T1.6: Popularity ranker and protocol stamp helpers** *(FR-7.1, FR-1.5; design D8; AC-18)*
  - **Code.** Add `EVAL_PROTOCOL`, `item_popularity`, `load_item_popularity`, `popularity_scores`
    and `evaluate_popularity` to `evaluation/metrics.py`, exactly as D8 specifies. Ranking uses
    `compute_metrics_batch`, so ties are tie-averaged.
  - **Tests.** Add AC-18 to `tests/test_eval_integrity.py`. One case is a hand-built
    `train_user_history` with pad/MASK absent and two items sharing a count. Another is a
    hand-computed tie-averaged NDCG.
  - **Done when:** AC-18 passes.

## Phase 2 — Autocast scoping (training-correctness fix)

- [x] **T2.1: `train_one_step` owns its autocast region; stale-cache regression test** *(FR-3.2,
  FR-3.4; design D3, D5; AC-08, AC-09)*
  - **Region.** In `training/pretrain.py::train_one_step`:
    - delete the `torch.is_autocast_enabled` assertion;
    - wrap everything from the first `model(masked_seq, …)` forward through the computation of
      `L_total` in `autocast_context(config, device_type=device.type)`. That includes the
      contrastive augmented forward, every MoE call and every loss.
  - **Outside the region.** The dtype asserts, finite check A, `zero_grad`, backward, the
    GradNorm update, clipping, check B, the optimizer and scheduler steps, and `moe.clear_cache()`
    all stay outside, unchanged and in the same order.
  - **Tests.** Add AC-08 and AC-09 to `tests/test_eval_integrity.py`. Use the real
    `train_one_step` on the in-memory tiny setup that `tests/test_training.py::TestMoECacheLifetime`
    / `TestMixedPrecision` use, with `mixed_precision=True`, `bf16` and AdamW `lr = 0.05`.
    - Register forward hooks on every `nn.Linear` in `model`, `moe` and the loss functions.
    - Snapshot the weights before step 1 and after step 1.
    - At step 2, assert that every Linear that ran gives output `torch.equal` to
      `F.linear(input, W_after_step1, b_after_step1)` computed in a fresh
      `autocast_context(cfg, "cpu")`, and not equal to the output with the pre-step-1 weights.
  - **Existing tests.** In `tests/test_training.py::TestMixedPrecision`, replace
    `test_step_refuses_to_run_without_autocast` with `test_step_runs_its_own_autocast_region`
    (no caller region, a hooked encoder Linear outputs bf16, losses are fp32). Keep
    `test_step_trains_fp32_master_weights_under_autocast` as it is.
  - **Done when:**
    - AC-08 and AC-09 pass;
    - `uv run python -m pytest tests/test_training.py -m "not slow" -q` passes;
    - `tests/test_determinism.py` passes unmodified (AC-13).

- [x] **T2.2: Every other forward site owns its region** *(FR-3.2; design D3, D5; AC-10, AC-12)*
  - **Code.** Add `autocast_context(config, device_type=<param device type>)` around exactly the
    D3 table regions in:
    - `training/pretrain.py::collect_validation_routing`;
    - `evaluation/gradient_diagnostics.py::run_gradient_diagnostics` (forwards and losses only;
      `autograd.grad` outside);
    - `evaluation/flops.py::benchmark_latency` (each warm-up and timed forward, inside the timed
      window);
    - `training/local_train.py::train_user_lora` (forward, MoE and loss; backward and the scaler
      steps outside).
  - **Tests.** Add AC-10 (parametrised over all seven sites, including the ones done in
    T1.3/T1.4/T2.1) and AC-12 (LoRA inside an enclosing raw autocast region) to
    `tests/test_eval_integrity.py`.
  - **Done when:** AC-10 and AC-12 pass. `tests/test_lora.py`, `tests/test_federated.py`,
    `tests/test_diagnostics.py` and `tests/test_training.py -m "not slow"` pass.

- [x] **T2.3: Remove the stage-wide autocast region** *(FR-3.1; design D3; AC-11, AC-13)*
  - **Code.** In `main.py::main`, remove `with autocast_context(config):` around
    `STAGE_MAP[args.stage](config, args)` and its comment. Keep the OOM handling, and replace the
    comment with one pointing at eval-integrity FR-3.
  - **Tests.** Add AC-11 to `tests/test_eval_integrity.py`: patch a stub stage into `STAGE_MAP`
    and `VALID_STAGES` via `monkeypatch`, run with `--mixed_precision`, and assert autocast is
    disabled inside it.
  - **Done when:** AC-11 passes, and `tests/test_cli.py` and `tests/test_determinism.py` pass.

- [x] **T2.4: Training-semantics version in the fingerprint** *(FR-4; design D6; AC-14)*
  - **Code.** In `training/ablation_runner.py`:
    - add `TRAINING_SEMANTICS_VERSION = 2`;
    - extract `_training_fields(cfg_dict)`;
    - make `_training_fingerprint` add `"_training_semantics"` to the hashed payload.

    `ablation_eval_seed` joins `POST_TRAINING_ONLY_FIELDS` in T3.1, together with the field.
  - **Tests.** Add AC-14 to `tests/test_eval_integrity.py`:
    - monkeypatching the version changes the fingerprint;
    - a backbone registered under the legacy fingerprint (`_training_fields` payload without the
      key) on the tiny fixture is not reused, which is checked by spying on `run_pretrain`.
  - **Done when:** AC-14 passes, and `tests/test_ablation_robustness.py`, `tests/test_cli.py`,
    `tests/test_training.py -m "not slow"` and `tests/test_scripts.py` pass unmodified.

- [x] **T2.5: Selection on the fixed metric; validation popularity reference** *(FR-5; design D8;
  AC-15)*
  - **Code.** In `run_pretrain`:
    - compute `val_popularity_reference` once, with `load_item_popularity` and
      `evaluate_popularity` over `val_candidate_tensors`;
    - write it to the per-epoch log once, as a header line through `epoch_logger`. It must not
      break `_truncate_epoch_log`, which only drops lines matching `epoch=N`;
    - return it in the result dict.

    Selection logic is otherwise unchanged; it is fixed by T1.1–T2.1.
  - **Tests.** Add AC-15 to `tests/test_eval_integrity.py`: tiny fixture, `tiny_config(...,
    mixed_precision=True, dtype="bf16", pretrain_epochs=3)`, CPU. For each epoch, reload
    `epoch_{e}.pt` into a fresh `build_model` and compare its `evaluate_model` NDCG to the stored
    `val_ndcg`.
  - **Done when:** AC-15 passes and `tests/test_resume.py` passes. The test fails on a checkout
    with T2.1 reverted; record that check in the BUILD note.
  - **BUILD note (2026-10-16).**
    - A literal revert of T2.1 restores the `torch.is_autocast_enabled` assertion. With no
      stage-wide region (T2.3), `run_pretrain` then aborts with "autocast is not active", so
      AC-15 fails.
    - The pre-fix autocast pattern was also simulated on the tiny fixture with a scratch
      script, run once each way (mixed bf16, 3 epochs, CPU):
      - pre-fix: one stage-wide raw `torch.autocast` region, and `autocast_context` with
        the cache ON in `evaluation.metrics`, `training.pretrain` and
        `models.ann_retrieval`;
      - fixed: the current code.
    - Stored `val_ndcg` vs post-load NDCG@10:

      | Epoch | Pre-fix pattern | Fixed code |
      |---|---|---|
      | 0 | 0.451784 vs 0.444028 | 0.400627 vs 0.400627 |
      | 1 | 0.451976 vs 0.451806 | 0.404857 vs 0.404857 |
      | 2 | 0.451202 vs 0.446483 | 0.419021 vs 0.419021 |

      AC-15's 1e-6 check therefore fails under the pre-fix pattern and passes on the fixed
      code.

## Phase 3 — Honest reporting

- [x] **T3.1: Ablation test on the full split, with test-eval record, popularity and protocol**
  *(FR-6, FR-7.2, FR-1.5; design D7, D8; AC-16, AC-17, part of AC-19)*
  - **Config.** In `configs/config.py`:
    - `ablation_eval_examples` defaults to 0 (= all rows); rewrite its comment;
    - add `ablation_eval_seed: int = 0`, and add it to `POST_TRAINING_ONLY_FIELDS` in
      `training/ablation_runner.py`;
    - `validate()` rejects negative values for either.
  - **Row selection.** In `training/ablation_runner.py`, add `select_eval_rows` (D7).
    `_run_single_config` scores the selected rows (gather only the selected candidate rows from
    the memory-mapped matrix) and records `test_eval`, `num_eval_examples`, `popularity_reference`
    (popularity over the same rows) and `eval_protocol`.
  - **Tests.** Add AC-16 and AC-17 to `tests/test_eval_integrity.py`.
  - **Done when:** AC-16 and AC-17 pass. `tests/test_ablation_robustness.py`,
    `tests/test_ablation_reseed.py`, `tests/test_ablation_parallel.py -m "not slow"` and
    `tests/test_cli.py` pass. If an existing test asserted prefix rows, update it and say why in
    the BUILD note.

- [x] **T3.2: Popularity in study summary and plot** *(FR-7.3, FR-7.4; design D8; part of AC-19)*
  - **Summary.** `_save_summary` adds per-study `best_ndcg`, `popularity_reference_ndcg` and
    `best_minus_popularity_ndcg`, and accepts an optional `extra` dict merged at top level.
  - **Plot.** `_plot_study` draws the dashed popularity line when available.
  - **Tests.** Add the summary, study-JSON and `axhline`-spy parts of AC-19 to
    `tests/test_eval_integrity.py`.
  - **Done when:** those parts pass and `tests/test_cli.py` passes.

- [x] **T3.3: Popularity and protocol in the evaluate, cold-start and long-tail outputs**
  *(FR-7.2, FR-1.5; design D8; part of AC-19)*
  - **`--stage evaluate`.** `main.run_evaluate_stage` adds `popularity_reference` over the scored
    candidate tensors (ANN candidates when ANN is on) and `eval_protocol` to
    `test_metrics_{model}.json`.
  - **Cold start.** `run_cold_start_evaluation` adds `cold_popularity` and `warm_popularity`.
  - **Long tail.** `run_longtail_evaluation` adds `popularity_head`, `popularity_mid` and
    `popularity_tail`.

    Both use the same per-bucket selectors as the model metrics, and both `_save_report`s write
    `eval_protocol`. The plots must ignore the new keys.
  - **Tests.** Add the remaining AC-19 parts to `tests/test_eval_integrity.py` (tiny fixture with
    a briefly trained or freshly built model).
  - **Done when:** AC-19 passes in full, and `tests/test_diagnostics.py` and `tests/test_cli.py`
    pass.

## Phase 4 — Rescore the finished run

- [x] **T4.1: Per-fingerprint rescore engine** *(FR-8.1, FR-8.3, FR-8.4, FR-8.9; design D9;
  AC-21, AC-23, AC-24, AC-25)*
  - **Module.** Create `training/ablation_rescore.py` with:
    - `RESCORE_CAVEAT` (the FR-8.7 text);
    - `config_from_checkpoint`;
    - `EvalData` (lazy shared val/test data, candidates, popularity references and test-row
      selection, keyed by `max_seq_length`);
    - `rescore_fingerprint`.
  - **`rescore_fingerprint` behaviour.**
    - Takes the `.rescore.lock` file lock in the store directory.
    - Reads `rescore.json` and checks it is valid under DEC-8.
    - Checks each epoch's identity against `best.pt` with `_training_fields` and
      `source_revision`.
    - Loads epochs into one reused model and evaluates val with `evaluate_model`.
    - Writes atomically after each epoch.
    - Selects the best epoch (earliest argmax), runs the test pass with popularity, and marks the
      record complete.
  - **Tests.** Add AC-21, AC-23, AC-24 and AC-25 to `tests/test_eval_integrity.py`, against a
    tiny `run_ablation` B study trained with `mixed_precision=True`. AC-24 may use fabricated arm
    records sharing one fingerprint.
  - **Done when:** these ACs pass.

- [x] **T4.2: Rescored arm records, collection and the CLI stage** *(FR-8.1, FR-8.2, FR-8.5–FR-8.8;
  design D9; AC-20, AC-22, AC-26)*
  - **Module.** In `training/ablation_rescore.py`:
    - `rescored_config` (`ablation_dir` and `plot_dir` with `rescored` appended);
    - `rescore_arm`: the `rescore_key`, skip when current, the record built per FR-8.5, the
      extras quarantined per FR-8.6, and an atomic write to the rescored root;
    - `run_ablation_rescore`: the caveat WARNINGs at start and end; skipped or failed arms
      reported; OOM re-raised; `collect_study` on the rescored config; `_save_summary(...,
      extra={"rescore_caveat": ...})` for `all`.
  - **CLI.** In `main.py`, add `"ablation_rescore"` to `VALID_STAGES`, add
    `run_ablation_rescore_stage` (exits 1 when any study is not written, like
    `run_ablation_collect_stage`) and add the `STAGE_MAP` entry. `--allow_incomplete` applies.
  - **Tests.** Add AC-20, AC-22 and AC-26 to `tests/test_eval_integrity.py`. AC-20 checks
    SHA-256 of every original arm file, study JSON and checkpoint before and after.
  - **Done when:** AC-20, AC-22 and AC-26 pass and `tests/test_cli.py` passes.

- [x] **T4.3: Rescore launch script and runbook** *(FR-9; design D11; AC-27)*
  - **Packager.** Add `render_rescore_script` to `scripts/package_ablation_run.py` and write
    `launch/rescore_{machine}.sh` for each machine (D11).
  - **README.** Add the section "Rescoring a finished run" under "Ablation run (10% users)". It
    covers:
    - copying the new code tarball over the old code without touching `outputs/` or `data/`;
    - running `bash launch/rescore_<machine>.sh`;
    - the output locations (`outputs/ablations/rescored/seed_42/`,
      `by_fingerprint/<fp>/rescore.json`, `outputs/plots/rescored/`);
    - local CPU/MPS use (MPS recommended: about 8 min per fingerprint, against about 3 h on CPU);
    - the FR-8.7 caveat, word for word;
    - that the next `--stage ablation` retrains every arm (FR-4).
  - **Tests.** Add AC-27 to `tests/test_scripts.py`.
  - **Done when:** AC-27 passes and every existing `tests/test_scripts.py` test passes unmodified.

## Phase 5 — Verification

- [x] **T5.1: Targeted sweep** *(all ACs except AC-28)*
  - **Command.** Check `df -h .` first, then run each file separately with `-m "not slow" -q`:
    `test_metrics`, `test_eval_integrity`, `test_determinism`, `test_training`,
    `test_ann_retrieval`, `test_diagnostics`, `test_moe`, `test_lora`, `test_federated`,
    `test_resume`, `test_ablation_robustness`, `test_ablation_reseed`, `test_ablation_parallel`,
    `test_cli`, `test_scripts`, `test_smoke`.
  - **Done when:** every listed file passes. The BUILD note records the pass and skip counts per
    file and the time `test_eval_integrity.py` takes, which must be under 3 min on the Mac.
  - **BUILD note (2026-10-16).**
    - Disk check before the sweep: 47 GiB free (90% used).
    - Each file was run separately with `uv run python -m pytest tests/<f>.py -m "not slow"
      -q -rs`. Nothing failed and nothing was skipped.

      | File | Result |
      |---|---|
      | test_metrics | 14 passed |
      | test_eval_integrity | 49 passed in 56.6 s, under 3 min |
      | test_determinism | 6 passed |
      | test_training | 26 passed, 2 deselected |
      | test_ann_retrieval | 9 passed |
      | test_diagnostics | 22 passed |
      | test_moe | 19 passed |
      | test_lora | 7 passed |
      | test_federated | 13 passed, 6 deselected |
      | test_resume | 15 passed |
      | test_ablation_robustness | 31 passed |
      | test_ablation_reseed | 2 passed |
      | test_ablation_parallel | 33 passed |
      | test_cli | 63 passed |
      | test_scripts | 31 passed |
      | test_smoke | 2 passed |

    - Environment: the worktree has no `data/processed*`. The data-dependent tests in
      `test_training` and `test_resume` ran with those directories symlinked read-only to
      the main checkout's; the symlinks were removed afterwards.
    - The review fixes (REV-001, REV-002, REV-004) added tests to `test_eval_integrity.py`
      after this sweep. Their re-run is recorded in the hand-off.

- [x] **T5.2: Mac capped real-data smoke** *(AC-28)*
  - **Precondition.** `df -h .` shows at least 15 GB free; otherwise stop and report.
  - **Set aside.** Move the existing `outputs/checkpoints/t_small/` (an old 200-step smoke) to
    `outputs/checkpoints/t_small.pre-eval-integrity/`, and move it back afterwards. The CLI has no
    output-directory flag.
  - **Run.** On MPS (`device="auto"`), against `data/processed_users10pct`:
    `uv run python main.py --stage pretrain --model t-small --user_sample_fraction 0.1 --dtype
    bf16 --mixed_precision --attention_backend sdpa --epochs 2 --max_steps 200 --no_resume`.
  - **Re-evaluate.** Write a throwaway script in the scratchpad that reloads each `epoch_00{e}.pt`
    and runs `evaluate_model` on the full val split.
  - **Done when:** each epoch's stored `val_ndcg` and its post-load value agree within 1e-4. The
    `run_pretrain` log header carries the val popularity reference. All numbers in the BUILD note
    are read back from the artifacts, not from the console.
  - **BUILD note (2026-10-16).**
    - 46 GiB free, so the 15 GB precondition held.
    - The worktree had no `outputs/checkpoints/t_small/`, so nothing needed setting aside.
      `data/processed_users10pct` was symlinked read-only to the main checkout's.
    - The listed command ran on MPS (`sdpa on mps`), exited 0 and took 2 min 53 s.
    - A scratchpad script reloaded each `epoch_00{e}.pt` into a freshly built model and ran
      `evaluate_model` on the full val split (100,591 rows, MPS). Values are read from the
      checkpoints:

      | Epoch | Stored `val_ndcg` | Post-load |
      |---|---|---|
      | 0 | 0.275768 | 0.275768 (difference 0.0) |
      | 1 | 0.313153 | 0.313153 (difference 0.0) |

    - `best.pt` has `best_epoch` 1 and `best_val_ndcg` 0.3131531774997711, mixed bf16.
    - Line 1 of `outputs/logs/pretrain_t_small.log` is the header `val_popularity_reference
      | popularity_ndcg@10=0.407031 | popularity_mrr@10=0.323166 |
      popularity_recall@10=0.679448 | popularity_precision@10=0.067945 | rows=100591 |
      eval_protocol_version=2`. Lines 2–3 are the `epoch=0` and `epoch=1` lines.
    - The smoke outputs and the symlink were deleted afterwards.
