# Review — Training Pipeline

## Review Status

**CHANGES_REQUIRED**

## Feature

`specs/training-pipeline` — `training/pretrain.py`: LR scheduling, model construction, the four-objective forward path, GradNorm integration, checkpointing, per-epoch logging, validation, and `run_pretrain`.

## Review Scope

- Spec directory: `specs/training-pipeline/` (requirements.md, design.md, tasks.md)
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`, `specs/BUILD_ORDER.md`
- Baseline: working tree vs `HEAD` = `32b0904` on `master` (`origin/master` at the same commit). All feature work is uncommitted.
- Files inspected: `training/pretrain.py` (904 lines, modified), `training/gradnorm.py`, `training/losses.py`, `data/sequence_generator.py`, `evaluation/metrics.py`, `tests/test_training.py` (592 lines, modified), `tests/test_smoke.py`
- Re-review of the prior `review.md`. `pretrain.py` (21:45), `tests/test_training.py` (21:59) and `tests/test_smoke.py` (21:59) were all modified after it.
- Date: 2026-09-15

## Summary

Three of the four prior CRITICAL/HIGH findings are genuinely fixed, and I verified each rather than trusting the inline comments. REV-057 (GradNorm leaking second-order gradients into the anchor) is closed — instrumenting a real step shows **zero** model parameters have their `.grad` altered by `gradnorm.update()`. REV-058 (the 60 GB dense metadata block) is closed — `RecDataset` holds the 3.2 MB `(vocab, F)` matrix and gathers per item. REV-059 (the MIM metadata leak) is closed — `mask_metadata` zeroes the metadata rows at masked positions and three tests cover it. REV-063 (one forward per example at validation) is closed — `evaluate_model` batches at `eval_batch_size`. REV-066 is closed: `requirements.md`, `steering/testing.md` and `BUILD_ORDER.md` all carry their committed wording again, and T6.2/T6.3 are correctly left unchecked.

What blocks acceptance is not this spec's own code. `uv run python -m pytest` fails two tests, both of them this spec's smoke gates, and both for the reason `specs/transformer-backbone` owns: FlexAttention has no backward pass on MPS, and BUILD correctly armed `require_attention_backend` rather than certifying SDPA silently. I confirmed the pipeline beneath is sound — with the requirement neutralised at runtime and no repository file touched, both gates pass. So AC-04 is not FAIL, it is NOT VERIFIABLE pending a spec decision.

## Git Changes

| File | State |
|---|---|
| `training/pretrain.py` | modified (+~770 lines vs `32b0904`) |
| `training/gradnorm.py`, `training/losses.py` | modified (reviewed under `specs/loss-functions`) |
| `data/sequence_generator.py` | modified (reviewed under `specs/data-pipeline`) |
| `evaluation/metrics.py` | modified (reviewed under `specs/evaluation-diagnostics`) |
| `tests/test_training.py`, `tests/test_smoke.py` | modified |
| `specs/training-pipeline/tasks.md` | modified — BUILD notes on T6.2/T6.3 |
| `specs/training-pipeline/review.md` | untracked (this file) |

`specs/training-pipeline/requirements.md`, `steering/testing.md` and `specs/BUILD_ORDER.md` show **no** modification against `32b0904` — the REV-066 reverts landed.

**Scope assessment:** in scope. No new dependencies. `outputs/checkpoints/` is empty, so nothing trained has been committed or left behind.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 LR scheduler | PASS | `build_scheduler` ([pretrain.py:48](training/pretrain.py:48)) is linear warmup over `warmup_ratio * total_steps` then cosine to 0; `total_steps = num_batches_per_epoch * pretrain_epochs`; `scheduler.step()` is called once per batch inside `train_one_step` ([:320](training/pretrain.py:320)); `log_epoch` records `lr=`. |
| FR-2 Per-model checkpoint dirs | PASS | `Config.model_checkpoint_dir()` → `outputs/checkpoints/{t_small\|t_base\|t_large}/`; `epoch_{N:03d}.pt` and `best.pt` written at [:866](training/pretrain.py:866) and [:876](training/pretrain.py:876). |
| FR-3 Checkpoint format | PASS | `save_checkpoint` ([:68](training/pretrain.py:68)) stores epoch, model/moe/optimizer/scheduler/gradnorm state, all four val metrics and `config`; `is_best=True` adds the full named metadata block (best_epoch, four best metrics, seed, model_name, moe_variant, use_gradnorm, num_experts). MoE and GradNorm state live in sibling keys rather than inside `model_state_dict` — functionally equivalent, and `load_checkpoint` restores all of it. `loss_fns_state_dict` additionally preserves the `MultiStepLoss` projections (REV-047). |
| FR-4 Best checkpoint selection | PASS | `if val_ndcg > best_val_ndcg` at [:873](training/pretrain.py:873) — strict improvement on validation NDCG@10 only. No test split is loaded anywhere in `run_pretrain`; the only datasets opened are `train_tensorised.pt` and `val_tensorised.pt`. |
| FR-5 Training loop | PASS | Negative pool loaded once outside the epoch loop ([:713](training/pretrain.py:713)) and never rebuilt; masking applied before the forward pass; contrastive second forward gated on `batch_idx % contrastive_interval`; all four losses combined per §7; `log_epoch` writes every required field to `outputs/logs/pretrain_{model}.log`. |
| FR-6 ARV-MMoE + GradNorm forward path | PASS | `train_one_step` ([:145](training/pretrain.py:145)) follows the sequence exactly: `return_all_hidden=True` → `pooled = all_hidden[:, -1, :]` (the last non-pad position under left padding) → per-objective MoE forward → per-objective loss → routing snapshots buffered → weighted sum → `backward(retain_graph=gradnorm is not None)` → `gradnorm.update` → `clip_grad_norm_` → `optimizer.step` → `scheduler.step`. |
| FR-7 BF16 training | PASS | `assert_bf16_params` before the loop; per-step dtype assertions on `all_hidden` and on each loss (with the documented fp32 InfoNCE exception); `zero_grad(set_to_none=True)` at [:299](training/pretrain.py:299) and [:695](training/pretrain.py:695); finite assertions on losses, total, gradients and lambdas. No autocast, no GradScaler. |
| FR-8 Validation | PASS | `evaluate_model` runs after every epoch under `no_grad` in eval mode ([:847-853](training/pretrain.py:847)) against the pre-stored `val_candidates.npy`, which was built from `eval_user_history` at preprocessing time. All four metrics reported. |
| FR-9 GradNorm lambda logging | PASS | `log_gradnorm_lambdas` every 100 steps to `outputs/logs/gradnorm_{model}.jsonl` (stale file removed at start of run, [:776](training/pretrain.py:776)); `plot_gradnorm_trajectory` writes `gradnorm_lambda_trajectory_{model}.png` at 150 DPI after the last epoch. |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 warmup→cosine, per-batch stepping | PASS | `TestLRScheduler::test_warmup_increases`, `test_cosine_reaches_near_zero`, `test_per_batch_stepping`. |
| AC-02 best checkpoint by val NDCG only, no test data | PASS | Verified by inspection of `run_pretrain`: only train/val tensorised splits and `val_candidates.npy` are opened; selection compares `val_ndcg` alone. `TestFewStepSmokeViaRunPretrain` asserts a finite val NDCG is produced. |
| AC-03 checkpoint roundtrip preserves state incl. GradNorm | PASS | `TestCheckpointRoundtrip::test_save_load_identical_output`. |
| AC-04 training loss decreases over epochs on real data | NOT VERIFIABLE | The capped-step gate that would show it (`TestFewStepSmokeViaRunPretrain`, which asserts `last_window_loss < first_window_loss`) currently fails on the armed backend gate. With that requirement neutralised at runtime it **passes**, so the loss does decrease — but the uncapped one-epoch run `steering/testing.md` Phase 17 asks for has not been performed, `outputs/checkpoints/` is empty, and no `pretrain_*.log` exists. See REV-106. |
| AC-05 pool loaded once, never regenerated | PASS | `TestNegativePoolReuse::test_run_pretrain_never_calls_build_once` monkeypatches `build_once` and asserts it is never reached; `test_round_trip_matches_recorded_fingerprint` and `test_tampered_pool_is_rejected` cover the fingerprint guard. |
| AC-06 routing distributions accumulated, not discarded | PASS | `_append_routing_snapshots` buffers detached `(B, K)` CPU tensors under a 256-batch FIFO cap; `_emit_routing_diagnostics` consumes and clears at `routing_diag_interval_epochs`; `collect_validation_routing` runs the comprehensive end-of-training pass. |
| AC-07 per-epoch log has all required fields | PASS | `log_epoch` ([:372](training/pretrain.py:372)) emits epoch, total and per-objective train losses, all four val metrics, lr, and per-objective lambdas. No real `pretrain_*.log` exists yet to inspect (AC-04). |
| AC-08 all finite assertions hold throughout training | PARTIAL | The assertions are present and correctly placed — per-objective loss, total, every trainable gradient (REV-022c), and `assert_finite_step` on lambdas. But nothing tests that they actually fire on a NaN; see `specs/loss-functions` REV-091, which this AC depends on. |

## Findings

```
REV-062 (cross-reference; owned by specs/transformer-backbone)
Severity: HIGH | Status: OPEN | Requirement: FR-5 | AC: AC-04
File: tests/test_smoke.py:30; tests/test_training.py:216; models/transformer.py:76-103
Problem: Both of this spec's smoke gates arm `require_attention_backend = True` and both
  FAIL on this host: "Preferred attention backend 'flex_attention' cannot execute on mps:
  NotImplementedError: FlexAttention does not support backward on MPS."
    tests/test_smoke.py::TestOneBatchEndToEnd::test_full_pipeline
    tests/test_training.py::TestFewStepSmokeViaRunPretrain::test_few_steps_via_run_pretrain
Impact: `steering/testing.md` Phases 16 and 17 cannot be signed off, so BUILD_ORDER Phase
  11 cannot authorise full MovieLens training. The repository's test suite does not pass.
Correction: Spec decision for the owner, tracked in full under
  `specs/transformer-backbone/review.md`. Recorded here because the failing tests belong
  to this spec's task list (T6.2, T6.3). Verified for the owner: with the backend
  requirement neutralised at runtime -- via a scratch pytest plugin, no repository file
  modified -- BOTH gates pass. Nothing in `run_pretrain` is broken. BUILD must not disarm
  the flag or edit the spec to make the suite green.
```

```
REV-106
Severity: MEDIUM | Status: OPEN | Requirement: FR-5, FR-8 | AC: AC-04, AC-07
File: outputs/ (empty checkpoints and no pretrain log); data/processed/ (artifact dates)
Problem: `run_pretrain` has never been executed at real scale with the current code.
  `outputs/checkpoints/` is empty, `outputs/logs/` holds only `run_config.json` and two
  stray `moe_parameter_report_*.json` files, and there is no `pretrain_t_*.log`,
  `gradnorm_t_*.jsonl` or `gradnorm_lambda_trajectory_*.png`. Separately, the artifacts in
  `data/processed/` were written at 01:14-01:27 while `data/sequence_generator.py` and
  `data/preprocessing.py` were last modified at 21:23-21:24, so even the inputs a real run
  would consume are the previous revision's output (see data-pipeline REV-090).
Impact: AC-04 ("training loss decreases over epochs on real data") and AC-07 ("per-epoch
  log contains all required fields") have no artifact behind them. The capped few-step
  gate is good evidence that the mechanics work, but it runs 1 epoch over a tiny synthetic
  subset -- it cannot show multi-epoch behaviour, the cosine decay actually reaching its
  tail, or the lambda trajectory plot rendering from a real JSONL.
Correction: Sequenced behind REV-062, which currently blocks any sanctioned run. Once the
  backend question is settled: re-run `--stage preprocess` so `data/processed/` matches the
  current source, then run the uncapped one-epoch T-Small gate (T6.3) on the training host
  and attach `pretrain_t_small.log`, `best.pt` and the lambda trajectory PNG as the
  evidence for AC-04 and AC-07. Leave T6.2/T6.3 unchecked until then -- which BUILD has
  correctly already done.
```

```
REV-107
Severity: LOW | Status: OPEN | Requirement: FR-6 | AC: AC-06
File: training/pretrain.py:806-822 (gradient-diagnostics call inside the training loop)
Problem: `run_gradient_diagnostics` is invoked mid-epoch every `grad_diag_interval_steps`.
  It calls `model.eval()` internally and never restores training mode; `train_one_step`'s
  dropout and the MoE's dropout would therefore be disabled for the rest of the epoch. The
  training loop compensates by calling `model.train()` / `moe.train()` immediately after
  ([pretrain.py:821-822](training/pretrain.py:821)), so the live path is correct -- but the
  invariant is maintained by the caller, not by the function that broke it, and the second
  caller (`main.py::run_gradient_diagnostics_stage`) does not restore it because it exits
  straight afterwards.
Impact: None today. It is a trap for the next caller: any future site that runs the
  diagnostic and then continues training without knowing to re-enter train mode silently
  trains without dropout.
Correction: Have `run_gradient_diagnostics` save `model.training` / `moe_layer.training` on
  entry and restore both on exit, the way `collect_validation_routing`
  ([pretrain.py:517-547](training/pretrain.py:517)) already does. The explicit
  `model.train()` in the loop can stay as belt-and-braces.
```

```
REV-108
Severity: LOW | Status: OPEN | Requirement: FR-5 | AC: —
File: training/pretrain.py:242-252 (contrastive augmentation in `train_one_step`);
       evaluation/gradient_diagnostics.py:202-213 (identical block)
Problem: The augmented contrastive view is built with a Python loop over the batch. Per
  contrastive batch it performs B `.tolist()` calls -- each a device-to-host copy and
  synchronisation -- then B Python-level `augment_sequence` calls and one
  `torch.tensor(list-of-lists)` construction. At `batch_size=256` that is 256 syncs plus
  256 list round-trips on every 4th step.
Impact: Throughput only; the arithmetic is correct. It is the same class of hot-path cost
  as the `apply_masking` loop that REV-069 removed, and it is now the largest remaining
  per-step Python/device round-trip in the training loop.
Correction: Either move the augmentation into the dataset (precompute an augmented view per
  example at tensorisation time, which also makes it reproducible from the seed alone), or
  vectorise the three strategies as tensor ops over the `(B, S)` batch. Neither is urgent;
  measure first against the one-epoch gate before deciding it is worth the complexity.
```

```
REV-092 (cross-reference; owned by specs/loss-functions)
Severity: MEDIUM | Status: OPEN | Requirement: FR-6 | AC: AC-08
File: training/gradnorm.py:72-74, reached from training/pretrain.py:302-304
Problem: `gradnorm.update` needs double-backward through the last block's attention kernel.
  With `config.device` resolving to CPU this raises "derivative for
  aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented".
Impact: Any CPU-only host cannot run the default `use_gradnorm=True` configuration; the
  H100/FlashAttention-4 combination is unverified.
Correction: Tracked in `specs/loss-functions/review.md` REV-092.
```

## Design Compliance

Matches `design.md` and the steering constraints. Specifically checked:

- **`conventions.md` left-padding** — `pooled = all_hidden[:, -1, :]` everywhere; no `seq_lens`-based position selection anywhere in `training/`.
- **`conventions.md` objective order** — `OBJECTIVE_ORDER` drives the MoE objective indices (0/1/2/3), `log_epoch`, `log_gradnorm_lambdas` and `plot_gradnorm_trajectory` identically.
- **`code-quality.md` config as single source of truth** — no hardcoded hyperparameters in `pretrain.py`. `ROUTING_BUFFER_MAX_BATCHES = 256` is a memory bound, not an experiment parameter, and is exposed as a keyword argument.
- **`code-quality.md` numerical safety** — `clip_grad_norm_(trainable, config.max_grad_norm)` covers the loss functions' own parameters too (REV-047), and `zero_grad(set_to_none=True)` is used at every site.
- **`architecture.md` training↔evaluation boundary** — no test data is touched by `run_pretrain`.
- **`testing.md` critical runtime assertions** — `train_user_history is not eval_user_history` is asserted at [:722](training/pretrain.py:722) against both really-loaded pickles.

One deliberate addition beyond `design.md`: `max_steps_per_epoch`. It is documented in `configs/config.py` as the dev-host smoke lever, defaults to 0 (uncapped, the production setting), and is correctly excluded from the production path. It does not relax any gate — T6.3 remains unchecked precisely because the capped run is not the one-epoch run Phase 17 asks for.

## Task Completion

16 of 18 tasks are `[x]`; T6.2 and T6.3 are correctly unchecked with accurate BUILD notes.

| Task | Verdict |
|---|---|
| T1.x scheduler | Complete, with three real tests. |
| T2.x model construction / loss fns | Complete. `collect_trainable_parameters` and the `_opt_ids` assertion at [:698-703](training/pretrain.py:698) prove every loss parameter is registered in the optimizer (REV-047). |
| T3.x GradNorm integration | Complete; isolation independently verified. |
| T4.x checkpointing, logging, routing snapshots | Complete. |
| T5.1 `run_pretrain` | Complete as an implementation. Never run at scale (REV-106). |
| T5.2 BF16 + finite assertions | Complete as code. The negative test proving the assertions fire is missing — loss-functions REV-091. |
| T6.1 checkpoint roundtrip test | Complete. |
| **T6.2 one-batch end-to-end smoke** | Correctly `[ ]`. The test exists and its training assertions pass; the gate fails on the backend. BUILD note is accurate. |
| **T6.3 one-epoch T-Small smoke** | Correctly `[ ]`. The BUILD note states plainly that the uncapped run has not been performed and that relaxing Phase 17 is the owner's call, not BUILD's. This is the right handling of the previous review's REV-066. |
| T6.4 negative pool reuse test | Complete, and the test is a genuine monkeypatch-based check rather than an inspection. |

No false-complete task claims found in this spec. The two unchecked boxes are honest.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` (full suite) | 231 passed, **2 failed**, 135 s. Both failures are this spec's smoke gates, blocked on `transformer-backbone` REV-062. |
| `uv run python -m pytest tests/test_training.py::TestFewStepSmokeViaRunPretrain -q` | 1 failed, 20 s — `AttentionBackendError ... require_attention_backend=True forbids falling back`. |
| Scratch: same two gates re-run with `require_attention_backend` neutralised at runtime via a scratch pytest plugin (no repository file modified) | **3 passed, 23 s.** `run_pretrain` completes its capped budget, validation metrics are computed, the loss decreases across the run, and the checkpoint is written. REV-062 is the sole blocker. |
| Scratch: instrument `gradnorm.update` against a real `TransformerModel` step | `model grads mutated by gradnorm.update: [] count=0`; active lambdas sum to 2.99999976 for 3 objectives — REV-057 confirmed fixed. |
| `git diff --stat 32b0904 -- specs/training-pipeline/requirements.md steering/testing.md specs/BUILD_ORDER.md` | No changes — REV-066's reverts are in place. |
| `ls outputs/checkpoints outputs/logs` | `checkpoints/` empty; `logs/` holds only `run_config.json` and two `moe_parameter_report_*.json` — REV-106 confirmed. |
| Direct inspection of `data/processed/` mtimes vs `data/*.py` mtimes | Artifacts 01:14–01:27, source 21:23–21:24 — the inputs predate the current pipeline. |

## Final Assessment

`training/pretrain.py` is in good shape. All four prior CRITICAL/HIGH findings are fixed, the fixes verify under instrumentation rather than on the strength of their comments, and the two unchecked tasks are unchecked for the right reason.

What BUILD should do, in order:

1. **Nothing on REV-062** — escalate it. It is a spec decision (sanction SDPA on MPS, or pin a PyTorch build where FlexAttention trains there). Do not disarm `require_attention_backend` and do not edit `requirements.md`, `steering/testing.md` or `BUILD_ORDER.md`. Note for the owner that both gates pass once the backend requirement is lifted, so the only open question is which kernel is sanctioned.
2. **REV-107** — restore train/eval mode inside `run_gradient_diagnostics`; three lines, and it removes a trap for the next caller.
3. **REV-106** — once REV-062 is settled: re-run `--stage preprocess`, then the uncapped one-epoch T-Small gate, and attach the log, checkpoint and lambda-trajectory plot as AC-04/AC-07 evidence. Only then re-mark T6.2 and T6.3.
4. **REV-108** — optional throughput work; measure against the one-epoch run before deciding.

AC-04 and AC-07 stay NOT VERIFIABLE / unevidenced until a real run happens; AC-08 stays PARTIAL until loss-functions REV-091 adds the negative assertion tests.
