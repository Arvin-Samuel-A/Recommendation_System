# Review — Training Pipeline

## Review Status

**PASS_WITH_WARNINGS**

(Up from `CHANGES_REQUIRED`. REV-138 and REV-139 are both resolved and independently
verified — the coverage rule is now defined by exclusion, both halves of the revision share
one predicate, and completeness is tested rather than assumed. What remains is REV-108
(LOW, a throughput item) and REV-125 (INFO, AC-04 deferred to Phase 14), plus one new INFO.)

## Spec Amendments — Status

The 2026-09-18 owner amendments (authored by REVIEW at the owner's instruction) are
implemented:

| File | Change | Status |
|---|---|---|
| `steering/code-quality.md` | "Artifact Provenance" | Implemented against the original enumeration |
| `steering/code-quality.md` | **"Coverage: a property, not a list"** (amended 2026-09-18) | **Implemented and verified — REV-138 closed** |
| `requirements.md` FR-3 + AC-09 | **Amended again 2026-09-18:** FR-3 no longer restates a coverage list — it references the steering rule; AC-09 now requires completeness to be tested and both halves to share one rule | Already satisfied by BUILD's implementation; verified below |
| `requirements.md` FR-3 | `source_revision` on every checkpoint | **Implemented and verified on the real `best.pt`** |
| `requirements.md` AC-09 | Checkpoint + every derived artifact carry the same value | **Implemented; verified by reading the checkpoint and all 9 derived artifacts and comparing** |

## Feature

`specs/training-pipeline` — `training/pretrain.py`: LR scheduler, model construction, the
four-objective training loop, contrastive interval gating, GradNorm integration,
validation, checkpointing, per-epoch and λ-trajectory logging, routing-snapshot
accumulation.

## Review Scope

- Spec directory: `specs/training-pipeline/` — `requirements.md` amended (FR-3 + AC-09),
  `tasks.md` modified by BUILD (T6.3 re-run record, new T6.5).
- Steering: `code-quality.md` (incl. the new "Artifact Provenance" section), `testing.md`,
  `conventions.md`, `architecture.md`.
- Baseline: `master` at `0bc67ee`; 34 files modified in the working tree after BUILD's
  round. This spec owns `training/pretrain.py` (+29) and `utils.py` (+132, the provenance
  helpers).
- Files inspected: `training/pretrain.py`, `utils.py` (read in full), `tests/test_training.py`,
  and the real artifacts: `best.pt`, `pretrain_t_small.log`, `gradnorm_t_small.jsonl`, and
  all nine derived JSON reports.
- Re-review (third pass). Prior findings carried in: REV-136, REV-129, REV-130 (OPEN),
  REV-125, REV-108 (OPEN), REV-106, REV-107, REV-126 (RESOLVED).
- Date: 2026-09-18 (post-BUILD-fix)

## Summary

The re-run and the provenance work both landed, and the record is now clean.

**REV-129 / REV-130 are resolved by removing the cause rather than by arguing the history.**
The gate was re-run at 15:25–15:29 on 2026-09-18 against current source, and every figure in
T6.3's note was read back out of the artifact. I checked all of them: `val_ndcg@10=0.091145`,
`mrr=0.064986`, `recall=0.178565`, `precision=0.017856`, and the five per-objective training
losses, all match `outputs/logs/pretrain_t_small.log` exactly. The three contradictory
accounts of `best.pt` are gone — `loss-functions` T3.7 and `personalization-federated` T6.1
now both defer to T6.3, and T6.3 is consistent with the files.

**REV-136 is implemented, and implemented well.** `source_revision` is on the checkpoint,
and I verified the echo by reading the checkpoint and every derived artifact and comparing
the fields: **all nine JSON reports MATCH**, plus the per-epoch log line and every record in
the λ JSONL. Two design choices are better than the minimum:

- Downstream stages call `artifact_source_revision(ckpt)` — echoing the *checkpoint's*
  revision — rather than `get_source_revision()`. So an artifact reports the provenance of
  the model, not of whatever code happened to evaluate it. That distinction is the whole
  point and it would have been easy to get backwards.
- A pre-AC-09 checkpoint degrades to an explicit "provenance is NOT established" note rather
  than borrowing the running revision. I verified this is load-bearing: breaking it to
  substitute the running revision fails `test_checkpoint_without_revision_is_recorded_as_unestablished`.

**What is not resolved is that the identifier does not identify enough.** `SOURCE_REVISION_DIRS`
is `("models", "training", "data", "configs")`. `evaluation/` is absent — and that is the
directory containing `longtail.py`, `cold_start.py`, `metrics.py`, `flops.py`,
`gradient_diagnostics.py` and `routing_diagnostics.py`, i.e. the code that produces every
derived artifact this project publishes. Measured: appending a line to
`evaluation/longtail.py` leaves `source_revision` **byte-identical** — same commit, same
`dirty`, same `source_hash`. The catalogue-wide bucketing change (T3.4) lived in exactly
that file, so two runs straddling it would be indistinguishable. Filed as REV-138.

The root cause is the steering enumeration, which **I authored**. BUILD followed it
literally — correctly, per §9 — and flagged the `main.py` / `utils.py` half of the gap
rather than widening the rule unilaterally. That was the right call. One claim in that
flag is wrong, though, and it matters: see REV-139.

## Git Changes

`training/pretrain.py` (+29): `source_revision` into `save_checkpoint`, the run manifest,
the epoch log, the λ JSONL and the routing-diagnostics call. `utils.py` (+132):
`get_source_revision`, `artifact_source_revision`, `_hash_source_tree`, `_git`.
`tests/test_training.py` (+180): `TestArtifactProvenance` (4 tests) plus an end-to-end echo
assertion in the few-step gate.

Artifact timeline, from `stat` — monotone and consistent with a single sequenced cascade:
preprocess 15:24–15:25 → pretrain 15:28:58 → personalize 15:29 → federated 15:31 →
evaluate 15:34 → cold_start / routing / gradient 15:38 → longtail 15:41 → efficiency 15:49.
Every derived artifact postdates the checkpoint it echoes.

**Scope assessment:** proportionate. No opportunistic edits.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 LR scheduler | PASS | `build_scheduler` is a `LambdaLR` with linear warmup over `int(total_steps * warmup_ratio)` then cosine to 0, clamped at `max(0.0, ...)`. `total_steps = num_batches_per_epoch * pretrain_epochs`, and `max_steps_per_epoch` correctly shrinks `num_batches_per_epoch` so a capped run executes a whole schedule rather than a truncated one. Stepped per batch inside `train_one_step` |
| FR-2 Per-model checkpoint dirs | PASS | `outputs/checkpoints/t_small/` exists with `best.pt` and `epoch_000.pt`; the path is built from `config.model_dir_name()` |
| FR-3 Checkpoint format (amended) | PASS | Read off the real `best.pt`: all of `epoch`, `model_state_dict`, `moe_state_dict`, `loss_fns_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `gradnorm_state_dict`, the four `val_*` metrics and `config` are present as sibling top-level keys, matching the REV-126 amendment. `gradnorm_state_dict` holds `log_lambdas`, `optimizer` and `initial_losses`. `loss_fns_state_dict` holds `multi_step`, so the step projections survive a reload. The best checkpoint carries all ten FR-3 metadata fields, **and now `source_revision`** — measured on the real file: `{commit: "0bc67ee…", dirty: true, source_hash: "3a59e7cb70a8be69"}`. `dirty: true` is correct and deliberate, since REV-137/REV-136 are uncommitted |
| FR-4 Best selection by val NDCG@10 only | PASS | `if val_ndcg > best_val_ndcg` is the sole selection condition; no other metric appears in a comparison anywhere in the loop, and no test metric is read during training |
| FR-5 Training loop | PASS | The pool is loaded once via `GlobalNegativePool.load` before the epoch loop and never rebuilt. Masking is applied before the forward pass; contrastive fires on `batch_idx % contrastive_interval == 0`. The epoch log carries every required field |
| FR-6 ARV-MMoE + GradNorm forward path | PASS | `return_all_hidden=True` → per-objective `moe(...)` at indices 0–3 → per-objective loss → weighted sum → `backward(retain_graph=(gradnorm is not None))` → `gradnorm.update` → clip → `optimizer.step` → `scheduler.step`. Routing snapshots are detached and appended to a capped per-epoch buffer |
| FR-7 BF16 training | PASS | `assert_bf16_params(model, moe)` runs before training and skips only `log_lambdas` (which is the controller's, not the model's). `optimizer.zero_grad(set_to_none=True)`. No autocast, no GradScaler |
| FR-8 Validation | PASS | Runs after each epoch against the stored `val_candidates.npy` and `eval_user_history`; all four metrics reported. The last run evaluated the full 993,372-example validation split |
| FR-9 GradNorm λ logging | PASS | `log_gradnorm_lambdas` every 100 steps to `outputs/logs/gradnorm_t_small.jsonl`; `outputs/plots/gradnorm_lambda_trajectory_t_small.png` exists, rendered from that JSONL |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 warmup → cosine, per-batch stepping | PASS | `TestLRScheduler` has all three tests: warmup increases linearly, cosine reaches near zero, and stepping is per batch. The `lr=0.00000000` in the epoch log is the schedule's endpoint under the capped run, not a defect — `max_steps_per_epoch` shrinks `total_steps` too |
| AC-02 best by val NDCG@10, test untouched | PASS | Single comparison on `val_ndcg`; no test artifact is opened by `run_pretrain` |
| AC-03 checkpoint roundtrip preserves state incl. GradNorm | PASS | `TestCheckpointRoundtrip::test_save_load_identical_output` covers model output identity and GradNorm state. The real `best.pt` was read directly and carries the full state |
| AC-04 training loss decreases over epochs on real data | **NOT VERIFIABLE** | Unchanged and still deliberately deferred. The owner's Phase 17 amendment states that a few-hundred-step window shows loss *rising* under LR warmup (measured 26.23 → 28.51 at 100 steps; 27.32 → 29.15 at 400) and that the reduced gate does not claim AC-04. It closes only against a multi-epoch run at BUILD_ORDER Phase 14 |
| AC-05 pool loaded once, never regenerated | PASS | One `GlobalNegativePool.load` call before the loop; `build_once` is never called in `pretrain.py`. `TestNegativePoolReuse::test_run_pretrain_never_calls_build_once` asserts this by monkeypatching `build_once` to fail, which is the right shape of test — it cannot pass if the call happens |
| AC-06 routing accumulated, not discarded | PASS | `_append_routing_snapshots` into `routing_buffer` under a hard cap; `_emit_routing_diagnostics` concatenates and clears. `outputs/logs/routing_diagnostics_t_small.json` and its three plots exist |
| AC-07 per-epoch log has all required fields | PASS | The real log line carries `train_loss_total`, all four per-objective losses, all four validation metrics, `lr` and all four `lambda_*` values. The in-suite gate asserts each field by name |
| AC-08 finite assertions hold throughout | PASS | `assert_finite_step` on every step, covering losses and lambdas; `TestFiniteAssertionsActuallyFire` (in `tests/test_losses.py`) confirms it fires before the optimizer steps |
| AC-09 `source_revision` on every checkpoint and every derived artifact, coverage complete and tested (amended) | PASS | Verified as AC-09 specifies — by reading the real checkpoint **and** each real derived artifact and comparing, not by asserting the writing code exists. All nine JSON reports MATCH the checkpoint exactly (`test_metrics`, `cold_start`, `longtail`, `efficiency`, `routing_diagnostics`, `gradient_diagnostics`, `personalization_gain`, `federated_rounds`, `run_config`), as do the epoch log line and every λ JSONL record. The amended half is met too: coverage is complete against the steering rule (27 files, all 7 `evaluation/` modules, both top-level modules, no `tests/`), the dirtiness check shares the hash's predicate, and completeness is tested observationally by recording which files the hash actually opens — REV-138 |

## Findings

```
REV-138
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-3 (amended) | AC: AC-09 (amended)
File: utils.py:56-140 (_is_covered_source, _covered_source_files, _hash_source_tree,
      _porcelain_paths); steering/code-quality.md ("Coverage: a property, not a list")
Resolution: implemented as an exclusion rule and verified empirically, not from the diff.
  * Coverage is enumerated from `git ls-files -- "*.py"` and filtered by ONE predicate,
    `_is_covered_source`, excluding `tests/`, `.venv/`, `__pycache__/`, `data/raw/`,
    `data/processed/`. Measured: **27 covered files**, including all 7 `evaluation/`
    modules and both top-level modules (`main.py`, `utils.py`); zero `tests/` files.
  * The property holds where it previously failed. Appending a line and restoring it:
    `evaluation/longtail.py` moves the hash (c6b41af8 -> 6b7c026e), `main.py` moves it
    (-> 7d363642), `utils.py` (-> 3b9d4dc0), `models/moe.py` (-> c3776e7c). Editing
    `tests/test_metrics.py` leaves it **unchanged**, as the rule requires.
  * The dirtiness check and the hash now share the SAME predicate rather than two rules
    that happen to agree -- `_porcelain_paths()` feeds `git status --porcelain` through
    `_is_covered_source`, with renames split on `->`, quoted paths unquoted, and untracked
    `??` entries skipped because the rule excludes anything not tracked.
  * Edge cases handled beyond what was asked: a tracked-but-deleted file still moves the
    hash (`<missing>` marker), and a non-git checkout falls back to a tree walk applying
    the same exclusions.
Completeness test -- the one the amendment newly required, and the strongest thing here:
  `test_hashed_set_equals_the_set_the_coverage_rule_defines` monkeypatches `builtins.open`
  to record which files `_hash_source_tree` ACTUALLY reads, and compares that observed set
  against an independently derived `git ls-files` set. That is observational rather than a
  restatement of intent, which is exactly what was needed -- the superseded allowlist
  shipped green precisely because the old tests asserted the mechanism worked rather than
  that it covered what it must.
Non-vacuity verified, not accepted: reinstating the `models|training|data|configs` allowlist
  inside `_is_covered_source` fails 3 of the 6 provenance tests --
  `test_hashed_set_equals_the_set_the_coverage_rule_defines`,
  `test_dirtiness_and_hash_share_one_coverage_rule`, and
  `test_dirty_tree_is_never_attributed_to_its_nearest_commit`. Restored; 6 pass.
BUILD's own runtime verification used a scratch output directory, so no published artifact
  was disturbed -- confirmed: `best.pt` and the epoch log still carry their 15:28 mtimes.
```

```
REV-139
Severity: LOW | Status: RESOLVED | Requirement: FR-3 (amended) | AC: AC-09
File: specs/training-pipeline/tasks.md (T6.5)
Resolution: the false mitigation is corrected, and corrected the right way -- the superseded
  claim is quoted and refuted in place rather than quietly deleted, so a later reader can see
  the claim was made and why it was wrong. The note now also names `evaluation/` as the
  consequential omission, which the original gap statement left out entirely.
```

```
REV-140
Severity: INFO | Status: OPEN | Requirement: FR-3 | AC: AC-09
File: utils.py (get_source_revision); outputs/ (existing artifacts)
Note, no action required. A `source_hash` does not record WHICH coverage rule produced it,
  and that rule has now changed once -- within a single round. The published artifacts carry
  `3a59e7cb70a8be69`, computed under the superseded allowlist; the same tree today hashes to
  `c6b41af8025ff7e9` under the widened rule. Nothing is broken by this: the hash is an opaque
  run identifier compared between artifacts, not recomputed and matched, and every published
  artifact still echoes its own checkpoint consistently (verified -- all 9 match). A false
  "same code" conclusion is not reachable; only a false "different code" one, which is the
  safe direction.
Recording it because the failure it would cause is the same class this whole thread has been
  about: a value that looks comparable across time but is not. If the rule is ever amended
  again, a `rule_version` alongside `source_hash` would make old hashes self-describing. Not
  worth doing on one change; worth doing on a second.
Re-stamping existing artifacts is NOT required and should not be done on this account.
```

```
REV-136
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-3 (amended) | AC: AC-09
File: utils.py:44-171; training/pretrain.py:107 and the derived-artifact writers
Resolution: implemented across the board and verified the way AC-09 demands -- by reading a
  real checkpoint and each real derived artifact and comparing the fields, not by asserting
  the writing code exists. All nine JSON reports match the checkpoint's
  `{commit, dirty, source_hash}` exactly; so do the epoch log line (`source_commit` /
  `source_hash` / `source_dirty`) and every record in `gradnorm_t_small.jsonl`. Per-arm
  ablation results and the federated per-round records are covered in code.
Two choices worth crediting, both the harder and more correct option:
  * Derived artifacts call `artifact_source_revision(ckpt)` -- the CHECKPOINT's revision --
    rather than `get_source_revision()`. An artifact therefore reports the provenance of the
    model, not of whatever code evaluated it. Getting this backwards would have looked
    identical in a green suite and been silently wrong.
  * A pre-AC-09 checkpoint degrades to an explicit "provenance is NOT established" note
    instead of borrowing the running revision. I verified this is load-bearing: patching it
    to substitute the running revision fails
    `test_checkpoint_without_revision_is_recorded_as_unestablished`, and restoring passes.
Residual: the recorded value's DISCRIMINATING POWER is incomplete -- REV-138. That is a
  defect in the rule I wrote, not in this implementation of it, which is why this closes.
Minor, not filed separately: `moe_parameter_report_*.json` carries no `source_revision`. It
  is a pure function of config and is not checkpoint-derived, so AC-09's literal text does
  not reach it; worth stamping with the running revision for consistency when REV-138 is
  addressed.
```

```
REV-129
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-5, FR-6 | AC: AC-07
File: specs/training-pipeline/tasks.md (T6.3); outputs/checkpoints/t_small/best.pt
Resolution: the gate was re-run on 2026-09-18 (preprocess 15:24-15:25, pretrain 15:25:59 ->
  15:29:08, 189 s, exit 0) against current source, and the record rewritten from the
  artifacts. Every figure in T6.3's note was checked against the files this pass and matches
  exactly: `val_ndcg@10=0.091145`, `mrr=0.064986`, `recall=0.178565`, `precision=0.017856`,
  `train_loss_total=28.176922`, `nextitem=4.594844`, `maskeditem=9.900937`,
  `multistep=9.905000`, `contrastive=4.433012`.
  The preprocess re-run was required rather than incidental: `data-pipeline` REV-137 changed
  the training data, so re-running pretrain alone would have certified a model trained on
  superseded inputs. BUILD sequenced the full cascade -- preprocess -> pretrain ->
  personalize -> federated -> six evaluation stages -- and the artifact mtimes confirm it ran
  in that order, every derived file postdating the checkpoint.
  The provenance dispute is resolved by removal rather than argument: the current `best.pt`
  was written at 15:28:58 by the command recorded in T6.3, and carries its own
  `source_revision`. The two contradicting notes now defer to T6.3. One account, and it is
  consistent with the files.
```

```
REV-130
Severity: LOW | Status: RESOLVED | Requirement: FR-5 | AC: AC-07
File: specs/training-pipeline/tasks.md (T6.3)
Resolution: the practice is corrected, and the correction is stated in the note itself --
  "every value below was read back out of the artifact AFTER the run, never transcribed from
  console output". Verified: all nine quoted figures match the files. The superseded
  2026-09-17 values are retained explicitly as superseded rather than deleted, so the
  supersession is checkable instead of merely asserted. Same treatment applied in
  `evaluation-diagnostics` T8.1, which closes REV-133.
```

```
REV-125
Severity: INFO | Status: OPEN | Requirement: FR-5 | AC: AC-04
File: specs/training-pipeline/requirements.md (AC-04); steering/testing.md (Phase 17)
Unchanged. AC-04 is outstanding by deliberate, dated, owner-authorised sequencing, not by
  omission, and the reduced Phase 17 gate explicitly disclaims it. No BUILD action is
  possible today. Re-recorded because the failure mode worth guarding against is a later
  review reading a green suite and marking AC-04 PASS. Close it at BUILD_ORDER Phase 14
  against the per-epoch `train_loss_total` series from a full multi-epoch run -- never
  against a capped run.
```

```
REV-108
Severity: LOW | Status: OPEN | Requirement: FR-5 | AC: —
File: training/pretrain.py:244-254; evaluation/gradient_diagnostics.py:202-213
Problem: unchanged and re-confirmed in the source this pass. The augmented contrastive
  view is built with a Python loop over the batch -- `for i in range(B): real =
  seq[i][seq[i] != 0].tolist()`, a Python-level `augment_sequence` per row, then one
  `torch.tensor(list-of-lists)`. Each `.tolist()` is a device-to-host copy and a
  synchronisation; at batch_size=256 that is 256 syncs plus 256 list round-trips on every
  4th step.
Impact: throughput only; the arithmetic is correct. Its relative weight is larger than it
  was, since REV-120 removed ~275 ms/step of GradNorm cost (0.641 s -> ~0.235 s measured).
Correction: either precompute an augmented view per example at tensorisation time (which
  also makes it reproducible from the seed alone) or vectorise the three strategies as
  tensor ops over the (B, S) batch. Not urgent; judge it against the Phase 14 full run
  with ~0.235 s/step as the baseline.
```

```
REV-106
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-5, FR-8 | AC: AC-07
Re-verified as a statement about the code and the in-suite gate: `run_pretrain` does
  execute end to end at real scale, computes validation over the full split, writes the
  checkpoint and the log. That remains true. The *artifacts* from the run this finding
  closed against are now stale -- tracked as REV-129 rather than by reopening this.
```

```
REV-107
Severity: LOW | Status: RESOLVED | Requirement: FR-6 | AC: AC-06
Re-confirmed. `run_gradient_diagnostics` saves and restores both `model.training` and
  `moe_layer.training`, so the invariant is held by the function that breaks it. The
  caller additionally restores `train()` after the call, which is belt-and-braces rather
  than a dependency.
```

```
REV-126
Severity: INFO | Status: RESOLVED | Requirement: FR-3 | AC: AC-03
Re-verified against the real `best.pt`: MoE and GradNorm state are sibling top-level keys,
  exactly as the amended FR-3 now describes, and `loss_fns_state_dict` is present
  alongside them.
```

```
REV-062 / REV-092 (cross-references)
Owned by specs/transformer-backbone and specs/loss-functions respectively. Both closed in
  their home specs this round; see those reviews.
```

## Design Compliance

Compliant. The loop implements the FR-6 sequence in the stated order, the anchor comes
from `get_anchor_layer_params(model, config)` so GradNorm operates on the last shared
Transformer block in every variant, and `retain_graph` is conditioned on the controller's
presence rather than hardcoded.

Steering compliance checked specifically:

- `optimizer.zero_grad(set_to_none=True)` — present, including after the gradient-
  diagnostics excursion, which would otherwise leave diagnostic gradients in the buffers.
- `clip_grad_norm_(..., max_norm=1.0)` on every step.
- No per-epoch negative pool regeneration — enforced by a test that fails if `build_once`
  is called.
- No cross-model state sharing — each `run_pretrain` builds its own model, optimizer,
  scheduler and controller.
- No test data touched: `run_pretrain` opens only the train and val artifacts.

## Task Completion

| Task | Marked | Verdict |
|---|---|---|
| T6.3 few-step T-Small gate | `[x]` | **Now genuinely complete.** Was false-complete last pass. The gate was re-run, the record rewritten from the artifacts, and all nine quoted figures re-checked against the files here. The note also explains the two things a reader could misread (`lr=0.00000000` as the schedule's endpoint; λ summing to 3.000001 over the three *active* objectives) and restates the AC-04 non-coverage rather than quietly claiming it |
| T6.5 artifact provenance | `[x]` | Complete. Verified as AC-09 specifies, by reading files rather than asserting code. Its "KNOWN GAP" bullet is right to exist and right to defer the decision to the owner — but it contains a false mitigation and omits the consequential directory: REV-139, REV-138 |
| T6.2 one-batch smoke, T6.4 pool reuse, T1.x–T5.x | `[x]` | Complete; unchanged this round and re-confirmed by the passing suite |

No false-complete tasks this pass.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **291 passed**, 0 failed, 53.15 s (was 288) |
| `uv run python -m pytest tests/test_training.py::TestArtifactProvenance -q` | **6 passed** |
| Same, **with the `models\|training\|data\|configs` allowlist reinstated** | **3 failed** — the completeness test, the shared-rule test, and the dirty-tree test. Restored; 6 pass |
| `_covered_source_files()` (direct) | **27 files**; 7 under `evaluation/`; `main.py` and `utils.py` present; zero `tests/` |
| Append + restore a probe line, re-read `source_hash` (direct) | `evaluation/longtail.py` → moves; `main.py` → moves; `utils.py` → moves; `models/moe.py` → moves; **`tests/test_metrics.py` → unchanged** |
| Compare `source_revision` across all published artifacts (direct) | all 9 checkpoint-derived artifacts MATCH the checkpoint; the two `moe_parameter_report_*` files carry the **running** revision, which is correct — they are a pure function of config, not checkpoint-derived |
| `stat` on `best.pt`, `pretrain_t_small.log` | still 15:28 — BUILD's runtime verification used a scratch output dir and disturbed no published artifact |
| `torch.load(best.pt)` (direct) | `source_revision = {commit: "0bc67ee…", dirty: true, source_hash: "3a59e7cb70a8be69"}` — the pre-widening hash, correctly left un-restamped (REV-140) |

## Final Assessment

**Nothing for BUILD. This spec is accepted.**

REV-138 is closed properly. The rule was inverted rather than patched, both halves of the
revision now share one predicate, and — the part that matters most — completeness is tested
*observationally*, by recording which files the hash actually opens and comparing that to an
independently derived set. That is the test the superseded allowlist could not have passed,
and it is the reason this class of gap should not recur silently.

BUILD also went past the brief in ways that are correct rather than gratuitous: a
tracked-but-deleted file still moves the hash, a non-git checkout falls back to a tree walk
under the same exclusions, rename and quoted-path entries are handled in the porcelain
parse, and the runtime verification was done into a scratch directory so no published
artifact was touched. I confirmed that last one — `best.pt` still carries its 15:28 mtime.

REV-139's correction is the right shape too: the false claim is quoted and refuted in place
rather than deleted, so the record shows the claim was made and why it was wrong.

**FR-3 was amended again this round** (by me, at the owner's instruction): it no longer
restates a coverage list, because a second copy of a list is a second thing to drift — the
exact failure this finding was about. It now references the steering rule, and AC-09 requires
completeness to be tested. BUILD's implementation already satisfies both.

Three things carried, none blocking:

1. **REV-140 (INFO, new).** A `source_hash` does not record which coverage rule produced it,
   and that rule has now changed once. Existing artifacts are internally consistent and must
   **not** be re-stamped; a `rule_version` field would be worth adding only if the rule is
   amended a second time.
2. **REV-108 (LOW).** The contrastive augmentation's per-row `.tolist()` device sync —
   throughput only, best judged against the Phase 14 full run at the ~0.235 s/step baseline.
3. **REV-125 (INFO).** AC-04 remains outstanding by the owner's deliberate sequencing and
   closes only against a real multi-epoch run, never a capped one.
