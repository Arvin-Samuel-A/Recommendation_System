# Review — Training Pipeline

## Review Status

**PASS_WITH_WARNINGS**

(Up from `CHANGES_REQUIRED`. REV-106, the MEDIUM finding that blocked this spec, is
resolved: `run_pretrain` has now been executed at real scale and its artifacts verified.
One LOW finding remains open, plus AC-04, which the spec owner has explicitly deferred to
full training.)

## Feature

`specs/training-pipeline` — `training/pretrain.py`: LR scheduler, model construction,
training loop over all four objectives, contrastive interval gating, validation,
checkpointing and best selection, per-epoch and GradNorm logging, `run_pretrain`
orchestration, runtime safety assertions.

## Review Scope

- Spec directory: `specs/training-pipeline/` — `requirements.md` and `design.md`
  unmodified; **`tasks.md` heavily amended (`+90/-27`)**, carrying the owner's Phase 17
  decision and BUILD's run record for T6.2 and T6.3.
- Steering: `steering/testing.md` (**amended by the owner: Phase 17 reduced from a full
  epoch to a few-step capped run, with a dated rationale**), `code-quality.md`,
  `architecture.md`, `conventions.md`.
- Baseline: working tree vs `HEAD` = `d438950` on `master`; `origin/master` at the same
  commit. 38 tracked files modified, no untracked files.
- Files inspected: `training/pretrain.py` (modified), `main.py` (modified),
  `tests/test_training.py` (modified `+26`), `tests/test_smoke.py` (modified `+15`),
  `configs/config.py`.
- The real run artifacts under `outputs/` were loaded and measured directly — the
  checkpoint was opened and its keys enumerated, the log read, the λ arithmetic redone.
- Re-review. Prior findings carried in: REV-062 (RESOLVED, cross-ref), REV-106 (MEDIUM,
  OPEN), REV-107 (RESOLVED), REV-108 (LOW, OPEN), REV-092 (cross-ref, superseded).
- Date: 2026-09-17

## Summary

REV-106 is closed. Two things happened, and it matters that they are distinguished.

First, the **spec owner** reduced the Phase 17 gate from an uncapped one-epoch T-Small run
to a few-step capped run on real data, recording the decision with its rationale and an
explicit reference to REV-066 — the earlier episode where BUILD rewrote this same gate to
fit what it had run. `steering/testing.md` now carries a dated owner decision saying
exactly that: "REV-066 correctly refused to let BUILD relax this gate on its own; this
change is the spec owner's". That is the right way to make this change, and it is the
opposite of what REV-066 flagged.

Second, **BUILD actually ran the reduced gate** and the artifacts exist. I verified them
against disk rather than against the task note:
`uv run python main.py --stage pretrain --model t-small --epochs 1 --max_steps 200`
produced `outputs/checkpoints/t_small/{best.pt,epoch_000.pt}`,
`outputs/logs/pretrain_t_small.log`, `gradnorm_t_small.jsonl`,
`routing_diagnostics_t_small.json`, and six plots including
`gradnorm_lambda_trajectory_t_small.png` — all dated 09-17 00:25–00:33, i.e. after the
preprocess re-run that REV-106 required as its step (1). The four validation metrics in
the log match the task note to six decimals (`val_ndcg@10=0.090022`,
`mrr=0.064059`, `recall=0.176741`, `precision=0.017674`), and `best.pt` carries every
field FR-3 lists, `config` included.

I re-derived the two things the task note asks a reader not to misread, rather than
accepting them. **`lr=0.00000000` is correct**: `training/pretrain.py:681-683` shrinks
`num_batches_per_epoch` — and therefore `total_steps` — by `max_steps_per_epoch`, so a
200-step run executes the entire warmup-plus-cosine schedule and lands on the schedule's
zero endpoint. **The λ sum is correct**: the three active λ are 0.259607 + 1.430659 +
1.309734 = exactly 3.000000 against `num_active = 3`, which is AC-05 holding; the 3.93
total simply includes the inactive contrastive λ, which is excluded by design.

What this run does **not** establish is AC-04. The owner's own amendment says so in
`steering/testing.md`, with measurements: under LR warmup a short window shows loss
*rising* (100 steps 26.23 → 28.51; 400 steps 27.32 → 29.15). AC-04 is unchanged and
remains outstanding until full training. This is a sequencing fact, not a defect, and
there is no BUILD action for it now — but it is why this spec is not a clean PASS.

## Git Changes

| File | Change |
|---|---|
| `training/pretrain.py` | `+15`: `probe_double_backward` call site removed (loss-functions T3.6) |
| `main.py` | `+52`: `--max_steps` added to `parse_args` with override logging |
| `tests/test_training.py` | `+26`: few-step smoke gate updates, SDPA comment sync |
| `tests/test_smoke.py` | `+15`: **comment-only** — verified; `require_attention_backend = True` preserved |
| `specs/training-pipeline/tasks.md` | `+90/-27`: owner's Phase 17 decision, T6.2/T6.3 run records |
| `steering/testing.md` | owner's Phase 17 amendment |

**Scope assessment:** proportionate. `--max_steps` is a genuine requirement of the reduced
gate — `conventions.md` forbids editing source to run it — and it is implemented as a CLI
override that logs the original and new value, which is what `code-quality.md` asks for.
Nothing unexpected is touched.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 LR scheduler | PASS | Linear warmup then cosine decay; `scheduler.step()` at `pretrain.py:322`, inside `train_one_step` — per batch, not per epoch; LR logged per epoch |
| FR-2 Per-model checkpoint dirs | PASS | `outputs/checkpoints/t_small/` created by the run, holding `epoch_000.pt` and `best.pt` |
| FR-3 Checkpoint format | PASS | Opened `best.pt` directly: all of `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `gradnorm_state_dict`, `val_{ndcg,mrr,recall,precision}`, `config`, plus the full best-metadata set (`best_epoch`, `best_val_*`, `seed`, `model_name`, `moe_variant`, `use_gradnorm`, `num_experts`). MoE lives in a sibling `moe_state_dict` key rather than nested inside `model_state_dict`; functionally complete, see REV-126 |
| FR-4 Best selection by val NDCG@10 | PASS | `pretrain.py:877` — strict `>` against `best_val_ndcg`, no other metric consulted; test data not loaded in this stage |
| FR-5 Training loop | PASS | Pool loaded once, never regenerated; masking before forward; contrastive on interval batches; per-epoch log written to `outputs/logs/pretrain_t_small.log` with every listed field |
| FR-6 ARV-MMoE + GradNorm forward path | PASS | Order verified line by line at `pretrain.py:301-322`: `zero_grad(set_to_none=True)` → `backward(retain_graph=(gradnorm is not None))` → `gradnorm.update` → gradient finiteness assertions → `clip_grad_norm_` → `optimizer.step()` → `scheduler.step()`. Routing snapshots accumulated at `:806` |
| FR-7 BF16 training | PASS | `assert_bf16_params(model, moe)` before training; `set_to_none=True`; no autocast or GradScaler; per-parameter gradient finiteness assertion at `:314-317` |
| FR-8 Validation | PASS | Post-epoch validation over the **full** 993,372-example split against the stored `val_candidates.npy`; `eval_user_history` used, not `train_user_history` |
| FR-9 GradNorm λ logging | PASS | `outputs/logs/gradnorm_t_small.jsonl` written from the real run; `outputs/plots/gradnorm_lambda_trajectory_t_small.png` rendered from it |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 warmup→cosine, per-batch stepping | PASS | `scheduler.step()` inside `train_one_step`; `total_steps` correctly derived from `num_batches_per_epoch * pretrain_epochs`, with the `max_steps_per_epoch` cap applied to both — which is why the capped run's terminal `lr=0.0` is the schedule's endpoint, not a bug |
| AC-02 best by val NDCG@10 only; no test data | PASS | Strict-improvement check on `val_ndcg` alone; the pretrain stage loads only train and val tensorised sets |
| AC-03 checkpoint roundtrip preserves state incl. GradNorm | PASS | `gradnorm_state_dict` present with `log_lambdas`, `optimizer`, `initial_losses`; roundtrip test in `tests/test_training.py` |
| **AC-04 training loss decreases over epochs on real data** | **NOT VERIFIABLE (outstanding)** | Explicitly not established by the reduced gate, and the owner's `steering/testing.md` amendment says so with measurements (loss *rises* under warmup at 100 and 400 steps). Requires the multi-epoch run at BUILD_ORDER Phase 14. Tracked as REV-125 |
| AC-05 pool loaded once, never regenerated | PASS | Fingerprint-matched load; `tests/test_training.py` T6.4 reuse test; no per-epoch regeneration path exists |
| AC-06 routing distributions accumulated | PASS | `routing_buffer` accumulated per step at `:806`, consumed by `_emit_routing_diagnostics` at `:868`, cleared only at `:895`. The real run produced `routing_diagnostics_t_small.json` and three routing plots |
| AC-07 per-epoch log has all required fields | PASS | Verified against the real log line, not a fixture: `train_loss_total`, the four per-objective losses, all four validation metrics, `lr`, and all four λ values |
| AC-08 finite assertions hold throughout | PASS | Total-loss assertion at `:298`, per-parameter gradient assertion at `:314-317`, plus `assert_finite_step`. A 200-step real run completed with exit 0 and no assertion firing |

## Findings

```
REV-106
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-5, FR-8 | AC: AC-07 (AC-04 carried to REV-125)
File: outputs/checkpoints/, outputs/logs/, outputs/plots/
Resolution: all three of the correction's ordered steps were performed.
  (1) `--stage preprocess` was re-run against the current source -- verified independently
      in the data-pipeline review this pass (artifacts 09-17 00:13-00:15; the three stale
      facts all flipped).
  (2) The Phase 17 gate was run against that fresh data. The gate itself was REDUCED by
      the spec owner from an uncapped epoch to a capped few-step run, recorded with a
      dated rationale in `steering/testing.md` and cross-referenced to REV-066. That is an
      owner decision, correctly made and correctly attributed -- not the BUILD-side
      relaxation REV-066 flagged. Checked specifically: the reduction is in the owner's
      steering doc and BUILD_ORDER, not only in tasks.md.
  (3) The artifacts exist and were verified against disk: `best.pt` + `epoch_000.pt` with
      the full FR-3 key set, `pretrain_t_small.log` with every AC-07 field,
      `gradnorm_t_small.jsonl`, `gradnorm_lambda_trajectory_t_small.png` rendered from
      that real JSONL, `routing_diagnostics_t_small.json` and three routing plots. All
      dated 09-17 00:25-00:33, after the preprocess re-run.
  The two potential misreadings in the log were independently re-derived rather than
  accepted: terminal `lr=0.0` follows from `max_steps_per_epoch` shrinking `total_steps`
  (pretrain.py:681-683), and the 3.93 lambda total decomposes into exactly 3.000000 over
  the three ACTIVE objectives, which is AC-05 holding.
  AC-04 is NOT closed by this and is not claimed to be -- see REV-125.
```

```
REV-125
Severity: INFO | Status: OPEN | Requirement: FR-5 | AC: AC-04
File: specs/training-pipeline/requirements.md (AC-04); steering/testing.md (Phase 17)
Problem: AC-04 ("training loss decreases over epochs on real data") is not satisfied and
  cannot be satisfied by the current gate. The owner's Phase 17 amendment states this
  directly and backs it with dev-host measurements showing loss RISING over short windows
  under LR warmup (100 steps 26.23 -> 28.51; 400 steps 27.32 -> 29.15). The reduced gate
  explicitly no longer claims AC-04.
Impact: None on correctness, and no BUILD action is possible today. Recorded because it
  is the single reason this spec cannot reach PASS: an acceptance criterion is
  outstanding by deliberate sequencing, deferred to the multi-epoch run at BUILD_ORDER
  Phase 14. Filed as INFO rather than as a defect because the gap is documented, dated
  and owner-authorised -- the failure mode worth guarding against is a later review
  reading the green suite and marking AC-04 PASS.
Correction: None now. At Phase 14, record the per-epoch `train_loss_total` series from
  `outputs/logs/pretrain_t_small.log` across the full run and close AC-04 against it.
  Do not close AC-04 on a capped run.
```

```
REV-107
Severity: LOW | Status: RESOLVED | Requirement: FR-6 | AC: AC-06
File: evaluation/gradient_diagnostics.py:143-146, :249-250
Resolution: re-confirmed. `run_gradient_diagnostics` saves and restores both
  `model.training` and `moe_layer.training`, so the invariant is held by the function
  that breaks it rather than by its callers.
```

```
REV-108
Severity: LOW | Status: OPEN | Requirement: FR-5 | AC: —
File: training/pretrain.py:242-255 (contrastive augmentation in `train_one_step`);
      evaluation/gradient_diagnostics.py:202-213 (identical block)
Problem: unchanged. The augmented contrastive view is built with a Python loop over the
  batch: `for i in range(B): real = seq[i][seq[i] != 0].tolist()`, then a Python-level
  `augment_sequence` per row, then one `torch.tensor(list-of-lists)` construction. Each
  `.tolist()` is a device-to-host copy and a synchronisation. At `batch_size=256` that is
  256 syncs plus 256 list round-trips on every 4th step.
Impact: Throughput only; the arithmetic is correct. Its relative weight has GROWN since
  the last review: loss-functions REV-120 removed ~275 ms/step of GradNorm cost, taking
  the measured step from 0.641 s to ~0.235 s, so this block is now a larger share of what
  remains.
Correction: Either move the augmentation into the dataset (precompute an augmented view
  per example at tensorisation time, which also makes it reproducible from the seed
  alone), or vectorise the three strategies as tensor ops over the `(B, S)` batch.
  Still not urgent. The right moment to judge it is against the Phase 14 full run, and
  ~0.235 s/step is now the baseline to judge it against, not 0.641.
```

```
REV-126
Severity: INFO | Status: RESOLVED | Requirement: FR-3 | AC: AC-03
File: specs/training-pipeline/requirements.md (FR-3)
Resolution: closed by the spec owner on 2026-09-17. FR-3 now lists the checkpoint's actual
  top-level keys -- `model_state_dict`, `moe_state_dict`, `loss_fns_state_dict`,
  `gradnorm_state_dict` (holding log_lambdas, its optimizer state and initial_losses) --
  instead of implying MoE and GradNorm state are nested inside `model_state_dict`, with a
  note recording that the three-key layout is the correct one and the wording was what was
  wrong. Verified against `best.pt` this pass: every key the amended FR-3 names is present.
```

```
REV-062 (cross-reference; owned by specs/transformer-backbone)
Severity: HIGH | Status: RESOLVED | Requirement: FR-5 | AC: —
Note: the MPS backend blocker on T6.2 is resolved. Re-verified here that the resolution
  did not come at the gate's expense: `git diff HEAD -- tests/test_smoke.py` is
  comment-only and `require_attention_backend = True` is still set.
  NOTE for the reader: a NEW critical finding on the CUDA/FA4 attention path, REV-122, is
  filed in `specs/transformer-backbone/review.md`. It does not affect this spec's dev-host
  results (MPS/SDPA is correct) but it MUST be fixed before the Phase 14 H100 run that
  AC-04 depends on.
```

```
REV-092 (cross-reference; owned by specs/loss-functions)
Severity: MEDIUM | Status: NOT_APPLICABLE (superseded by REV-120)
Note: the double-backward probe called from `run_pretrain` is deleted, along with the
  requirement it checked. Verified: `grep -rn probe_double_backward` over the tree returns
  nothing.
```

## Design Compliance

Compliant. The FR-6 training-step contract is implemented in exactly the specified order,
which I checked line by line rather than by grep. The `steering/architecture.md` boundary
"Training ↔ Evaluation: test data untouched until model configuration and best checkpoint
are frozen" is respected — the pretrain stage loads only the train and val tensorised
datasets, and best selection reads `val_ndcg` alone.

The one behavioural subtlety worth recording is that `max_steps_per_epoch` deliberately
shrinks `total_steps` as well as the step budget, so a capped run is a complete LR
schedule compressed rather than a truncated prefix of the full one. That is the right
choice for a smoke gate — it exercises warmup, decay and the tail in one run — but it does
mean a capped run's LR trajectory is not a prefix of the uncapped run's, which is worth
knowing when comparing the two.

No unapproved dependency, no architectural drift.

## Task Completion

| Task | Verdict |
|---|---|
| T6.2 one-batch end-to-end smoke | **Newly checked, and the check is justified.** The gate was re-run unmodified against the narrowed backend surface, and I confirmed the claim that mattered: `tests/test_smoke.py` changed on comment lines only, `require_attention_backend = True` intact, no assertion touched |
| T6.3 few-step T-Small smoke on real data | **Newly checked, and justified.** The task text was reduced by the *owner*, with the decision recorded in `steering/testing.md` and `BUILD_ORDER` rather than only in tasks.md — the distinction REV-066 turned on. The run happened and its three claims were verified against artifacts on disk. The note is also honest about what the gate does not cover (AC-04), which is the part a weaker note would have quietly omitted |
| T6.1, T6.4 | Checked; verified |
| T1–T5 | Checked; unchanged and previously verified. Re-spot-checked the scheduler, best-selection and routing-buffer paths |

No false-complete tasks in this spec. T6.3's note is the most carefully evidenced task
record in the repository and it is worth saying so.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **267 passed**, 0 failed, 49.20 s |
| `uv run python -m pytest -q -m slow --collect-only` | 8 slow collected (incl. `TestFewStepSmokeViaRunPretrain`, `TestFewStepsSmoke`), 259 deselected — confirms the slow gates ran in the 267 |
| `ls -la outputs/checkpoints/t_small outputs/logs outputs/plots` | `best.pt`, `epoch_000.pt`, `pretrain_t_small.log`, `gradnorm_t_small.jsonl`, `routing_diagnostics_t_small.json`, 6 plots — all 09-17 00:25–00:33 |
| `torch.load("outputs/checkpoints/t_small/best.pt")`, enumerate keys (direct) | all FR-3 fields present incl. `config`; `gradnorm_state_dict` = `{log_lambdas, optimizer, initial_losses}` |
| `cat outputs/logs/pretrain_t_small.log` (direct) | metrics match the T6.3 note to 6 d.p.; every AC-07 field present |
| λ arithmetic re-derived from the log (direct) | 0.259607 + 1.430659 + 1.309734 = **3.000000** over 3 active objectives — AC-05 confirmed |
| `grep -n "max_steps" main.py` | `--max_steps` in `parse_args` at `:59`, override logged at `:96-100` |
| `sed -n '298,325p' training/pretrain.py` | FR-6 step order confirmed line by line |
| `uv run python -m compileall -q training main.py` | clean |

## Final Assessment

Nothing blocking for BUILD. The finding that held this spec open is resolved, and the run
behind it is properly evidenced.

One optional item for BUILD:

1. **REV-108 (LOW)** — vectorise or precompute the contrastive augmentation. Judge it
   against the Phase 14 full run; the baseline is now ~0.235 s/step, not 0.641.

Two for the **spec owner**:

2. **REV-125 (INFO)** — AC-04 stays outstanding until the Phase 14 multi-epoch run. Close
   it against the per-epoch loss series from that run, not from a capped one.
3. **REV-126 (INFO)** — reword FR-3's "model_state_dict (including MoE + ...)" to match
   the three sibling keys actually written.

One cross-spec dependency worth flagging here because it lands on this spec's critical
path: **REV-122** (`specs/transformer-backbone`, CRITICAL) breaks causal masking on the
CUDA/FlashAttention-4 backend. Dev-host results are unaffected, but it must be fixed
before the H100 run that AC-04 depends on.
