# Review — Personalization & Federated Simulation

## Review Status

**PASS**

(Was `CHANGES_REQUIRED`. REV-131's provenance question is resolved — both stages were re-run
off a single freshly written checkpoint that records its own `source_revision`, and both
artifacts echo it. Only REV-114 (INFO) remains.)

## Feature

`specs/personalization-federated` — `models/lora.py` (`LoRALinear`, `inject_lora`,
`get_lora_state_dict`, `load_lora_state_dict`, `count_lora_parameters`),
`training/local_train.py` (`run_local_personalization`, `train_user_lora`),
`training/federated.py` (`weighted_fedavg`, `run_federated`, `compute_frozen_checksum`).

## Review Scope

- Spec directory: `specs/personalization-federated/` (requirements.md, design.md,
  tasks.md). None modified in the working tree. AC-05 and AC-06 carry the owner's
  2026-09-17 smoke-gate amendment.
- Steering: `architecture.md` (Global ↔ Local boundary), `testing.md` (Critical Runtime
  Assertions), `code-quality.md`, `conventions.md`.
- Baseline: `master` at `0bc67ee`, `origin/master` at the same commit. No file owned by
  this spec is modified in the working tree.
- Files inspected: `models/lora.py` (70 lines, read in full), `training/federated.py`
  (439 lines), `training/local_train.py` (488 lines), `tests/test_lora.py`,
  `tests/test_federated.py`, and the real artifacts
  `outputs/logs/personalization_gain_t_small.json`,
  `outputs/logs/federated_rounds_t_small.json` and the three plots.
- Provenance established with `git log --date=format`, `git show <commit> -- <file>` and
  artifact `stat` times, because REV-131 turns on ordering.
- Re-review. Prior findings carried in: REV-127 (MEDIUM, OPEN), REV-114 (INFO, OPEN),
  REV-117, REV-111, REV-112, REV-113 (RESOLVED).
- Date: 2026-09-18

## Summary

Closed. Both stages were re-run on 2026-09-18 inside a single sequenced cascade off one
freshly written checkpoint, and the provenance question that reopened this spec is now
answerable from the files.

Re-measured here rather than accepted:

- **Provenance.** `personalization_gain_t_small.json` and `federated_rounds_t_small.json`
  both carry a `source_revision` that MATCHES the checkpoint's exactly
  (`{commit: "0bc67ee…", dirty: true, source_hash: "3a59e7cb70a8be69"}`). Artifact mtimes
  confirm the ordering: checkpoint 15:28:58, personalize 15:29, federated 15:31.
- **FR-4.** All six fields; 10 users, 0 skipped; `gain.mean = -0.00980`; `per_user_gains`
  1 positive / 5 negative / 4 zero — not constant, so AC-05's smoke-gate condition holds.
- **FR-7 / FR-8.** 3 rounds, `val_examples_scored` uniform at 5000, `val_ndcg_full` on the
  final round only (0.09461), `upload_kb = 48.0`, `total_round_kb = 960.0`. I re-derived
  the cost independently of the log, which does not record `lora_params`:
  `count_lora_parameters` gives 24,576 at `lora_rank=8`, and `24576 × 2 / 1024 = 48.0`.

These numbers differ from the previous round's because the training data changed underneath
(`data-pipeline` REV-137 restored 6,634 previously-deleted interactions). BUILD does not
present them as comparable, which is correct.

The discipline on the results themselves is unchanged and still right: the negative mean
gain and the per-round NDCG are recorded as observations with the RQ3/RQ4 claims explicitly
withheld to Phase 14.

## Git Changes

`training/local_train.py` (+23) and `training/federated.py` (+12): both now read
`artifact_source_revision(backbone_ckpt)` and stamp it into their reports. No change to the
LoRA, aggregation, checksum or communication-accounting logic — confirmed by diff and by the
26 tests in `tests/test_federated.py` + `tests/test_lora.py` passing unchanged.

Artifact mtimes place both stages inside the 2026-09-18 cascade, after the checkpoint they
echo.

**Scope assessment:** proportionate; provenance plumbing only.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 LoRA implementation | PASS | `LoRALinear` clones weight and bias as `requires_grad=False` parameters, adds `lora_A = randn(d_in, r) * 0.02` and `lora_B = zeros(r, d_out)`, and forwards `F.linear(x, w, b) + x @ A @ B`. `inject_lora` replaces `q_proj` and `v_proj` in every block, freezes everything, then unfreezes only the four LoRA tensors per block |
| FR-2 Parameter isolation | PASS | Measured on a t-small build after injection: 12 trainable tensors, **all** matching the LoRA pattern; zero trainable backbone parameters. `get_lora_state_dict` filters on the LoRA name pattern and additionally excludes the four `moe.*` prefixes; measured output is 12 keys, all `blocks.N.attn.{q,v}_proj.lora_{A,B}`, with no MoE key |
| FR-3 Local personalization | PASS | Per user: frozen backbone, adapter reset, LoRA-only AdamW at `lora_lr` with `weight_decay=0`, train on `train_user_history`, then evaluate with and without LoRA on the stored candidate rows. Cold/warm is recorded per user (`is_cold` present in every `per_user_gains` entry) |
| FR-4 Report & plots | PASS | All six fields present in the real JSON; both plots exist |
| FR-5 Federated simulation | PASS | Persistent `global_lora_state` across rounds; clients load it, train, and upload; weighted FedAvg by sample count with a simple-mean branch. `TestWeightedFedAvg` checks both against hand-computed values |
| FR-6 Client state integrity | PASS | `compute_frozen_checksum` is taken before round 1 and asserted at four points — on each client model before local training, after each round, and at the end. The real 3-round run exited 0, so none fired. MoE-key exclusion is asserted on every client update, and `TestMoEExclusionGuard` proves the assertion *fires* by injecting both a synthetic and a real MoE key |
| FR-7 Communication cost | PASS | Verified independently of the log (see Summary): 48.0 / 48.0 / 960.0 match the formula exactly at `lora_rank=8` |
| FR-8 Convergence plot | PASS | `federated_convergence_t_small.png` exists; every plotted round is scored on the same 5,000-example prefix, so the series is one population |
| FR-9 LoRA training constraints | PASS | `lora_lr` is a separate config field, `weight_decay=0`, optimizer is AdamW over LoRA parameters only |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 Q and V wrapped; output changes after training | PASS | `test_q_v_wrapped`, `test_forward_changes_after_training`. Measured: exactly `q_proj` and `v_proj` in all 3 blocks are `LoRALinear` |
| AC-02 state dict has only LoRA keys | PASS | Measured directly: 12 keys, zero MoE keys. `TestLoRAStateDict` plus `TestMoEExclusionGuard`'s three tests |
| AC-03 count matches formula | PASS | Measured: `count_lora_parameters` = 24,576 = `2 × 3 layers × (256×8 + 8×256)`, matching the hand formula exactly |
| AC-04 frozen checksum identical per user | PASS | `test_run_local_personalization_preserves_frozen_backbone` and `test_frozen_checksum_unchanged_after_lora`; the real 10-user run exited 0 with the assertion armed |
| AC-05 *(smoke gate)* personalization completes, six fields, both plots, gains not constant | PASS | Verified against the file: all six fields, 10 users, 0 skipped, and a gain array with both signs (4 pos / 3 neg / 3 zero). The distribution is **not** claimed, as the amendment requires. Provenance caveat: see REV-131 |
| AC-06 *(smoke gate)* federated completes, checksum holds, uniform `val_examples_scored`, `val_ndcg_full` final only, plot renders | PASS | Verified against the file: 3 rounds, `val_examples_scored` uniform at 5000, `val_ndcg_full` `[None, None, 0.11288]`, plot present, run exited 0. Convergence is **not** claimed. Provenance caveat: see REV-131 |
| AC-07 communication cost formula | PASS | Re-derived independently, not read back from the log |
| AC-08 checksum unchanged through all rounds | PASS | Asserted at four points in `run_federated`; `test_frozen_backbone_survives_run_federated` |
| AC-09 `global_lora_state` persists; clients resume | PASS | `TestFederatedConvergence::test_run_federated_two_rounds`; `TestWeightedFedAvg` covers the aggregation |

## Findings

```
REV-131
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-4, FR-7, FR-8 | AC: AC-05, AC-06
File: outputs/logs/personalization_gain_t_small.json; outputs/logs/federated_rounds_t_small.json
Resolution: resolved by removing the cause, not by settling the argument. Both stages were
  re-run on 2026-09-18 as part of one sequenced cascade off a single freshly written
  checkpoint (personalize 15:29, federated 15:31, checkpoint 15:28:58 -- mtimes confirm the
  ordering), and both artifacts now carry a `source_revision` that MATCHES the checkpoint's
  exactly: `{commit: "0bc67ee...", dirty: true, source_hash: "3a59e7cb70a8be69"}`. Verified
  by reading the checkpoint and each artifact and comparing the fields.
  The three contradictory accounts are gone: T6.1 and `loss-functions` T3.7 now both defer to
  `specs/training-pipeline` T6.3, which is consistent with the files. One account, and it is
  checkable from the artifacts rather than from prose -- which is the durable half of the fix.
Re-measured this pass on the regenerated artifacts:
    * personalization: 10 users, 0 skipped, all six FR-4 fields; `gain.mean = -0.00980`;
      `per_user_gains` has 1 positive / 5 negative / 4 zero -- still not constant, so AC-05's
      smoke-gate condition holds. Note this run sits on the REV-137 training data, so the
      numbers are not comparable to the previous run's and BUILD does not present them as
      such.
    * federated: 3 rounds, `val_examples_scored` UNIFORM at 5000, `val_ndcg_full` present on
      the final round ONLY (0.09461), `upload_kb = download_kb = 48.0`, `total_round_kb =
      960.0`. AC-07's formula re-derived independently of the log: `count_lora_parameters`
      gives 24,576 at `lora_rank=8`, and 24576 * 2 / 1024 = 48.0 exactly.
    * The frozen-state checksum assertion did not fire; both runs exited 0.
Discipline maintained, and it is the right one: the negative mean gain and the per-round NDCG
  are recorded as observations with the research claim withheld to Phase 14, exactly as the
  owner's smoke-gate amendment requires.
```
REV-127
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-4, FR-7, FR-8 | AC: AC-05, AC-06
File: outputs/logs/, outputs/plots/
Resolution: both preconditions cleared and both stages run. Precondition (a),
  `ablations-cli` REV-131's missing CLI surface, is closed -- `--eval_users`,
  `--clients_per_round` and `--local_epochs` exist, all default to `None` so the
  configured values stay authoritative when absent, and all four overrides were logged
  original -> new. `configs/config.py` is untouched, which I confirmed against `git diff`.
  All five artifacts now exist and every claim BUILD made about them was independently
  re-measured here and holds -- including `upload_kb`, which I re-derived from
  `count_lora_parameters` rather than reading back from the log that reports it.
Note: precondition (b) -- running against a regenerated checkpoint -- is the part that is
  now in question, tracked as REV-131. The artifacts exist and are internally consistent;
  what is unresolved is which model produced them.
```

```
REV-117
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-5, FR-8 | AC: AC-06
Re-verified on a real run for the first time, which is what it needed. Measured on
  `federated_rounds_t_small.json`: `val_examples_scored` is 5000 on every round including
  the last, and `val_ndcg_full` appears only on the final round. The plotted series is one
  population end to end. `test_scores_stored_candidates_and_bounds_each_round` pins it, and
  `test_zero_means_full_split_every_round` covers the `federated_eval_examples = 0` case.
```

```
REV-111 / REV-112 / REV-113
Severity: LOW / MEDIUM / LOW | Status: RESOLVED
Re-confirmed. Negatives are drawn from one run-scoped RandomState threaded into
  `train_user_lora`; evaluation reads the stored fingerprinted candidate rows via
  `load_or_build_eval_candidates` with `expected_rows`; and the FR-4 field is
  `num_users_skipped` in the code, the tests and the real artifact.
```

```
REV-114
Severity: INFO | Status: OPEN | Requirement: FR-3 | AC: —
File: training/local_train.py (`pristine_lora` reuse)
Unchanged, and now observable in a real artifact. FR-3 says "inject fresh LoRA" per user;
  the implementation builds the model once and resets via
  `load_lora_state_dict(model, pristine_lora)`. Because `lora_B` is zero-initialised a
  reset adapter is an exact identity, so no user's trained adapter leaks into another's
  and the "without LoRA" measurement is exact -- the requirement's intent is met. What
  differs from a literal reading is that `lora_A` is the same random draw for every user.
Impact: none, and arguably better -- it removes per-user initialisation variance from the
  gain distribution. Worth stating in the paper's method section, since the 10-user
  histogram now exists and a reader will assume per-user re-randomisation.
Correction: none required.
```

## Design Compliance

Compliant with the spec and with `steering/architecture.md`'s Global ↔ Local boundary:
"only LoRA adapter state crosses the wire; backbone, MoE, router, experts, objective
embeddings are frozen global state". That boundary is enforced three ways rather than
one — by `requires_grad` after injection, by the name filter plus explicit MoE-prefix
exclusion in `get_lora_state_dict`, and by a runtime assertion on every client update — and
the third is proven to fire by a test that injects a real MoE key.

`steering/testing.md`'s Critical Runtime Assertions for this area are all present and
armed: the LoRA state dict carries no `moe.router` / `moe.experts` /
`moe.objective_embeddings` / `moe.gates` keys, and the federated frozen-state checksum is
asserted unchanged through every round.

The gate-scale reduction was done entirely through CLI overrides with `configs/config.py`
untouched, which is what `conventions.md`'s Config-Driven Design requires and what made
the earlier version of this run impossible.

## Task Completion

| Task | Marked | Verdict |
|---|---|---|
| T6.1 run both stages at gate scale | `[x]` | **Complete as to execution, but its provenance claim is not supported.** Both stages did run, the artifacts exist, and every measurable claim in the note checks out against the files — I re-derived `upload_kb` independently rather than reading it back. The one claim I could not confirm is the one that matters most: "against the REGENERATED best.pt … corrected multi-step objective", which `specs/loss-functions` T3.7 directly contradicts. See REV-131 |
| T1.1–T1.4, T2.1–T2.4 (LoRA) | `[x]` | Complete. Injection, filtering, counting and freeze enforcement all re-measured directly this pass |
| T3.1–T3.4 (personalization) | `[x]` | Complete, now with artifacts behind them |
| T4.1–T4.5 (federated) | `[x]` | Complete. T4.5's REV-117 invariant is confirmed on a real run |
| T5.1–T5.4 (tests) | `[x]` | Complete; 26 tests pass. T5.4 is stronger than the task text asks — it proves the guard fires on a real MoE key, not only on a synthetic one |

No task is false-complete on execution. T6.1's *record* is what needs correcting.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **275 passed**, 0 failed, 55.74 s |
| `uv run python -m pytest tests/test_federated.py tests/test_lora.py -q` | **26 passed**, 27.25 s |
| `inject_lora` + `count_lora_parameters` on t-small (direct) | 24,576 = `2 × 3 × (256×8 + 8×256)`; `24576 × 2 / 1024 = 48.0` |
| `get_lora_state_dict` on the injected model (direct) | 12 keys, all `blocks.N.attn.{q,v}_proj.lora_{A,B}`; zero MoE-prefixed keys |
| `named_parameters()` trainable filter (direct) | 12 trainable tensors, all LoRA; zero trainable backbone parameters |
| Read `personalization_gain_t_small.json` (direct) | six fields; 10 evaluated, 0 skipped; `gain.mean=-0.004738`, min −0.09635, max 0.07887; `per_user_gains` 4 pos / 3 neg / 3 zero, each entry carrying `user_id`, `gain`, `is_cold` |
| Read `federated_rounds_t_small.json` (direct) | 3 rounds; `val_examples_scored` 5000/5000/5000; `val_ndcg_full` None/None/0.11288; `upload_kb` 48.0, `download_kb` 48.0, `total_round_kb` 960.0, `num_clients` 10 in every round |
| `ls outputs/plots/` | `personalization_gain_histogram_t_small.png`, `global_vs_personalized_t_small.png`, `federated_convergence_t_small.png` all present |
| `git show daffb41 -- main.py` / `-- training/losses.py` | one commit carries both the CLI flags the gate run used and the `pos = S - 1` change — the basis for REV-131's ordering analysis |

## Final Assessment

**Nothing for BUILD. This spec is accepted.**

REV-131 is closed the durable way: not by reconstructing which checkpoint produced which
number, but by making that question a field lookup. Both artifacts now name their source,
and the name matches the checkpoint they came from.

REV-114 (INFO) stands unchanged and needs no action — every user starts from an identical
adapter that is an exact identity, so `lora_A` being one shared draw removes initialisation
variance from the gain distribution rather than confounding it. State it in the method
section; the 10-user histogram now exists and a reader will otherwise assume per-user
re-randomisation.

Two things to carry, neither this spec's work:

1. **`training-pipeline` REV-138** — the provenance identifier does not move when
   `evaluation/` changes. `training/local_train.py` and `training/federated.py` *are*
   covered, so these two artifacts were correctly attributed even before the fix. **REV-138 is
   now resolved** — coverage is an exclusion rule and every first-party module is in scope, so
   the caveat is retired.
2. **RQ3 and RQ4 still have no acceptance evidence.** AC-05 and AC-06 are pipeline-integrity
   gates at this scale and say so. Ten users cannot characterise a gain distribution and
   three rounds over ten clients cannot show convergence. A later review must not read these
   green artifacts as satisfying the research claims — that is Phase 14's job.
