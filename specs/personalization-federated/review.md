# Review — Personalization & Federated Simulation

## Review Status

**CHANGES_REQUIRED**

(REV-117, the MEDIUM finding that held this spec open, is fully resolved. What keeps the
status is REV-127: neither stage has ever been run, so the required artifacts do not exist.
The owner reduced AC-05 and AC-06 to smoke gates on 2026-09-17, which makes that run cheap
— but it is now blocked on `ablations-cli` REV-131, because the reduced scale it specifies
has no CLI surface.)

## Feature

`specs/personalization-federated` — `models/lora.py` (`LoRALinear`, `inject_lora`,
`get_lora_state_dict`, `load_lora_state_dict`, `count_lora_parameters`),
`training/local_train.py` (per-user LoRA training, 1000-user evaluation, report, plots),
`training/federated.py` (weighted FedAvg, round loop, checksum enforcement, communication
logging, convergence plot).

## Review Scope

- Spec directory: `specs/personalization-federated/` — `requirements.md` and `design.md`
  unmodified; `tasks.md` edited (T3.3 field-name sync, T4.5 REV-117 note).
- Steering: `steering/architecture.md` (LoRA-only federated communication),
  `code-quality.md`, `testing.md` (LoRA key filtering, frozen-state checksum).
- Baseline: working tree vs `HEAD` = `d438950` on `master`; `origin/master` at the same
  commit. Of this spec's files, `training/federated.py` (`+99`) and
  `training/local_train.py` (`+59`) are modified, plus `tests/test_federated.py` (`+235`)
  and `configs/config.py` (`federated_eval_examples`).
- Files inspected: `training/federated.py`, `training/local_train.py`, `models/lora.py`,
  `tests/test_federated.py`, `tests/test_lora.py`, `main.py`
  (`accuracy_vs_communication`), `configs/config.py`.
- `outputs/logs/` and `outputs/plots/` were listed directly to check for this spec's
  artifacts.
- Re-review. Prior findings carried in: REV-111 (RESOLVED), REV-112 (RESOLVED),
  REV-113 (RESOLVED), REV-114 (INFO, OPEN), REV-117 (MEDIUM, OPEN).
- Date: 2026-09-17

## Summary

REV-117 is closed, and the fix took the exact shape the correction asked for rather than
an approximation of it. Every round — the final one included — is now scored on the same
bounded `federated_eval_examples` prefix, so the series `_plot_convergence` draws is one
population end to end. The final round *additionally* runs the full split into
`val_ndcg_full`, and that field is what the headline number reads: a new
`final_val_ndcg(round_logs)` helper prefers `val_ndcg_full` (falling back to the prefix
only for an interrupted run), and both `run_federated`'s return value and `main.py`'s
`accuracy_vs_communication` go through it. The plot deliberately does *not* draw
`val_ndcg_full`, with a comment saying why, and labels the series with the example count
when it is uniform. All three consumers of the number are consistent.

The regression test is genuinely a regression test, which is worth noting because the
correction specifically asked for one. `test_scores_stored_candidates_and_bounds_each_round`
asserts the evaluation count (3 = two bounded rounds + one full-split pass), asserts each
per-round matrix is *the stored prefix* by `array_equal`, asserts
`{log["val_examples_scored"]} == {federated_eval_examples}` — the assertion that makes the
curve readable — and asserts `final_val_ndcg == logs[-1]["val_ndcg_full"]`. It also
asserts `val_ndcg_full` is absent from round 0, so the "only the final round runs the full
split" half cannot silently regress either.

REV-111, REV-112 and REV-113 remain closed; I re-checked the candidate-loader routing and
the run-scoped negative RNG rather than inheriting them.

What has not changed is that **neither stage has ever been executed**. `outputs/logs/` and
`outputs/plots/` contain no `personalization_gain_*.json`,
`personalization_gain_histogram_*.png`, `global_vs_personalized_*.png`,
`federated_rounds_*.json` or `federated_convergence_*.png`. Previously this was arguably
blocked — there was no pretrained checkpoint to personalize from. There is now:
`outputs/checkpoints/t_small/best.pt` was produced on 09-17 and verified in the
training-pipeline review. So the gap is actionable, and it is filed as REV-127.

## Git Changes

| File | Change | Attribution |
|---|---|---|
| `training/federated.py` | `+99`: bounded prefix on every round, additional final-round full-split pass, `val_ndcg_full` / `val_examples_scored_full`, new `final_val_ndcg()` helper, plot comment + dynamic series label | REV-117, this spec |
| `training/local_train.py` | `+59`: run-scoped `neg_rng`, stored candidate rows with an alignment assertion, `num_skipped` → `num_users_skipped` | REV-111, REV-113 |
| `main.py` | `accuracy_vs_communication` now reads `final_val_ndcg(rounds)` | REV-117 |
| `configs/config.py` | `federated_eval_examples: int = 5000` with rationale | REV-112(b) |
| `tests/test_federated.py` | `+235`: the REV-117 regression assertions, full-split-when-zero case | this spec |
| `specs/.../tasks.md` | T3.3 field name, T4.5 REV-117 note | spec sync |

**Scope assessment:** in scope and proportionate. No new dependency, no committed
artifact. The `tasks.md` edits are a one-word FR-4 sync and a note recording a fix — not a
relaxation of either task.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 LoRA implementation | PASS | `LoRALinear` with `A ~ 0.02·randn`, `B = zeros`; Q and V wrapped in every attention layer; freeze-all-then-unfreeze-LoRA; `tests/test_lora.py` |
| FR-2 Parameter isolation | PASS | Optimizer built from LoRA params only; zero-trainable-backbone assertion before training; `get_lora_state_dict` filters to `lora_`/`.A`/`.B`, with two negative tests asserting injected and real MoE keys are rejected |
| FR-3 Local personalization | PASS (code) | Seeded 1000-user sample, per-user reset to a pristine adapter, with/without-LoRA evaluation on the stored test rows, run-scoped negative RNG (REV-111). Literal "fresh injection" nuance is REV-114 |
| FR-4 Personalization report & plots | **PARTIAL** | The writer produces every field FR-4 lists and both plots are implemented; **no report or plot has ever been produced** (REV-127) |
| FR-5 Federated simulation | PASS | Persistent `global_lora_state`, per-round client sampling, weighted FedAvg with the simple-mean fallback, zero-sample clients skipped; `test_run_federated_two_rounds` |
| FR-6 Client state integrity | PASS | `compute_frozen_checksum` asserted at client construction, before/after every round, and after the loop; `test_frozen_backbone_survives_run_federated` |
| FR-7 Communication cost | **PARTIAL** | Formula implemented exactly (`lora_params * 2 / 1024`, doubled, × clients) and unit-tested; MoE params excluded by construction. **`federated_rounds_{model}.json` has never been written** (REV-127) |
| FR-8 Federated convergence plot | **PARTIAL** | `_plot_convergence` is correct and now internally consistent (REV-117). **The PNG has never been produced** (REV-127) |
| FR-9 LoRA training constraints | PASS | AdamW over LoRA params only, `lora_lr` independent of backbone LR, `weight_decay=0` |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 `inject_lora` wraps Q and V; output changes after training | PASS | `tests/test_lora.py` |
| AC-02 LoRA state dict has no MoE/router/expert/embedding keys | PASS | Name-pattern filter plus two negative tests — one injecting a fake MoE key, one using a real `moe.*` key |
| AC-03 `count_lora_parameters` matches A+B sizes | PASS | `tests/test_lora.py` |
| AC-04 frozen checksum identical per user | PASS (unit) | `test_run_local_personalization_preserves_frozen_backbone`. No artifact from a real 1000-user run (REV-127) |
| AC-05 *(smoke gate, amended 2026-09-17)* personalization completes, report + both plots written, gains not all-constant | **NOT VERIFIABLE** | Still no artifact. The owner reduced this from a distribution claim to a pipeline-integrity gate at `personalization_eval_users = 10`, which is achievable — but running it needs a flag that does not exist (REV-131). The distribution claim (RQ3) now belongs to Phase 14 |
| AC-06 *(smoke gate, amended 2026-09-17)* federated completes, checksum holds, round log uniform with `val_ndcg_full` on the last round, curve renders | **NOT VERIFIABLE** | Still no artifact. Reduced by the owner from a convergence claim to a pipeline-integrity gate at `clients_per_round = 10`; blocked on REV-131. The REV-117 fix makes the curve readable when it is produced, which is a precondition, not evidence. The convergence claim (RQ4) now belongs to Phase 14 |
| AC-07 comm cost matches formula | PASS (unit) | Formula verified in code and test; no round log from a real run (REV-127) |
| AC-08 frozen checksum unchanged through all rounds | PASS (unit) | Asserted before and after every round in `run_federated`; `test_frozen_backbone_survives_run_federated` drives a real two-round loop |
| AC-09 `global_lora_state` persists; clients resume | PASS (unit) | `test_run_federated_two_rounds` |

## Findings

```
REV-117
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-5, FR-8 | AC: AC-06
File: training/federated.py:155-175, :261-320, :397-430; main.py:601-604
Resolution: implemented exactly as the correction specified, in all three places the
  number is consumed.
  - Every round, final included, scores `round_val_candidates` = the bounded prefix. The
    `eval_sequences = val_sequences if is_final_round else round_val_sequences` branch is
    gone.
  - The final round additionally runs the full split into `val_ndcg_full`,
    `val_mrr_full`, `val_examples_scored_full`, and short-circuits to reuse `val_metrics`
    when `federated_eval_examples = 0` (so the zero case does not pay for two identical
    passes).
  - `_plot_convergence` plots the prefix series only, with a comment stating that
    `val_ndcg_full` is deliberately excluded, and labels the series with the example count
    when it is uniform across rounds.
  - The headline number goes through a new `final_val_ndcg(round_logs)` helper that
    prefers `val_ndcg_full` and falls back to `val_ndcg` only when no full-split pass was
    recorded. Both `run_federated`'s return value and `main.py`'s
    `accuracy_vs_communication` use it -- I checked the latter specifically, since it was
    the second consumer the correction named.
  The requested test extension is present and non-vacuous:
  `test_scores_stored_candidates_and_bounds_each_round` asserts the evaluation COUNT is
  3, that each per-round matrix equals the stored prefix by `array_equal`, that
  `{val_examples_scored} == {federated_eval_examples}` across all plotted points, that
  `val_ndcg_full` is absent from round 0 and present on the last, and that
  `final_val_ndcg == logs[-1]["val_ndcg_full"]`.
```

```
REV-127
Severity: MEDIUM | Status: OPEN | Requirement: FR-4, FR-7, FR-8 | AC: AC-05, AC-06 (and the artifact half of AC-04, AC-07)
File: outputs/logs/ (no personalization_gain_*.json, no federated_rounds_*.json);
      outputs/plots/ (no personalization_gain_histogram_*.png,
      no global_vs_personalized_*.png, no federated_convergence_*.png)
Problem: unchanged in substance -- neither `--stage personalize` nor `--stage federated`
  has ever been run, so three of this spec's five required output artifacts and both
  personalization plots do not exist.
  Two things changed around it on 2026-09-17, and they pull in opposite directions.
  (1) The spec owner reduced AC-05 and AC-06 from research claims to pipeline-integrity
      smoke gates, at `personalization_eval_users = 10`, `clients_per_round = 10`,
      `local_epochs = 1` and a few rounds. That makes the run cheap and removes the reason
      it kept being deferred.
  (2) That reduced scale has NO CLI surface -- `ablations-cli` REV-131 -- so running it
      today would mean editing `configs/config.py`, which `conventions.md` forbids and
      which would change the configured values every later run inherits.
  So this finding is now blocked on REV-131 rather than on cost or on a missing checkpoint.
Impact: FR-4's report, FR-7's round log and FR-8's convergence plot still have no
  existence evidence, and the communication-efficiency claim RQ4 rests on a round log no
  real run has written. Note what the owner's amendment did and did not do: it moved the
  RQ3/RQ4 *claims* to Phase 14, so AC-05 and AC-06 no longer assert an outcome -- but it
  did not remove the requirement that the artifacts exist and the invariants hold, which
  is what this finding tracks.
Correction: TWO preconditions before the runs, both outside this spec.
  (a) `ablations-cli` REV-131 -- the flags below do not exist yet.
  (b) `loss-functions` REV-105 changes the multi-step objective, which makes the CURRENT
      `outputs/checkpoints/t_small/best.pt` a model trained under a superseded objective.
      These stages consume that checkpoint. Re-run the ~190 s T6.3 pretrain gate after
      REV-105 lands, and run these stages against the regenerated checkpoint -- otherwise
      the artifacts describe an experiment the code no longer implements, and nothing in
      the pipeline will flag it.
  Then run both stages at the gate scale and verify against the files rather than the
  console:
      uv run python main.py --stage personalize --model t-small --eval_users 10 --local_epochs 1
      uv run python main.py --stage federated  --model t-small --clients_per_round 10 --epochs 3
  Confirm: `personalization_gain_t_small.json` carries all six FR-4 fields (including
  `num_users_skipped`, per REV-113) and `per_user_gains` contains both positive and
  negative entries; both plots render; `federated_rounds_t_small.json` records
  `upload_kb == lora_params * 2 / 1024`, a UNIFORM `val_examples_scored` across rounds and
  `val_ndcg_full` on the final round only (the REV-117 invariant, on a real run for the
  first time); the frozen checksum assertion does not fire; and
  `federated_convergence_t_small.png` renders.
  Do NOT reduce the defaults in `configs/config.py` to achieve this, and do not record the
  resulting numbers as evidence for AC-05's distribution or AC-06's convergence -- the
  amended criteria explicitly disclaim both at this scale.
```

```
REV-111
Severity: LOW | Status: RESOLVED | Requirement: FR-3, FR-9 | AC: —
Resolution: re-confirmed. One run-scoped `np.random.RandomState(config.seed)` is threaded
  into `train_user_lora`, so the adapter no longer meets an identical 100 negatives on
  every local epoch; the evaluation half reads the stored candidate rows.
  `test_training_negatives_are_run_scoped_and_evaluation_uses_stored_rows` covers both.
```

```
REV-112
Severity: MEDIUM | Status: RESOLVED (both parts) | Requirement: FR-5 | AC: AC-06
Resolution: (a) re-confirmed -- `run_federated` routes through
  `load_or_build_eval_candidates` with `expected_rows`, so federated scores the same
  fingerprinted rows as every other stage and no longer rebuilds them under a hardcoded
  seed. (b) bounded by `federated_eval_examples`, and the follow-on defect that fix
  introduced is now itself resolved (REV-117).
```

```
REV-113
Severity: LOW | Status: RESOLVED | Requirement: FR-4 | AC: —
Resolution: re-confirmed. `num_skipped` → `num_users_skipped` throughout, matching FR-4;
  the `tests/test_cli.py` keyword call site was updated and `tasks.md` T3.3 was synced.
  No stale readers remain.
```

```
REV-114
Severity: INFO | Status: OPEN | Requirement: FR-3 | AC: —
File: training/local_train.py:274 (`pristine_lora` reuse)
Problem: unchanged. FR-3 says "inject fresh LoRA" per user. The implementation builds the
  model once and resets between users via `load_lora_state_dict(model, pristine_lora)`.
  Because `lora_B` is zero-initialized a reset adapter is an exact identity, so the
  "without LoRA" measurement is unaffected and no user's trained adapter leaks into
  another's -- the requirement's intent is met. What differs from a literal reading is
  that `lora_A` is the SAME random draw for every user.
Impact: None, and arguably better: a shared starting point removes per-user initialization
  variance from the gain distribution AC-05's histogram reads. Worth keeping in mind when
  REV-127's run produces that histogram.
Correction: None. If the spec owner wants per-user re-randomization it is a one-line
  change in the reset, but the current behaviour is the better experiment.
```

## Design Compliance

Compliant on the boundary that matters most: only LoRA adapter state crosses the wire.
`get_lora_state_dict` filters to `lora_`/`.A`/`.B` keys, two negative tests assert MoE keys
are rejected, and `compute_frozen_checksum` is asserted at client construction, per round,
and after the loop — so `steering/architecture.md`'s "backbone, MoE, router, experts,
objective embeddings are frozen global state" is enforced rather than assumed, and
`steering/testing.md`'s "Federated: global frozen-state checksum unchanged through every
round" has a real runtime assertion behind it.

The REV-117 resolution also improves design honesty in a way worth recording: the round
log now carries both the measurement population (`val_examples_scored`) and the flag
distinguishing the headline pass (`is_final_round`, `val_examples_scored_full`), so the
artifact is self-describing rather than requiring the reader to know the config.

No unapproved dependency, no architectural drift.

## Task Completion

All tasks are marked `[x]`. Spot-checked this pass:

| Task | Verdict |
|---|---|
| T3.3 personalization report | Checked; field names now match FR-4. But the report has never been **written** by a real run — the task claims implementation, which is true, so this is not false-complete; the gap is REV-127 |
| T3.4 personalization plots | Same: implemented, never produced |
| T4.4 communication cost logging | Implemented and unit-tested; never written by a real run (REV-127) |
| T4.5 federated convergence plot | Checked. The REV-117 note appended to it is accurate in every clause I verified — same prefix on every round, full split additionally on the last, and the test asserting a single plotted population |
| T1.x–T2.x LoRA | Checked; unchanged and previously verified |

No false-complete tasks: every task claims implementation, and every implementation exists.
The missing piece is execution, which no task in this spec currently asks for — see the
Final Assessment.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **267 passed**, 0 failed, 49.20 s |
| `uv run python -m pytest -q -m slow --collect-only` | 6 of the 8 slow tests belong to this spec (`TestFederatedConvergence`, `TestFederatedValidationProtocol`, `TestPersonalizationEntryPoint`, `TestPersonalizationNegativeSampling`) and ran in the 267 |
| `git diff training/federated.py` (read in full) | REV-117 fix confirmed clause by clause |
| `sed -n '302,384p' tests/test_federated.py` | regression assertions confirmed non-vacuous |
| `grep -n "accuracy_vs_communication" -A 12 main.py`; `sed -n '590,620p' main.py` | third consumer routes through `final_val_ndcg(rounds)` |
| `ls outputs/logs outputs/plots \| grep -i "federated\|personalization"` | **no matches** — REV-127 |
| `uv run python -m compileall -q training models` | clean |

## Final Assessment

For BUILD, in order:

1. **`ablations-cli` REV-131 (MEDIUM)** — add `--eval_users`, `--clients_per_round` and
   `--local_epochs`. This is not this spec's finding, but it is this spec's blocker.
2. **REV-127 (MEDIUM, blocking)** — run both stages at the owner's gate scale and verify
   the five artifacts against the files. The commands and the checks are in the finding.

Nothing else. REV-111, REV-112, REV-113 and REV-117 are all resolved and were re-verified
against the source this pass rather than carried forward. REV-117's fix was implemented in
all three places the number is consumed, with a regression test that asserts the plotted
series is a single population — the specific thing that made the curve unreadable.

For the **spec owner**:

3. **REV-114 (INFO)** — decide whether FR-3's "inject fresh LoRA" should be reworded to
   match the shared-`lora_A` reset, which is the better experiment. Worth settling before
   the gate run produces the first real gain histogram.
4. Consider adding explicit *execution* tasks to `tasks.md`. Every task in this spec asks
   for implementation and all of them are honestly complete; nothing asks anyone to run
   either stage, which is why this gap persisted quietly across four reviews.
   `training-pipeline` T6.3 is the model to follow.
