# Review — Personalization & Federated

## Review Status

**CHANGES_REQUIRED**

## Feature

`specs/personalization-federated` — `models/lora.py`, `training/local_train.py`, `training/federated.py`: LoRA injection, 1000-user local personalization, weighted FedAvg, frozen-state checksums, communication accounting, convergence plotting.

## Review Scope

- Spec directory: `specs/personalization-federated/` (requirements.md, design.md, tasks.md)
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`
- Baseline: working tree vs `HEAD` = `32b0904` on `master`. All feature work is uncommitted.
- Files inspected: `models/lora.py` (70 lines, unchanged), `training/local_train.py` (447, untracked), `training/federated.py` (356, untracked), `tests/test_lora.py`, `tests/test_federated.py` (400, untracked), `configs/config.py`
- Re-review of the prior `review.md`. `local_train.py` (21:31), `federated.py` (15:45) and `tests/test_federated.py` (15:57) were all modified after it.
- Date: 2026-09-15

## Summary

Both prior CRITICALs are genuinely fixed, and the fixes are backed by tests that check the property rather than restating it. REV-003 — the per-user LoRA being fitted to the very test examples its gain was then measured on — is closed: `build_local_training_examples` draws from the user's training interactions (or validation, per `local_train_source`), the test split is rejected outright by `Config.validate`, and `TestLocalTrainingSourceIsNotTheEvaluationSplit` asserts the training and evaluation index sets are disjoint. REV-004 is closed the same way for the federated client loop. REV-061 (the positional argument binding a negative count to `normalize_embeddings`) and REV-062b (the metadata-free serving path) are both fixed, and REV-070's dangling config flags are gone.

Three new issues remain, all in the federated/local evaluation plumbing rather than in the isolation guarantees, which are solid. The largest is that `run_federated` rebuilds the entire validation candidate set on every call and then scores the full 993,372-example validation split on every one of 20 rounds — which is both a violation of the "generated once and reused" constraint and not tractable at real scale.

## Git Changes

| File | State |
|---|---|
| `models/lora.py` | unchanged from `32b0904` |
| `training/local_train.py` | untracked (new, 447 lines) |
| `training/federated.py` | untracked (new, 356 lines) |
| `tests/test_federated.py` | untracked (new, 400 lines) |
| `tests/test_lora.py` | modified |
| `configs/config.py` | modified — `local_train_source`, `local_train_max_examples`; `use_lora`/`use_federated` removed |
| `specs/personalization-federated/tasks.md` | modified |
| `specs/personalization-federated/review.md` | untracked (this file) |

**Scope assessment:** in scope. No new dependencies. `models/lora.py` needed no change, which is the right outcome — the prior findings were all in the consumers, not in the LoRA primitives.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 LoRA implementation | PASS | `LoRALinear` ([lora.py:8](models/lora.py:8)) stores the frozen weight/bias, adds `A = randn(d_in, rank) * 0.02` and `B = zeros(rank, d_out)`, forwards `F.linear(x, w, b) + x @ A @ B`. `inject_lora` wraps Q and V in every block, freezes everything, then unfreezes only the four LoRA tensors per block. `get_lora_state_dict`, `load_lora_state_dict` and `count_lora_parameters` all present. |
| FR-2 Parameter isolation | PASS | `inject_lora` sets `requires_grad = False` across the whole model before selectively unfreezing; `train_user_lora` builds its optimizer from `[p for p in model.parameters() if p.requires_grad]` and asserts `_get_trainable_backbone_count(model) == 0` first ([local_train.py:108](training/local_train.py:108)). `MoELayer` is never a submodule of `TransformerModel`, so MoE keys cannot enter a LoRA state dict structurally; `assert_no_moe_keys` ([federated.py:44](training/federated.py:44)) additionally checks the update against the MoE's real key space, which is what makes the guard reachable rather than vacuous (REV-026c). |
| FR-3 Local personalization | PASS | `run_local_personalization` samples `personalization_eval_users` seeded from `config.seed`, evaluates without LoRA, trains the adapter on the user's TRAIN (or VAL) interactions, re-evaluates, records NDCG with/without, gain, and cold/warm. The frozen checksum is asserted per user. See REV-111 on the "fresh LoRA" wording. |
| FR-4 Report & plots | PARTIAL | `personalization_gain_{model}.json` carries `num_users_evaluated`, with/without stats, gain stats and `per_user_gains` (plus `per_user_scores` for Ablation E). Both plots are written with the y=x diagonal and the x=0/mean lines. One field name differs from the spec — REV-113. |
| FR-5 Federated simulation | PASS | `global_lora_state` persists across rounds and is reloaded into the client model per client ([federated.py:211](training/federated.py:211)); `clients_per_round` sampled from a seeded RNG; `weighted_fedavg` weights by local sample count, or takes a simple mean when `use_weighted_fedavg=False`. Clients with no training examples are skipped and excluded from aggregation, as the Edge Cases require. |
| FR-6 Client state integrity | PASS | `compute_frozen_checksum` hashes every non-LoRA model parameter plus every MoE parameter; asserted before each round ([federated.py:186](training/federated.py:186)), after the last ([:267](training/federated.py:267)), and on the reusable client model at construction ([:178](training/federated.py:178)). `assert_no_moe_keys` runs on every client update. |
| FR-7 Communication cost | PASS | `upload_kb = lora_param_count * 2 / 1024`, download equal, `total_round_kb = (upload + download) * config.clients_per_round` — the configured multiplier FR-7 specifies, with `actual_round_kb` recorded alongside so neither reading is lost (REV-036). No MoE parameters enter the count. Written to `federated_rounds_{model}.json`. |
| FR-8 Convergence plot | PASS | `federated_convergence_{model}.png`: round on x, val NDCG@10 on y, with a dashed horizontal line at the genuine non-federated personalized baseline resolved from `personalization_gain_{model}.json`, falling back to the checkpoint's `best_val_ndcg`, and omitted entirely when neither exists (REV-023's fix holds — it no longer draws round 0 against itself). |
| FR-9 LoRA training constraints | PASS | `lora_lr = 1e-3` and `lora_weight_decay = 0.0` are their own config fields, independent of `learning_rate`; optimizer is AdamW over the LoRA parameters only. |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 Q and V wrapped; output changes after training | PASS | `TestLoRAInjection::test_q_v_wrapped`, `test_forward_changes_after_training`. |
| AC-02 state dict has only LoRA keys | PASS | `TestLoRAStateDict::test_no_moe_keys`, `test_only_lora_keys`; `TestMoEExclusionGuard` (3 tests) covers both a synthetically injected MoE key and a real MoE state key. |
| AC-03 `count_lora_parameters` matches the formula | PASS | `TestParameterCount::test_count_matches_expected`. |
| AC-04 frozen checksum identical before and after personalization | PASS | Asserted per user in the loop ([local_train.py:306](training/local_train.py:306)); `TestPersonalizationEntryPoint::test_run_local_personalization_preserves_frozen_backbone` runs the real entry point end-to-end. |
| AC-05 gain histogram shows a realistic distribution | PARTIAL | The histogram is produced and the gain is now honestly computed (REV-003 fixed), so negative gains are possible where before memorisation guaranteed positives. But no real run exists — `outputs/logs/` has no `personalization_gain_*.json` — so the distribution itself is unobserved. Blocked behind the training gate, not behind this spec. |
| AC-06 federated NDCG improves over rounds | NOT VERIFIABLE | `TestFederatedConvergence::test_run_federated_two_rounds` proves rounds execute, NDCG is finite and logged, and state persists. Whether it *improves* needs a real multi-round run against a real checkpoint; none exists. REV-112 additionally makes such a run impractical as written. |
| AC-07 communication cost matches the formula | PASS | Verified in `TestFederatedConvergence::test_run_federated_two_rounds`; the implementation is the literal formula with `clients_per_round`. |
| AC-08 global checksum unchanged through all rounds | PASS | `TestFederatedConvergence::test_frozen_backbone_survives_run_federated`. |
| AC-09 `global_lora_state` persists across rounds | PASS | Covered by the same two-round test plus `TestWeightedFedAvg` for the aggregation step. |

## Findings

```
REV-112
Severity: MEDIUM | Status: OPEN | Requirement: FR-5 | AC: AC-06
File: training/federated.py:135-146 (validation setup), :233-239 (per-round evaluation)
Problem: Two coupled problems in the federated validation path.
  (a) `val_candidates = generate_eval_candidates(val_examples, neg_pool, eval_user_history,
      ...)` rebuilds the entire validation candidate set on every call to `run_federated`,
      even though `data/processed/val_candidates.npy` already holds exactly that matrix,
      fingerprinted, and `training/pretrain.py::_load_or_build_val_candidates` loads it.
      This is the same "never regenerated" constraint the evaluation-diagnostics spec
      states and that REV-041 was raised about. It also passes no `seed=`, so it silently
      uses `generate_eval_candidates`'s default of 42 rather than `config.seed` -- a
      `--seed 123` run would evaluate against seed-42 candidates while every other stage
      used 123.
  (b) `val_sequences = val_dataset.sequences` is the FULL validation split -- 993,372
      examples -- and `evaluate_model` is called on all of it inside the round loop, i.e.
      `federated_rounds = 20` times.
Impact: At real scale the candidate rebuild alone is ~1M sequential `filter_negatives`
  calls and an 800 MB allocation, repeated per invocation; the 20 full-split evaluations
  then dominate everything else the stage does. `--stage federated` is not tractable as
  written, so AC-06 cannot be demonstrated and Ablation H6 (which calls `run_federated`
  through `system_eval="communication"`) inherits the same cost. The seed mismatch is
  small but makes a multi-seed federated run quietly inconsistent with its own manifest.
Correction: Load `val_candidates.npy` via `load_eval_candidates` with the same
  row-count check `_load_or_build_val_candidates` performs, and fall back to
  `generate_eval_candidates(..., seed=config.seed)` only when the file is absent. For the
  per-round cost, evaluate on a fixed, seeded subset of the validation split -- a
  `federated_eval_examples` config field mirroring `ablation_eval_examples`, with the SAME
  prefix used every round so the convergence curve stays comparable across rounds -- and
  keep the full-split pass for the final round only.
```

```
REV-111
Severity: LOW | Status: OPEN | Requirement: FR-3, FR-9 | AC: —
File: training/local_train.py:123-127 (`train_user_lora`); :266 (candidate construction)
Problem: Both `filter_negatives` call sites in this module omit `rng=`, so the function
  falls back to `RandomState(int(target) % ...)` -- seeded from the TARGET ITEM. In
  `train_user_lora` that means the adapter sees the identical 100 negatives for a given
  target on every one of `local_epochs` passes, and every user who has interacted with
  that item trains against the same negative set. In the candidate-construction loop it
  means the per-user evaluation candidates differ from the stored `test_candidates.npy`
  that `--stage evaluate` scores against.
  Verified on a toy pool: two examples sharing a target and differing in user and history
  received byte-identical negative sets, and neither matched the stored sets.
Impact: The training half wastes two thirds of the local epochs -- the sampled-softmax
  contrastive signal is identical on passes 2 and 3 -- which understates what LoRA
  personalization can achieve and therefore understates the headline RQ4 gain. The
  evaluation half puts the personalization numbers on a different candidate protocol from
  the headline test metrics, so `with_lora`/`without_lora` NDCG cannot be quoted alongside
  `test_metrics_{model}.json`. This is the same root cause as evaluation-diagnostics
  REV-093.
Correction: Thread a run-scoped `np.random.RandomState(config.seed)` through
  `run_local_personalization` into both call sites, so negatives vary per draw and the run
  stays reproducible from the seed. For the evaluation half, prefer loading the stored
  `test_candidates.npy` rows for the user's example indices, matching what
  `ablation_runner._run_single_config` already does.
```

```
REV-113
Severity: LOW | Status: OPEN | Requirement: FR-4 | AC: —
File: training/local_train.py:331, 339 (`_build_report`)
Problem: FR-4 names the report's fields literally: "num_users_evaluated,
  num_users_skipped, with_lora stats, without_lora stats, gain stats, per_user_gains
  array". The implementation writes the key as `num_skipped`. (T3.3 in tasks.md also says
  `num_skipped`, so the task and the code agree with each other and both differ from the
  requirement.)
Impact: Cosmetic in isolation, but `personalization_gain_{model}.json` is a declared
  artifact that `main.py::run_lora_rank_sweep_stage` and
  `federated.py::_resolve_personalized_baseline` both read, and anything written against
  the spec's field list would miss it.
Correction: Rename to `num_users_skipped` (one line, plus the two reads in
  `run_lora_rank_sweep_stage` if it consumes the field), or take it to the spec owner as a
  wording change. Do not leave the spec and the artifact disagreeing.
```

```
REV-114
Severity: INFO | Status: OPEN | Requirement: FR-3 | AC: —
File: training/local_train.py:245-249, :274 (`pristine_lora` reuse)
Problem: FR-3 says "inject fresh LoRA" per user and "never reuse another user's LoRA".
  REV-024b replaced the per-user rebuild with one model built once, plus
  `load_lora_state_dict(model, pristine_lora)` to reset between users. Because `lora_B` is
  zero-initialized, a reset adapter is an exact identity, so the "without LoRA"
  measurement is unchanged and no user's trained adapter leaks into another's -- the
  requirement's actual intent is met. What differs from a literal reading is that
  `lora_A` is the SAME random draw for every user rather than being re-randomized.
Impact: None, and arguably better: a shared starting point removes per-user initialization
  variance from the gain distribution, which is what AC-05's histogram is reading. Noted
  so the deviation from "fresh" is on the record and can be stated in the paper.
Correction: None. If the spec owner wants per-user re-randomization, it is a one-line
  change (re-draw `lora_A` in the reset), but the current behaviour is the better
  experiment.
```

## Design Compliance

Matches `design.md` and the steering constraints. Checked specifically:

- **`architecture.md` decision 4, "LoRA-only federated communication"** — only `get_lora_state_dict` output crosses `weighted_fedavg`; router, experts, objective embeddings and GradNorm state are never touched. The frozen checksum covers all of them (it hashes every MoE parameter unconditionally).
- **`conventions.md` global↔local boundary** — enforced structurally (MoE is a sibling module, not a submodule) and defensively (`assert_no_moe_keys` against the real key space).
- **`testing.md` critical runtime assertions** — "Federated: global frozen-state checksum unchanged through every round" is asserted before every round and after the last; "LoRA state dict: no keys starting with `moe.*`" is asserted on every client update.
- **`code-quality.md` device handling** — `get_device(config)` throughout, no `.cuda()`.
- **Non-Goals respected** — no real distributed infrastructure, no differential privacy, no client dropout beyond sampling.

One addition beyond `design.md`: `local_train_source` / `local_train_max_examples`. These are the mechanism for REV-003/REV-004's fix — the split a client trains on is now an explicit, validated config dimension rather than an implicit choice — and `Config.validate` refuses the test split outright with a message saying so. That is a strengthening of the design's leakage boundary, not a departure from it.

## Task Completion

All 21 tasks are `[x]`. Twenty verify; one is partial.

| Task | Verdict |
|---|---|
| T1.1–T1.4 LoRA primitives | Complete, and unchanged from `32b0904` — correctly so. |
| T2.1–T2.4 LoRA tests | Complete. `TestFreezeEnforcement::test_zero_trainable_backbone` is a real assertion, not an inspection. |
| T3.1 `run_local_personalization` | Complete. The "fresh LoRA per user" wording is satisfied in substance (REV-114). |
| T3.2 per-user training function | Complete. Optimizer is LoRA-only, AdamW, `lora_lr`, `weight_decay=0`; the zero-trainable-backbone assertion runs before training; negatives are drawn against `train_user_history`, as the task specifies. Negative-sampling determinism is REV-111. |
| **T3.3 personalization report** | **Partially false-complete** — the field is `num_skipped`, not FR-4's `num_users_skipped` (REV-113). Everything else the task lists is present. |
| T3.4 personalization plots | Complete; `plot_personalization_scatter` was generalised to multiple series for Ablation E without losing the single-series case. |
| T4.1 `weighted_fedavg` | Complete; `TestWeightedFedAvg` checks both weighted and simple-mean paths against manual arithmetic. |
| T4.2 federated loop | Complete as an implementation; the per-round evaluation cost is REV-112. |
| T4.3 frozen-state checksum | Complete. |
| T4.4 communication cost logging | Complete, with both the configured and actual multipliers recorded. |
| T4.5 convergence plot | Complete, with a genuine external baseline. |
| T5.1–T5.4 tests | Complete. `TestLocalTrainingSourceIsNotTheEvaluationSplit` (5 tests) is the regression suite for REV-003/REV-004 and checks index disjointness directly, which is the right property. |

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` (full suite) | 231 passed, 2 failed, 135 s. Neither failure is in this spec — both are `transformer-backbone` REV-062's armed MPS backend gate. |
| `uv run python -m pytest tests/test_lora.py tests/test_federated.py -q` | All pass (included in the run above), including the two end-to-end runs of `run_federated` and `run_local_personalization` against a synthetic processed directory. |
| Scratch: compare `filter_negatives` with and without an explicit `rng` on a toy pool | Without `rng`, two examples sharing a target received identical negatives and neither matched `generate_eval_candidates`'s stored output — REV-111 confirmed. |
| Source trace: `grep -rn "load_eval_candidates\|generate_eval_candidates"` across non-test source | `federated.py:140` regenerates; `pretrain.py:620` and `main.py:194` load — REV-112(a) confirmed. |
| Source read: `federated.py:135-146` and `:233-239` | `val_sequences` is the full split and `evaluate_model` sits inside the round loop — REV-112(b) confirmed. |
| `ls outputs/logs` | No `personalization_gain_*.json` and no `federated_rounds_*.json`; `outputs/checkpoints/` is empty — no real run has happened, so AC-05 and AC-06 have no artifacts. |

## Final Assessment

The isolation guarantees — which are what this spec exists to establish — are solid and well tested, and both prior CRITICALs are properly fixed rather than papered over. The remaining work is in the evaluation plumbing around them.

BUILD should, in order:

1. **REV-112** — load `val_candidates.npy` instead of rebuilding it, pass `seed=config.seed` on the fallback, and bound the per-round validation to a fixed seeded subset. Without this, `--stage federated` cannot be run at real scale and AC-06 stays undemonstrable.
2. **REV-111** — thread a run-scoped RNG into both `filter_negatives` call sites, and prefer the stored candidate rows for the per-user evaluation. Coordinate with evaluation-diagnostics REV-093; it is the same root cause and should be fixed once, consistently.
3. **REV-113** — rename `num_skipped` to `num_users_skipped`, or take the wording to the spec owner.
4. **REV-114** — no action; record the shared-`lora_A` choice in `design.md` or the paper.

AC-05 and AC-06 stay unevidenced until a real backbone exists to personalize and federate — blocked behind `specs/transformer-backbone` REV-062 and tracked in `specs/training-pipeline/review.md` REV-106, not by anything in this spec.
