# Review — Loss Functions & GradNorm

## Review Status

**PASS**

(Was `CHANGES_REQUIRED` on REV-105, the owner's 2026-09-17 multi-step horizon amendment.
That work has landed and is verified below. No finding in this spec is open above INFO,
and nothing is left for BUILD.)

## Feature

`specs/loss-functions` — `training/losses.py` (`SampledSoftmaxLoss`, `MaskedItemLoss`,
`MultiStepLoss`, `InfoNCELoss`, `apply_masking`, `mask_metadata`, `static_weighted_sum`,
`has_valid_signal`) and `training/gradnorm.py` (`GradNormController`,
`get_anchor_layer_params`), with `assert_finite_step` in `training/pretrain.py`.

## Review Scope

- Spec directory: `specs/loss-functions/` (requirements.md, design.md, tasks.md). All
  three unmodified in the working tree; FR-3/AC-03 carry the owner's 2026-09-17
  amendment and FR-5/AC-10/AC-11 the REV-120 amendment.
- Steering: `architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`.
- Baseline: `master` at `0bc67ee`, `origin/master` at the same commit. No file owned by
  this spec is modified in the working tree.
- Files inspected: `training/losses.py` (279 lines, read in full),
  `training/gradnorm.py` (142 lines, read in full), `tests/test_losses.py` (707 lines),
  plus the `assert_finite_step` and loss-combination call sites in `training/pretrain.py`
  and the relevant `configs/config.py` validators.
- Re-review. Prior findings carried in: REV-105 (MEDIUM, OPEN), REV-124 (INFO, OPEN),
  REV-120, REV-091, REV-121, REV-056 (RESOLVED), REV-092 (NOT_APPLICABLE).
- Date: 2026-09-18

## Summary

Both open findings are closed, and the substantive one is closed correctly.

REV-105 asked for the amended FR-3 rule: every multi-step step reads position `S-1`.
`MultiStepLoss.forward` now hoists `pos = S - 1` out of the loop and nothing else in the
loss changed — the per-step projections, the valid-target masking and the
mean-over-valid-steps reduction are untouched, which is exactly what the amendment scoped.

**The replacement test was verified to discriminate by me, not taken on BUILD's word.**
The superseded rule (`all_hidden_states[:, S - 1 - step, :]`) was reinstated in the
working copy and the suite re-run: `test_every_step_reads_position_S_minus_1` FAILS on it
(3.2969 against the expected value) and passes on the fix, with the other 38 tests
unaffected. `training/losses.py` was restored immediately; `git diff` on it is empty. This
matters because the *old* test — `test_correct_position_usage`, which asserted only
`torch.isfinite` — held under both rules, which is how the mismatch survived three
reviews. The replacement does not have that weakness.

REV-124 was a stale docstring, now reworded on first-order grounds with no assertion
changed.

REV-120's amended FR-5 holds on re-inspection: `create_graph=True` appears nowhere in the
controller, the probe helpers are gone from the entire repository, and a full `update()`
completes on CPU.

## Git Changes

No file owned by this spec appears in `git status`. The four modified files belong to
`evaluation-diagnostics`. The T3.7 and T3.8 work was committed before this baseline
(`daffb41`, `aa04dee`).

**Scope assessment:** proportionate and complete. T3.7 changed one expression in
`MultiStepLoss` and two tests, which is precisely the amendment's footprint — no
opportunistic edits rode along.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Sampled softmax | PASS | Tied weights consumed as `item_emb_weight[pos_ids]` / `[neg_ids]`; logits are `cat([pos_score, neg_scores])` with label 0. Both sides of every dot product normalized when `normalize_embeddings` is on. `test_loss_decreases_with_optimizer` and `test_high_positive_score_low_loss` |
| FR-2 Masked item modeling | PASS | Cross-entropy over full-vocabulary logits at masked positions only; `hidden_states[mask_positions]` with an explicit zero-length short-circuit returning a detached `0.0`. `apply_masking` uses `mask_id` supplied as `num_items + 1` and keeps ≥1 real item unmasked per row. `mask_metadata` zeroes the metadata rows at masked positions, closing the REV-059 leak |
| FR-3 Multi-step (amended) | PASS | `pos = S - 1` hoisted out of the step loop, so all k projections read one fixed context vector and horizons are uniform at 2..k+1. Per-step `nn.Linear(D, D, bias=False)` in an `nn.ModuleList`; only non-zero targets contribute; mean over valid steps; detached `0.0` when no step is valid |
| FR-4 Contrastive InfoNCE | PASS | Both views L2-normalized, `(2B, 2B)` similarity with temperature, diagonal masked to `-1e9`, labels `[B..2B) ++ [0..B)` giving positive pairs `(i, i+B)` and `(i+B, i)`. Fired on `batch_idx % contrastive_interval == 0` (`pretrain.py:239`), so the second forward pass is genuinely skipped otherwise. `contrastive_interval=0` with `use_contrastive=True` raises — measured |
| FR-5 GradNorm controller (amended) | PASS | `log_lambdas` over the fixed `OBJECTIVE_ORDER`, separate Adam, `lambda_t = exp(log_lambda_t)`, initial losses captured on first valid step. `G_t` is built in the factored form `lambdas[idx] * grad_norm` with the norm taken at `create_graph=False` and `.detach()`ed before scaling. λ gradient via `torch.autograd.grad(gradnorm_loss, [log_lambdas])`, `.detach()`ed before assignment. Post-step renormalization over active indices only. Anchor is the last Transformer block |
| FR-6 Loss combination | PASS | `L_total.backward(retain_graph=(gradnorm is not None))` at `pretrain.py:302` — retained exactly when GradNorm needs per-objective gradients. `static_weighted_sum` applies the four configured weights with `next_item` unweighted, and skips absent objectives. `assert_finite_step` checks every reported loss value and the lambdas on every step |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 sampled softmax decreases on toy data | PASS | `TestSampledSoftmax::test_loss_decreases_with_optimizer` runs real optimizer steps and compares; `test_high_positive_score_low_loss` pins the direction |
| AC-02 MIM targets only masked positions | PASS | `test_only_masked_contribute` and `test_zero_masked_returns_zero`. The selection is `hidden_states[mask_positions]` / `true_item_ids[mask_positions]`, so non-masked positions cannot enter the cross-entropy at all |
| AC-03 every step reads `S-1`, horizons uniform 2..k+1 | PASS | `test_every_step_reads_position_S_minus_1` scrambles every position `< S-1` and asserts the loss is **exactly** unchanged, with a non-vacuity assertion that perturbing `S-1` does move it. **Independently verified to discriminate:** it fails on the reinstated `S-1-step` rule and passes on the fix. `test_all_k_steps_contribute_distinct_projections` separately guards against the k steps collapsing into one prediction, which reading a single position could otherwise mask |
| AC-04 InfoNCE similarity matrix with masked self-similarity | PASS | `test_similarity_matrix_shape` checks `(2B, 2B)` and the masked diagonal; `test_identical_pairs_low_loss` / `test_random_pairs_higher_loss` check the direction |
| AC-05 lambdas sum to num_active after every update | PASS | Measured directly this pass on CPU across 4, 2 and 1 active objectives: sums `3.9999997616`, `2.0000000000`, `1.0000000000` — within 1e-4 in every case. `TestGradNormLambdaConstraint::test_lambda_sum_after_update` is parametrized over the same three |
| AC-06 inactive objectives excluded from gradient computation | PASS | `update()` iterates `active_losses.keys()` only, so no `autograd.grad` call is made for an inactive objective and no zero placeholder reaches autograd. `has_valid_signal` is the gate callers use to drop degenerate objectives. `TestGradNormActiveOnly::test_inactive_excluded` |
| AC-07 anchor resolves in all variants incl. `moe_variant="none"` | PASS | `get_anchor_layer_params` reads `model.blocks[-1]`, which exists in every variant because the MoE layer is a separate module. `TestGradNormAnchorResolution::test_anchor_exists` is parametrized over `none`/`mmoe`/`arv_mmoe` and checks parameter identity by `data_ptr`, not just non-emptiness |
| AC-08 static fallback matches manual weighted sum | PASS | `TestStaticWeightedSum::test_matches_manual` and `test_partial_active` |
| AC-09 all losses finite, asserted | PASS | `assert_finite_step` called at `pretrain.py:800` on every step. `TestFiniteAssertionsActuallyFire` has three tests, including `test_nan_objective_loss_aborts_the_step_before_the_optimizer` — which checks the assertion fires *before* the optimizer steps, not merely that it exists |
| AC-10 no second-order differentiation; `update()` runs on CPU+SDPA | PASS | `grep create_graph` over `training/gradnorm.py` returns the literal `create_graph=False` and one comment naming it — no `True`. A repository-wide grep for `probe_double_backward`, `_raise_double_backward` and `DoubleBackwardUnsupportedError` returns **nothing**, so T3.6's deletions are complete rather than partial. A full `update()` was run here on CPU across three activity levels and completed in every case |
| AC-11 λ gradient matches autograd; no graph on `.grad` | PASS | `TestLambdaGradientEquivalence::test_matches_closed_form` is parametrized over all-active, partial and the single-objective `G_t == G*_t` boundary. Measured directly this pass after each of three `update()` calls: `log_lambdas.grad.requires_grad is False` and `grad_fn is None` |

## Findings

```
REV-105
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-3 (amended) | AC: AC-03 (amended)
File: training/losses.py:140-150 (MultiStepLoss.forward)
Resolution: `pos = S - 1` is hoisted out of the step loop; the per-step projections, the
  valid-target masking and the mean-over-valid-steps reduction are unchanged, which is
  the exact scope the amendment specified. Horizons are now uniform at 2, 3, 4 rather
  than 2, 4, 6.
Verification (independent): `pos = S - 1 - step` was reinstated in the working copy and
  `tests/test_losses.py` re-run -- `test_every_step_reads_position_S_minus_1` fails on the
  superseded rule (3.2969) and passes on the fix, 38 other tests unaffected. Source
  restored; `git diff training/losses.py` is empty. The test therefore has the property
  its predecessor lacked: it distinguishes the two rules rather than merely asserting
  finiteness.
Residual, correctly flagged by BUILD rather than by me: `outputs/checkpoints/t_small/best.pt`
  was trained under the superseded objective (`train_loss_multistep=9.912500` in
  `outputs/logs/pretrain_t_small.log`), so the artifacts predate this change. BUILD routed
  that to training-pipeline T6.3 rather than silently leaving stale numbers in place.
  Tracked there; see that spec's review.
```

```
REV-124
Severity: INFO | Status: RESOLVED | Requirement: FR-5 | AC: —
File: tests/test_losses.py:342-353 (TestGradNormGradientIsolation docstring)
Resolution: reworded on first-order grounds. It now states that `update()` runs between
  `L_total.backward()` and `optimizer.step()`, so a plain `.backward()` on `gradnorm_loss`
  would accumulate into the model `.grad` buffers the main AdamW is about to step on. No
  assertion changed, and the two isolation tests still pass. This was the last place in
  the repository asserting the second-order form still existed.
```

```
REV-120
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-5 (amended) | AC: AC-10, AC-11
File: training/gradnorm.py:66-125
Re-verified rather than carried on trust. The factored form is present as written
  (`grad(..., create_graph=False)` -> `.norm(p=2).detach()` -> scaled by `self.lambdas[idx]`),
  `gradnorm_loss` runs its graph only through `log_lambdas`, the λ gradient is taken with
  `autograd.grad` and `.detach()`ed before assignment, `zero_grad(set_to_none=True)`
  precedes it, and the renormalization is untouched. All five sub-corrections confirmed
  still in place, including the deletion of the double-backward probe helpers -- a
  repository-wide grep finds no trace of them.
```

```
REV-091
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-6 | AC: AC-09
File: training/pretrain.py:351-358, :800
Re-verified. `assert_finite_step` checks every non-underscore loss value and, when a
  controller is present, the lambdas. It is called on every step, and
  `test_nan_objective_loss_aborts_the_step_before_the_optimizer` pins the ordering rather
  than just the existence of the check.
```

```
REV-092
Severity: MEDIUM | Status: NOT_APPLICABLE | Requirement: FR-5
Superseded by REV-120's amendment of FR-5. Retained so the ID is not reused.
```

```
REV-121
Severity: INFO | Status: RESOLVED | Requirement: FR-5 | AC: —
Re-confirmed. (Note: an unrelated INFO in specs/data-pipeline also carries the number
REV-121 in this round's numbering; the two are independent and neither is actionable.)
```

```
REV-056
Severity: INFO | Status: RESOLVED | Requirement: FR-5 | AC: —
Re-confirmed.
```

## Design Compliance

Compliant, including the two amended decisions.

- **D7 (factored `G_t`, REV-120)** — implemented exactly, down to the trap the decision
  names: the `.detach()` before `log_lambdas.grad = ...`. A tensor derived from
  `self.lambdas[idx] * ...` carries a graph through `log_lambdas`, and PyTorch accepts
  assigning it to `.grad` silently; the code detaches, and `test_grad_carries_no_autograd_graph`
  asserts the result.
- **The amended FR-3 horizon rule** — implemented as a one-line change, as the owner's
  note scoped it.
- **Objective order** — `OBJECTIVE_ORDER` is imported from `configs/config.py` and used
  for both `_obj_to_idx` and the `log_lambdas` layout, so the controller cannot drift from
  the global order that `steering/conventions.md` fixes.
- **Anchor** — the last Transformer block, which exists before the optional MoE in every
  variant, satisfying the constraint that the anchor exist in all variants.
- **Normalization** — applied at the scoring sites rather than inside the model, per the
  transformer spec's D4, and `normalize_item_table` is computed once per step and shared
  by MIM and multi-step rather than recomputed per objective. The docstring records why it
  is deliberately *not* cached across steps (the result is part of the step's autograd
  graph, and gradient diagnostics and LoRA personalization call these losses with no
  optimizer step between).

All loss computation stays in the model's dtype; the float casts in the controller are on
scalars for numerical stability of the λ arithmetic, not a dtype substitution in the model
path.

## Task Completion

All tasks T1.1–T4.11 are marked `[x]`. Verified this pass:

| Task | Verdict |
|---|---|
| T3.6 factored `G_t` | Complete. Every element re-checked, including the deletions |
| T3.7 read `S-1` at every step | **Complete, and the non-vacuity claim is true.** Independently reproduced: the new test fails on the reinstated old rule. BUILD also recorded the downstream consequence (the t-small checkpoint predates the change) rather than leaving it implicit |
| T3.8 docstring rewording | Complete. Reworded, no assertion touched |
| T4.3 multi-step test | Complete. The task text itself records that the original per-step-position injection no longer distinguishes anything under the amended rule, and the replacement asserts the property only the amended rule has |
| T4.10 / T4.11 λ-gradient and first-order tests | Complete. Both exist, both pass, and both were re-derived here independently of the test file |
| T1.1–T1.4, T2.1, T3.1–T3.5, T4.1–T4.9 | Complete. Verified against the code and the 39 passing tests |

No false-complete tasks. Notably, the two tasks whose text was superseded by amendments
(T1.3's position rule, T4.3's injection test) carry explicit "superseded by" pointers
rather than being silently rewritten, so the history of the decision stays readable.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **275 passed**, 0 failed, 55.74 s |
| `uv run python -m pytest tests/test_losses.py -q` | **39 passed**, 1.64 s |
| `uv run python -m pytest tests/test_losses.py -q` **with `pos = S - 1 - step` reinstated** | **1 failed, 38 passed** — `test_every_step_reads_position_S_minus_1`. Source restored; `git diff training/losses.py` empty |
| `grep -n create_graph training/gradnorm.py` | one `create_graph=False` and one comment; no `True` |
| `grep -rn 'probe_double_backward\|DoubleBackwardUnsupported\|_raise_double_backward'` over the repo | no matches — T3.6's deletions are complete |
| `GradNormController.update()` on CPU, 4 / 2 / 1 active objectives (direct) | active-λ sums `3.9999997616` / `2.0000000000` / `1.0000000000`; `gradnorm_loss` finite in all three; `log_lambdas.grad.requires_grad False`, `grad_fn None` after each |
| `Config(use_contrastive=True, contrastive_interval=0).validate()` (direct) | `ValueError: contrastive_interval=0 is invalid; use use_contrastive=False to disable` |
| `grep -n retain_graph training/pretrain.py` | `L_total.backward(retain_graph=(gradnorm is not None))` at line 302 |

## Final Assessment

**Nothing for BUILD in this spec.** It is accepted.

Both findings that reopened it are closed, and the one that mattered was closed with a
test that actually separates the two candidate behaviours — which is the specific failure
mode that let the original mismatch survive three reviews. That is the right lesson to
have drawn from REV-105, and it was drawn.

One consequence to keep visible, already correctly routed by BUILD rather than hidden: the
existing `t_small` checkpoint and its logged multi-step loss were produced under the
superseded objective. They are not evidence about the current objective, and any figure or
table derived from them needs regenerating. That work is tracked as training-pipeline T6.3
and is assessed in that spec's review, not here.
