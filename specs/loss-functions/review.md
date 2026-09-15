# Review — Loss Functions & GradNorm

## Review Status

**CHANGES_REQUIRED**

## Feature

`specs/loss-functions` — `training/losses.py` (sampled softmax, MIM, multi-step, InfoNCE, `apply_masking`, `mask_metadata`, static fallback) and `training/gradnorm.py` (`GradNormController`, anchor resolution).

## Review Scope

- Spec directory: `specs/loss-functions/` (requirements.md, design.md, tasks.md)
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`
- Baseline: working tree vs `HEAD` = `32b0904` on `master`. All feature work is uncommitted.
- Files inspected: `training/losses.py` (273 lines, modified), `training/gradnorm.py` (127 lines, modified), `tests/test_losses.py` (427 lines, modified), and the consuming call sites in `training/pretrain.py`, `training/local_train.py`, `evaluation/gradient_diagnostics.py`
- Re-review of the prior `review.md`. `losses.py` (21:42) and `gradnorm.py` (15:34) were both modified after it.
- Date: 2026-09-15

## Summary

The prior CRITICAL is genuinely fixed, and I verified it rather than reading the comment. `GradNormController.update` now computes the lambda gradient in isolation with `torch.autograd.grad(gradnorm_loss, [self.log_lambdas])` instead of calling `.backward()` on the double-backward graph. I instrumented a real `TransformerModel` step: **zero** model parameters had their `.grad` altered by `update()`, and the active lambdas summed to 3.0000 as AC-05 requires. REV-075 (`set_to_none=True`), REV-069 (the vectorised mask guard — the Python loop and its ≥512 device syncs per step are gone) and REV-077 (one shared normalized item table per step) are all resolved too.

Two things remain. **T4.9 is marked complete but the test it describes does not exist anywhere in the suite** — nothing verifies that a NaN/Inf loss actually trips an assertion, so AC-09's safety net is unproven in the one direction that matters. And `update()`'s `create_graph=True` path needs double-backward through the attention kernel, which raises outright when `config.device` resolves to CPU.

## Git Changes

| File | State |
|---|---|
| `training/losses.py` | modified (+~140 lines vs `32b0904`) |
| `training/gradnorm.py` | modified (+~30 lines) |
| `tests/test_losses.py` | modified (+~230 lines) |
| `configs/config.py` | modified — `gradnorm_anchor` |
| `specs/loss-functions/review.md` | untracked (this file) |

`specs/loss-functions/tasks.md` is unchanged from `32b0904`.

**Scope assessment:** in scope. No new dependencies. `mask_metadata` is new in `losses.py` but belongs there — it is the masking-integration half of FR-2 and is consumed by `pretrain.py` and `gradient_diagnostics.py`.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Sampled softmax | PASS | `SampledSoftmaxLoss.forward` ([losses.py:61](training/losses.py:61)) uses `item_emb_weight` directly (tied), builds `[pos_score, neg_scores]` and cross-entropies against label 0. Signature matches the spec's `(user_emb, pos_ids, neg_ids, item_emb_weight)`. REV-049's constructor-arg trap is gone — `neg_ids.shape[-1]` implies K. |
| FR-2 Masked item modeling | PASS | `apply_masking` ([losses.py:202](training/losses.py:202)) masks only non-padding positions with `mask_id = num_items + 1`, returns `(masked_seq, mask_positions, true_item_ids)`, and keeps ≥1 real item unmasked. `MaskedItemLoss` takes full-vocab logits via tied weights and returns a detached `0.0` when there are no masked positions. `mask_metadata` zeroes the metadata rows at masked positions, closing the REV-059 leak. |
| FR-3 Multi-step | PASS | `MultiStepLoss` ([losses.py:112](training/losses.py:112)) holds k learnable `nn.Linear(D, D, bias=False)` projections and reads `pos = S - 1 - step`. No `seq_lens`-based indexing anywhere in the repository. Invalid (zero) targets are masked out; mean over valid steps. |
| FR-4 InfoNCE | PASS | `InfoNCELoss` ([losses.py:165](training/losses.py:165)) L2-normalizes both views, builds the `(2B, 2B)` similarity matrix with temperature, masks the diagonal to `-1e9`, labels `(i, i+B)` / `(i+B, i)`. `train_one_step` gates the second forward pass on `batch_idx % contrastive_interval == 0`; `Config.validate` rejects `contrastive_interval == 0` with `use_contrastive=True`. |
| FR-5 GradNorm controller | PASS | `log_lambdas` over the fixed `OBJECTIVE_ORDER`, own Adam optimizer, `lambda = exp(log_lambda)`, initial losses stored on first sight, G_t / Lhat / r_t / G* computed per the spec, `sum|G_t - detach(G*_t)|`, renormalization to `num_active`. Anchor is the last Transformer block, config-selectable via `gradnorm_anchor`. Inactive objectives excluded through `has_valid_signal`. Verified empirically — see Testing. |
| FR-6 Loss combination | PASS | `train_one_step` ([pretrain.py:291-320](training/pretrain.py:291)) follows the contract exactly: active_losses → weighted sum → `backward(retain_graph=gradnorm is not None)` → `gradnorm.update` → finite-gradient assertion → `clip_grad_norm_` → `optimizer.step` → `scheduler.step`. `static_weighted_sum` matches the spec's formula. Finite assertions on losses, total, lambdas and gradients are all present. |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 sampled softmax decreases on toy data | PASS | `TestSampledSoftmax::test_loss_decreases_with_optimizer` and `test_high_positive_score_low_loss`. |
| AC-02 MIM targets only masked positions | PASS | `TestMaskedItemLoss::test_only_masked_contribute` and `test_zero_masked_returns_zero`. `TestMaskedMetadataLeak` (3 tests) additionally proves the masked positions no longer carry the true item's genome vector. |
| AC-03 multi-step reads `S-1-step` | PASS | `TestMultiStepLoss::test_correct_position_usage` injects distinguishable values at S-1/S-2/S-3 and checks each step reads its own. See REV-105 for a horizon observation that is not a defect. |
| AC-04 InfoNCE similarity matrix | PASS | `TestInfoNCE::test_similarity_matrix_shape`, `test_identical_pairs_low_loss`, `test_random_pairs_higher_loss`. |
| AC-05 lambdas sum to `num_active` | PASS | `TestGradNormLambdaConstraint::test_lambda_sum_after_update`, parametrised over 2/3/4 active objectives. Independently reproduced: 3 active objectives on a real t-small model gave `sum = 2.99999976` (bf16 tolerance). |
| AC-06 inactive objectives excluded | PASS | `TestGradNormActiveOnly::test_inactive_excluded`. `has_valid_signal` ([losses.py:190](training/losses.py:190)) is the exclusion test and `train_one_step` records skips in `loss_values["_skipped"]`. |
| AC-07 anchor resolves in all variants | PASS | `TestGradNormAnchorResolution::test_anchor_exists`, parametrised over `none`/`mmoe`/`arv_mmoe`. The anchor is `model.blocks[-1]`, which exists in every variant because `MoELayer` is a separate module. |
| AC-08 static fallback matches manual sum | PASS | `TestStaticWeightedSum::test_matches_manual` and `test_partial_active`. |
| AC-09 all losses finite, asserted | PARTIAL | The assertions exist and fire on every step ([pretrain.py:276](training/pretrain.py:276), :296, :313-315, and `assert_finite_step`). What is missing is any test that a NaN/Inf input actually trips them — see REV-091. |

## Findings

```
REV-091
Severity: MEDIUM | Status: OPEN | Requirement: FR-6 | AC: AC-09
File: specs/loss-functions/tasks.md:60 (T4.9); tests/ (no such test exists)
Problem: T4.9 is marked [x] -- "Finite assertion tests: Verify that NaN/Inf losses trigger
  assertion failures. (AC-09)". No such test exists. Grepping the whole suite for
  `nan`/`inf`/`isfinite`/`AssertionError` returns 25 hits; every one of them asserts that a
  value IS finite on well-behaved input, which is the opposite direction. The only
  `pytest.raises(AssertionError, ...)` uses in the suite are the two MoE-key guards in
  `tests/test_federated.py`, which belong to a different spec.
Impact: AC-09's whole purpose is the safety net that stops a NaN from silently corrupting
  weights through a long run. `code-quality.md` lists these assertions under "Numerical
  Safety", and training-pipeline AC-08 depends on them too. Nothing verifies they would
  actually fire -- an assertion accidentally weakened (e.g. `torch.isfinite(p.grad).any()`
  instead of `.all()`, or a guard moved above the point where the bad value appears) would
  pass CI unnoticed.
Correction: Add negative tests. Three cheap ones cover it: (a) call
  `training.pretrain.assert_finite_step({"total": float("nan")}, None)` and assert it
  raises; (b) build a `GradNormController`, set `log_lambdas.data` to inf and assert
  `assert_finite_step` raises on the lambda branch; (c) monkeypatch one loss function to
  return `torch.tensor(float("nan"))` and assert `train_one_step` raises at its
  per-objective `torch.isfinite(loss)` check rather than completing the optimizer step.
  Then re-mark T4.9. Until such tests exist T4.9 should be unchecked.
```

```
REV-092
Severity: MEDIUM | Status: OPEN | Requirement: FR-5 | AC: AC-05, AC-07
File: training/gradnorm.py:72-74 (`torch.autograd.grad(..., create_graph=True)`)
Problem: Computing G_t requires a differentiable gradient of each weighted loss w.r.t. the
  anchor parameters, i.e. double-backward through everything between the loss and
  `model.blocks[-1]` -- which includes that block's attention kernel. Not every kernel
  supports it. Reproduced on this host with `config.device = "cpu"`:
    RuntimeError: derivative for
    aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented
  raised from `gradnorm.py:102`. The same call on MPS (where SDPA dispatches to a
  different kernel) succeeds.
Impact: `config.device` defaults to `"auto"`, which resolves CUDA > MPS > CPU. On any
  CPU-only host -- a CI runner, a collaborator's machine, a container without an
  accelerator -- every run with `use_gradnorm=True` (the default, and the arm the paper's
  central claim rests on) dies on the first optimizer step with an error that names an
  ATen kernel rather than the actual problem. The same class of risk applies to the H100
  path: `flash_attn_varlen_func` is not guaranteed to support `create_graph=True`, and
  that combination has never been exercised (no CUDA host here), so the production
  GradNorm run is unverified.
Correction: Two parts. (a) Make the failure legible: wrap the `create_graph=True` grad
  call and re-raise as a clear error naming the attention backend and suggesting
  `use_gradnorm=False` or a different backend -- an operator should not have to decode an
  ATen symbol. (b) Add a startup check alongside `AttentionBackend.self_test()` that runs
  one double-backward through a single block when `use_gradnorm=True`, so the run fails at
  second zero rather than after dataset loading. The H100 half must be verified on the
  H100 before the final run; flag it for the BUILD_ORDER Phase 13 gate.
```

```
REV-105
Severity: INFO | Status: OPEN | Requirement: FR-3 | AC: AC-03
File: training/losses.py:137-145 (`MultiStepLoss.forward`);
       data/sequence_generator.py:141-152 (`generate_sliding_window_examples`)
Problem: The two halves of the multi-step objective pair a growing horizon. For an example
  whose context is `items[:i]`, the generator writes
  `targets_k = [items[i+1], items[i+2], items[i+3]]`, while the loss reads step `s` from
  position `S-1-s`, which holds `items[i-1-s]`. So step 0 predicts 2 interactions ahead,
  step 1 predicts 4 ahead, step 2 predicts 6 ahead, rather than a uniform 1/2/3.
Impact: None on correctness -- every target is strictly in the future of the position it
  is predicted from, so there is no leakage, and the position rule is exactly what
  `conventions.md` and FR-3 prescribe. But "Predict next k items" in FR-3's opening line
  reads as horizons 1..k, and the paper would describe the objective wrongly if written
  from that sentence.
Correction: No code change without a spec decision. Either reword FR-3 to state the
  horizon explicitly, or (if uniform 1..k horizons were intended) have every step read
  position `S-1` and let the per-step projections carry the horizon. Raise with the spec
  owner; BUILD should not pick one unilaterally.
```

```
REV-056
Severity: INFO | Status: RESOLVED | Requirement: FR-5 | AC: —
File: training/gradnorm.py:68-90
Note: The G_t / Lhat / r_t / G* block now matches FR-5's formulas line for line, including
  the `L0 < 1e-12` guard the Edge Cases ask for. Carried forward from the prior review and
  closed.
```

## Design Compliance

Matches `design.md` and the steering constraints:

- **All four losses live in `training/losses.py`, GradNorm in `training/gradnorm.py`** — as `conventions.md`'s file layout requires.
- **Tied weights** — no separate output projection exists; every objective scores against `model.item_emb_weight`.
- **`mask_token = num_items + 1`** — `train_one_step`, `run_gradient_diagnostics` and `local_train` all compute it that way; `tests/test_augmentation.py::test_mask_token_not_num_items` guards the augmentation side.
- **GradNorm anchor on the shared Transformer block** — matches `architecture.md`'s decision 5.
- **Separate GradNorm optimizer** — `torch.optim.Adam([log_lambdas])`, never the model optimizer, and now genuinely isolated from the model's gradients.
- **BF16** — `train_one_step` asserts each loss's dtype is bf16, with the documented fp32 exception for InfoNCE (whose similarity matrix is numerically fragile in bf16). That exception is recorded in `design.md`, so it is not a silent dtype substitution.

Two additions beyond `design.md`, both benign: `normalize_item_table` / `_shared_normalized_table` (the REV-077 sharing helper, deliberately not cached across steps for the documented autograd reason) and `mask_metadata` (the REV-059 fix). Both are consumed uniformly.

## Task Completion

18 of 19 tasks verify; T4.9 does not.

| Task | Verdict |
|---|---|
| T1.1–T1.4 four losses | Complete, each with the signature the task specifies. |
| T2.1 `apply_masking` | Complete. The "keep at least 1 unmasked real item" guard is now vectorised and correct — since `mask_positions ⊆ real_items`, `n_mask >= n_real` implies every real item is masked, so the randomly chosen keep-position is always one that was masked. `TestApplyMaskingVectorisedGuard::test_guard_holds_at_mask_prob_one` covers the extreme. |
| T3.1–T3.2 GradNorm controller and update | Complete and independently verified (see Testing). |
| T3.3 anchor resolution | Complete. `gradnorm_anchor` is a real config flag with two legal values and `ANCHOR_CHOICES` validation. |
| T3.4 state_dict / load_state_dict | Complete; round-tripped through `save_checkpoint`/`load_checkpoint` and covered by `TestCheckpointRoundtrip`. |
| T3.5 static fallback | Complete. |
| T4.1–T4.8 tests | All eight exist and test what they name. `TestGradNormGradientIsolation` (2 tests) is a genuine regression test for REV-057, checking both that anchor gradients are untouched and that lambdas still move. |
| **T4.9 finite assertion tests** | **False-complete.** No test in the suite verifies that a NaN/Inf loss triggers an assertion failure (REV-091). |

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` (full suite) | 231 passed, 2 failed, 135 s. Neither failure is in this spec — both are `transformer-backbone` REV-062's armed MPS backend gate. |
| `uv run python -m pytest tests/test_losses.py -q` | All pass (included in the run above). |
| Scratch: instrument `GradNormController.update` against a real `TransformerModel` on MPS — snapshot every `model.named_parameters()` `.grad` before and after | **`model grads mutated by gradnorm.update: [] count=0`** — REV-057 confirmed fixed, `.backward()` no longer leaks second-order gradients into the anchor. |
| Same run, lambda renormalization with 3 active objectives | `lambdas = [1.00664, 0.98671, 1.00664, 1.0]`, `sum(active) = 2.99999976` vs expected 3 — AC-05 confirmed. |
| Same script with `config.device = "cpu"` | `RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented` at `gradnorm.py:102` — REV-092 confirmed. |
| `grep -rn "nan\|NaN\|inf\|isfinite\|AssertionError" tests/*.py` | 25 hits, all asserting finiteness on good input; zero negative tests — REV-091 confirmed. |
| Scratch: `generate_sequences` on a synthetic user, cross-checked against `MultiStepLoss`'s `S-1-step` indexing | Horizons are 2/4/6, not 1/2/3 — REV-105 confirmed as an observation. |

## Final Assessment

The losses and GradNorm are correct, and the prior CRITICAL is fixed in substance, not just in comments. Two actions for BUILD, in order:

1. **REV-091** — write the three negative assertion tests and only then re-mark T4.9. Until they exist, T4.9 should be unchecked; it is currently the only false-complete claim in this spec.
2. **REV-092** — wrap the `create_graph=True` grad call so a kernel without double-backward support fails with a legible message, and add the one-block double-backward probe to startup when `use_gradnorm=True`. Flag the H100 half for the Phase 13 gate — GradNorm over FlashAttention-4 has never been exercised.

**REV-105** goes to the spec owner (multi-step horizon wording), not to BUILD. **REV-056** is closed.

AC-09 stays PARTIAL until REV-091 is done; everything else passes with evidence.
