# Review — Loss Functions & GradNorm

## Review Status

**CHANGES_REQUIRED**

(REV-120 is fully implemented and verified — that finding is closed. What reopens the spec
is REV-105: the owner resolved the multi-step horizon question on 2026-09-17 by amending
FR-3 and AC-03, and the implementation does not yet match the amended requirement. This is
new BUILD work created by a deliberate spec decision, not a regression.)

## Feature

`specs/loss-functions` — `training/losses.py` (`SampledSoftmaxLoss`, `MaskedItemLoss`,
`MultiStepLoss`, `ContrastiveInfoNCELoss`, `static_weighted_sum`, `assert_finite_step`)
and `training/gradnorm.py` (`GradNormController`, `get_anchor_layer_params`).

## Review Scope

- Spec directory: `specs/loss-functions/` — **`requirements.md` and `design.md` were both
  amended by the owner since the last review** (the REV-120 decision: FR-5 factored-`G_t`
  wording, new AC-10 and AC-11, new design D7). `tasks.md` gained T3.6, T4.10, T4.11.
- Steering: `steering/code-quality.md` (numerical safety, `set_to_none`),
  `architecture.md` (GradNorm anchor), `conventions.md` (`OBJECTIVE_ORDER`),
  `testing.md` Phase 10/12.
- Baseline: working tree vs `HEAD` = `d438950` on `master`; `origin/master` at the same
  commit. 38 tracked files modified, no untracked files.
- `training/losses.py` is **not** modified in the working tree (verified against `HEAD`).
  `training/gradnorm.py` is modified `-116/+?` and `tests/test_losses.py` `+149`.
- Verification was executed, not inferred: the λ gradient was recomputed from scratch in
  an independent process and compared against the shipped value on three input regimes.
- Re-review. Prior findings carried in: REV-091 (RESOLVED), REV-092 (RESOLVED/SUPERSEDED),
  REV-105 (INFO, OPEN), REV-120 (MEDIUM, OPEN), REV-121 (INFO, OPEN), REV-056 (RESOLVED).
- Date: 2026-09-17

## Summary

REV-120 is closed, and closed completely — all five sub-corrections (a)–(e) landed.

`G_t` is now built factored: `torch.autograd.grad(L_t, anchor_params,
create_graph=False)`, `.norm(p=2).detach()`, then scaled by `self.lambdas[idx]`. The λ
gradient is taken with `torch.autograd.grad(gradnorm_loss, [self.log_lambdas])` and — the
trap D7 explicitly names — `.detach()`ed before assignment to `log_lambdas.grad`.
`probe_double_backward`, `_raise_double_backward`, `DoubleBackwardUnsupportedError` and
the `run_pretrain` probe call are all gone; `grep -rn "probe_double_backward\|
DoubleBackwardUnsupported"` over the tree returns nothing.

I did not take the equivalence on trust. Recomputing `sign(G_t − G*_t) · G_t` from
scratch in a separate process, against the controller's pre-update λ, the shipped
`log_lambdas.grad` matches **bit-identically (max difference 0.0)** in all three regimes
the owner's AC-11 names — all four objectives active, three active, and the single-active
case that sits exactly on the `G_t == G*_t` subgradient boundary. In every case
`grad.requires_grad` is `False` and `grad_fn` is `None`, inactive objectives receive
exactly zero, and `sum(active λ) == num_active` after the step (4.0 / 3.0 / 1.0 against
targets 4 / 3 / 1). AC-10's static half holds too: the only two `create_graph` strings in
the module are the explicit `create_graph=False` and a comment explaining it.

The two AC tests BUILD added are non-vacuous. `TestLambdaGradientEquivalence._closed_form`
recomputes the gradient independently rather than restating the controller, and
`test_full_update_runs_on_cpu` exercises the exact configuration that previously raised
`aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented`.

Everything else in the spec is unchanged and still correct. The three remaining findings
are INFO and all belong to the spec owner.

## Git Changes

| File | Change |
|---|---|
| `training/gradnorm.py` | `-116` net: factored `G_t`, detached λ gradient, deletion of the double-backward probe, the raiser and its exception type |
| `training/pretrain.py` | probe call site removed (part of a larger diff owned by `training-pipeline`) |
| `tests/test_losses.py` | `+149`: `TestLambdaGradientEquivalence` (3 parametrised cases), `test_create_graph_true_absent_from_the_controller`, `test_full_update_runs_on_cpu` |
| `specs/loss-functions/{requirements,design,tasks}.md` | owner's FR-5 amendment, AC-10/AC-11, design D7; BUILD's T3.6/T4.10/T4.11 |

**Scope assessment:** proportionate and complete. The diff is confined to the GradNorm
controller and its tests; `training/losses.py` was correctly not touched, since REV-120
was never a defect in the four objectives.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Sampled softmax | PASS | Positive logit concatenated ahead of `(B, N)` negative scores, label 0, `cross_entropy`; both sides L2-normalized when enabled |
| FR-2 Masked item modelling | PASS | Full-vocabulary softmax over masked positions; empty-mask batches return a detached zero and are excluded from the GradNorm active set by the caller (REV-048) |
| FR-3 Multi-step prediction (**amended 2026-09-17**) | **FAIL** | FR-3 now requires every step to read position `S-1`, giving uniform horizons 2..k+1. `MultiStepLoss.forward` still reads `S - 1 - step` (`training/losses.py:138`), giving horizons 2, 4, 6. Per-step bias-free projections and zero-target exclusion are unchanged and correct (REV-105) |
| FR-4 Contrastive InfoNCE | PASS | Symmetric similarity, self-similarity masked, temperature applied; gated by `contrastive_interval` |
| FR-5 GradNorm (amended) | PASS | Factored `G_t` per D7; no `create_graph=True`; λ gradient written to `log_lambdas.grad` only, detached; post-step renormalization unchanged and under `no_grad()`; anchor is the last shared Transformer block; inactive objectives excluded entirely |
| FR-6 Loss combination | PASS | `weighted_sum` / `static_weighted_sum`; `retain_graph=config.use_gradnorm`; `assert_finite_step` covers losses, total and lambdas |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 sampled softmax behaves on toy data | PASS | `tests/test_losses.py` — low loss for a high-scoring positive, high when reversed, decreasing under optimization |
| AC-02 MIM recovers masked items | PASS | Toy test |
| AC-03 multi-step reads `S-1` for **every** step | **FAIL** | The amended AC-03 requires all k projections to receive the same input vector. The implementation reads a different position per step, and the existing position test asserts the *old* rule, so it will need inverting (REV-105) |
| AC-04 InfoNCE symmetric, self-similarity masked | PASS | Structural test on the similarity matrix |
| AC-05 λ sum constraint after update | PASS | Re-measured directly: `sum(active λ)` = 4.0 / 3.0 / 1.0 against `num_active` 4 / 3 / 1 |
| AC-06 inactive objectives excluded from GradNorm | PASS | Re-measured: inactive entries of `log_lambdas.grad` are exactly zero; no placeholder losses reach `autograd.grad` |
| AC-07 anchor resolves in all variants incl. `moe_variant="none"` | PASS | `get_anchor_layer_params` resolves `model.blocks[-1]`, which exists in every variant because MoE is a separate module; pointer-identity test |
| AC-08 static fallback equals manual weighted sum | PASS | `tests/test_losses.py` |
| AC-09 all losses finite, asserted | PASS | `assert_finite_step` raises on a NaN total, an inf per-objective loss and an inf `log_lambdas`; negative tests assert each raises (REV-091) |
| AC-10 no second-order differentiation **(new)** | PASS | `create_graph=True` appears nowhere in `training/gradnorm.py` (only an explicit `create_graph=False` and its comment); `test_full_update_runs_on_cpu` completes a full `update()` with `config.device="cpu"` — the configuration that previously raised on the missing SDPA CPU double-backward |
| AC-11 λ gradient matches `autograd.grad`, no graph **(new)** | PASS | Independently recomputed closed form matches the shipped gradient **bit-identically (0.0)** on all-active, partially-active and the `G_t == G*_t` boundary; `requires_grad is False`, `grad_fn is None` in all three |

## Findings

```
REV-120
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-5 (amended) | AC: AC-10, AC-11
File: training/gradnorm.py:67-120
Resolution: all five sub-corrections implemented and verified.
  (a) G_t factored: `torch.autograd.grad(active_losses[name].float(), anchor_params,
      retain_graph=True, create_graph=False)` -> `.norm(p=2).detach()` ->
      `self.lambdas[idx] * grad_norm`.
  (b) lambda gradient via `torch.autograd.grad(gradnorm_loss, [self.log_lambdas])` --
      D7's option 1, the maintainable one.
  (c) `.detach()` applied at the assignment: `self.log_lambdas.grad =
      lambda_grad.detach()`. Verified the consequence rather than the line -- shipped
      `log_lambdas.grad` has `requires_grad is False` and `grad_fn is None` after
      `update()` in all three regimes, so no graph is pinned across steps.
  (d) `probe_double_backward`, `_raise_double_backward` and
      `DoubleBackwardUnsupportedError` are deleted, and so is the `run_pretrain` call
      site. `grep -rn "probe_double_backward|DoubleBackwardUnsupported" --include=*.py`
      over the tree returns nothing.
  (e) Both tests added, and both are non-vacuous: the AC-11 test recomputes the closed
      form independently of the controller, and the AC-10 test runs a real `update()` on
      cpu.
  Equivalence re-verified independently this pass (not inherited from the prior review's
  measurement): max |shipped - closed form| = 0.0 for num_active = 4, 3 and 1.
  The renormalization was correctly left untouched.
```

```
REV-091
Severity: MEDIUM | Status: RESOLVED | Requirement: FR-6 | AC: AC-09
Resolution: re-confirmed. `assert_finite_step` has negative tests in both directions --
  a NaN total, an inf per-objective loss and an inf-valued `log_lambdas` each raise with
  a matching message. An assertion accidentally weakened to `.any()` would fail CI.
```

```
REV-092
Severity: MEDIUM | Status: NOT_APPLICABLE (superseded by REV-120) | Requirement: FR-5
File: training/gradnorm.py (formerly `probe_double_backward`)
Note: this finding asked for a clear error and an early probe when the host's attention
  kernel had no double-backward. REV-120 removed the double-backward entirely, so there
  is no longer an invariant to probe. The probe, the raiser and the exception type were
  deleted by T3.6 -- which is the correct outcome: leaving them would imply a constraint
  the code no longer has. Closing as no longer describing the code.
```

```
REV-105
Severity: MEDIUM (raised from INFO) | Status: OPEN | Requirement: FR-3 (amended) | AC: AC-03 (amended)
File: training/losses.py:137-145 (`MultiStepLoss.forward`)
Problem: the severity changed in kind because the spec did. Through the previous three
  reviews this was an INFO note that FR-3's prose ("Predict next k items") did not match
  its own position rule (`S-1-step`), with no defect either way and no authority to pick
  one. The spec owner resolved it on 2026-09-17: FR-3 and AC-03 now require **every step to
  read position `S-1`**, so all k per-step projections receive one fixed context
  representation and the horizons are uniform at 2, 3, ..., k+1.
  The implementation still does the old thing:
      pos = S - 1 - step          # training/losses.py:138
  which pairs step `s` with context `items[:i-s]` and target `items[i+1+s]`, i.e. horizons
  2, 4, 6 -- each step simultaneously moving the target further out and shortening the
  context. That is now a requirement violation rather than an ambiguity.
Evidence: verified the arithmetic against the generator this pass rather than inheriting
  it. `generate_sliding_window_examples` emits `context = items[:i]`, `target = items[i]`,
  `targets_k[s] = items[i+k]` for k = 1..K. Position `S-1` holds `items[i-1]`; position
  `S-1-s` holds `items[i-1-s]`. So the current horizon is `(i+1+s) - (i-1-s) = 2 + 2s`,
  and reading from `S-1` throughout gives `2 + s`. Neither is 1..k, which is why the
  amended FR-3 states the horizons explicitly as 2..k+1 -- horizon 1 is the next-item
  objective's target and is deliberately not duplicated.
Impact: No leakage under either scheme -- every target is strictly in the future of the
  position it is predicted from -- so nothing currently trained is *wrong*. What changes is
  which experiment runs, and how the objective can be described. Fixing it now is free;
  fixing it after Phase 14 means retraining, which is why the owner settled it before full
  training rather than after.
Correction: In `MultiStepLoss.forward`, read the same position for every step:
      pos = S - 1                      # was S - 1 - step
  leaving the per-step projections, the valid-target masking and the mean-over-valid-steps
  reduction exactly as they are. Then INVERT the existing position test: it currently
  asserts the `S-1-step` rule, so it will pass on the old behaviour and fail on the
  required one. Replace it with an assertion that all k steps read an identical hidden
  vector -- e.g. that swapping the contents of positions < S-1 leaves the multi-step loss
  unchanged, which is a property only the corrected form has.
  Also update `specs/moe-routing` FR-10's consumer line if it is read as normative; it has
  already been synced to `Z_{u,forecast}[:, S-1, :]` by the owner.
DOWNSTREAM CONSEQUENCE -- read this before running anything:
  `outputs/checkpoints/t_small/best.pt` was trained under the OLD scheme. Multi-step was
  live in that run -- `outputs/logs/pretrain_t_small.log` records
  `train_loss_multistep=9.912500` -- so once this fix lands, that checkpoint is a model
  trained under a different objective from the one the code implements.
  `personalization-federated` REV-127 and `evaluation-diagnostics` REV-129 both CONSUME
  that checkpoint. Running them against the existing `best.pt` after this fix would produce
  artifacts describing a superseded experiment, and nothing in the pipeline would flag it.
  The T6.3 gate is ~190 s, so the remedy is cheap. Required order:
    1. REV-122 (transformer-backbone, CRITICAL) -- independent; does NOT invalidate any
       existing artifact, since everything on disk was produced on MPS/SDPA, which is the
       correct path. It is a blocker for the H100 run only.
    2. This finding (REV-105), plus REV-128 (evaluation-diagnostics, fp32 cosines),
       REV-131 (ablations-cli, CLI flags) and REV-124 -- all code/test, no runs.
    3. Re-run the T6.3 gate:
         uv run python main.py --stage pretrain --model t-small --epochs 1 --max_steps 200
       This regenerates `best.pt` under the corrected objective AND the gradient
       diagnostics with fp32 cosines, in one pass.
    4. Only then the REV-127 and REV-129 stage runs, against the fresh checkpoint.
  Step 3 is the step that is easy to skip and expensive to discover later.
```

```
REV-121
Severity: INFO | Status: RESOLVED | Requirement: FR-5 | AC: -
File: specs/loss-functions/design.md ("Training Loop Integration" pseudocode)
Resolution: closed by the spec owner on 2026-09-17. The pseudocode's
  `if use_gradnorm and step % gradnorm_update_interval == 0:` is replaced by
  `if use_gradnorm:` with a note recording that no such field ever existed in `Config`,
  that `training/pretrain.py` has always updated lambda on every step, and that the clause
  was removed rather than implemented because an update interval would change the
  algorithm. The design and the implementation now agree.
```

```
REV-124
Severity: INFO | Status: OPEN | Requirement: FR-5 | AC: —
File: tests/test_losses.py:298-306 (`TestGradNormGradientIsolation` docstring)
Problem: the class docstring still explains the REV-057 isolation requirement in terms of
  the old implementation: "The GradNorm loss is built from `torch.autograd.grad(...,
  create_graph=True)`, so a plain `.backward()` on it propagates second-order gradients
  into every model parameter in that graph." As of T3.6 there is no `create_graph=True`
  and no second-order graph.
Impact: None on behaviour -- the test itself is still correct and still valuable, because
  the isolation requirement (write into `log_lambdas.grad` only) survives REV-120
  unchanged. Only the stated reason is stale. Flagged because it is the one place left in
  the repository that asserts the second-order form still exists, one file away from the
  AC-10 test that asserts it does not.
Correction: reword the docstring to justify isolation on first-order grounds -- a plain
  `.backward()` on `gradnorm_loss` would still accumulate into model `.grad` buffers the
  caller is about to step on. One-line test-comment edit; no assertion change.
```

```
REV-056
Severity: INFO | Status: RESOLVED | Requirement: FR-5 | AC: —
Resolution: carried forward from the prior review; re-confirmed, no change.
```

## Design Compliance

Compliant with the amended design. D7 is implemented as written, including the parts
that are easy to skip: the norm is detached *before* being scaled by λ (not after), the λ
gradient uses D7's option 1, and the detach-before-assign trap is handled explicitly
rather than incidentally. D7's "unchanged by this decision" list is respected — the
post-step renormalization still runs under `no_grad()` after `optimizer.step()`, D4's
active-only semantics hold, D5's anchor is still the last shared Transformer block, and
D6's separate Adam over `log_lambdas` is intact.

D7's final clause — "`probe_double_backward` ... become dead code and should be removed
rather than left as a check of an invariant the code no longer relies on" — was followed.

The one live divergence between design and implementation is REV-121's
`gradnorm_update_interval` pseudocode, which predates this decision.

## Task Completion

| Task | Verdict |
|---|---|
| T3.2 GradNorm update step | Checked. Its `G_t` expression is superseded, and the task text now says so explicitly rather than silently contradicting the code — the right way to handle it without rewriting history |
| T3.3 anchor resolution | Checked, verified; pointer-identity test against `model.blocks[-1]` |
| T3.6 factored `G_t` **(new)** | Checked, verified true clause by clause: factored build, `create_graph=False`, detached norm, `autograd.grad` for λ, `.detach()` before assignment, all four deletions performed, renormalization untouched |
| T4.10 λ-gradient equivalence **(new)** | Checked, and verified non-vacuous — the closed form is recomputed independently, and I reproduced its result in a separate process |
| T4.11 first-order-only **(new)** | Checked, verified true. The CPU test is the regression guard that keeps this fixed |
| T4.1–T4.9 | Checked; unchanged and previously verified |

No false-complete tasks in this spec.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **267 passed**, 0 failed, 49.20 s |
| `uv run python -m pytest tests/test_losses.py -q` | passed as part of the above |
| **Independent λ-gradient recomputation** (separate process, closed form built from scratch) | `max abs(shipped − closed form)` = **0.0** for num_active = 4, 3, 1; `requires_grad=False`, `grad_fn=None` in all three; inactive grads exactly 0; `sum(active λ)` = 4.0 / 3.0 / 1.0 |
| `grep -rn "create_graph" --include=*.py .` | `training/gradnorm.py` has only `create_graph=False` plus its comment; the remaining hits are in tests (one assertion, one stale docstring — REV-124) |
| `grep -rn "probe_double_backward\|DoubleBackwardUnsupported" --include=*.py .` | **no matches** — T3.6(d) confirmed |
| `uv run python -m compileall -q training` | clean |

## Final Assessment

One thing for BUILD, and it is new work rather than a leftover:

1. **REV-105 (MEDIUM, blocking)** — implement the amended FR-3: `pos = S - 1` for every
   step in `MultiStepLoss.forward`, and invert the position test, which currently asserts
   the superseded rule and would otherwise pass on the wrong behaviour. Nothing else in
   the loss changes. Do this **before** any full training run — after Phase 14 it costs a
   retrain.

There is currently no task in `tasks.md` covering REV-105. BUILD should add one under
Phase 3 next to T3.2, whose `G_t` clause was superseded the same way — that task is the
precedent for how to record a requirement amendment without rewriting history.

REV-120 is closed and was the reason this spec was previously reopened; the equivalence was
independently re-verified this pass rather than carried forward. REV-121 and REV-091 are
closed. REV-124 (stale `TestGradNormGradientIsolation` docstring describing the deleted
second-order implementation) remains an optional one-line test-comment cleanup — worth
folding into the REV-105 commit since both touch this area.

Nothing left for the spec owner: REV-105 and REV-121 were the two open owner decisions and
both were made on 2026-09-17.
