# Review — Transformer Backbone

## Review Status

**PASS_WITH_WARNINGS**

(Was `CHANGES_REQUIRED`. REV-122, the sole CRITICAL, is fixed and independently verified
below. The only remaining items are INFO, plus AC-03, which cannot be verified on this
host and whose task is correctly left unchecked.)

## Feature

`specs/transformer-backbone` — `models/transformer.py`: attention backend wrapper
(SDPA / FlashAttention-4), Pre-LN blocks, three size variants, embedding and positional
layers with tied output weights, metadata projection, L2 embedding normalization, BF16
throughout.

## Review Scope

- Spec directory: `specs/transformer-backbone/` (requirements.md, design.md, tasks.md).
  All three unmodified in the working tree.
- Steering: `architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`.
- Baseline: `master` at `0bc67ee`, `origin/master` at the same commit. No file owned by
  this spec is modified in the working tree, so the code below is `HEAD`'s content.
- Files inspected: `models/transformer.py` (490 lines, read in full),
  `tests/test_transformer.py` (707 lines), `configs/config.py` backend validation,
  `README.md` hardware sections, `outputs/logs/run_config.json`.
- Re-review. Prior findings carried in: REV-122 (CRITICAL, OPEN), REV-062, REV-080,
  REV-102, REV-082 (RESOLVED), REV-103 (NOT_APPLICABLE), REV-123 (INFO, OPEN).
- Date: 2026-09-18

## Summary

REV-122 is genuinely fixed, and the fix is the one the finding asked for — correction (b),
not a patch over the symptom. `AttentionBackend.forward`/`_dispatch` no longer accept an
`attn_mask` parameter at all; the caller passes `(padding_mask, is_causal)` and each
backend formats the mask its own kernel reads. `CausalSelfAttention.forward` now passes
`is_causal=True` unconditionally on both branches. This is design D2 point 3 implemented
as written, which is what the previous review recorded as the underlying design violation.

**The regression tests were verified to be non-vacuous by me, not taken on BUILD's word.**
I reinstated the pre-fix contract in the working copy (`is_causal=(padding_mask is None)`)
and re-ran the new test class: `test_future_token_cannot_reach_earlier_positions[sdpa]`,
`[flash_attention_4]` and `test_flash_attention_kernel_receives_causal_true_on_padded_batches`
all FAIL on it — the spy records `causal=[False, True]`, exactly the `False` on the padded
call that the finding described — and all pass on the fix. `models/transformer.py` was
restored immediately afterwards and `git diff` on it is empty.

`_probe` now covers both the unpadded and the left-padded branch, so the startup gate
exercises the path training actually takes. That closes the second half of REV-122: the
old probe would have certified a backend that trains non-causally.

What remains is AC-03. The FlashAttention-4 path is exercised only against a stubbed
reference kernel; no CUDA device and no `flash-attn` build exist on this host. T1.1 is
correctly left `[ ]` and its BUILD note says so plainly. That is honest reporting, not a
defect — but it means this spec is not fully accepted until the H100 self-test runs.

## Git Changes

No file owned by this spec appears in `git status`. The four modified files
(`evaluation/longtail.py`, `specs/evaluation-diagnostics/tasks.md`,
`tests/test_diagnostics.py`, `tests/test_preprocessing.py`) all belong to
`evaluation-diagnostics`. The REV-122 fix and its tests were committed before this
baseline, in `daffb41` / `aa04dee`.

**Scope assessment:** nothing unexpected, nothing missing.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Embedding layer + tied output | PASS | `nn.Embedding(num_items + 2, hidden_dim)` and learnable `nn.Embedding(max_seq_length, hidden_dim)`; no separate output matrix exists anywhere in the module. `item_emb_weight` is a property returning `self.item_embedding.weight`, verified identical by object identity on all three variants |
| FR-2 Pre-LN block, 4× FFN | PASS | `TransformerBlock.forward` is `x + drop(attn(norm1(x)))` then `x + drop(ffn(norm2(x)))` — norms before, not after. FFN is `Linear(D, 4D) → GELU → Dropout → Linear(4D, D)` with `ffn_expansion` from config (measured 4) |
| FR-3 Three size variants | PASS | Measured via `apply_model_config`: t-small (3, 256, 4), t-base (4, 320, 4), t-large (5, 384, 6). `hidden_dim % num_heads == 0` for all three, asserted in `CausalSelfAttention.__init__` |
| FR-4 Attention backends | PASS (CUDA half NOT VERIFIABLE) | `VALID_BACKENDS == ("auto", "flash_attention_4", "sdpa")`. `_preferred_backend` returns `flash_attention_4` on CUDA and `sdpa` otherwise. Explicit requests never fall back; `require_attention_backend=True` also forbids substituting for an unavailable `auto` preference. `_probe` runs forward **and** backward over both an unpadded and a left-padded input. Resolved backend recorded in `outputs/logs/run_config.json` (`attention_backend_resolved: "sdpa"`, `fallback_reason: null`). The FA4 kernel itself has not executed — see AC-03 |
| FR-5 Masking | PASS | Causal intent travels as `is_causal=True` from `CausalSelfAttention` to every backend; `_sdpa_additive_mask` folds causality and key padding into SDPA's additive mask and returns `None` when unpadded so the native fast path is kept. Padding mask is `(seq != 0)`, computed once and passed to every block only when the batch actually contains padding |
| FR-6 Dual output mode | PASS | `return_all_hidden=True` → `(B, S, D)`; `False` → `(B, D)` pooled from index `S-1` per the left-padding convention. Measured on all three variants |
| FR-7 Metadata integration | PASS | Bias-free `nn.Linear(feature_dim, hidden_dim)`, added to item + positional embeddings. `use_metadata=True` with `feature_dim <= 0` raises rather than silently producing a metadata-free model. A dimension mismatch raises naming both dimensions. A metadata tensor handed to a `use_metadata=False` model is ignored, with a once-per-process warning so the mismatch is not actually silent |
| FR-8 Embedding normalization | PASS | `l2_normalize` computes in fp32 and casts back, applied to the pooled representation and at the scoring sites. Hidden states are deliberately returned un-normalized because they feed the MoE layer; the `[-1, 1]` bound is enforced where dot products happen |
| FR-9 BF16 | PASS | `self.to(dtype=resolve_dtype(config), device=get_device(config))`. Measured: every parameter and both output modes are `torch.bfloat16` on all three variants. No autocast, no GradScaler anywhere in the module |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 shapes for 3 variants × 2 output modes | PASS | Measured directly this pass: pooled `(4, 256)` / `(4, 320)` / `(4, 384)`, all-hidden `(4, 50, 256)` / `(4, 50, 320)` / `(4, 50, 384)`. `TestModelShapes::test_output_shapes` is parametrized over all three |
| AC-02 SDPA self-test (fwd+bwd) on MPS; `auto` → `sdpa` with no fallback reason | PASS | Constructed on this MPS host: `backend_name == "sdpa"`, `fallback_reason is None`. `_probe` ran forward and backward over both branches during construction. `run_config.json` records the same triple. `test_mps_auto_resolves_to_sdpa_with_the_gate_armed` pins it |
| AC-03 FlashAttention-4 self-test on H100 | **NOT VERIFIABLE** | No CUDA device and no `flash-attn` package on this host. Everything in the FA4 path is exercised against `TestFlashAttentionVarlen._install_reference_flash_attn`, a dense-attention stub — which verifies the unpad/cu_seqlens/repad plumbing and the `causal=` flag, but not the real kernel. T1.1 is correctly left unchecked |
| AC-04 causal mask + padding contribute nothing | PASS | `test_future_token_cannot_reach_earlier_positions` drives `CausalSelfAttention.forward` on a left-padded input, perturbs only position `S-1`, and asserts every earlier real position moves by **exactly 0.0**, with a non-vacuity assertion that `S-1` itself moves — parametrized over both backends. `test_padding_mask_zeros_padded_key_contributions` covers the padding half. Both verified to fail on the pre-REV-122 contract |
| AC-05 tied weights by identity | PASS | `model.item_emb_weight is model.item_embedding.weight` measured `True` on all three variants; `TestTiedWeights` asserts it |
| AC-06 metadata on/off | PASS | `test_metadata_changes_output` (True ⇒ output differs) and `test_no_metadata_ignores_tensor` (False ⇒ output identical with and without the tensor), plus `test_no_metadata_warns_once_when_tensor_supplied` |
| AC-07 scores bounded to [-1, 1] | PASS | Measured on all three variants against the normalized item table: ranges `[-0.171, 0.631]`, `[-0.157, 0.606]`, `[-0.164, 0.602]` — inside the bound. `TestEmbeddingNormalization::test_scores_bounded` asserts it |
| AC-08 BF16 parameters and computation | PASS | Measured: `all(p.dtype == torch.bfloat16)` and both outputs `torch.bfloat16`, all three variants |
| AC-09 `flex_attention` rejected; no reachable code path | PASS | Measured: `Config(attention_backend="flex_attention").validate()` raises `ValueError: Unknown attention_backend 'flex_attention'. Choose from 'auto', 'flash_attention_4', 'sdpa'`. A repository-wide grep for `flex_attention`/`FlexAttention` over `main.py`, `models/`, `training/`, `evaluation/`, `configs/`, `data/`, `tests/` and `README.md` returns **only** four explanatory comments (three in `transformer.py`, one in `config.py`) and the dedicated rejection tests. No kernel, no dispatch branch, no enum member |

## Findings

```
REV-122
Severity: CRITICAL | Status: RESOLVED | Requirement: FR-4, FR-5 | AC: AC-04, AC-03
File: models/transformer.py:146-198 (_dispatch, _sdpa_additive_mask), :326-345
      (CausalSelfAttention.forward), :113-141 (_probe)
Resolution: fixed as correction (b), the one the finding recommended. `forward`/`_dispatch`
  take `(padding_mask, is_causal)` and no longer accept an `attn_mask` at all, so an
  SDPA-dialect additive mask can no longer be the channel causality travels in.
  `CausalSelfAttention.forward` passes `is_causal=True` on both branches and builds no
  mask. `_sdpa_additive_mask` is SDPA's own formatter and returns `None` when unpadded,
  preserving the native `is_causal` fast path. `_probe` was extended to the left-padded
  case and labels which case failed.
Verification (independent, not BUILD's): the pre-fix contract was reinstated in the
  working copy and the new test class re-run. Three of its four tests fail on the old
  contract -- including the spy asserting the flag the kernel actually received, which
  records `causal=[False, True]` -- and all four pass on the fix. The four pre-existing
  `TestFlashAttentionVarlen` tests pass on BOTH, which is exactly why they could not
  catch this. `models/transformer.py` restored; `git diff` on it is empty.
Residual: this closes the *correctness contract*. It does not close AC-03, and BUILD's
  note says so explicitly rather than claiming otherwise.
```

```
REV-062
Severity: HIGH | Status: RESOLVED | Requirement: FR-4 | AC: AC-02, AC-09
File: models/transformer.py:13-16, :57-63; configs/config.py:297
Re-verified. `VALID_BACKENDS` has exactly three members, `_preferred_backend` returns
  `sdpa` for every non-CUDA device, and the config rejection message names only the three
  sanctioned values. `require_attention_backend` and the fail-fast paths are intact --
  `_resolve_backend` still raises on an explicit unavailable backend and still raises for
  `auto` when the gate is armed. Documentation is synced: README's ASCII block reads
  "SDPA (MPS) or FA4 (H100)", both Mac rows of the hardware table read SDPA, and the
  fail-fast paragraph names SDPA (T5.1). T5.2's smoke-gate comments are updated.
```

```
REV-080
Severity: LOW | Status: RESOLVED | Requirement: FR-5 | AC: AC-04
File: models/transformer.py:470-473
Re-verified. `padding_mask = (seq != 0)` marks every pad position, and
  `test_padding_mask_marks_every_pad_position` pins it. The mask is passed to the blocks
  only when the batch actually contains padding, which is an optimisation rather than a
  semantic change -- an unpadded batch needs no key mask.
```

```
REV-102
Severity: LOW | Status: RESOLVED | Requirement: FR-7 | AC: AC-06
File: models/transformer.py:381-383, :443-458
Re-verified. The once-per-process warning is guarded by a class attribute and the tensor
  is set to `None` before it can enter the computation, so AC-06's "must not affect
  computation" holds exactly. Covered by `test_no_metadata_warns_once_when_tensor_supplied`.
```

```
REV-082
Severity: INFO | Status: RESOLVED | Requirement: FR-2 | AC: —
File: models/transformer.py:347-351; specs/transformer-backbone/design.md (D-dropout)
Re-verified. Attention dropout is applied to the attention OUTPUT rather than to the
  weight matrix, and design.md carries the D-dropout note recording it. FR-2 pins only
  the rate, so this conforms; the note exists so the paper's method section states it.
```

```
REV-103
Severity: INFO | Status: NOT_APPLICABLE | Requirement: FR-5 | AC: —
Superseded by REV-122 and closed there. Retained so the ID is not reused.
```

```
REV-123
Severity: INFO | Status: OPEN | Requirement: FR-5 | AC: —
File: models/transformer.py:177-198 (_sdpa_additive_mask)
Problem: a fully-masked query row is reachable -- at a padded query position every key is
  blocked by the padding half, and every later key by the causal half -- so that row of
  the additive mask is entirely `-inf`.
Impact: none observed. Re-measured independently this pass on MPS in bf16 with a
  left-padded batch: the dispatched output is finite at both the padded and the real
  positions. PyTorch's SDPA returns a safe value for fully-masked rows rather than NaN.
Correction: none required. Recorded because the safety of this path is a property of the
  SDPA kernel rather than of anything this repository asserts, and the Edge Cases clause
  ("sequence entirely padding ... valid hidden state at S-1 still computed") is satisfied
  by that external behaviour. If it ever needs pinning, the place is an assertion in
  `_probe`, which already runs a padded case.
```

## Design Compliance

Compliant on every point, including the one that was violated last pass.

- **D1 Pre-LN** — implemented as specified.
- **D2 backend abstraction** — all four of its listed jobs are now done by the wrapper.
  Point 3 ("format masks for the selected backend") was the violation behind REV-122 and
  is now literally true: `_sdpa_additive_mask` is SDPA's formatter, the varlen unpad/repad
  is FA4's, and the caller supplies neither.
- **D2b SDPA as the sanctioned MPS backend** — implemented; the enum, the preference
  function, the config validator and the documentation all agree.
- **D3 additive bias-free metadata projection** — `nn.Linear(..., bias=False)`, preserving
  the zero-vector semantics for zero-metadata cold items.
- **D4 normalization at the scoring sites** — pooled output normalized, hidden states left
  raw for the MoE layer.
- **D-dropout** — matches the code.

No unapproved dependencies; `flash_attn` is imported lazily inside the CUDA branch only,
so the module imports cleanly on a host without it. No `.cuda()` calls; device comes from
`get_device(config)` throughout.

## Task Completion

| Task | Marked | Verdict |
|---|---|---|
| T1.1 attention backend wrapper | `[ ]` | **Correctly unchecked.** The SDPA half is complete and verified; the CUDA half is implemented and plumbed but the real kernel has never run. BUILD's note states this plainly and the REV-122 note explicitly refuses to claim the fix earns AC-03. This is the kind of honest incompleteness a review should confirm, not penalise |
| T1.3 remove FlexAttention | `[x]` | Verified complete. The claims in the BUILD note were re-measured: the exact `ValueError` text, the MPS `auto` → `sdpa` resolution with `fallback_reason is None`, and the absence of any FlexAttention code path |
| T1.4 test the narrowed surface | `[x]` | Verified. `TestNarrowedBackendSurface` has all four tests the task names; `TestAttentionBackendGate` is retained and its comments name SDPA |
| T1.5 route causal intent to every backend | `[x]` | **Verified complete, including the non-vacuity claim.** BUILD asserted the new tests fail on the reinstated old contract; I reproduced that independently. Every element the task specifies is present: the `attn_mask` parameter is gone, `_sdpa_additive_mask` exists and short-circuits when unpadded, `is_causal=True` on both branches, the regression test goes through `CausalSelfAttention.forward` rather than the private helpers, and `_probe` covers the padded branch |
| T1.2, T2.1–T2.2, T3.1–T3.3, T4.1–T4.6 | `[x]` | Verified against the code and the 30 passing tests in `tests/test_transformer.py`. Shapes, tied-weight identity, metadata on/off, BF16 and the `[-1, 1]` bound were all re-measured directly this pass rather than inferred from the tests |
| T5.1 README sync | `[x]` | Verified against `README.md`: line 63 ASCII block, lines 491–492 hardware table, line 497 fail-fast paragraph. No FlexAttention mention survives in the file |
| T5.2 smoke-gate comments | `[x]` | Verified; `require_attention_backend = True` is still set in both gates |

No false-complete tasks. T1.1's open state is accurate.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **275 passed**, 0 failed, 55.74 s |
| `uv run python -m pytest tests/test_transformer.py -q` | **30 passed**, 0.85 s |
| `uv run python -m pytest tests/test_transformer.py::TestCausalIntentReachesEveryBackend -q` **with the pre-REV-122 contract reinstated** | **3 failed, 1 passed** — the spy reports `causal=[False, True]`. Source restored; `git diff models/transformer.py` empty |
| Construct `AttentionBackend(Config())` on this host (direct) | device `mps`, backend `sdpa`, `fallback_reason None` |
| `_dispatch("sdpa", ..., padding_mask=left-padded, is_causal=True)` (direct) | output finite at padded and real positions — REV-123 re-measured |
| Build all three variants, forward both modes (direct) | shapes, dtypes, tied-weight identity and score bounds as tabulated above |
| `Config(attention_backend="flex_attention").validate()` (direct) | `ValueError` naming exactly the three sanctioned values |
| `grep -rn 'flex_attention\|FlexAttention'` over all source, tests and README | 4 explanatory comments + the rejection tests; no code path |
| Read `outputs/logs/run_config.json` (direct) | `torch 2.14.0`, device `mps`, dtype `torch.bfloat16`, requested `auto`, resolved `sdpa`, fallback `null` |

## Final Assessment

**Nothing for BUILD on this host.** The CRITICAL that blocked this spec is closed, and it
was closed properly — the contract changed so the defect class cannot recur, not just the
one call site.

Two things remain, neither of them BUILD's to do now:

1. **AC-03 stays unverified and T1.1 stays open until the H100 run.** Everything that can
   be checked without the hardware has been: the varlen plumbing matches dense causal
   attention against a reference stub, the `causal=` flag reaching the kernel is asserted
   on both the padded and unpadded calls, and the startup probe now exercises the padded
   branch so the gate cannot certify a non-causal configuration. When the H100 is
   available, run the self-test, confirm `attention_backend_resolved == "flash_attention_4"`
   with a null fallback reason in the manifest, and re-run
   `TestCausalIntentReachesEveryBackend` against the real kernel rather than the stub —
   that last step is what actually retires AC-03, because REV-122 lived precisely in the
   gap between the stub and the kernel.
2. **REV-123 is INFO and needs no action.** It is recorded so that the repository's
   reliance on SDPA's fully-masked-row behaviour is a known, measured fact rather than an
   accident.

This spec is accepted for all dev-host work. It is not fully accepted for the H100 path,
which is the correct state and is what the unchecked T1.1 communicates.
