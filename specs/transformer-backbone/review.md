# Review — Transformer Backbone

## Review Status

**CHANGES_REQUIRED**

(Downgraded from the prior `PASS_WITH_WARNINGS`. One new CRITICAL finding, REV-122, on
the CUDA/H100 attention path. Everything the prior review verified still holds — this is
a defect the prior review did not reach, not a regression.)

## Feature

`specs/transformer-backbone` — `models/transformer.py` (`AttentionBackend`,
`CausalSelfAttention`, `TransformerBlock`, `TransformerModel`, size factory), plus the
`attention_backend` surface in `configs/config.py`.

## Review Scope

- Spec directory: `specs/transformer-backbone/` (requirements.md, design.md, tasks.md) —
  all three were edited by the owner/BUILD since the review before last (the SDPA
  decision); unchanged since the last review.
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`.
- Baseline: working tree vs `HEAD` = `d438950` on `master`; `origin/master` at the same
  commit. 38 tracked files modified, no untracked files.
- Files inspected: `models/transformer.py` (443 lines, modified `-55/+22`),
  `configs/config.py`, `tests/test_transformer.py` (modified, `+91`), `tests/test_smoke.py`,
  `tests/test_training.py`, `README.md`.
- Verification was executed, not inferred: the full suite was run, backend resolution and
  config rejection were exercised directly on this host (MPS, torch 2.14.0), and the FA4
  path was driven end-to-end through `TransformerModel` with the CUDA kernel replaced by
  an exact SDPA reference.
- Re-review. Prior findings carried in: REV-062 (HIGH, RESOLVED), REV-080 (RESOLVED),
  REV-102 (RESOLVED), REV-082 (INFO, OPEN), REV-103 (INFO, NOT_APPLICABLE).
- Date: 2026-09-17

## Summary

The REV-062 resolution holds up under re-inspection, and it was done the right way. SDPA
is the sanctioned MPS/CPU kernel, `flex_attention` is gone from the enum and the class,
and — the part that mattered — the gate was not disarmed to achieve it:
`require_attention_backend = True` is still set in both smoke tests and no assertion was
altered. Re-measured here: `VALID_BACKENDS == ('auto', 'flash_attention_4', 'sdpa')`;
`Config(attention_backend="flex_attention").validate()` raises with a message offering
only the three supported values; `auto` on MPS resolves to `sdpa` with
`fallback_reason is None` and `self_test()` returns `status: 'passed'`;
`hasattr(AttentionBackend, "_flex_attention")` is `False`.

What the prior review did not reach is the branch none of the tests exercise end-to-end.
**`CausalSelfAttention.forward` hardcodes `is_causal=False` on the padded path**, relying
on the additive `attn_mask` to carry causality. The SDPA branch reads that mask, so MPS
is correct. The FA4 branch **ignores `attn_mask` entirely** and takes causality only from
the `is_causal` flag — so on CUDA, every left-padded batch runs *bidirectional*
attention. Measured directly (REV-122): changing only the last token of a left-padded
sequence shifts the hidden states at four earlier real positions by ~0.13, which must be
exactly zero under a causal mask. Since every training batch is left-padded, this is the
H100 production path, and it breaks AC-04 and FR-4's "changes only the backend, not the
mathematical operation".

The existing `TestFlashAttentionVarlen` tests cannot catch it: all four call
`_flash_attention_4` / `_flash_attention_4_varlen` **directly with `is_causal=True`**,
never through `CausalSelfAttention.forward`, which is the only place the flag is set to
`False`. The self-test probe likewise passes `is_causal=True` with no mask, so the armed
gate stays green.

## Git Changes

| File | Change |
|---|---|
| `models/transformer.py` | `-55/+22`: `VALID_BACKENDS` narrowed, `_flex_attention` kernel and its `_dispatch` branch deleted, `_preferred_backend()` returns `sdpa` for every non-CUDA device, class docstring rewritten |
| `configs/config.py` | `attention_backend` validation list and error message narrowed to three values |
| `tests/test_transformer.py` | `+91`: FlexAttention-rejection tests, the REV-080 padding test, the REV-102 metadata tests |
| `README.md`, `tests/test_smoke.py`, `tests/test_training.py` | documentation/comment sync (T5.1, T5.2) |

**Scope assessment:** proportionate to T1.3/T1.4/T5.1/T5.2 and nothing more. I checked
specifically that the smoke-test edits are comment-only, since "BUILD must not disarm the
flag to make the suite green" was the load-bearing clause of REV-062 — `git diff HEAD --
tests/test_smoke.py tests/test_training.py` confirms comment-only, with
`require_attention_backend = True` preserved in both.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Embedding layer, tied weights | PASS | `nn.Embedding(num_items + 2, hidden_dim)`, learnable `position_embedding`, `item_emb_weight` property returns the embedding weight itself; identity test present |
| FR-2 Pre-LN block, 4× FFN | PASS | `TransformerBlock` applies `norm1`→attn→residual, `norm2`→FFN→residual; `ffn_dim = hidden_dim * ffn_expansion` |
| FR-3 Three size variants | PASS | Size factory; `assert hidden_dim % num_heads == 0` in `CausalSelfAttention.__init__` |
| FR-4 Attention backends | **FAIL** | The enum, resolution, self-test (forward+backward) and fail-fast behaviour are all correct. But the wrapper does **not** "change only the backend, not the mathematical operation": on a padded batch SDPA computes causal attention and FA4 computes bidirectional attention (REV-122) |
| FR-5 Masking | **PARTIAL** | Causal + padding masking is correct under SDPA and verified. Under FA4, padding is handled (via unpad/repad) but causality is not (REV-122) |
| FR-6 Dual output mode | PASS | `return_all_hidden=True` → `(B, S, D)`; `False` → pooled `x[:, -1, :]`; shape tests for all three sizes |
| FR-7 Metadata integration | PASS | Bias-free `nn.Linear(feature_dim, hidden_dim)` added to inputs; `feature_dim` validated; a tensor passed to a `use_metadata=False` model is ignored with a once-per-process warning (REV-102) |
| FR-8 Embedding normalization | PASS | `l2_normalize` on the pooled representation; hidden states returned un-normalized by design (REV-044), with normalization applied at the scoring sites |
| FR-9 BF16 | PASS | `self.to(dtype=resolve_dtype(config))`; `resolve_dtype` raises on non-bf16; no autocast or GradScaler anywhere |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 shapes for all sizes, both modes | PASS | Shape tests for T-Small/T-Base/T-Large × both `return_all_hidden` modes |
| AC-02 SDPA self-test fwd+bwd on MPS, `auto`→`sdpa`, no fallback | PASS | Reproduced directly: `{'backend': 'sdpa', 'requested_backend': 'auto', 'fallback_reason': None, 'device': 'mps', 'dtype': 'torch.bfloat16', 'status': 'passed'}`. `_probe` runs `out.float().sum().backward()` and checks gradient finiteness |
| AC-03 FA4 self-test on H100 | NOT VERIFIABLE | No CUDA device and no flash-attn build on this host. T1.1 is correctly left unchecked. **Note:** REV-122 means this gate would pass on the H100 while the model still trains non-causally — the probe uses an unpadded input, so it never exercises the broken branch |
| AC-04 causal mask prevents future attention; padding contributes nothing | **FAIL** | Padding half: PASS, and now genuinely tested (REV-080 fix drives `CausalSelfAttention` with identity projections and asserts exact zero). Causal half: PASS under SDPA, **FAIL under FA4** — measured leakage of 0.133 at positions that must be unaffected (REV-122) |
| AC-05 tied weights by identity | PASS | `item_emb_weight` returns `self.item_embedding.weight`; identity test |
| AC-06 metadata on/off | PASS | Output bit-identical to `metadata=None` when `use_metadata=False`; test supplies a real tensor rather than `None` (REV-102 fix) |
| AC-07 L2 bounds scores to [-1, 1] | PASS | `tests/test_transformer.py` normalization test |
| AC-08 BF16 throughout | PASS | Parameter dtype check; `resolve_dtype` raises otherwise |
| AC-09 `flex_attention` rejected, no reachable path | PASS | Rejection message reproduced verbatim; `hasattr(AttentionBackend, "_flex_attention")` is `False`; `grep -rni flex` over `models/ configs/ training/ evaluation/ data/ main.py utils.py` returns only comments |

## Findings

```
REV-122
Severity: CRITICAL | Status: OPEN | Requirement: FR-4, FR-5 | AC: AC-04, AC-03
File: models/transformer.py:288-301 (`CausalSelfAttention.forward`, padded branch);
      models/transformer.py:141-150 (`AttentionBackend._dispatch`);
      models/transformer.py:186-227 (`_flash_attention_4_varlen`)
Problem: on the padded branch, `CausalSelfAttention.forward` builds an additive
  `attn_float` mask that encodes causal + key-padding, and then calls
      self.backend.forward(q, k, v, attn_mask=attn_float,
                           padding_mask=padding_mask, is_causal=False)
  `is_causal=False` is hardcoded because the mask is expected to carry causality. That
  contract holds for exactly one backend. `_dispatch` forwards `attn_mask` to SDPA (which
  reads it) but passes only `padding_mask` and `is_causal` to `_flash_attention_4`, which
  discards the additive mask and derives causality solely from the flag. The varlen call
  therefore runs with `causal=False`.
  `TransformerModel.forward` passes a padding mask whenever the batch contains any pad
  token, and every training batch is left-padded to `max_seq_length`, so this is the
  normal path on CUDA, not an edge case.
Evidence (measured, not inferred): driving `TransformerModel` with
  `attention_backend="flash_attention_4"` and the CUDA kernel replaced by an exact SDPA
  reference, on a left-padded row `[0,0,0,11,12,13,14,15]`, changing ONLY position 7:
      pos 0-2 (pad):  0.000000        <- correctly excluded
      pos 3:          0.129807        <- must be 0 under a causal mask
      pos 4:          0.132777        <- must be 0
      pos 5:          0.120646        <- must be 0
      pos 6:          0.086955        <- must be 0
      pos 7:          1.517679        <- expected
  The `causal=` flags the kernel actually received were [True, False, False]: True for
  the unpadded self-test probe, False for both real forward passes.
Impact: On the H100 -- the declared production training hardware -- the backbone attends
  bidirectionally. Every objective that reads an interior position (masked item modelling,
  multi-step targets at S-1-k) sees tokens that come after its own target, which is direct
  label leakage. Results would be inflated and not comparable with the MPS/SDPA numbers,
  since the two backends compute different functions -- precisely what FR-4's "changes
  only the backend, not the mathematical operation" exists to prevent. Nothing is wrong on
  the dev host, so this is invisible to every test and every smoke gate currently run, and
  it would first appear as an unexplained H100-vs-MPS discrepancy in the paper's numbers.
  AC-03 does not protect against it either: `_probe` attends an UNPADDED input with
  `is_causal=True`, so the H100 self-test would pass while training ran non-causally.
Correction: Stop routing causality through a mask only one backend reads. Either
  (a) pass the real causal intent down -- keep a `is_causal=True` argument on the padded
  branch and let the SDPA branch combine `is_causal` with a padding-only additive mask
  (SDPA accepts `attn_mask` and `is_causal` together is NOT allowed, so build the causal
  part into the mask for SDPA only, while FA4 receives `is_causal=True`); or
  (b) give `_dispatch` the unambiguous pair `(padding_mask, is_causal)` and let each
  backend construct its own mask, which is what the "thin wrapper formats masks for the
  selected backend" design (D2, point 3) actually describes.
  Add a regression test that goes through `CausalSelfAttention.forward` (NOT through
  `_flash_attention_4*` directly) with the stubbed kernel and asserts that perturbing
  position S-1 leaves positions < S-1 bit-identical. The four existing
  `TestFlashAttentionVarlen` tests all call the private helpers with `is_causal=True`, so
  none of them can fail on this defect -- that is why it survived two reviews.
  Also worth extending `_probe` to exercise a padded input, so the H100 gate covers the
  branch that actually runs in training.
```

```
REV-062
Severity: HIGH | Status: RESOLVED | Requirement: FR-4 | AC: AC-02, AC-09
File: models/transformer.py; configs/config.py:294-303
Resolution re-verified this pass. The owner's decision (a) is reflected in requirements
  FR-4/AC-02/AC-09, design D2b, steering/code-quality.md and BUILD_ORDER Phase 4; BUILD's
  T1.3/T1.4/T5.1/T5.2 implement it. Re-checked the specific failure mode the finding
  warned about: `git diff HEAD -- tests/test_smoke.py tests/test_training.py` is
  comment-only in both, `require_attention_backend = True` is still present in each, and
  no assertion was altered. The suite is green because the preferred backend executes,
  not because the requirement was relaxed.
```

```
REV-080
Severity: LOW | Status: RESOLVED | Requirement: FR-5 | AC: AC-04
File: tests/test_transformer.py:38-85
Resolution: re-confirmed. The replacement drives `CausalSelfAttention` directly with
  identity v/out projections, places the distinctive value at padded key positions only,
  asserts the real-query output is exactly zero, and includes a non-vacuity assertion
  that the same value DOES reach real queries when the padding mask is omitted.
  Note the contrast with REV-122: this test earns its PASS precisely because it goes
  through `CausalSelfAttention.forward`. The FA4 tests do not, which is the gap.
```

```
REV-102
Severity: LOW | Status: RESOLVED | Requirement: FR-7 | AC: AC-06
File: models/transformer.py:398-411; tests/test_transformer.py:294-340
Resolution: re-confirmed. A metadata tensor supplied to a `use_metadata=False` model is
  ignored (output identical to `metadata=None`) with a once-per-process warning, so it is
  ignored but not silently. The test supplies a real tensor.
```

```
REV-082
Severity: INFO | Status: RESOLVED | Requirement: FR-2 | AC: -
File: specs/transformer-backbone/design.md (new section "D-dropout")
Resolution: closed by the spec owner on 2026-09-17. `design.md` now records that
  `attn_dropout` is applied to the attention OUTPUT rather than to the attention weights,
  why (SDPA's `dropout_p` is not usable uniformly across the two sanctioned backends, so
  output-side dropout preserves FR-4's "changes only the backend, not the mathematical
  operation" property), and that FR-2 pins only the rate and not the placement, so the
  implementation conforms. The deviation from textbook attention-weight dropout is now on
  the record in the design rather than only in a review.
```

```
REV-103
Severity: INFO | Status: NOT_APPLICABLE | Requirement: FR-5 | AC: —
Note: this recorded that the `(B, H, S, S)` additive mask was built and then discarded on
  the FlexAttention and FA4 branches. It is superseded: what it described as a harmless
  inefficiency on the FA4 branch is, on inspection, the CRITICAL defect now filed as
  REV-122. Closing this ID and tracking the real problem there.
```

```
REV-123
Severity: INFO | Status: OPEN | Requirement: FR-5 | AC: —
File: models/transformer.py:288-297
Note: a fully-padded query row (position 0 of a left-padded sequence can only attend to
  key 0, which is itself padding) produces an all `-inf` mask row. This does NOT produce
  NaN: PyTorch's SDPA returns zeros for fully-masked rows, verified on both cpu and mps in
  bf16, and a whole-batch check on an all-pad sequence returned finite hidden states and a
  finite pooled vector. Recorded only because the safety of this path is a property of the
  SDPA kernel rather than of anything this repository asserts, and the requirements' Edge
  Cases section ("sequence entirely padding ... valid hidden state at S-1 still computed")
  is currently satisfied by that external behaviour. No action required.
```

## Design Compliance

Compliant with the amended design on every point except one, and that one is REV-122.

D2b is implemented as written: MPS/CPU resolve to `sdpa`, CUDA to `flash_attention_4`, no
third option in the enum, `require_attention_backend` and the fail-fast paths untouched.
D2's amended point 2 ("forward *and* backward") is implemented in `_probe`. D1 (Pre-LN),
D3 (additive bias-free metadata projection), D4 (normalization at scoring sites) are all
respected. The new testing-strategy line about the backend enum has a matching test.

The violation is **D2 point 3** — "the wrapper's job is ... format masks for the selected
backend". The wrapper does not do this: the *caller* formats one mask in SDPA's dialect
and hands it down, and the FA4 branch silently receives a semantic (`is_causal=False`)
that is false. This is the design decision that, if actually implemented, would have
prevented REV-122.

No unapproved dependency, no architectural drift.

## Task Completion

| Task | Verdict |
|---|---|
| T1.1 attention wrapper | Correctly **unchecked**, and REV-122 vindicates that. The task's own text — "delegates to the selected backend in its expected argument format" — is not satisfied for FA4, so this must stay unchecked regardless of H100 access |
| T1.2 backend tests | Checked. The padding half it claims is genuinely tested now. But the task also claims causal-mask coverage, and that coverage does not extend to the FA4 branch — see REV-122's note on why the four varlen tests cannot fail on it |
| T1.3 remove FlexAttention | Checked, verified true. Every clause performed: kernel deleted, `_dispatch` branch deleted, `VALID_BACKENDS` narrowed, config list and message narrowed, `_preferred_backend()` changed, docstring rewritten |
| T1.4 test narrowed surface | Checked, verified true. Both assertions exist; the MPS assertion is not skipped on this host and passes |
| T2.x–T4.x | Checked; unchanged and previously verified. Re-spot-checked tied weights, BF16 and normalization |
| T5.1 README sync | Checked, verified true — all three locations updated, nothing else changed |
| T5.2 smoke-gate comments | Checked, verified true, and verified comment-only |

No false-complete tasks. T1.1 being unchecked is the correct state and now has a second
reason to remain so.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` | **267 passed**, 0 failed, 49.20 s |
| `uv run python -m pytest -q -m slow --collect-only` | 8 slow collected, 259 deselected — confirms the 267 run included the slow gates |
| `Config(attention_backend="flex_attention").validate()` (direct) | raises `ValueError: Unknown attention_backend 'flex_attention'. Choose from 'auto', 'flash_attention_4', 'sdpa'` |
| `AttentionBackend(cfg).self_test()`, `auto` + `require_attention_backend=True` (direct) | `{'backend': 'sdpa', 'requested_backend': 'auto', 'fallback_reason': None, 'device': 'mps', 'dtype': 'torch.bfloat16', 'status': 'passed'}` |
| `hasattr(AttentionBackend, "_flex_attention")` (direct) | `False` |
| `git diff HEAD -- tests/test_smoke.py tests/test_training.py` | comment-only; `require_attention_backend = True` preserved in both |
| `grep -rni flex models/ configs/ training/ evaluation/ data/ main.py utils.py` | comments only; no reachable code path |
| **FA4 leakage probe** — `TransformerModel` with `attention_backend="flash_attention_4"`, CUDA kernel stubbed with an exact SDPA reference, left-padded input, perturb position S-1 only (direct) | **positions 3,4,5,6 shift by 0.130 / 0.133 / 0.121 / 0.087 — must be 0. Kernel received `causal=[True, False, False]`.** REV-122 |
| All-padding and left-padded finiteness check under SDPA (direct) | all hidden states and pooled outputs finite on cpu and mps — REV-123 |

## Final Assessment

For BUILD, in order:

1. **REV-122 (CRITICAL, blocking)** — fix the causal-intent plumbing between
   `CausalSelfAttention.forward` and the FA4 branch, per correction (a) or (b). This must
   land before any H100 run: it silently invalidates CUDA training results and the AC-03
   self-test cannot catch it, because `_probe` attends an unpadded input.
2. **REV-122 regression test** — add a test that goes through `CausalSelfAttention.forward`
   with the stubbed FA4 kernel and asserts positions `< S-1` are bit-identical when
   position `S-1` changes. Extend `_probe` to include a padded input so the H100 gate
   covers the branch training actually takes. Do not close the finding on the
   implementation fix alone: the reason this survived two reviews is that all four
   existing varlen tests call the private helpers directly with `is_causal=True`.
3. **T1.1** — keep it unchecked. It now has two open reasons: unverified FA4 on H100, and
   REV-122.

Nothing for the spec owner. REV-062 was resolved by owner decision in the previous round
and re-verified here; REV-082 was closed by the owner on 2026-09-17. REV-123 is an
observation carried for the record with no action attached.
