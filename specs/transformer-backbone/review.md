# Review — Transformer Backbone

## Review Status

**CHANGES_REQUIRED**

## Feature

`specs/transformer-backbone` — `models/transformer.py`: `AttentionBackend`, `CausalSelfAttention`, `TransformerBlock`, `TransformerModel`, the size variants, metadata projection, and embedding normalization.

## Review Scope

- Spec directory: `specs/transformer-backbone/` (requirements.md, design.md, tasks.md)
- Steering: `steering/architecture.md`, `code-quality.md`, `conventions.md`, `testing.md`
- Baseline: working tree vs `HEAD` = `32b0904` on `master`. All feature work is uncommitted.
- Files inspected: `models/transformer.py` (457 lines, modified), `tests/test_transformer.py` (453 lines, modified), `configs/config.py`, `utils.py`, and the scoring call sites in `evaluation/metrics.py` and `training/losses.py`
- Hardware: Apple Silicon, MPS, PyTorch 2.14.0 — the dev host `steering/architecture.md` names
- Re-review of the prior `review.md`. `transformer.py` (21:51) and `tests/test_transformer.py` (21:52) were both modified after it.
- Date: 2026-09-15

## Summary

REV-081 is genuinely fixed: the FlashAttention-4 path now routes left-padded batches through `flash_attn_varlen_func` with unpad/cu_seqlens/repad plumbing, and `TestFlashAttentionVarlen` exercises that plumbing against dense causal attention with the kernel stubbed. The masking implementation itself is correct — I verified directly that padded key positions contribute exactly zero to every real query position.

Two things are not resolved. **REV-062** has moved from "silently wrong" to "honestly blocked": the gate is now armed in the two smoke tests, and they FAIL on this host because FlexAttention has no backward pass on MPS in PyTorch 2.14.0. BUILD handled this correctly — it left T1.1, T6.2 and T6.3 unchecked and escalated rather than editing the spec — but the repository's test suite does not pass, and the decision (sanction SDPA, or pin a PyTorch build where FlexAttention trains on MPS) is the owner's. **REV-080** is still open: the test named for padding masking compares a tensor against *itself*, so AC-04's padding half remains unverified by the suite even though the property holds.

## Git Changes

| File | State |
|---|---|
| `models/transformer.py` | modified (+~320 lines vs `32b0904`) |
| `tests/test_transformer.py` | modified (+~330 lines) |
| `configs/config.py` | modified — `require_attention_backend`, `"sdpa"` in the `attention_backend` enum |
| `specs/transformer-backbone/tasks.md` | modified (BUILD notes on T1.1) |
| `specs/transformer-backbone/review.md` | untracked (this file) |

**Scope assessment:** in scope. No new runtime dependencies — `flash_attn` is imported lazily inside the CUDA path and is not required to import the module.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Embedding layer | PASS | `nn.Embedding(num_items + 2, hidden_dim)` and learnable `nn.Embedding(max_seq_length, hidden_dim)` ([transformer.py:373](models/transformer.py:373)). No separate output matrix anywhere; `item_emb_weight` is a property returning `self.item_embedding.weight`. |
| FR-2 Pre-LN block | PASS | `TransformerBlock.forward` ([transformer.py:354](models/transformer.py:354)) is `x + drop(attn(norm1(x)))` then `x + drop(ffn(norm2(x)))`; FFN is Linear→GELU→Dropout→Linear at `ffn_expansion`. |
| FR-3 Size variants | PASS | `MODEL_CONFIGS` matches the table exactly; `Config.validate` raises on `hidden_dim % num_heads != 0`; `TestModelShapes` parametrises all three. |
| FR-4 Attention backends | PARTIAL | Resolution, probing and manifest recording are all real (REV-043's fix holds: every backend is genuinely invoked). MPS→FlexAttention cannot execute a backward pass on this host, so `"auto"` resolves to SDPA — see REV-062. CUDA→FA4 is implemented but has never been run (no CUDA device here). |
| FR-5 Masking | PASS | Verified directly, not only by inspection: with `v` set to 100.0 at the two padded positions and zero elsewhere, the output at all four real query positions is exactly 0 — padded keys contribute nothing. Causal masking covered by `test_causal_masking`. See REV-080 for the test-coverage gap. |
| FR-6 Hidden state output | PASS | `return_all_hidden=True` → `(B, S, D)`; `False` → pooled `x[:, -1, :]` with optional L2 normalization ([transformer.py:445-457](models/transformer.py:445)). `TestModelShapes::test_output_shapes` asserts both. |
| FR-7 Metadata integration | PARTIAL | `nn.Linear(feature_dim, hidden_dim, bias=False)` added to the input sum; `feature_dim <= 0` with `use_metadata=True` raises a clear error (REV-050). But supplying a metadata tensor to a `use_metadata=False` model raises rather than being ignored — see REV-102. |
| FR-8 Embedding normalization | PASS | `l2_normalize` is applied at the scoring sites — `score_candidates_batch` ([metrics.py:117](evaluation/metrics.py:117)) and the loss functions via `_maybe_normalize` — which is where the dot products happen. `TestEmbeddingNormalization::test_scores_bounded` asserts the [-1, 1] bound. |
| FR-9 BF16 | PASS | `self.to(dtype=resolve_dtype(config))` in `__init__`; `resolve_dtype` raises off bf16; `TestBF16` asserts both parameter and output dtype. No autocast, no GradScaler anywhere in the repo. |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 output shapes for both modes, all three sizes | PASS | `tests/test_transformer.py::TestModelShapes::test_output_shapes`, parametrised over t-small/t-base/t-large. |
| AC-02 FlexAttention self-test passes on MPS | **FAIL** | `AttentionBackend._probe("flex_attention")` returns `NotImplementedError: FlexAttention does not support backward on MPS`. Reproduced directly and in both failing gate tests. `"auto"` falls back to SDPA. |
| AC-03 FlashAttention-4 self-test passes on H100 | NOT VERIFIABLE | No CUDA device and no `flash_attn` build on this host. The varlen plumbing is tested against a stubbed kernel (`TestFlashAttentionVarlen`), which covers everything this repository owns, but the real kernel has not run. T1.1 is correctly left unchecked for this reason. |
| AC-04 causal mask + padding contribute nothing | PASS (implementation) / NOT VERIFIED (suite) | Property confirmed directly — see FR-5 above. The suite's `test_padding_masking` does not test it (REV-080). |
| AC-05 tied weights by identity | PASS | `TestTiedWeights::test_item_embedding_is_output_weight` asserts `model.item_emb_weight is model.item_embedding.weight`. |
| AC-06 metadata on/off | PARTIAL | The `use_metadata=True` half passes (`test_metadata_changes_output`). The `False` half is not tested as written and the implementation raises rather than ignoring — REV-102. |
| AC-07 normalized scores bounded to [-1, 1] | PASS | `TestEmbeddingNormalization::test_scores_bounded`. |
| AC-08 all parameters and computation BF16 | PASS | `TestBF16::test_parameters_are_bf16` and `test_output_is_bf16`; `training/pretrain.py::assert_bf16_params` re-asserts at training time. |

## Findings

```
REV-062
Severity: HIGH | Status: OPEN | Requirement: FR-4 | AC: AC-02
File: models/transformer.py:76-103 (`_resolve_backend`); configs/config.py:57
       (`require_attention_backend`); tests/test_smoke.py:30; tests/test_training.py:216
Problem: FlexAttention cannot run a backward pass on MPS in this PyTorch build, so the
  project's declared dev host cannot train on the backend `steering/architecture.md`
  specifies. BUILD's response was correct -- arm the gate in the two smoke tests, leave
  T1.1/T6.2/T6.3 unchecked, and escalate rather than relax the spec. The consequence is
  that `uv run python -m pytest` now exits with 2 failures:
    tests/test_smoke.py::TestOneBatchEndToEnd::test_full_pipeline
    tests/test_training.py::TestFewStepSmokeViaRunPretrain::test_few_steps_via_run_pretrain
  both with "Preferred attention backend 'flex_attention' cannot execute on mps:
  NotImplementedError: FlexAttention does not support backward on MPS."
Impact: `steering/testing.md`'s Phase 16 and 17 gates cannot be signed off on this host,
  so BUILD_ORDER Phase 11 cannot authorise full MovieLens training. Leaving the flag at
  its `False` default instead would let training proceed on SDPA -- exactly the silent
  fallback `code-quality.md` forbids for the MPS gate.
Correction: This is a SPEC DECISION for the owner, not a BUILD fix. Two viable
  resolutions: (a) sanction SDPA as the MPS training backend and reword FR-4/AC-02 and
  `code-quality.md`'s "Attention Backend Discipline" accordingly, keeping
  `require_attention_backend` armed for the H100 run only; or (b) pin a PyTorch build in
  which FlexAttention trains on MPS and re-run the gates. Verified for the owner's
  benefit: with the requirement neutralised at runtime (no repository file touched),
  tests/test_smoke.py and TestFewStepSmokeViaRunPretrain BOTH PASS -- so nothing beneath
  the backend choice is broken. BUILD must not edit requirements.md or steering/ to
  unblock this.
```

```
REV-080
Severity: LOW | Status: OPEN | Requirement: FR-5 | AC: AC-04
File: tests/test_transformer.py:38-64 (`test_padding_masking`),
       tests/test_transformer.py:88-105 (`test_padded_prefix_does_not_change_valid_outputs`)
Problem: Both tests compare a tensor against itself. `seq_a` and `seq_b` in
  `test_padding_masking` are both `[[0, 0] + suffix]` -- byte-identical -- despite the
  docstring saying they carry "DIFFERENT content in the two padded positions"; padding is
  always id 0, so there is no different content to carry. `torch.equal(out_a, out_b)`
  therefore holds for any deterministic model whether or not padding is masked. The same
  applies to `seq_a`/`seq_b` in `test_padded_prefix_does_not_change_valid_outputs`
  (`[[0]*6 + items]` twice). The only non-vacuous assertion in either test is that a
  prefix of REAL items changes the output -- which is true whether or not pads are masked.
  The implementation IS correct: I confirmed separately that with `v = 100.0` at the two
  padded positions, the attention output at all real query positions is exactly 0.
Impact: AC-04's padding half is unverified by the suite. Every training sequence is
  left-padded, so a regression in the mask construction at
  `CausalSelfAttention.forward:315-323` would pass CI and silently corrupt every batch.
Correction: Assert the property directly at the layer, not through two identical
  sequences. Build q/k/v with a distinctive value placed ONLY at padded key positions and
  assert the output at every real query position is zero -- this is a four-line test and
  it is what the scratch check above did. Additionally unmark T1.2 or amend its wording
  until such a test exists, since it claims "Test: padding mask zeros out padded key
  contributions".
```

```
REV-102
Severity: LOW | Status: OPEN | Requirement: FR-7 | AC: AC-06
File: models/transformer.py:421-427 (`TransformerModel.forward`);
       tests/test_transformer.py:217-231 (`test_no_metadata_ignores_tensor`)
Problem: `requirements.md`'s Edge Cases say "`use_metadata=False` with metadata tensor
  supplied: must not affect computation", and AC-06 says "with `False`, output is
  identical regardless of metadata tensor". The implementation instead raises ValueError
  ("Refusing to silently ignore it"). The test named for this criterion never supplies a
  metadata tensor -- it calls `model(seq, return_all_hidden=False)` twice with no metadata
  and asserts the two results match -- so neither the spec's behaviour nor the
  implementation's is actually covered.
Impact: Minor. The raising behaviour is arguably safer than silent ignoring and no live
  call site hits it (`evaluation/longtail.py` passes `metadata=None`, and
  `cold_start.py`'s meta_off arm passes None deliberately). But AC-06 as written is not
  met, and its test proves nothing either way.
Correction: One of two, both cheap: (a) keep the raise and ask the spec owner to reword
  the Edge Case and AC-06 to "supplying metadata to a metadata-free model is an error";
  or (b) match the spec and ignore the tensor with a warning. Either way, replace
  `test_no_metadata_ignores_tensor` with a test that actually passes a metadata tensor to
  a `use_metadata=False` model and asserts the chosen behaviour.
```

```
REV-082
Severity: INFO | Status: OPEN | Requirement: FR-2 | AC: —
File: models/transformer.py:334 (`CausalSelfAttention.forward`)
Problem: `attn_dropout` is applied to the attention OUTPUT rather than to the attention
  weights, because SDPA's `dropout_p` is not usable uniformly across the three backends.
  This is documented inline and is a reasonable backend-agnostic choice; FR-2 only
  specifies "Dropout rate: config.dropout" and does not pin the placement.
Impact: None observed. Noted so the deviation from textbook attention-weight dropout is
  on the record for the paper's method section.
```

```
REV-103
Severity: INFO | Status: OPEN | Requirement: FR-5 | AC: —
File: models/transformer.py:315-323 (`CausalSelfAttention.forward`)
Problem: The `(B, H, S, S)` additive float mask is built on every padded forward pass and
  handed to `_dispatch` alongside `padding_mask`, but only the SDPA branch reads it --
  the FlexAttention branch builds its own block mask from `padding_mask` and the FA4
  branch uses `padding_mask` for unpadding. On the two hardware-preferred backends the
  mask is constructed and discarded.
Impact: At batch_size 256, S=50, H=4 in bf16 that is ~5 MB allocated and freed per block
  per step -- negligible next to the activations, and zero on the SDPA path that actually
  runs here. Recorded as an observation, not a defect.
```

## Design Compliance

Matches `design.md`. Pre-LN ordering, the `AttentionBackend` indirection that "changes only the backend, not the mathematical operation", learnable positional embeddings (no sinusoidal), tied output weights, and the `get_device`/`resolve_dtype` discipline are all as designed. No `.cuda()` or `.to("cuda")` calls anywhere in `models/`.

Two additions beyond `design.md`, both justified in the source:
- `"sdpa"` as a first-class backend plus `require_attention_backend`. This is the mechanism that makes "no silent fallback" enforceable rather than assumed; `design.md` did not anticipate a third backend. The fallback is loud (WARNING) and recorded in `run_config.json` via `save_config_snapshot`. Acceptable.
- `_flash_attention_4_varlen`. Required to make the CUDA path work at all on left-padded data; a correct implementation of FR-5 for that backend rather than a redesign.

## Task Completion

12 of 13 tasks are marked `[x]`; T1.1 is correctly unchecked with a BUILD note explaining that AC-03 cannot be signed off without a CUDA host.

| Task | Verdict |
|---|---|
| T1.1 attention backend wrapper | Correctly `[ ]`. The MPS/SDPA halves are done and tested; the CUDA half is implemented but unverifiable here. The BUILD note is accurate. |
| T1.2 test attention backends | **False-complete.** Claims "Test: padding mask zeros out padded key contributions"; no such test exists (REV-080). The self-test and causal-mask halves are genuinely covered. |
| T2.1 embedding layer | Complete, with tied weights verified by identity. |
| T2.2 metadata projection | Complete for the `True` path including the `feature_dim <= 0` validation; the `False`-with-tensor path deviates (REV-102). |
| T3.1 transformer block | Complete. |
| T3.2 full model | Complete. |
| T4.x / T5.x tests | Complete except as noted under T1.2. `TestAttentionBackendGate` is a real test of the `require_attention_backend` mechanism in both directions, and `TestFlashAttentionVarlen` is a genuinely useful stubbed-kernel test of the repository's own varlen plumbing. |

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest -q` (full suite) | 231 passed, **2 failed**, 135 s. Both failures are this spec's REV-062 gate. |
| `uv run python -m pytest tests/test_transformer.py -q` | All pass (included above). |
| Scratch: build `TransformerModel` on MPS, forward a batch with 40 padded positions | All outputs finite, gradients finite — no fully-masked-row NaN on the SDPA path. Same on CPU. |
| Scratch: `AttentionBackend.forward` with `v = 100.0` at padded positions only | Output at all real query positions is exactly `0.0` — FR-5/AC-04 property confirmed. |
| Scratch: re-run `tests/test_smoke.py` + `TestFewStepSmokeViaRunPretrain` with `require_attention_backend` neutralised at runtime via a scratch pytest plugin (no repository file modified) | **3 passed.** Confirms REV-062 is the sole blocker and the pipeline beneath it is sound. |
| Scratch: `AttentionBackend._probe("flex_attention")` on this host | `(False, "NotImplementedError: FlexAttention does not support backward on MPS...")` |

## Final Assessment

The backbone itself is correct — masking, tied weights, normalization, dtype discipline and the varlen FA4 plumbing all check out. Two actions:

1. **REV-062 goes to the spec owner, not to BUILD.** BUILD must not disarm the gate or edit `requirements.md` / `steering/code-quality.md` to make the suite green. Record in the escalation that the underlying pipeline passes once the backend requirement is neutralised, so the decision is purely "which kernel is sanctioned on MPS".
2. **BUILD should fix REV-080** — replace the two self-comparing padding tests with a direct assertion on the layer, then either re-check T1.2 or leave it unchecked. **REV-102** is a smaller follow-up: pick a behaviour for `use_metadata=False` with a metadata tensor and write the test that actually exercises it.

Status stays `CHANGES_REQUIRED` while REV-062 is open and the suite does not pass. AC-02 is FAIL on this hardware and AC-03 is NOT VERIFIABLE until an H100 run happens.
