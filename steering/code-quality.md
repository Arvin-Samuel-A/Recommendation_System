# Code Quality

## Dtype Discipline

- All model and data computation uses `torch.bfloat16`. No FP16, no autocast, no GradScaler.
- `resolve_dtype(config)` must raise if dtype is not `"bf16"`.
- Never silently change dtype if a numerical issue occurs — fail explicitly.

## Device Handling

- Always use `get_device(config)` → `torch.device`. Never call `.cuda()` or `.to("cuda")` directly.
- `config.device = "auto"` resolves CUDA > MPS > CPU.
- Device synchronization is mandatory before latency timing: `torch.mps.synchronize()` on MPS, `torch.cuda.synchronize(device)` on CUDA.

## Attention Backend Discipline

- MPS/CPU → SDPA (`F.scaled_dot_product_attention`). CUDA/H100 → FlashAttention-4. `"auto"` resolves by hardware.
- SDPA is the sanctioned local/dev kernel, not a fallback. FlexAttention is not a supported backend in this project: it cannot run a backward pass on MPS in the pinned PyTorch build, so it can never satisfy the training gates on the dev host.
- Each backend must pass a startup self-test — forward **and** backward — before training.
- Final H100 run must fail fast if FA4 cannot initialize — no silent fallback.
- MPS smoke test must fail fast if SDPA cannot execute — no silent fallback.
- "No silent fallback" is about *honesty of the manifest*, not about kernel preference: the backend that actually ran is the backend that gets recorded, and `require_attention_backend=True` turns any substitution into a hard failure.
- Record PyTorch version, device, dtype, and resolved attention backend in every run manifest.

## Artifact Provenance

**Owner decision (2026-09-18).** Every artifact must be able to say which code produced it, from
the file itself and without reference to prose.

- Every checkpoint records a `source_revision`: the git commit when the tree is clean; otherwise
  the commit plus a content hash over the covered sources and a `dirty: true` marker. A run from a
  dirty tree is recorded as such, never silently as its nearest commit.
- Every artifact derived from a checkpoint — metrics, diagnostics, reports, round logs — echoes
  that checkpoint's `source_revision` verbatim.
- A verification note that quotes a number quotes it by reading the artifact back after the run,
  never from console output. A run that is later superseded then leaves a record that is visibly
  wrong rather than plausibly right.

### Coverage: a property, not a list (amended 2026-09-18)

**The governing rule is a property: `source_revision` must change whenever any code that can
change a published artifact changes.** Everything below is how that property is realised; if the
two ever conflict, the property wins and the realisation is the thing that is wrong.

Coverage is therefore defined by **exclusion, not enumeration**: every first-party `.py` file in
the repository is covered, except

- `tests/` — a test cannot change a published artifact;
- `.venv/`, `__pycache__/`, and anything not tracked by git;
- `data/raw/` and `data/processed/` — data, not code.

An allowlist was tried first and failed within one round: it named `models/`, `training/`,
`data/`, `configs/` and thereby omitted `evaluation/`, which is where every derived artifact in
this project is actually produced. Measured at the time: editing `evaluation/longtail.py` left
`source_revision` byte-identical. A denylist covers new code by default, so drift now requires
someone to write an explicit exclusion rather than merely to forget one.

Two constraints that follow, and are not optional:

- **The dirtiness check and the content hash use the same coverage rule**, so the two halves of a
  revision can never disagree about what "this code" means. A change to a covered file that moves
  neither the hash nor the `dirty` flag is a defect in the rule's implementation.
- **Coverage completeness is tested, not assumed.** A test asserts that the set of files actually
  hashed equals the set this rule defines. Asserting only that *some* change moves the hash proves
  the mechanism works, not that it covers what it must — that is precisely the gap that let the
  allowlist ship green.

**Why this exists.** Commit time is not edit time, and a change landing between a run and its
write-up leaves no trace in the artifact. That happened: three task notes gave three different
accounts of which objective trained a single checkpoint, none matched the files, and the question
could not be settled from the repository at all — the checkpoint stores `config`, which the change
in question did not touch. Provenance belongs in the artifact, not in prose about the artifact.

## Configuration as Single Source of Truth

- `configs/config.py` (`Config` dataclass) holds all hyperparameters.
- No hardcoded hyperparameters in model/training/evaluation code.
- CLI flags override config values; log which values were overridden and original vs. new.

## Numerical Safety

- Assert finite losses, finite GradNorm loss, and finite active lambdas on every training step.
- Gradient clipping: `clip_grad_norm_(model.parameters(), max_norm=1.0)` on every step.
- `optimizer.zero_grad(set_to_none=True)` — use `set_to_none` for memory efficiency.

## Error Handling

- Missing Kaggle credentials with no local dataset → fail with clear auth message; never substitute another dataset.
- Missing checkpoint at evaluate stage → print skip message with remedy command, return — never raise exception.
- Negative pool exhaustion → fail with clear dataset/configuration error.
- `contrastive_interval = 0` is invalid; use `use_contrastive = False` to disable.

## Anti-Patterns (Forbidden)

- No external recommendation libraries (RecBole, Surprise, pytorch-fm, etc.)
- No sinusoidal positional embeddings (learnable `nn.Embedding` only)
- No GradScaler or FP16 mixed-precision policy
- No `num_items` as mask token (that collides with the last real item ID)
- No test data for model selection, threshold tuning, or diagnostic decisions
- No silent dtype/backend substitution
- No worker initialization when `num_workers=0`
- No per-epoch negative pool regeneration
- No cross-model state sharing between independently trained models
