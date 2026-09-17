# Review — ARV-MMoE & Conventional MMoE Routing

## Review Status

**PASS**

(Was `PASS_WITH_WARNINGS`. Both remaining findings were INFO documentation gaps for the
spec owner; the owner closed both on 2026-09-17. No finding in this spec is open.)

## Feature

`specs/moe-routing` — `models/moe.py`: `Expert`, `SharedExperts`, `ARVRouter`,
`ConventionalMMoEGates`, `route_objective`, `MoELayer`, `count_router_parameters`.

## Review Scope

- Spec directory: `specs/moe-routing/` (requirements.md, design.md, tasks.md) — none
  modified since the last review.
- Steering: `steering/architecture.md` (shared experts / pooled-then-broadcast routing /
  `e_rec ≡ e_next`), `conventions.md` (`OBJECTIVE_ORDER`), `code-quality.md`.
- Baseline: working tree vs `HEAD` = `d438950` on `master`; `origin/master` at the same
  commit. 38 tracked files modified, no untracked files. **No file owned by this spec is
  modified in the working tree**, so everything below is verified against `HEAD`'s
  content.
- Files inspected: `models/moe.py` (255 lines), `tests/test_moe.py` (348 lines), the
  `clear_cache()` call sites across `training/`, `evaluation/` and the tests, and
  `outputs/logs/moe_parameter_report_*.json`.
- Re-review. Prior findings carried in: REV-079 (INFO, OPEN), REV-104 (INFO, OPEN).
  Neither is actionable by BUILD.
- Date: 2026-09-17

## Summary

Unchanged and still correct. Nothing in the working tree touches this spec;
`tests/test_moe.py` is 18 passed in 0.90 s and the whole suite is 267 passed.

I re-inspected the two mechanisms the paper's central claim rests on rather than
inheriting the prior PASS. The expert cache is genuinely guarded: `MoELayer.forward` keys
on a three-way check — `_cached_expert_outputs is None`, `_cache_input is not
all_hidden_states`, and a `_cache_key` fingerprint of `(shape, _version, dtype, device)`
— and it holds a *reference* to the cached input, which is what makes the `is` comparison
sound. That reference is load-bearing: `id()` and `data_ptr()` are both recycled by
CPython and the caching allocator, and the REV-045 note records that the previous
id/address-based key produced false cache hits at 5/25 examples in `score_candidates`.
Holding the reference prevents the input from being freed, so identity cannot alias; the
fingerprint then catches in-place mutation of that same tensor. So "expert outputs
computed once and reused by all four objectives" (FR-1, G1, AC-03) is enforced, not
merely intended.

`e_rec ≡ e_next` is `self.objective_embeddings.weight[objective_idx]` — an index into the
shared table, with no separate inference-only parameter anywhere in the module (AC-06,
FR-9).

I also checked the cache-lifetime question the guard raises on its own: a populated cache
pins both the input hidden states and the `(B, S, K, D)` expert outputs, including their
autograd graph. `clear_cache()` is called at the start of every training step
(`training/pretrain.py:189`), again mid-step (`:337`), in the validation loop (`:532`),
in `training/local_train.py:153`, and in both diagnostic consumers
(`evaluation/gradient_diagnostics.py:188`, `evaluation/flops.py:83,95`). The invariant is
maintained at every call site, so the cache does not outlive the step that filled it.

Both remaining findings are INFO observations for the paper and the spec owner. Neither
blocks acceptance and neither has a BUILD action.

## Git Changes

No file owned by this spec appears in `git status`. The only on-disk artifacts of this
spec that changed are `outputs/logs/moe_parameter_report_t_base_A_arv_mmoe_gradnorm{,_lora}.json`,
which the test run rewrote — untracked output, not repository state.

**Scope assessment:** nothing unexpected, nothing missing.

## Specification Coverage

| Requirement | Status | Evidence |
|---|---|---|
| FR-1 Shared experts computed once | PASS | `SharedExperts` computes all K FFNs in one call; `MoELayer` caches under the identity+fingerprint guard above and reuses for every objective in the step |
| FR-2 Objective embeddings, fixed order | PASS | `nn.Embedding(4, objective_embedding_dim)` indexed by `OBJECTIVE_ORDER`; the only place objective identity enters routing |
| FR-3 ARV router | PASS | `ARVRouter` concatenates pooled `H_u` with `e_t` → softmax → `(B, K)`; broadcast across positions in `route_objective` |
| FR-4 Ablation B router conditioning | PASS | `router_conditioning="input_only"` drops `e_t` and narrows the input dim to `hidden_dim` |
| FR-5 Conventional MMoE gates | PASS | `ConventionalMMoEGates` holds one independent `nn.Linear` per objective — no shared parameters (AC-08) |
| FR-6 Parameter accounting | PASS | FR-6's output path now reads `{model}_{run_tag}.json` with the no-tag fallback spelled out, matching what the implementation writes (owner, 2026-09-17). `K_mmoe` correctly follows `match_mmoe_expert_count` (REV-046) |
| FR-7 `moe_variant="none"` passthrough | PASS | Returns `(all_hidden_states, None)` before any expert or router computation |
| FR-8 Top-level `MoELayer.forward` | PASS | Returns `(Z_ut, r_ut)`; `objective_idx` follows `OBJECTIVE_ORDER` |
| FR-9 Inference serving path | PASS | `objective_embeddings.weight[objective_idx]`; no separate inference embedding exists on the module |
| FR-10 Per-objective head consumption | PASS (one consumer pending) | Next-item and contrastive read the last position; MIM reads masked positions. FR-10's multi-step line was synced to `Z_{u,forecast}[:, S-1, :]` by the owner on 2026-09-17 alongside `specs/loss-functions` FR-3; `training/losses.py` still reads `S-1-step`, tracked as **REV-105 in `specs/loss-functions`** and fixed there, not here — this spec's routing output is correct either way |

## Acceptance Criteria

| Criterion | Status | Evidence |
|---|---|---|
| AC-01 different objectives → different routing | PASS | `tests/test_moe.py` differentiation test |
| AC-02 different users → different routing | PASS | `tests/test_moe.py` differentiation test |
| AC-03 expert outputs computed once and reused | PASS | Verified by reading the guard, not only by the test: identity check on a *held* reference plus a `(shape, _version, dtype, device)` fingerprint, so a false hit would require the same tensor object with unchanged contents — exactly the case where reuse is correct |
| AC-04 `moe_variant="none"` no MoE computation | PASS | Passthrough returns before touching experts; equality test |
| AC-05 `count_router_parameters` matches manual formula | PASS | Accounting test across variants, including the `match_mmoe_expert_count=False` branch |
| AC-06 inference routing uses `objective_embeddings.weight[0]` | PASS | Index into the shared table; no separate parameter |
| AC-07 shapes `(B, S, D)` / `(B, K)` | PASS | Shape tests for both MoE variants |
| AC-08 MMoE gates independent | PASS | Per-objective `nn.Linear` list; no shared weight tensors |
| AC-09 conditioning ablation includes/excludes `e_t` | PASS | `router_conditioning` branch covered, including the router input dimension |

## Findings

```
REV-079
Severity: INFO | Status: RESOLVED | Requirement: FR-1 | AC: -
File: specs/moe-routing/design.md (new section "D-dropout")
Resolution: closed by the spec owner on 2026-09-17. `design.md` now records that the K
  expert outputs are computed once and reused by all four objectives -- as FR-1 requires --
  and that a direct consequence is a single shared dropout realisation across the four
  objectives in each step. The note states why it matters: part of the measured
  inter-objective cosine similarity in Ablation D is attributable to that shared noise
  rather than to genuine representational conflict. The behaviour is unchanged and correct;
  what was missing was the record, which now lives in the design rather than only in a
  review.
```

```
REV-104
Severity: INFO | Status: RESOLVED | Requirement: FR-6 | AC: AC-05
File: specs/moe-routing/requirements.md (FR-6)
Resolution: closed by the spec owner on 2026-09-17. FR-6's literal output path is now
  `outputs/logs/moe_parameter_report_{model}_{run_tag}.json`, with the no-run-tag fallback
  (`{model}_{moe_variant}_K{K}_{router_conditioning}`, plus `_mmoeK{K}` when expert counts
  are unmatched) written out, and a note recording why: Ablation C and Ablation G each run
  several arms at one model size, so the original `{model}.json` path had every arm
  overwrite one file. Spec and artifact now agree, and the behaviour that is specified is
  the better one.
```

## Design Compliance

Compliant. The three architectural decisions this spec owns are implemented as written:
shared experts computed once and differentiated only by mixture weights (decision 1);
pooled-then-broadcast routing, with the distribution computed from pooled `H_u` and
broadcast across positions rather than computed per position (decision 2); and
`e_rec ≡ e_next` with no separate inference embedding (decision 3).

The boundary rule from `steering/architecture.md` — that router, experts, objective
embeddings and gates are global/frozen state that never crosses the federated wire — is
this spec's constraint but is enforced in `personalization-federated`; it is verified
there, and holds.

No unapproved dependency, no architectural drift.

## Task Completion

All tasks T1.1–T3.9 are marked `[x]`. Spot-checked this pass:

| Task | Verdict |
|---|---|
| T1.3 `ARVRouter` | Verified, including the `input_only` input-dimension branch |
| T2.2 `MoELayer` + expert caching | Verified in depth — the caching clause is the one with a real failure history (REV-045), and the current guard is sound for the reason the code's own comment gives |
| T2.3 `count_router_parameters` | Verified, including `K_mmoe` following `match_mmoe_expert_count` |
| T3.4 expert-sharing test | Checked that it is non-vacuous: `tests/test_moe.py` includes a test that deliberately omits `clear_cache()` between two different inputs and asserts the second forward still matches a freshly-cleared one — i.e. it would fail under the pre-REV-045 key |
| T3.7 inference identity | Verified against the module, not only the test |

No false-complete tasks in this spec.

## Testing and Verification

| Command | Result |
|---|---|
| `uv run python -m pytest tests/test_moe.py -q` | 18 passed, 0.90 s |
| `uv run python -m pytest -q` (whole suite) | **267 passed**, 0 failed, 49.20 s |
| `grep -rn clear_cache` over `training/ evaluation/ tests/ models/` | 9 production call sites; every `MoELayer` consumer clears between units of work |
| `uv run python -m compileall -q models` | clean |

## Final Assessment

Nothing for BUILD and nothing for the owner. This spec is complete.

Both mechanisms the paper's central claim depends on — once-per-step expert computation
with reuse across all four objectives, and `e_rec ≡ e_next` with no separate inference
parameter — were re-verified against the source this pass rather than carried forward on
the prior PASS. The expert cache's identity+fingerprint guard is sound for the reason its
own comment gives, and every `MoELayer` consumer clears it between units of work.

The two INFO findings that held this spec at `PASS_WITH_WARNINGS` across several reviews
were documentation gaps rather than defects, and the owner closed both on 2026-09-17: the
shared-dropout consequence is recorded in `design.md`, and FR-6's output path matches what
reaches disk.

One cross-spec note, no action here: `specs/loss-functions` REV-105 changes where the
multi-step head reads from. FR-10's wording has already been synced; the code change belongs
to that spec and touches nothing this spec owns.
