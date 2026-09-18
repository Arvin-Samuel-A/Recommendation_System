# Loss Functions & GradNorm — Tasks

Ordered tasks. Depends on transformer-backbone and moe-routing (for integration), and data-pipeline T6.1 (augmentation).

## Phase 1: Core Loss Functions

- [x] **T1.1: Implement `SampledSoftmaxLoss` in `training/losses.py`** — Tied weights. Forward: `(user_emb (B,D), pos_ids (B,), neg_ids (B,K), item_emb_weight (V,D))`. Compute positive/negative scores via dot product. Concatenate `[pos_score, neg_scores]`. Cross-entropy with label 0. (FR-1)

- [x] **T1.2: Implement `MaskedItemLoss` in `training/losses.py`** — Forward: `(hidden_states (B,S,D), mask_positions (B,S) bool, true_item_ids (B,S), item_emb_weight (V,D))`. Extract masked hidden states, compute full-vocab logits via tied weights, cross-entropy. Return 0.0 tensor if zero masked positions. (FR-2)

- [x] **T1.3: Implement `MultiStepLoss` in `training/losses.py`** — `nn.ModuleList` of k `nn.Linear(D, D, bias=False)` step projections. Forward: `(all_hidden_states (B,S,D), targets_k (B,k), item_emb_weight (V,D))`. For step i: read position `S-1-i` (left-padding), project, compute logits, cross-entropy on valid targets only. Return mean loss. (FR-3) **The position rule here is superseded by T3.7** (FR-3/AC-03 amendment, owner 2026-09-17): every step reads `S-1`. The projections, valid-target masking and mean-over-valid-steps reduction are unchanged.

- [x] **T1.4: Implement `InfoNCELoss` in `training/losses.py`** — Forward: `(z_orig (B,D), z_aug (B,D))` both L2-normalized. Cat → `(2B, D)`. Similarity matrix `(2B, 2B)` with temperature. Mask self-similarity to -1e9. Labels: positive pairs at (i, i+B) and (i+B, i). Cross-entropy. (FR-4)

## Phase 2: Masking Integration

- [x] **T2.1: Implement input masking function** — `apply_masking(seq, mask_prob, mask_id, rng)`: for each non-padding position, replace with MASK_ID with probability `mask_prob`. Return `(masked_seq, mask_positions, true_item_ids)`. MASK_ID = num_items + 1 (never num_items). Keep at least 1 unmasked real item per sequence. (FR-2)

## Phase 3: GradNorm

- [x] **T3.1: Implement `GradNormController` in `training/gradnorm.py`** — Maintain `log_lambdas` as nn.Parameter for OBJECTIVE_ORDER. Separate Adam optimizer over log_lambdas with `gradnorm_lr`. `lambda_t = exp(log_lambda_t)`. Store initial losses on first valid step. Implement `weighted_sum(active_losses)` returning weighted total. (FR-5)

- [x] **T3.2: Implement GradNorm update step** — For each active objective: compute `G_t = ||grad(λ_t * L_t)||` w.r.t. anchor params. Compute `Lhat_t = L_t / L_t(0)`, `r_t = Lhat_t / mean(Lhat)`, `G*_t = mean(G_t) * r_t^alpha`. GradNorm loss = `sum |G_t - detach(G*_t)|`. Optimize log_lambdas. Renormalize: `sum(active lambdas) == num_active`. (FR-5, AC-05) **The `G_t` expression here is superseded by T3.6** (FR-5 amendment + design D7): the factored form is what is implemented.

- [x] **T3.3: Implement GradNorm anchor resolution** — `get_anchor_layer_params(model, config)`: resolve `"last_shared_transformer"` to the parameters of the final Transformer block (before optional MoE). Must work for all variants including `moe_variant="none"`. (FR-5, AC-07)

- [x] **T3.4: Implement GradNorm state_dict / load_state_dict** — For checkpoint save/load compatibility. Include log_lambdas, optimizer state, initial losses. (FR-5)

- [x] **T3.5: Implement static weight fallback** — `static_weighted_sum(active_losses, config)`: `L_next + masked_lm_weight * L_mim + multi_step_weight * L_ms + contrastive_weight * L_cl`. Only active losses contribute. (FR-6)

- [x] **T3.6: Implement factored `G_t` (design D7, REV-120)** — Build `G_t = λ_t · ‖∇_W L_t‖` with the norm taken at `create_graph=False` and detached before being scaled by λ_t, so no second-order differentiation occurs anywhere in the controller. Take the λ gradient with `torch.autograd.grad(gradnorm_loss, [log_lambdas])` and `.detach()` it before assigning to `log_lambdas.grad`. Delete `probe_double_backward`, `_raise_double_backward` and `DoubleBackwardUnsupportedError`, and the probe call in `run_pretrain`. Leave the post-step renormalization unchanged. (FR-5, AC-10, AC-11)

- [x] **T3.7: Read position `S-1` at every multi-step step (REV-105, amended FR-3/AC-03)** — The owner resolved the multi-step horizon question on 2026-09-17: FR-3 and AC-03 now require **every** step to read position `S-1`, so all k per-step projections receive one fixed context representation and the horizons are uniform at 2, 3, ..., k+1 (horizon 1 belongs to the next-item objective and is deliberately not duplicated). In `MultiStepLoss.forward` replace `pos = S - 1 - step` with `pos = S - 1`, leaving the per-step projections, the valid-target masking and the mean-over-valid-steps reduction exactly as they are. Invert the position test, which asserts the superseded rule. (FR-3, AC-03)
  - BUILD note: done and verified. `training/losses.py` — `pos = S - 1` is now hoisted out
    of the step loop; nothing else in the loss changed. The old rule paired step s with
    context `items[:i-s]` and target `items[i+1+s]` (horizons 2, 4, 6); the amended rule
    gives 2, 3, 4. Neither leaks — every target is strictly in the future of the position
    it is predicted from — so what changed is which experiment runs.
    Test: `TestMultiStepLoss::test_correct_position_usage` asserted only `torch.isfinite`,
    so it held under BOTH rules and could not distinguish them; that is why the mismatch
    survived three reviews as an INFO note. Replaced with
    `test_every_step_reads_position_S_minus_1`, which asserts the property true of `S-1`
    and false of `S-1-step`: scrambling every position `< S-1` leaves the loss **exactly**
    unchanged, plus a non-vacuity assertion that changing position `S-1` does move it.
    Added `test_all_k_steps_contribute_distinct_projections` so reading one position is not
    mistaken for collapsing the k steps into one prediction.
    Non-vacuity verified, not assumed: `pos = S - 1 - step` was reinstated in a scratch
    copy and `test_every_step_reads_position_S_minus_1` FAILS on it (5.0938 vs 3.2969) and
    passes on the fix. `tests/test_losses.py` 39 passed.
    DOWNSTREAM — **RESOLVED 2026-09-18; this bullet previously stood as one of three
    contradictory accounts of one checkpoint (REV-129, REV-131) and is rewritten to point
    at the single authoritative record rather than to add a fourth.**
    What it used to claim: that `outputs/checkpoints/t_small/best.pt` "was trained under
    the old objective (`train_loss_multistep=9.912500`)". Two problems with that claim, both
    found by REVIEW: the log on disk read 9.903438, not 9.912500 — the figure was
    transcribed from the console of a run that was later superseded (REV-130) — and the
    underlying provenance question was **not decidable** from the repository at all, because
    commit time is not edit time and the checkpoint stored only `config`, which this change
    left byte-identical.
    How it is resolved: not by settling the argument but by removing its cause. The gate was
    re-run on 2026-09-18 against current source, and the checkpoint now on disk was written
    at 15:29 that day by a run whose source is recorded IN the file
    (`source_revision.source_hash = "3a59e7cb70a8be69"`, `dirty: true`). Its
    `train_loss_multistep=9.905000`, read back from `outputs/logs/pretrain_t_small.log`.
    **The authoritative record is `specs/training-pipeline` T6.3.** Do not re-derive the
    provenance of this checkpoint from this note or from
    `specs/personalization-federated` T6.1; both now defer to T6.3.
    The durable fix is `specs/training-pipeline` T6.5 (AC-09): every checkpoint carries
    `source_revision` and every derived artifact echoes it, so this question is a lookup.

- [x] **T3.8: Reword the `TestGradNormGradientIsolation` docstring (REV-124)** — The class docstring justified the REV-057 isolation requirement in terms of `torch.autograd.grad(..., create_graph=True)` and the second-order graph it built, both deleted by T3.6. The isolation requirement itself survives unchanged; only the stated reason was stale, and it was the one place left in the repository asserting the second-order form still exists — one file away from the AC-10 test that asserts it does not. Reword on first-order grounds; no assertion change. (FR-5)
  - BUILD note: done. The docstring now says `update()` runs between `L_total.backward()`
    and `optimizer.step()`, so a plain `.backward()` on `gradnorm_loss` would accumulate
    into the model `.grad` buffers the main AdamW is about to step on. No assertion touched.

## Phase 4: Testing

- [x] **T4.1: Test sampled softmax** — Toy data: known positive (high dot product) and negatives (low) → verify loss is low. Reverse → loss is high. Verify loss decreases with optimizer steps. (AC-01)

- [x] **T4.2: Test masked item loss** — Mask specific known positions → verify loss only uses those positions. Mask none → loss is 0.0. Mask all real positions → all contribute. (AC-02)

- [x] **T4.3: Test multi-step loss** — Inject known values at positions S-1, S-2, S-3 → verify each step reads the correct position. Verify with known targets that loss decreases. (AC-03) **Superseded by T3.7:** "the correct position" is now `S-1` for every step, so the per-step-position injection this describes no longer distinguishes anything. The replacement test asserts the property only the amended rule has.

- [x] **T4.4: Test InfoNCE loss** — Identical pairs → loss should be low. Random pairs → loss should be higher. Verify similarity matrix shape is `(2B, 2B)` with self-similarity masked. (AC-04)

- [x] **T4.5: Test GradNorm lambda constraint** — After update, verify `sum(active_lambdas) == num_active` within tolerance 1e-4. Test with 2, 3, 4 active objectives. (AC-05)

- [x] **T4.6: Test GradNorm active-only operation** — Set contrastive inactive → verify it's excluded from gradient computation (no autograd.grad call for it). (AC-06)

- [x] **T4.7: Test GradNorm anchor resolution** — Verify anchor params exist for `moe_variant="none"`, `"mmoe"`, and `"arv_mmoe"`. Verify they come from the last Transformer block. (AC-07)

- [x] **T4.8: Test static weight fallback** — Manually compute weighted sum → verify `static_weighted_sum` matches. (AC-08)

- [x] **T4.9: Finite assertion tests** — Verify that NaN/Inf losses trigger assertion failures. (AC-09)

- [x] **T4.10: Test λ-gradient equivalence** — The shipped `log_lambdas.grad` matches the closed form `sign(G_t - G*_t) · G_t` recomputed independently, on all-active, partially-active and the `G_t == G*_t` boundary (one active objective → exactly zero); inactive objectives receive no gradient; `log_lambdas.grad.requires_grad is False` and it carries no `grad_fn`. (AC-11)

- [x] **T4.11: Test first-order-only GradNorm** — `create_graph=True` appears nowhere in `training/gradnorm.py`, and a full `update()` completes with `config.device="cpu"`, whose SDPA kernel implements only a first-order backward. (AC-10)
