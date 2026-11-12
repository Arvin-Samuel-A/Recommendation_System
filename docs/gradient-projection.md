# Next-item-protected gradient projection (deferred experiment)

**Status:** implemented and tested, not yet run at scale. Commit `f75535f` (branch `claude/run11-followup-36a3c6`) has the code; no ablation arm uses it yet.

## The idea

Next-item prediction is the objective the model is evaluated on. Masked-item, multi-step and contrastive are auxiliary. Instead of weighting the four objectives with GradNorm, we keep fixed weights and **remove only the part of each auxiliary gradient that works against next-item**:

- Each objective gets its own gradient: one backward pass per objective.
- For each auxiliary gradient `g_a`, if `g_a · g_next < 0` (a conflict), it becomes `g_a − (g_a · g_next / ‖g_next‖²) · g_next`. Otherwise it is kept as is.
- The update is `g_next + Σ g_a`.

This is an asymmetric version of PCGrad (Yu et al., 2020):
- Next-item's gradient is never changed.
- Every auxiliary is projected against the *original* next-item gradient, so the order of the objectives doesn't matter.
- Auxiliaries are never projected against each other.

The dot products run over all trainable parameters. A parameter that only one objective reaches (an objective-specific gate, say) is zero in the other gradients, so the projection effectively acts on the shared parameters, as in PCGrad.

## Why

Run11 and the β sweep (20-epoch schedule, scored at the best epoch within 10) showed:

- **GradNorm, as originally used, switched next-item off.** In A:arv_mmoe_gradnorm, λ_next_item fell from 0.62 to exactly 0 by epoch 12. Validation NDCG@10 peaked at 0.682 (epoch 3) and then declined.
- **Equal weights also decline after an early peak.** arv_mmoe peaked at 0.688 (epoch 5) and fell to 0.669 by epoch 16. So the problem isn't only GradNorm: auxiliary gradients with as much weight as next-item hurt it later in training.
- **Fixing λ_next_item at 1 helps.** With GradNorm balancing only the auxiliaries under a budget β, the decline is much smaller. β = 0.3 reached 0.694 validation / 0.656 test, against the baseline's 0.690 / 0.657.

Projection attacks the same problem directly: auxiliaries keep their full weight where they agree with next-item, and lose only the conflicting part. Weighting can't do that, because it scales the helpful and the harmful parts of a gradient together.

## Evidence so far

Capped local smoke run: 150 steps, A:arv_mmoe settings, 10% data, τ = 0.07, contrastive every step, MPS.

| | arv_mmoe (equal weights) | + projection |
|---|---|---|
| Next-item loss, first → last 10% of steps | 4.76 → 3.64 | 4.75 → 3.63 |
| Step time | 0.47 s | 0.79 s (**1.67×**) |

Conflicts with next-item over those steps:

| Auxiliary | Mean cosine with g_next | Steps projected |
|---|---|---|
| masked-item | +0.049 | 15% |
| multi-step | +0.035 | 24% |
| contrastive | −0.029 | **59%** |

Contrastive conflicts with next-item on most steps, which fits GradNorm driving λ_contrastive to about 0. These are early-training numbers only: 150 steps, one seed. Never base a design decision on proxy-scale gradient statistics (see docs/ablation-run-errors-and-fixes.md §6.2).

## How to run it

- **Config:** `grad_projection = "next_item"`, which requires `use_gradnorm = False` (validated). It uses the static weights `masked_lm_weight`, `multi_step_weight` and `contrastive_weight`, all 1.0 by default. fp16 loss scaling isn't supported. The default `""` leaves every training fingerprint unchanged.
- **Code:** `training/grad_projection.py` and `train_one_step` in `training/pretrain.py`. Tests are in `tests/test_grad_projection.py`.
- **Logging:** each epoch line gets `proj_cos_<aux>` (mean cosine before projection) and `proj_frac_<aux>` (fraction of steps projected).
- **Adding the arm:** add it to study A in `training/ablation_runner.py`, after `arv_mmoe_gradnorm`:
  ```python
  {"label": "arv_mmoe_projection", "use_masked_lm": True, "use_multi_step": True, "use_contrastive": True, "moe_variant": "arv_mmoe", "use_gradnorm": False, "grad_projection": "next_item"},
  ```
  Then run `main.py --stage ablation --study A --arms arv_mmoe_projection` with the same flags as the other arms (`--epochs`, `--logit_temperature 0.07 --contrastive_interval 1`, ...). Arms whose fingerprints didn't change are reused.
- **Cost:** about 1.7× the arv_mmoe step time, so roughly 1.7 h of training for a 10-epoch schedule on one L4 at 10% users (arv_mmoe takes about 1.0 h).

## Planned comparison

All arms on the same schedule and seed, compared against the popularity reference. Test NDCG@10 at the best validation epoch.

| Arm | What it isolates |
|---|---|
| single_obj_baseline | next-item alone |
| arv_mmoe | equal weights, no balancing |
| arv_mmoe_gradnorm (next-item primary, chosen β) | weighting |
| **arv_mmoe_projection** | conflict removal at equal weights |

Possible follow-ups:
- Projection with the β budget as static weights (0.1 each), which compares projection with GradNorm at the same total weight.
- Projection combined with next-item-primary GradNorm.
- Symmetric PCGrad as a reference.
