# Future scope

Two experiments were left out of the run11 re-run (master `4247429`, 10% users, t-base, seed 42, τ = 0.07, contrastive every step) to keep its cost down.

## 1. Sequence length 200 (study H, arm H2_seq200)

**Why it was dropped.** At τ = 0.07 with contrastive every step, seq200 takes **770 ms/step**, which is **13.75 h** for 20 epochs on one L4 and about 11.9 GB of memory. That one arm is 13.75 of the 59 GPU-hours, and it made the makespan about 16.3 h (about $75) instead of 13.1 h (about $60). Run11 still covers the sequence-length study with seq25, seq50 (the base config) and seq100.

**How to add it later** (only seq200 needs training):
1. Package from master with the default grid, i.e. **without** `--sequence-length-grid`, so study H's arms are seq25/50/100/200. Keep the same flags otherwise: `--logit-temperature 0.07 --contrastive-interval 1`.
2. Run only H2_seq200 on one L4, in run11's directory, so the existing arms are reused:
   `main.py --stage ablation --study H --arms H2_seq200 <run11 flags without --sequence_length_grid>`
   seq25/50/100 have the same training fingerprints as in run11, so they are reused rather than retrained.
3. Then run `--stage ablation_collect --study H` with the full grid, to rebuild study H's JSON and plots with all four points.
- **Cost.** About 13.75 h on one L4. A cheaper single-GPU instance can be used, or the shared H100, where flash-attention-3 should be much faster.
- **The data already supports it.** The 10% data is stored at width 200 (`stored_seq_length: 200`).

## 2. Temperature (τ) sensitivity sweep

**Why.** Run11 fixes τ = 0.07 for all four objectives, matching the contrastive temperature common in the literature (MoCo uses 0.07; CLIP initialises its learnable temperature at 0.07). Reviewers will expect evidence that the results aren't fragile around that value.

**What we know so far.**
- τ = 1 is ruled out by arithmetic. With cosine logits, the lowest possible loss is 2.68 for next-item (chance 4.62) and 7.51 for masked-item and multi-step (chance 9.51).
- AWS smoke test, 2 epochs, test NDCG@10. Popularity scores 0.3736:

  | Arm | τ = 1 | τ = 0.07 |
  |---|---|---|
  | single_obj_baseline | 0.375 | 0.499 |
  | arv_mmoe_gradnorm | 0.375 | 0.610–0.613 |

- In a short local run the learnable scale drifted from 14.3 to 22 (τ ≈ 0.045), which hints that the best τ may be slightly below 0.07.

**Proposed sweep.**
- **Values:** τ ∈ {0.05, 0.07, 0.1, 0.2}, plus `--learnable_logit_scale` initialised at 0.07.
- **Models:** single_obj_baseline **and** arv_mmoe_gradnorm. Tuning only the proposed model would handicap the baseline.
- **Settings:** run11's flags (`--contrastive_interval 1`, bf16 mixed precision, batch 256), 10% users.
- **Selection: choose τ on validation NDCG, never on test.** Report test only for the chosen value.
- **Stage 1 (screen):** 2 epochs per arm. That's 10 short arms on 4 L4s, about 30 min and about $3.
- **Stage 2 (confirm):** full 20 epochs for the best two τ values, if the screen leaves them within noise.
- **Seeds:** use at least 2–3 seeds for any claim that results are "insensitive within noise".

**What it gives the paper.** A sensitivity table in the appendix, and a sentence of the form: *"We use τ = 0.07 for all objectives, matching the contrastive temperature (He et al., 2020). Results vary by at most Δ NDCG@10 for τ ∈ [0.05, 0.2] (Appendix X)."* If another value wins clearly on validation, switch to it and re-run the affected studies.

## Evidence behind the ideas below (as of 2026-10-17)

All single seed, 10% users, test NDCG@10 at the best validation epoch. Popularity scores 0.374.

**run11** (20-epoch schedule):

| Arm | Test NDCG@10 | Note |
|---|---|---|
| next-item-only baseline | 0.657 | best epoch 18 |
| conv_mmoe | 0.651 | |
| arv_mmoe (equal weights) | 0.649 | |
| arv_mmoe_gradnorm (original GradNorm) | 0.644 | λ_next_item fell to exactly 0 from epoch 12 |
| 4obj_no_routing | 0.610 | |

**β sweep** (next-item-primary GradNorm, 20-epoch schedule, scored at the best epoch ≤ 9):

| β | Val | Test |
|---|---|---|
| 0.3 | 0.6939 | 0.6562 |
| 1 | 0.6912 | 0.6496 |
| 0.1 | 0.6883 | 0.6524 |
| 3 | 0.6879 | 0.6501 |

**run12** (the 10-epoch A–H with β = 0.3) was launched 2026-10-17. Re-check every idea below against its numbers.

What this suggests:
1. No multi-objective arm has beaten next-item alone on test.
2. Multi-objective arms peak early and decline when the auxiliaries carry as much weight as next-item.
3. Routing vs no routing is a large gap (about 0.04); ARV vs MMoE routing is noise.

## 3. Next-item-protected gradient projection

Each auxiliary gradient that conflicts with next-item's (negative dot product) loses its component along next-item's gradient, and next-item's gradient is never changed: an asymmetric PCGrad. It is already implemented (`grad_projection = "next_item"`, commit `f75535f`) and is **not used by any arm**. The idea, evidence, run instructions and planned comparison are in **docs/gradient-projection.md**.

- **Why it might help:** it targets the post-peak decline directly. It removes only the conflicting part of each auxiliary gradient, while weighting scales the helpful and harmful parts together.
- **Why the effect may be small:**
  - Measured conflicts are weak: mean cosines +0.05 (masked-item), +0.035 (multi-step) and −0.03 (contrastive) over a 150-step smoke run, so projection removes only a few percent of each auxiliary gradient.
  - Equal-weight training declines anyway, which points to interference that first-order conflict doesn't capture: Adam's per-coordinate scaling, and gradient clipping of the total norm.
  - Comparison studies (Kurin et al. 2022; Xin et al. 2022) find PCGrad-style methods often don't beat well-tuned fixed weights.
- **Test:** one arm compared with arv_mmoe (equal weights) and next-item-primary GradNorm at β = 0.3, on the same schedule. The decisive check: does `proj_frac` rise over the epochs while validation stops declining? Cost: about 1.7× the arv_mmoe step time.

## 4. One objective per expert (fixed routing)

Expert k serves objective k through a fixed one-hot gate: no learned routing. We have K = 4 experts and 4 objectives, so this is parameter-matched with MMoE and ARV.

- **Why:**
  - With `moe_variant = "none"` every objective reads the same final hidden states. With MMoE or ARV, each gets its own mixture of experts.
  - The large routing vs no-routing gap, next to the ARV ≈ MMoE tie, suggests the benefit may come entirely from **giving each objective its own output transform**, not from learned mixing. Fixed experts test that directly.
  - Next-item's own expert would get no auxiliary gradients at all, which is a structural version of what projection does.
- **What it decides:**
  - If fixed ≈ MMoE ≈ ARV, the benefit is task decoupling, and learned routing (including ARV) adds nothing. That's an honest finding, but it weakens ARV as the contribution.
  - If ARV clearly beats fixed experts, that's the strongest evidence for ARV.
- **Before building:** check run12's routing statistics (studies B, C, G). If ARV's gates are already close to one objective per expert, this arm ≈ what we already have.
- **Cost:** a fixed-gate `moe_variant` (about 20 lines) and one arm in study C.

## 5. Recurrent Looped Transformer for next-item only (per-token recurrence)

This is based on the Recurrent Looped Transformer (RLT; Zhang et al., arXiv 2610.07591).
- The layers are split into a parallel causal encoder and a recurrent decoder.
- At each item, a gated merge `u_t = e_t + α·g_t ⊙ W_s·RMSNorm(s_{t−1})` feeds **the previous item's final decoder state** back in.
- The decoder cross-attends to the encoder's memory and uses sliding-window self-attention.
- The recurrence runs **across tokens**, not over depth.
- Variants: RLT-1 (feedback at every token), RLT-2 (once per chunk of B tokens), RLT-0 (no feedback, the control).

- **Why it fits recommendation:**
  - A user's preferences are a state updated with every interaction, the "update at every input" structure RLT is built for.
  - It allows **streaming updates at serving time**: a new interaction updates the state at a fixed per-token cost, without re-encoding the history.
  - It's consistent with next-item alone being the strongest arm so far.
- **Why it may not transfer:**
  - The paper's gains are on algorithmic state tracking (parity, S5 permutations, modular arithmetic) that needs exact composition over *longer sequences than seen in training*. There are no results on natural data.
  - Recommendation trains and tests at the same lengths, and self-attention recommenders usually beat recurrent ones, which suggests state-tracking capacity isn't the bottleneck.
  - Training needs T·L_D sequential decoder stages: about 100 for sequences of 50 with a 2-layer decoder, against 4 parallel layers now. The paper doesn't measure wall-clock speed. Chunking (RLT-2) helps speed but lost most of RLT's S5 gain in the paper (100% → 20% with chunks of 4).
  - With one objective, "MMoE" no longer applies: ARV reduces to input-conditioned or state-conditioned routing.
  - It would replace the multi-objective thesis rather than extend it.
- **Cheap test:**
  - Next-item only, 4 layers total (for example a 2 + 2 or 3 + 1 split), 10% users, a few epochs.
  - **Measure step time first.** More than 5× slower on an L4 likely rules it out at full scale.
  - Then compare **RLT-1 vs RLT-0 vs the current Transformer.** If per-token feedback doesn't beat no feedback on NDCG, the recurrence isn't helping recommendation.
- **Cost:** a new architecture (gated merge, encoder memory, sliding-window decoder, backpropagation through the recurrent history), the largest build of these ideas.

## 6. Looped single MoE layer with hard routing, next-item only (recurrence over depth)

One MoE layer, at layer k, whose output feeds back into its own input for N passes. On each pass a **hard top-1 gate** picks one expert, so each user (or token) follows a *path* through the experts, for example E2 → E4 → E1. Next-item is the only objective.

- **Why:**
  - **Composition:** K experts and N passes give K^N paths (64 for K = 4, N = 3) for the parameters of K experts, so much more effective depth per parameter.
  - **Interpretability:** our ARV routing is per user (from the pooled state), so a hard path per user is a learned "user segment program". Path usage, path diversity, and how paths relate to long-tail or cold-start users all make good analysis.
  - **It builds on ARV:** the router's objective embedding e_t becomes a **loop-step embedding**, so the router knows which pass it's on. That's an evolution of the existing contribution, not a pivot.
- **Design decisions and risks:**
  - **Hard gates aren't differentiable.** Use Switch-Transformer-style top-1 (scale the chosen expert's output by its gate probability) or Gumbel-softmax with a straight-through estimator. Avoid REINFORCE.
  - **Routing collapse, made worse by the loop.** If the router picks the same expert every pass, the model reduces to one weight-tied FFN applied N times. It needs a load-balancing term (a regulariser, not an objective) and the step embedding, and you should track how many distinct paths are actually used.
  - **What loops matters most.** Looping only the MoE-FFN processes each position independently, so there's no attention between items across passes, and it may be no stronger than one larger FFN. Looping the **whole block** (attention + MoE-FFN) re-attends on every pass and is the closer analogue of a deeper Transformer. Start with the whole block.
  - **Stability:** residual connection plus normalisation, small N (2–4); gradients flow back through all N passes.
  - **Compute:** a hard gate saves compute only with real dispatch to the chosen expert. The current MoE computes all K experts and mixes them, so a naive hard gate costs the same. Either way the cost is about N times the layer.
  - **Is depth the bottleneck?** Check first that a deeper ordinary baseline (for example 6 layers, or t-large) beats t-base. If it doesn't, a looped layer can still win only as a regularised, parameter-efficient alternative, and that has to be shown.
  - **Related work to position against:** Routing Networks (Rosenbaum et al., 2018; a router recursively picks function blocks), MoEUT (MoE inside weight-tied Universal Transformers), Mixture-of-Recursions (routing over how many times a shared block is applied), and adaptive computation time if the loop later gets early stopping. The defensible angle: **per-user expert paths for sequential recommendation, with a step-conditioned ARV router.**
- **Minimal experiment** (next-item only, 10% users, 10-epoch schedule, K = 4, last Transformer block replaced by a looped MoE block):

  | Arm | Isolates |
  |---|---|
  | N = 1, hard gate | a plain MoE layer, no loop |
  | N = 2, 3, hard, step-conditioned | the loop and the paths |
  | N = 3, soft (ARV mixing on each pass) | hard vs soft gating |
  | baseline (t-base) and a deeper baseline (6 layers) | whether depth helps at all |

  **Success:** the looped hard arms beat both N = 1 and the deeper baseline at a similar parameter count, *and* path statistics show many distinct paths in use. If N = 2 or 3 ≈ N = 1, the loop isn't helping.
- **Cost:** one looped block plus a router change. That's a smaller build than idea 5, and it keeps ARV central.

## Suggested order

After run12, and once the full-dataset run is settled:
1. **Idea 4**, fixed experts: cheapest, and it decides how ARV should be framed.
2. **Idea 3**, projection: already built, one arm.
3. **Idea 6**, looped hard-routed MoE: the architecture idea most connected to the thesis. Run the depth-scaling check first.
4. **Idea 5**, RLT: only if the step-time check passes.

Any result that changes a headline conclusion needs 2–3 seeds before it goes in the paper: most gaps here are about 0.01.
