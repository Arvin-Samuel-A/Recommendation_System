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
