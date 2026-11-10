"""Per-user paired tests on MovieLens-20M over seeds 42/43/44 (protocol v3, full ranking, 138,493 users).

For each arm A and contrast B: per seed, the paired difference of the per-user metric (A - B)
and its paired-t p. Pooled over seeds: each user's metric averaged over the 3 seeds first, then a
paired t-test and Wilcoxon signed-rank test over users, a 95% CI of the mean difference, and a
Holm-adjusted p across all contrasts of that metric. "consistent" = same sign in all 3 seeds.

usage: paired_tests_3seed.py NPZ_DIR [NPZ_DIR ...]   (files seed{42,43,44}_{STUDY}_{ARM}.npz)
"""
import glob
import os
import sys

import numpy as np
from scipy import stats

SEEDS = (42, 43, 44)
REF = "A_arv_mmoe_gradnorm"
runs = {}  # tag -> seed -> npz
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "seed4[234]_*.npz")):
        b = os.path.basename(f)[:-4]
        seed, tag = int(b[4:6]), b[7:]
        runs.setdefault(tag, {})[seed] = np.load(f)
tags = sorted(t for t, v in runs.items() if all(s in v for s in SEEDS))
skipped = sorted(t for t in runs if t not in tags)
uids = runs[REF][42]["user_ids"]
for t in tags:
    for s in SEEDS:
        assert np.array_equal(runs[t][s]["user_ids"], uids), f"user order differs: {t} seed {s}"


def holm(ps):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


pairs = [(t, REF) for t in tags if t != REF] + [
    ("D_none_gradnorm", "A_4obj_no_routing"),
    ("A_arv_mmoe", "A_conv_mmoe"),
    ("G_K6", "G_K2"), ("G_K8", "G_K6"),
    ("H_H2_seq50", "H_H2_seq25"), ("H_H2_seq100", "H_H2_seq50"),
]
pairs = [p for p in pairs if p[0] in tags and p[1] in tags]

for metric in ("ndcg", "recall"):
    rows = []
    for a, b in pairs:
        per_seed = []
        for s in SEEDS:
            x = runs[a][s][metric].astype(np.float64)
            y = runs[b][s][metric].astype(np.float64)
            per_seed.append(((x - y).mean(), stats.ttest_rel(x, y).pvalue))
        x = np.mean([runs[a][s][metric].astype(np.float64) for s in SEEDS], axis=0)
        y = np.mean([runs[b][s][metric].astype(np.float64) for s in SEEDS], axis=0)
        d = x - y
        se = d.std(ddof=1) / np.sqrt(len(d))
        p_t = stats.ttest_rel(x, y).pvalue
        nz = d[d != 0]
        p_w = stats.wilcoxon(nz).pvalue if len(nz) else 1.0
        signs = {np.sign(m) for m, _ in per_seed}
        rows.append((a, b, x.mean(), y.mean(), d.mean(), se, p_t, p_w, per_seed, len(signs) == 1))
    adj = holm([r[6] for r in rows])
    print(f"\n===== {metric}@10, 3 seeds pooled per user (reference = {REF}) =====")
    print(f"{'arm':22s} {'vs':22s} {'arm':>7s} {'vs':>7s} {'diff':>9s} {'rel':>7s} {'95% CI':>22s} "
          f"{'t p':>8s} {'Holm p':>8s} {'Wilcoxon p':>10s}  per-seed diff (t p)  consistent")
    for (a, b, xm, ym, dm, se, pt, pw, ps, cons), pa in zip(rows, adj):
        seeds_txt = "  ".join(f"{m:+.4f}({p:.1g})" for m, p in ps)
        print(f"{a:22s} {b:22s} {xm:7.4f} {ym:7.4f} {dm:+9.5f} {100 * dm / ym:+6.1f}% "
              f"[{dm - 1.96 * se:+.5f}, {dm + 1.96 * se:+.5f}] {pt:8.2g} {pa:8.2g} {pw:10.2g}  {seeds_txt}  "
              f"{'yes' if cons else 'NO'}")
if skipped:
    print(f"\nskipped (not all 3 seeds): {skipped}")
