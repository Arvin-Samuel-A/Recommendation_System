"""Gradient-conflict analysis of `grad_conflict.py` outputs (per-batch Gram matrices).

For each input file, each parameter scope and each weighting, over the batches:
  cos(next, aux)       mean ± sd of the cosine between next-item and each auxiliary
                       gradient, and the share of batches where it is negative (PCGrad's
                       conflict test)
  weighted aux sum     A = sum_a w_a g_a with the arm's weights: cos(g_next, A), |A|/|g_next|,
                       and "next-item progress" = (g_next . d) / |g_next|^2 for the update
                       d = g_next + A (1 = as fast as next-item alone; < 1 = the auxiliaries
                       slow next-item down; < 0 = the update increases next-item loss)
  worst objective      min_i cos(g_i, d): the objective that d helps least (negative = d
                       increases that objective's loss to first order)
  CAGrad (c = 0.5)     the conflict-averse direction of Liu et al. 2021 from the same Gram
                       matrix, with its next-item progress and worst-objective cosine
  PCGrad               next-item progress after projecting each auxiliary off g_next where
                       they conflict (the asymmetric, next-item-protected variant)

usage: gc_analysis.py LABEL=FILE[:w_masked,w_multi,w_contrastive] ...
       (weights default to 1,1,1, i.e. equal; next-item's weight is always 1)
"""
import itertools
import json
import sys

import numpy as np

C = 0.5
GRID = np.array([w for w in itertools.product(np.linspace(0, 1, 26), repeat=4)
                 if abs(sum(w) - 1) < 1e-9])  # simplex, step 0.04


def gram(rec):
    cos = np.array(rec["cos"], dtype=float)
    n = np.array(rec["norms"], dtype=float)
    return cos * np.outer(n, n), n


def cagrad(G):
    T = len(G)
    one = np.ones(T) / T
    g0_sq = one @ G @ one
    obj = GRID @ G @ one + np.sqrt(C * g0_sq) * np.sqrt(np.einsum("ki,ij,kj->k", GRID, G, GRID).clip(0))
    w = GRID[np.argmin(obj)]
    gw_norm = np.sqrt(max(w @ G @ w, 1e-30))
    # d = g0 + (sqrt(c)|g0| / |g_w|) g_w, as coefficients on the task gradients
    return one + np.sqrt(C * g0_sq) / gw_norm * w


def direction_stats(G, coef):
    """For d = sum_i coef_i g_i: next-item progress and the worst objective cosine."""
    gd = G @ coef                        # g_i . d
    d_norm = np.sqrt(max(coef @ G @ coef, 1e-30))
    cos_i = gd / (np.sqrt(np.diag(G)) * d_norm)
    return gd[0] / G[0, 0], cos_i.min(), int(np.argmin(cos_i))


def pcgrad_next_protected(G, w):
    """next-item progress with each conflicting auxiliary projected off g_next."""
    coef = np.array([1.0, *w])
    for a in range(1, len(G)):
        dot = G[a, 0]
        if dot < 0:                      # g_a <- g_a - (g_a.g_n/|g_n|^2) g_n
            coef[0] -= coef[a] * dot / G[0, 0]
    return (G @ coef)[0] / G[0, 0]


def analyse(label, path, w):
    d = json.load(open(path))
    names = d["objectives"]
    out = {"label": label, "ckpt": d["ckpt"], "seed": d["seed"], "batches": len(d["batches"]), "scopes": {}}
    for scope in d["batches"][0]:
        cosines = np.array([b[scope]["cos"][0][1:] for b in d["batches"]])        # (B, 3)
        rows = {"cos_next_aux_mean": cosines.mean(0), "cos_next_aux_sd": cosines.std(0),
                "conflict_frac": (cosines < 0).mean(0)}
        prog, worst, worst_i, cprog, cworst, pprog, acos, aratio = ([] for _ in range(8))
        for b in d["batches"]:
            G, n = gram(b[scope])
            coef = np.array([1.0, *w])
            p, wc, wi = direction_stats(G, coef)
            prog.append(p); worst.append(wc); worst_i.append(wi)
            A = np.array([0.0, *w])
            a_norm = np.sqrt(max(A @ G @ A, 1e-30))
            acos.append((G @ A)[0] / (np.sqrt(G[0, 0]) * a_norm)); aratio.append(a_norm / np.sqrt(G[0, 0]))
            cp, cw, _ = direction_stats(G, cagrad(G))
            cprog.append(cp); cworst.append(cw)
            pprog.append(pcgrad_next_protected(G, w))
        rows.update({
            "aux_sum_cos_next": (np.mean(acos), np.std(acos)), "aux_sum_norm_ratio": np.mean(aratio),
            "next_progress": (np.mean(prog), np.std(prog)), "worst_cos": np.mean(worst),
            "worst_obj": names[int(np.bincount(worst_i).argmax())],
            "cagrad_next_progress": np.mean(cprog), "cagrad_worst_cos": np.mean(cworst),
            "pcgrad_next_progress": np.mean(pprog),
        })
        out["scopes"][scope] = rows
    return names, out


def main():
    results = []
    for arg in sys.argv[1:]:
        label, rest = arg.split("=", 1)
        path, _, ws = rest.partition(":")
        w = [float(x) for x in ws.split(",")] if ws else [1.0, 1.0, 1.0]
        names, r = analyse(label, path, w)
        r["weights"] = w
        results.append(r)
    aux = names[1:]
    for scope in ("anchor", "encoder"):
        print(f"\n=== scope: {scope} (cos with next-item: mean ± sd [share of batches < 0]) ===")
        print(f"{'run':30s} " + " ".join(f"{a:>24s}" for a in aux) +
              f" {'cos(next, sum w*aux)':>21s} {'|sum|/|next|':>12s} {'next progress':>14s} {'worst obj (cos)':>22s}"
              f" {'PCGrad prog':>11s} {'CAGrad prog / worst':>20s}")
        for r in results:
            s = r["scopes"][scope]
            cells = " ".join(f"{m:+.3f} ± {sd:.3f} [{f:4.0%}]".rjust(24) for m, sd, f in
                             zip(s["cos_next_aux_mean"], s["cos_next_aux_sd"], s["conflict_frac"]))
            worst = f"{s['worst_obj']} ({s['worst_cos']:+.3f})"
            print(f"{r['label']:30s} {cells} {s['aux_sum_cos_next'][0]:+21.3f} {s['aux_sum_norm_ratio']:12.2f}"
                  f" {s['next_progress'][0]:+14.2f} {worst:>22s}"
                  f" {s['pcgrad_next_progress']:+11.2f} {s['cagrad_next_progress']:+9.2f} / {s['cagrad_worst_cos']:+.3f}")


if __name__ == "__main__":
    main()
