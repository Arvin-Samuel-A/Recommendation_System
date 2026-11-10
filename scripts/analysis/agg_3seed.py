"""3-seed (42/43/44) mean +- std of test metrics for every study arm (MovieLens-20M, protocol v3).

Reads ROOT/seed_{42,43,44}/{S}_t_base.json (each a list of arm results) and prints one markdown
table per study: per-seed NDCG@10, then mean +- sample std (ddof=1) of NDCG, Recall and MRR @10.
Also lists the code revisions seen, so mixed-code seeds would show up.

usage: agg_3seed.py ROOT [OUT_MD]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = sys.argv[1]
SEEDS = (42, 43, 44)
STUDIES = "ABCDEFGH"
REF = ("A", "arv_mmoe_gradnorm")

res = defaultdict(dict)           # (study, label) -> seed -> metrics dict
order = defaultdict(list)         # study -> labels in file order
revs = defaultdict(set)
pop = {}
for s in SEEDS:
    for st in STUDIES:
        p = os.path.join(ROOT, f"seed_{s}", f"{st}_t_base.json")
        if not os.path.exists(p):
            print(f"missing {p}", file=sys.stderr)
            continue
        for a in json.load(open(p)):
            lab = a["label"]
            if lab not in order[st]:
                order[st].append(lab)
            res[(st, lab)][s] = {m: a["metrics"][m]["mean"] for m in ("ndcg", "recall", "mrr")}
            revs[("train " + str((a.get("source_revision") or {}).get("source_hash")), "eval " + str((a.get("code_revision") or {}).get("source_hash")))].add(s)
            if a.get("popularity_reference"):
                pop[s] = a["popularity_reference"]


def ms(vals):
    v = np.asarray(vals, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))


ref_mean = ms([res[REF][s]["ndcg"] for s in SEEDS if s in res[REF]])[0]
out = []
out.append("# MovieLens-20M, protocol v3: 3-seed results (seeds 42/43/44)\n")
out.append(f"Reference = {REF[0]}:{REF[1]}, mean NDCG@10 {ref_mean:.4f}. "
           "Mean +- sample std over seeds; Delta = mean NDCG@10 minus the reference mean.\n")
for st in STUDIES:
    out.append(f"\n## Study {st}\n")
    out.append("| Arm | NDCG s42 | s43 | s44 | NDCG@10 mean ± std | Δ vs ref | Recall@10 | MRR@10 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for lab in order[st]:
        r = res[(st, lab)]
        cells = [f"{r[s]['ndcg']:.4f}" if s in r else "—" for s in SEEDS]
        seeds_ok = [s for s in SEEDS if s in r]
        n_m, n_s = ms([r[s]["ndcg"] for s in seeds_ok])
        r_m, r_s = ms([r[s]["recall"] for s in seeds_ok])
        m_m, m_s = ms([r[s]["mrr"] for s in seeds_ok])
        flag = "" if len(seeds_ok) == 3 else f" (n={len(seeds_ok)})"
        out.append(f"| {lab}{flag} | {' | '.join(cells)} | {n_m:.4f} ± {n_s:.4f} | {n_m - ref_mean:+.4f} "
                   f"| {r_m:.4f} ± {r_s:.4f} | {m_m:.4f} ± {m_s:.4f} |")
out.append("\n## Provenance\n")
for rev, ss in sorted(revs.items(), key=lambda x: str(x[0])):
    out.append(f"- source hash {rev[0]} / {rev[1]}: seeds {sorted(ss)}")
if pop:
    out.append(f"- popularity reference NDCG@10: " + ", ".join(f"s{k} {v['ndcg']['mean']:.4f}" for k, v in sorted(pop.items())))
text = "\n".join(out) + "\n"
print(text)
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(text)
