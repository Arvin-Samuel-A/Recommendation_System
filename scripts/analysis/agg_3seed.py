"""3-seed (42/43/44) mean +- std of test metrics for every study arm.

Reads ROOT/seed_{42,43,44}/{S}_t_base.json (each a list of arm results) and prints one markdown
table per study: per-seed NDCG@10, then mean +- sample std (ddof=1) of NDCG, Recall and MRR @10.
Also lists the code revisions seen, so mixed-code seeds would show up.

The LoRA / federated arms (A:arv_mmoe_gradnorm_lora, study E, H:H6_communication) are left out;
that code was removed on 2026-11-12.

usage: agg_3seed.py ROOT [OUT_MD] [--title TITLE] [--epochs]
                     [--extra NAME=ARM_JSON_GLOB ...]
  --title   first heading (default: the MovieLens-20M protocol-v3 title)
  --epochs  add a best-epoch column (mean over seeds; for early-stopped runs)
  --extra   a single arm trained outside the study files, e.g. the fixed-weight Beauty run:
            NAME=ROOT2/seed_{seed}/arms/A_t_base/arv_mmoe.json ({seed} is filled in)
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

SEEDS = (42, 43, 44)
STUDIES = "ABCDEFGH"
REF = ("A", "arv_mmoe_gradnorm")
REMOVED = {("A", "arv_mmoe_gradnorm_lora"), ("H", "H6_communication")}
REMOVED_STUDIES = {"E"}

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("out_md", nargs="?")
p.add_argument("--title", default="MovieLens-20M, protocol v3: 3-seed results (seeds 42/43/44)")
p.add_argument("--epochs", action="store_true")
p.add_argument("--extra", action="append", default=[])
args = p.parse_args()

res = defaultdict(dict)           # (study, label) -> seed -> metrics dict
order = defaultdict(list)         # study -> labels in file order
revs = defaultdict(set)
pop = {}


def record(key, seed, a):
    res[key][seed] = {m: a["metrics"][m]["mean"] for m in ("ndcg", "recall", "mrr")}
    res[key][seed]["epoch"] = a.get("best_epoch")
    revs[("train " + str((a.get("source_revision") or {}).get("source_hash")),
          "eval " + str((a.get("code_revision") or {}).get("source_hash")))].add(seed)


for s in SEEDS:
    for st in STUDIES:
        if st in REMOVED_STUDIES:
            continue
        path = os.path.join(args.root, f"seed_{s}", f"{st}_t_base.json")
        if not os.path.exists(path):
            print(f"missing {path}", file=sys.stderr)
            continue
        for a in json.load(open(path)):
            lab = a["label"]
            if (st, lab) in REMOVED:
                continue
            if lab not in order[st]:
                order[st].append(lab)
            record((st, lab), s, a)
            if a.get("popularity_reference"):
                pop[s] = a["popularity_reference"]

extras = []
for spec in args.extra:
    name, pattern = spec.split("=", 1)
    extras.append(name)
    for s in SEEDS:
        path = pattern.format(seed=s)
        if os.path.exists(path):
            record(("extra", name), s, json.load(open(path)))
        else:
            print(f"missing {path}", file=sys.stderr)


def ms(vals):
    v = np.asarray(vals, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))


ref_mean = ms([res[REF][s]["ndcg"] for s in SEEDS if s in res[REF]])[0]
head = "| Arm | NDCG s42 | s43 | s44 | NDCG@10 mean ± std | Δ vs ref | Recall@10 | MRR@10 |"
rule = "|---|---|---|---|---|---|---|---|"
if args.epochs:
    head += " Best epoch |"
    rule += "---|"


def row(lab, r):
    cells = [f"{r[s]['ndcg']:.4f}" if s in r else "—" for s in SEEDS]
    seeds_ok = [s for s in SEEDS if s in r]
    n_m, n_s = ms([r[s]["ndcg"] for s in seeds_ok])
    r_m, r_s = ms([r[s]["recall"] for s in seeds_ok])
    m_m, m_s = ms([r[s]["mrr"] for s in seeds_ok])
    flag = "" if len(seeds_ok) == 3 else f" (n={len(seeds_ok)})"
    line = (f"| {lab}{flag} | {' | '.join(cells)} | {n_m:.4f} ± {n_s:.4f} | {n_m - ref_mean:+.4f} "
            f"| {r_m:.4f} ± {r_s:.4f} | {m_m:.4f} ± {m_s:.4f} |")
    if args.epochs:
        eps = [r[s]["epoch"] for s in seeds_ok if r[s]["epoch"] is not None]
        line += f" {np.mean(eps):.0f} ({'/'.join(str(e) for e in eps)}) |" if eps else " — |"
    return line


out = [f"# {args.title}\n",
       f"Reference = {REF[0]}:{REF[1]}, mean NDCG@10 {ref_mean:.4f}. "
       "Mean +- sample std over seeds; Delta = mean NDCG@10 minus the reference mean."
       + (" Best epoch = mean over seeds (per-seed values in brackets)." if args.epochs else "") + "\n",
       "The LoRA / federated arms (A:arv_mmoe_gradnorm_lora, study E, H6_communication) are left out; "
       "that code was removed on 2026-11-12. Their arm JSONs stay with the run's outputs.\n"]
for st in STUDIES:
    if st in REMOVED_STUDIES:
        continue
    out += [f"\n## Study {st}\n", head, rule]
    out += [row(lab, res[(st, lab)]) for lab in order[st]]
if extras:
    out += ["\n## Additional runs\n", head, rule]
    out += [row(name, res[("extra", name)]) for name in extras]
out.append("\n## Provenance\n")
for rev, ss in sorted(revs.items(), key=lambda x: str(x[0])):
    out.append(f"- source hash {rev[0]} / {rev[1]}: seeds {sorted(ss)}")
if pop:
    out.append("- popularity reference NDCG@10: "
               + ", ".join(f"s{k} {v['ndcg']['mean']:.4f}" for k, v in sorted(pop.items())))
text = "\n".join(out) + "\n"
print(text)
if args.out_md:
    open(args.out_md, "w").write(text)
