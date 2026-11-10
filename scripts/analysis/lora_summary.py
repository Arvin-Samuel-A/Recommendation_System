"""Summarise lora_diag.py output: per training setup, before -> after on the test target and
on the user's own local training examples. usage: lora_summary.py DIAG_JSONL"""
import json
import math
import statistics as st
import sys

recs = [json.loads(l) for l in open(sys.argv[1])]
setups = list(recs[0]["after"])
print(f"{len(recs)} users; before: median test rank {st.median(r['before']['rank'] for r in recs):.0f}, "
      f"mean NDCG@10 {st.mean(r['before']['ndcg'] for r in recs):.4f}, mean test logp {st.mean(r['before']['logp'] for r in recs):.3f}, "
      f"mean local-train loss {st.mean(r['before']['train_loss'] for r in recs):.3f}")


def ms(xs):
    m = st.mean(xs)
    return m, (st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan"))


hdr = (f"{'setup':9s} {'dNDCG@10':>16s} {'dlogp(test)':>16s} {'logp up/down':>12s} {'median rank b->a':>17s} "
       f"{'dtrain loss':>16s} {'emb cos':>8s} {'|BA|/|W|':>9s} {'corr(dtrain,dtest)':>18s}")
print(hdr)
for s in setups:
    dn = [r["after"][s]["ndcg"] - r["before"]["ndcg"] for r in recs]
    dl = [r["after"][s]["logp"] - r["before"]["logp"] for r in recs]
    dt = [r["after"][s]["train_loss"] - r["before"]["train_loss"] for r in recs]
    up = sum(x > 0 for x in dl); down = sum(x < 0 for x in dl)
    rb = st.median(r["before"]["rank"] for r in recs); ra = st.median(r["after"][s]["rank"] for r in recs)
    cos = st.mean(r["after"][s]["emb_cos"] for r in recs); rel = st.mean(r["after"][s]["rel_norm"] for r in recs)
    # does fitting the user's own past better (more negative dtrain) go with a better test logp?
    mx, my = st.mean(dt), st.mean(dl)
    cov = sum((a - mx) * (b - my) for a, b in zip(dt, dl))
    corr = cov / math.sqrt(sum((a - mx) ** 2 for a in dt) * sum((b - my) ** 2 for b in dl))
    (n_m, n_se), (l_m, l_se), (t_m, t_se) = ms(dn), ms(dl), ms(dt)
    print(f"{s:9s} {n_m:+.4f} ± {n_se:.4f} {l_m:+.3f} ± {l_se:.3f}   {up:4d}/{down:<4d}  {rb:7.0f} -> {ra:<7.0f}"
          f" {t_m:+.3f} ± {t_se:.3f} {cos:8.3f} {rel:9.3f} {corr:+18.2f}")
# by history length (default setup)
print("\nby train-history length (default setup): n, mean dNDCG, mean dlogp")
for lo, hi in ((0, 50), (50, 200), (200, 10 ** 9)):
    g = [r for r in recs if lo <= r["n_train_hist"] < hi]
    if g:
        print(f"  [{lo},{hi if hi < 10**9 else 'inf'}): n={len(g):3d}  dNDCG {st.mean(r['after']['default']['ndcg'] - r['before']['ndcg'] for r in g):+.4f}"
              f"  dlogp {st.mean(r['after']['default']['logp'] - r['before']['logp'] for r in g):+.3f}")
