"""Aggregate phone benchmark JSONs per series: mean +- sd over runs of each statistic."""
import glob
import json
import statistics as st
import sys
from collections import defaultdict

runs = defaultdict(list)
for p in sorted(glob.glob(sys.argv[1] + "/bench_*.json")):
    r = json.load(open(p))
    runs[(r["runtime"]["threads"], r["series"]["id"])].append(r)


def ms(xs):
    return f"{st.mean(xs):.1f} ± {st.stdev(xs):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"


print("| Threads | Series start | Runs | Inference median (ms) | p90 | p99 | Mean | End-to-end median | Throughput /s | Per-run medians |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (t, sid), rs in sorted(runs.items()):
    inf = [r["inference_ms"] for r in rs]
    e2e = [r["end_to_end_ms"] for r in rs]
    meds = " ".join(f"{i['p50']:.0f}" for i in inf)
    print(f"| {t} | {sid[11:19]} | {len(rs)} of {rs[0]['series']['total']} "
          f"| {ms([i['p50'] for i in inf])} | {ms([i['p90'] for i in inf])} | {ms([i['p99'] for i in inf])} "
          f"| {ms([i['mean'] for i in inf])} | {ms([e['p50'] for e in e2e])} "
          f"| {ms([r['throughput_per_s'] for r in rs])} | {meds} |")
print()
print("| Threads | Series start | Load total s | Download s | Session create s | First inference ms | JS heap after MB (max) |")
print("|---|---|---|---|---|---|---|")
for (t, sid), rs in sorted(runs.items()):
    m = rs[0]["model"]
    print(f"| {t} | {sid[11:19]} | {m['load_total_s']} | {m['download_s']} | {m['session_create_s']} "
          f"| {m['first_inference_ms']} | {max(r['js_heap_mb']['after'] for r in rs)} |")
