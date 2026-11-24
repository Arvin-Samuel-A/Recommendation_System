"""Markdown tables for everything the 3-seed accuracy tables don't show, for one dataset.

Reads the study JSONs of a finished 3-seed run (ROOT/ablations/seed_{42,43,44}/{S}_t_base.json),
the run's GradNorm logs (ROOT/checkpoints/ablations/by_fingerprint/<fp>/logs/) and the
processed data directory, and prints:

  dataset statistics, training setup, GradNorm's learned weights, efficiency (parameters,
  model size, FLOPs), router parameter accounting, cold-start, long-tail, routing
  diagnostics and gradient diagnostics.

Every per-arm number is the mean +- sample std over the three seeds.

usage: results_extras.py NAME ROOT DATA_DIR [OUT_MD]
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import re

import numpy as np

NAME, ROOT, DATA = sys.argv[1:4]
OUT = sys.argv[4] if len(sys.argv) > 4 else None
SEEDS = (42, 43, 44)
OBJ = ("next_item", "masked_item", "multi_step", "contrastive")
REF = ("A", "arv_mmoe_gradnorm")
# The LoRA / federated arms; that code was removed on 2026-11-12.
REMOVED = {("A", "arv_mmoe_gradnorm_lora"), ("H", "H6_communication")}
USE_FLAG = {"masked_item": "use_masked_lm", "multi_step": "use_multi_step", "contrastive": "use_contrastive"}

arms = defaultdict(dict)  # (study, label) -> seed -> arm result
for s in SEEDS:
    for st in "ABCDFGH":
        p = os.path.join(ROOT, "ablations", f"seed_{s}", f"{st}_t_base.json")
        if os.path.exists(p):
            for a in json.load(open(p)):
                if (st, a["label"]) not in REMOVED:
                    arms[(st, a["label"])][s] = a

# Arms that differ only in post-training settings share one trained model (same fingerprint).
by_fp = defaultdict(list)
for key in sorted(arms):
    by_fp[arms[key][42]["training_fingerprint"]].append(key)


def models():
    """One key per trained model, preferring the study-A name, with the other arm names."""
    for keys in by_fp.values():
        keys = sorted(keys, key=lambda k: (k[0] != "A", k))
        alias = ", ".join(f"{k[0]}:{k[1]}" for k in keys[1:])
        yield keys[0], f"{keys[0][0]}:{keys[0][1]}" + (f" (= {alias})" if alias else "")

out = []


def ms(xs, fmt=".4f"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:{fmt}}"
    return f"{np.mean(xs):{fmt}} ± {np.std(xs, ddof=1):{fmt}}"


def get(key, path):
    """Values of a nested field of one arm across the seeds."""
    vals = []
    for s in SEEDS:
        v = arms[key].get(s)
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        vals.append(v)
    return vals


def table(head, rows):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    out.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    out.append("")


# ------------------------------------------------------------------ dataset statistics
stats = json.load(open(os.path.join(DATA, "dataset_stats.json")))
manifest = json.load(open(os.path.join(DATA, "preprocessing_manifest.json")))
splits = pickle.load(open(os.path.join(DATA, "splits.pkl"), "rb"))
buckets = pickle.load(open(os.path.join(DATA, "popularity_buckets.pkl"), "rb"))
cold = pickle.load(open(os.path.join(DATA, "cold_items.pkl"), "rb"))
users = list(splits["test"])
lengths = np.array([len(splits["train"].get(u, [])) + len(splits["val"].get(u, []))
                    + len(splits["test"].get(u, [])) for u in users])
test_items = [t for u in users for t in splits["test"][u]]
n_u, n_i, n_x = stats["num_users"], stats["num_items"], stats["total_interactions"]

out.append(f"### {NAME}: dataset statistics\n")
table(["Statistic", "Value"], [
    ["Users", f"{n_u:,}"],
    ["Items", f"{n_i:,}"],
    ["Interactions", f"{n_x:,}"],
    ["Density", f"{n_x / (n_u * n_i):.4%}"],
    ["Interactions per user: mean / median", f"{lengths.mean():.1f} / {np.median(lengths):.0f}"],
    ["Interactions per user: min / p90 / max", f"{lengths.min()} / {np.percentile(lengths, 90):.0f} / {lengths.max()}"],
    ["Split", "leave-one-out: last item → test, second-to-last → validation, rest → training"],
    ["Test rows (one per user)", f"{len(test_items):,}"],
    ["Item metadata features", manifest.get("feature_dim", "—")],
    ["Cold items (held out of training)", f"{len(cold):,} ({manifest.get('cold_start_strategy', '—')})"],
    ["Test rows whose target is cold", f"{sum(t in cold for t in test_items):,}"],
    ["Popularity buckets (items): head / mid / tail",
     " / ".join(f"{len(buckets[b]):,}" for b in ("head", "mid", "tail"))],
    ["Test rows by target bucket: head / mid / tail",
     " / ".join(f"{sum(t in buckets[b] for t in test_items):,}" for b in ("head", "mid", "tail"))],
])

# ------------------------------------------------------------------ setup
ref_cfg = arms[REF][42]["resolved_config"]
keys = [
    ("model_name", "Model"), ("num_layers", "Transformer layers"), ("hidden_dim", "Hidden size"),
    ("num_heads", "Attention heads"), ("max_seq_length", "Sequence length (reference)"),
    ("num_experts", "Experts K"), ("moe_variant", "MoE variant"),
    ("objective_embedding_dim", "Objective embedding size"), ("use_metadata", "Item metadata"),
    ("batch_size", "Batch size"), ("learning_rate", "Learning rate"), ("warmup_ratio", "Warm-up ratio"),
    ("min_lr_ratio", "Final LR ratio (cosine)"), ("weight_decay", "Weight decay"),
    ("max_grad_norm", "Gradient clipping"), ("pretrain_epochs", "Epochs (max)"),
    ("multi_step_k", "Multi-step horizon k"), ("contrastive_temperature", "Contrastive temperature"),
    ("contrastive_interval", "Contrastive every n steps"), ("mask_prob", "Masking probability"),
    ("use_gradnorm", "GradNorm"), ("gradnorm_alpha", "GradNorm α"), ("gradnorm_lr", "GradNorm LR"),
    ("gradnorm_primary_objective", "GradNorm primary objective"),
    ("gradnorm_aux_budget", "GradNorm auxiliary budget"), ("gradnorm_anchor", "GradNorm anchor"),
    ("dtype", "Precision"), ("mixed_precision", "Mixed precision"),
]
out.append(f"### {NAME}: training setup (reference arm A:arv_mmoe_gradnorm)\n")
table(["Setting", "Value"], [[label, ref_cfg.get(k, "—")] for k, label in keys if k in ref_cfg])

# ------------------------------------------------------------------ training length
out.append(f"### {NAME}: training length (best validation epoch)\n")
out.append("Epochs are counted from 0. One row per trained model; arms that reuse the same "
           "model are listed in brackets.\n")
rows = []
for key, label in sorted(models(), key=lambda m: m[1]):
    eps = get(key, ["best_epoch"])
    if all(e is not None for e in eps):
        rows.append([label, ms(eps, ".1f"), "/".join(str(e) for e in eps),
                     ms(get(key, ["best_val_ndcg"]))])
table(["Arm", "Best epoch", "Per seed", "Best validation NDCG@10"], rows)

# ------------------------------------------------------------------ GradNorm weights
out.append(f"### {NAME}: GradNorm's learned loss weights (last logged step)\n")
out.append("Next-item's weight is fixed at 1; GradNorm sets the three auxiliary weights, which sum "
           "to the auxiliary budget.\n")
rows = []
for key, label in sorted(models(), key=lambda m: m[1]):
    cfg = arms[key][42]["resolved_config"]
    if not cfg.get("use_gradnorm"):
        continue
    lam = defaultdict(list)
    for s in SEEDS:
        fp = arms[key][s]["training_fingerprint"]
        p = os.path.join(ROOT, "checkpoints", "ablations", "by_fingerprint", fp, "logs", "gradnorm_t_base.jsonl")
        if os.path.exists(p):
            last = json.loads(open(p).read().strip().splitlines()[-1])
            for o in OBJ:
                if o in last["lambdas"]:
                    lam[o].append(last["lambdas"][o])
    if lam:
        rows.append([label] + [ms(lam[o], ".3f") if cfg.get(USE_FLAG[o], True) and lam[o] else "off"
                               for o in OBJ[1:]])
table(["Arm", "masked-item", "multi-step", "contrastive"], rows)

# ------------------------------------------------------------------ efficiency
out.append(f"### {NAME}: efficiency\n")
out.append("Parameters include the MoE layer. Sizes are in decimal MB (10^6 bytes) at 4 bytes per "
           "parameter. FLOPs are analytical, for one recommendation (full input sequence, scoring "
           "the whole catalogue).\n")
rows = []
for key in sorted(k for k in arms if arms[k][42].get("parameters")
                  and k not in {("H", "H3_cold_start"), ("H", "H4_longtail")}):
    rows.append([
        f"{key[0]}:{key[1]}",
        ms([v / 1e6 for v in get(key, ["parameters", "total_parameters"])], ".3f"),
        ms(get(key, ["parameters", "router_parameters"]), ".0f"),
        ms([v / 1e6 for v in get(key, ["parameters", "expert_parameters"])], ".3f"),
        ms([v * 4 / 1e6 for v in get(key, ["parameters", "total_parameters"])], ".1f"),
        ms([v / 1e9 for v in get(key, ["flops_per_inference"])], ".3f"),
    ])
table(["Arm", "Parameters (M)", "Router params", "Expert params (M)", "Size fp32 (MB)",
       "GFLOPs / recommendation"], rows)

# ------------------------------------------------------------------ size breakdown
import torch  # noqa: E402

GROUPS = [  # (component, regex over "model."/"moe."/"loss." prefixed state-dict keys)
    ("Item embedding table (also scores the catalogue)", r"^model\.item_embedding\."),
    ("Position embeddings", r"^model\.position_embedding\."),
    ("Item metadata projection", r"^model\.metadata_projection\."),
    ("Transformer self-attention (4 blocks)", r"^model\.blocks\.\d+\.attn\."),
    ("Transformer feed-forward (4 blocks)", r"^model\.blocks\.\d+\.ffn\."),
    ("Layer norms", r"^model\.(blocks\.\d+\.norm|final_norm)"),
    ("MoE experts", r"^moe\.experts\."),
    ("ARV router", r"^moe\.router\."),
    ("Objective embeddings", r"^moe\.objective_embeddings\."),
]
ref_fp = arms[REF][42]["training_fingerprint"]
ck = torch.load(os.path.join(ROOT, "checkpoints", "ablations", "by_fingerprint", ref_fp, "t_base", "best.pt"),
                map_location="cpu", weights_only=False)
params = {}
for prefix, sd in (("model.", "model_state_dict"), ("moe.", "moe_state_dict"), ("loss.", "loss_fns_state_dict")):
    for k, v in (ck.get(sd) or {}).items():
        if torch.is_tensor(v):
            params[prefix + k] = v.numel()
        elif isinstance(v, dict):
            params.update({f"{prefix}{k}.{k2}": v2.numel() for k2, v2 in v.items() if torch.is_tensor(v2)})
counts, used = [], set()
for label, rx in GROUPS:
    keys = [k for k in params if re.search(rx, k)]
    used.update(keys)
    counts.append((label, sum(params[k] for k in keys)))
deployed = sum(n for _, n in counts)
leftover = {k: n for k, n in params.items() if k not in used and not k.startswith("loss.")}
assert not leftover, f"unassigned parameters: {leftover}"
# The reference's parameter count is recorded on H5 (same trained model, efficiency extras).
assert deployed == int(arms[("H", "H5_latency")][42]["parameters"]["total_parameters"]), \
    "breakdown does not add up to the total"
train_only = sum(n for k, n in params.items() if k.startswith("loss."))
cfg = arms[REF][42]["resolved_config"]
meta_shape = np.load(os.path.join(DATA, "item_metadata.npy"), mmap_mode="r").shape


def mb(n, b):
    return f"{n * b / 1e6:.2f}"


out.append(f"### {NAME}: model size breakdown (reference model, seed 42)\n")
out.append("Parameter counts read from the checkpoint. Sizes in decimal MB at fp32 (4 bytes), "
           "fp16 (2 bytes) and int8 (1 byte) per parameter.\n")
rows = [[label, f"{n:,}", f"{100 * n / deployed:.1f}%", mb(n, 4), mb(n, 2), mb(n, 1)] for label, n in counts]
rows.append(["**Total (deployed model)**", f"**{deployed:,}**", "**100%**",
             f"**{mb(deployed, 4)}**", f"**{mb(deployed, 2)}**", f"**{mb(deployed, 1)}**"])
table(["Component", "Parameters", "Share", "fp32 MB", "fp16 MB", "int8 MB"], rows)
out.append(f"- **Item embedding table:** {cfg['vocab_size']:,} rows × {cfg['hidden_dim']} "
           f"(the {n_i:,} items plus padding and mask tokens). It grows with the catalogue; "
           "everything else is fixed by the architecture.")
out.append(f"- **Training only, not in the deployed model:** the multi-step objective's projection "
           f"heads ({train_only:,} parameters, {mb(train_only, 4)} MB fp32).")
out.append(f"- **Phone (ONNX) model:** also carries the item metadata table the model reads at input "
           f"({meta_shape[0]:,} × {meta_shape[1]} values, {mb(meta_shape[0] * meta_shape[1], 4)} MB fp32) "
           f"and one precomputed norm per item ({mb(meta_shape[0], 4)} MB).\n")

out.append(f"### {NAME}: router parameter accounting (study C, K = 4)\n")
rp = arms[("C", "arv_mmoe")][42].get("router_parameters") or {}
table(["Routing component", "Parameters"], [
    ["MMoE: one gate per objective (4 gates)", rp.get("mmoe_gate_params")],
    ["ARV-MMoE: shared router", rp.get("arv_router_params")],
    ["ARV-MMoE: objective embeddings (4 × 32)", rp.get("arv_objective_embedding_params")],
    ["ARV-MMoE: total", rp.get("arv_total_router_params")],
    ["Saving of ARV-MMoE over MMoE", rp.get("difference")],
])


# ------------------------------------------------------------------ cold-start
def metric_rows(key, field, parts):
    rows = []
    for part, label in parts:
        rows.append([label] + [ms(get(key, [field, part, m, "mean"])) for m in ("ndcg", "recall", "mrr")])
    return rows


key = ("H", "H3_cold_start")
out.append(f"### {NAME}: cold-start (H3)\n")
cs = arms[key][42]["cold_start"].get("cold_set")
if cs:
    out.append(f"Cold set: {cs['num_cold_items']} cold items, {cs['num_cold_test_rows']} cold test rows, "
               f"{cs['num_warm_test_rows']:,} warm test rows.\n")
else:
    out.append(f"Cold set: {len(cold)} cold items, {sum(t in cold for t in test_items)} cold test rows "
               "(this run's result file does not record the counts; they are recomputed from the data).\n")
table(["Test rows", "NDCG@10", "Recall@10", "MRR@10"], metric_rows(key, "cold_start", [
    ("warm_meta_on", "warm targets, model with metadata"),
    ("warm_meta_off", "warm targets, metadata input off"),
    ("warm_popularity", "warm targets, popularity"),
    ("cold_meta_on", "cold targets, model with metadata"),
    ("cold_meta_off", "cold targets, metadata input off"),
    ("cold_popularity", "cold targets, popularity"),
]))

# ------------------------------------------------------------------ long-tail
key = ("H", "H4_longtail")
out.append(f"### {NAME}: long-tail (H4)\n")
out.append("Items bucketed by training popularity: head = top 20% of the catalogue, mid = next 20%, "
           "tail = bottom 60%. Rows are test rows whose target is in the bucket.\n")
table(["Bucket", "NDCG@10", "Recall@10", "MRR@10"], metric_rows(key, "longtail", [
    ("head", "head, model"), ("popularity_head", "head, popularity"),
    ("mid", "mid, model"), ("popularity_mid", "mid, popularity"),
    ("tail", "tail, model"), ("popularity_tail", "tail, popularity"),
]))

# ------------------------------------------------------------------ routing
out.append(f"### {NAME}: routing diagnostics (validation set)\n")
out.append("Entropy of the routing distribution in nats (maximum ln K: 0.693 for K=2, 1.386 for "
           "K=4, 1.792 for K=6, 2.079 for K=8). Utilization is each expert's average routing weight. "
           "Routing similarity is the cosine between objectives' average routing vectors.\n")
rows = []
for key in sorted(k for k in arms if arms[k][42].get("routing")):
    util = np.mean([arms[key][s]["routing"]["overall_utilization"] for s in SEEDS if s in arms[key]], axis=0)
    cos = np.mean([arms[key][s]["routing"]["objective_cosine_similarity"] for s in SEEDS], axis=0)
    rows.append([f"{key[0]}:{key[1]}", ms(get(key, ["routing", "overall_entropy_mean"]), ".3f")]
                + [ms(get(key, ["routing", "per_objective_entropy", o]), ".3f") for o in OBJ]
                + [" / ".join(f"{u:.2f}" for u in util)]
                + [f"{cos[0, 1]:.2f} / {cos[0, 2]:.2f} / {cos[0, 3]:.2f}"])
table(["Arm", "Entropy (all)", "next-item", "masked-item", "multi-step", "contrastive",
       "Expert utilization", "Routing sim. next vs masked / multi / contrastive"], rows)

# ------------------------------------------------------------------ gradient diagnostics
out.append(f"### {NAME}: gradient diagnostics (study D, anchor layer, best checkpoint)\n")
out.append("Cosine between next-item's gradient and each auxiliary gradient, and each objective's "
           "gradient norm, over 10 training batches.\n")
rows = []
for key in sorted(k for k in arms if arms[k][42].get("gradient")):
    cos = [np.array(arms[key][s]["gradient"]["cosine_similarity"]) for s in SEEDS]
    names = arms[key][42]["gradient"]["objective_names"]
    row = [f"{key[0]}:{key[1]}"]
    for o in OBJ[1:]:
        j = names.index(o)
        row.append(ms([c[0, j] for c in cos], ".3f"))
    for o in OBJ:
        row.append(ms(get(key, ["gradient", "gradient_norms", "per_objective", o]), ".3f"))
    rows.append(row)
table(["Arm", "cos next/masked", "cos next/multi", "cos next/contrastive",
       "norm next", "norm masked", "norm multi", "norm contrastive"], rows)

text = "\n".join(out) + "\n"
print(text)
if OUT:
    open(OUT, "w").write(text)
