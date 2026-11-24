#!/usr/bin/env python3
"""Status of study-X baseline arms: baseline_status.py RUN_DIR [OUTPUT_ROOT]"""
import glob, json, os, re, sys
root = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "outputs/amazon_beauty_2014"
store = os.path.join(root, out, "checkpoints/ablations/by_fingerprint")
for d in sorted(glob.glob(os.path.join(store, "*"))):
    log = os.path.join(d, "logs/pretrain_t_base.log")
    if not os.path.exists(log):
        continue
    t = open(log).read()
    c = {int(e): (float(n), float(r)) for e, n, r in re.findall(
        r"epoch=(\d+) \|.*?val_ndcg@10=([0-9.]+) \|.*?val_recall@10=([0-9.]+)", t)}
    loss = re.findall(r"train_loss_(\w+)=", t)
    kind = {"bce": "sasrec", "bpr": "gru4rec", "nce": "duorec"}
    name = next((kind[k] for k in loss if k in kind), None)
    if name is None:
        m = re.search(r"train_loss_ce=([0-9.]+)", t)
        name = "ce-model"
    done = os.path.exists(os.path.join(d, "backbone.json"))
    if c:
        b = max(c, key=lambda e: (c[e][0], -e))
        print(f"{os.path.basename(d)} {name:9s} last {max(c):3d} best {b:3d} val_ndcg {c[b][0]:.4f} val_hr {c[b][1]:.4f} {'DONE' if done else ''}")
for f in sorted(glob.glob(os.path.join(root, out, "ablations/seed_*/arms/X_t_base/*.json"))):
    a = json.load(open(f))
    m = a.get("metrics", {})
    print(f"{f.split('/')[-4]} {a['label']:9s} {a['status']} test NDCG@10 {m.get('ndcg',{}).get('mean',0):.4f} "
          f"HR@10 {m.get('recall',{}).get('mean',0):.4f} best_epoch {a.get('best_epoch')} fp {a.get('training_fingerprint')} "
          f"params {a.get('parameters',{}).get('total_parameters')} early {a.get('early_stopped')}")
