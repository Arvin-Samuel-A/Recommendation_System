"""Train next-item loss vs validation NDCG per epoch for Beauty arms (GradNorm vs not, fixed weights).
usage: train_val_curves.py  (run on the VM)"""
import json, re
ROOTS = {"GN": "/root/beauty14", "noGN": "/root/beauty14", "fixedW": "/root/beauty14_fixedw"}
ARM = {"GN": "arv_mmoe_gradnorm", "noGN": "arv_mmoe", "fixedW": "arv_mmoe"}
pat = re.compile(r" epoch=(\d+) \|.*?train_loss_nextitem=([0-9.]+).*?train_loss_multistep=([0-9.]+).*?val_ndcg@10=([0-9.]+)")
print(f"{'arm':7s} {'seed':4s} {'best ep':>7s} {'best val':>8s} {'train next@best':>15s} {'train next@ep20':>15s} {'val@ep20':>8s} {'train next@ep40':>15s} {'val@ep40':>8s} {'test':>7s}")
for arm in ("GN", "noGN", "fixedW"):
    for s in (42, 43, 44):
        a = json.load(open(f"{ROOTS[arm]}/outputs/amazon_beauty_2014/ablations/seed_{s}/arms/A_t_base/{ARM[arm]}.json"))
        log = f"{ROOTS[arm]}/outputs/amazon_beauty_2014/checkpoints/ablations/by_fingerprint/{a['training_fingerprint']}/logs/pretrain_t_base.log"
        rows = {}
        for line in open(log):
            m = pat.search(line)
            if m:
                rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        be = max(rows, key=lambda e: rows[e][2])
        g = lambda e, i: rows[e][i] if e in rows else float("nan")
        print(f"{arm:7s} {s:4d} {be:7d} {rows[be][2]:8.4f} {rows[be][0]:15.3f} {g(20,0):15.3f} {g(20,2):8.4f} {g(40,0):15.3f} {g(40,2):8.4f} {a['metrics']['ndcg']['mean']:7.4f}")
