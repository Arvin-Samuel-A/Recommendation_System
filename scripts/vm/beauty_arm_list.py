"""Arm JSONs of every distinct trained Beauty model (one per training fingerprint), 3 seeds."""
import json

root = "outputs/amazon_beauty_2014/ablations"
removed = {("A", "arv_mmoe_gradnorm_lora"), ("H", "H6_communication")}
paths = []
for s in (42, 43, 44):
    seen = set()
    for st in "ABCDFGH":
        for a in json.load(open(f"{root}/seed_{s}/{st}_t_base.json")):
            fp = a["training_fingerprint"]
            if (st, a["label"]) in removed or fp in seen:
                continue
            seen.add(fp)
            paths.append(f"{root}/seed_{s}/arms/{st}_t_base/{a['label']}.json")
print(" ".join(paths))
