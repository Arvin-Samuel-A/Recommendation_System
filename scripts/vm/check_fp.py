"""Recompute each arm result's training fingerprint with this checkout's code."""
import dataclasses, glob, json, sys
from configs.config import Config
from training.ablation_runner import _training_fingerprint
names = {f.name for f in dataclasses.fields(Config)}
same = diff = 0
bad = []
for root in sys.argv[1:]:
    for p in glob.glob(f"{root}/**/arms/**/*.json", recursive=True):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if not isinstance(r, dict) or "training_fingerprint" not in r or "resolved_config" not in r:
            continue
        cfg = Config()
        for k, v in r["resolved_config"].items():
            if k in names:
                setattr(cfg, k, v)
        if _training_fingerprint(cfg) == r["training_fingerprint"]:
            same += 1
        else:
            diff += 1; bad.append((p, r["training_fingerprint"], _training_fingerprint(cfg)))
print(f"match {same}, differ {diff}")
for b in bad[:10]:
    print(*b)
