#!/bin/bash
# Per-user test metrics for every distinct Beauty model (3 seeds) and the fixed-weight run, on GPU 0.
set -u
export CUDA_VISIBLE_DEVICES=0
PY=$HOME/run13/venv/bin/python
OUT=$HOME/early_stop/per_user_beauty_all
mkdir -p $OUT $OUT/fixedw
cd $HOME/beauty14
LIST=$($PY - <<"PYEOF"
import json, os
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
            paths.append(f"{root}/seed_{s}/arms/{st}_t_base/{a[label]}.json")
print(" ".join(paths))
PYEOF
)
echo "arms: $(echo $LIST | wc -w)"
$PY $HOME/early_stop/per_user_eval.py $OUT $LIST
cd $HOME/beauty14_fixedw
$PY $HOME/early_stop/per_user_eval.py $OUT/fixedw outputs/amazon_beauty_2014/ablations/seed_{42,43,44}/arms/A_t_base/arv_mmoe.json
for s in 42 43 44; do mv $OUT/fixedw/seed${s}_A_arv_mmoe.npz $OUT/seed${s}_X_fixed_weights.npz; done
echo DONE
