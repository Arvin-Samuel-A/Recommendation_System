#!/usr/bin/env bash
# MovieLens gradient geometry (seed 43, the one with epoch snapshots; epoch_N is 0-indexed): reference through training, arv_mmoe at epoch 5 and best.
# Each run waits for a GPU with >= 26 GB free (full batch 256), then runs pinned to it.
cd /root/run13
A=outputs/ablations/seed_43/arms/A_t_base
OUT=/root/early_stop/gc; mkdir -p $OUT
run() {  # arm ckpt
  while :; do
    g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 26624 {print $1; exit}')
    [ -n "$g" ] && break; sleep 30
  done
  echo "$(date -Is) GPU $g: $1 $2"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g venv/bin/python -u /root/early_stop/grad_conflict.py $A/$1.json $2 48 $OUT/ml_$1_$2_s43.json 2>&1 | grep -E "DONE|Error|Traceback"
}
for c in init epoch_0 epoch_4 epoch_9 best; do run arv_mmoe_gradnorm $c; done
for c in epoch_4 best; do run arv_mmoe $c; done
echo ALL DONE
