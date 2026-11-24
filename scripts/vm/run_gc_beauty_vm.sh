#!/usr/bin/env bash
# Beauty gradient geometry on the VM: reference (GradNorm) and arv_mmoe (no GradNorm) at best.pt,
# plus the untrained init (same architecture and seed for both arms), seeds 42-44, 48 batches each.
cd /root/beauty14
OUT=/root/early_stop/gc; mkdir -p $OUT
run() {  # arm_json ckpt out
  while :; do
    g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 >= 6144 {print $1; exit}')
    [ -n "$g" ] && break; sleep 30
  done
  echo "$(date -Is) GPU $g: $1 $2"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g venv/bin/python -u /root/early_stop/grad_conflict.py $1 $2 48 $3 2>&1 | grep -E "DONE|Error|Traceback"
}
for s in 42 43 44; do
  A=outputs/amazon_beauty_2014/ablations/seed_$s/arms/A_t_base
  run $A/arv_mmoe.json init $OUT/beauty_init_s$s.json
  for arm in arv_mmoe_gradnorm arv_mmoe; do run $A/$arm.json best $OUT/beauty_${arm}_best_s$s.json; done
done
echo ALL DONE
