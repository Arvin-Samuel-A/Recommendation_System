#!/usr/bin/env bash
# Beauty A:arv_mmoe with GradNorm off and loss weights fixed at 0.80 / 2.12 / 0.08, seeds 42-44.
cd /root/beauty14_fixedw
for s in 42 43 44; do
  SEED=$s STUDIES="A:arv_mmoe" NUM_WORKERS=0 NEED_MIB=7700 GPUS="0 1" START_GAP=120 \
    EXTRA_ARGS="--dataset_preset amazon_beauty_2014" \
    setsid nohup bash launch/run_h100_gated.sh > logs/fixedw_seed$s.log 2>&1 < /dev/null &
  echo "seed $s launcher $!"
done
setsid nohup python3 -u /root/early_stop/early_stop_fixedw.py launch/none.txt 42 43 44 \
  > /root/early_stop/watcher_fixedw.log 2>&1 < /dev/null &
echo "watcher $!"
