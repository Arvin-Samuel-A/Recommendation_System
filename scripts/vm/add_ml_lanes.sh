#!/usr/bin/env bash
# 2 more MovieLens lanes (32 GB start threshold: K-variant arms reach ~30 GB), and a
# MovieLens final pass that waits for all 6 MovieLens workers.
set -u
cd /root/run13
[ "$(readlink /proc/1152/cwd)" = /root/run13 ] && kill 1152
NEW=""
for i in 5 6; do
  QUEUE=launch/queue_seeds.txt GATED=launch/run_h100_gated_seed.sh NUM_WORKERS=2 NEED_MIB=32768 GPUS="0 1" START_GAP=240 \
    setsid nohup bash launch/queue_worker_seeded.sh > logs/worker_seeds_ml_$i.log 2>&1 < /dev/null &
  NEW="$NEW $!"
done
WAIT_PIDS="1148 1149 1150 1151$NEW" SEEDS="43 44" GATED=launch/run_h100_gated_seed.sh NUM_WORKERS=2 NEED_MIB=25600 GPUS="0 1" START_GAP=240 \
  setsid nohup bash launch/final_pass_seeds.sh > logs/final_pass_seeds.log 2>&1 < /dev/null &
echo "ml workers:$NEW final pass: $!" | tee -a /root/early_stop/launch_seeds.pids
