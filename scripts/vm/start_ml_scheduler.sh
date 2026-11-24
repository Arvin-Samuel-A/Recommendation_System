#!/usr/bin/env bash
# Hand the remaining MovieLens queue to ml_scheduler.py (old lanes exit after their current
# arm) and restart the MovieLens final pass to wait for the old lanes AND the scheduler.
set -u
cd /root/run13
flock launch/queue.lock bash -c "cp launch/queue_seeds.txt launch/queue_sized.txt && : > launch/queue_seeds.txt"
K2_PEAK=19500 LAST_START="0=$(date -d 2026-11-08T21:17:28 +%s),1=$(date -d 2026-11-08T21:27:14 +%s)" \
  setsid nohup python3 -u /root/early_stop/ml_scheduler.py launch/queue_sized.txt > logs/ml_scheduler.log 2>&1 < /dev/null &
S=$!
[ "$(readlink /proc/389504/cwd)" = /root/run13 ] && kill 389504
WAIT_PIDS="1148 1149 1150 1151 389502 389503 $S" SEEDS="43 44" GATED=launch/run_h100_gated_seed.sh NUM_WORKERS=2 NEED_MIB=25600 GPUS="0 1" START_GAP=240 \
  setsid nohup bash launch/final_pass_seeds.sh > logs/final_pass_seeds.log 2>&1 < /dev/null &
echo "scheduler $S final pass $!" | tee -a /root/early_stop/launch_seeds.pids
