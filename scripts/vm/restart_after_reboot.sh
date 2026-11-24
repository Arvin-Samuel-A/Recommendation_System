#!/usr/bin/env bash
# 2026-11-09: the container restarted at 06:18 and killed every run. Requeue the 9 interrupted
# MovieLens runs first (8 resume from last.pt), then restart the scheduler and final pass.
set -u
cd /root/run13
{ printf "%s\n" "43 G:K6" "44 G:K6" "43 A:arv_mmoe" "44 A:arv_mmoe" "43 H:H2_seq100" "44 H:H2_seq100" \
    "44 H:H2_seq50" "43 H:H2_seq25" "44 H:H2_seq25"; cat launch/queue_sized.txt; } > launch/queue_sized.txt.new
mv launch/queue_sized.txt.new launch/queue_sized.txt
rm -f launch/.start_gate
SETTLE_MIN=15 RESERVE_MIB="1=16384" K2_PEAK=19500 \
  setsid nohup python3 -u /root/early_stop/ml_scheduler.py launch/queue_sized.txt >> logs/ml_scheduler.log 2>&1 < /dev/null &
S=$!
WAIT_PIDS="$S" SEEDS="43 44" GATED=launch/run_h100_gated_seed.sh NUM_WORKERS=2 NEED_MIB=25600 GPUS="0 1" START_GAP=240 \
  setsid nohup bash launch/final_pass_seeds.sh >> logs/final_pass_seeds.log 2>&1 < /dev/null &
echo "$(date -Is) restart: scheduler $S final pass $!" | tee -a /root/early_stop/launch_seeds.pids
