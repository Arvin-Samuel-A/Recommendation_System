#!/usr/bin/env bash
# Replace ml_scheduler.py (15-min settle) with ml_scheduler2.py (peak budget, no wait).
set -u
cd /root/run13
kill 406 2>/dev/null; pkill -P 406 2>/dev/null
kill 405; sleep 2
RESERVE_MIB="1=16384" setsid nohup python3 -u /root/early_stop/ml_scheduler2.py launch/queue_sized.txt >> logs/ml_scheduler.log 2>&1 < /dev/null &
S=$!
WAIT_PIDS="$S" SEEDS="43 44" GATED=launch/run_h100_gated_seed.sh NUM_WORKERS=2 NEED_MIB=25600 GPUS="0 1" START_GAP=240 \
  setsid nohup bash launch/final_pass_seeds.sh >> logs/final_pass_seeds.log 2>&1 < /dev/null &
echo "$(date -Is) swap: scheduler2 $S final pass $!" | tee -a /root/early_stop/launch_seeds.pids
