#!/usr/bin/env bash
# Kaggle jobs are done: restart ml_scheduler2.py with no GPU 1 reserve (2 GB margin stays).
set -u
cd /root/run13
OLD_S=923; OLD_FP=924
# launchers of the trainers already running (started by the old scheduler), so the final pass waits for them too
LAUNCHERS=$(cd /root/early_stop && python3 -c 'import sys; sys.argv=["x","/dev/null"]; import ml_scheduler2 as m; print(" ".join(str(pp) for pp,g,sa in m.trainers().values()))')
kill $OLD_FP; pkill -P $OLD_FP 2>/dev/null
kill $OLD_S; sleep 2
setsid nohup python3 -u /root/early_stop/ml_scheduler2.py launch/queue_sized.txt >> logs/ml_scheduler.log 2>&1 < /dev/null &
S=$!
WAIT_PIDS="$S $LAUNCHERS" SEEDS="43 44" GATED=launch/run_h100_gated_seed.sh NUM_WORKERS=2 NEED_MIB=25600 GPUS="0 1" START_GAP=240 \
  setsid nohup bash launch/final_pass_seeds.sh >> logs/final_pass_seeds.log 2>&1 < /dev/null &
echo "$(date -Is) swap3 (no reserve): scheduler2 $S final pass $! waits on $S $LAUNCHERS" | tee -a /root/early_stop/launch_seeds.pids
