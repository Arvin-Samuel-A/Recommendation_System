#!/usr/bin/env bash
# 2 extra Beauty lanes pinned to GPU 0 (after the 3rd MovieLens arm ramps up), and a Beauty
# final pass that waits for all 7 Beauty workers instead of the first 5.
set -u
cd /root/beauty14
[ "$(readlink /proc/1158/cwd)" = /root/beauty14 ] && kill 1158
NEW=""
for i in 6 7; do
  QUEUE=launch/queue_seeds.txt GATED=launch/run_h100_gated.sh NUM_WORKERS=0 NEED_MIB=7168 GPUS="0" START_GAP=60 \
    EXTRA_ARGS="--dataset_preset amazon_beauty_2014" START_DELAY=240 \
    setsid nohup bash launch/queue_worker_seeded.sh > logs/worker_seeds_be_$i.log 2>&1 < /dev/null &
  NEW="$NEW $!"
done
WAIT_PIDS="1153 1154 1155 1156 1157$NEW" SEEDS="43 44" GATED=launch/run_h100_gated.sh NUM_WORKERS=0 NEED_MIB=7168 GPUS="0 1" START_GAP=60 \
  EXTRA_ARGS="--dataset_preset amazon_beauty_2014" \
  setsid nohup bash launch/final_pass_seeds.sh > logs/final_pass_seeds.log 2>&1 < /dev/null &
echo "new beauty workers:$NEW final pass: $!" | tee -a /root/early_stop/launch_seeds.pids
