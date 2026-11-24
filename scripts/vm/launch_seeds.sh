#!/usr/bin/env bash
# Seeds 43/44: 4 MovieLens workers (run13) + 5 Beauty workers (beauty14), the Beauty
# early-stop watcher, and one final all-studies pass per dataset after its workers finish.
set -u
cd /root/run13
ML=""
for i in 1 2 3 4; do
  QUEUE=launch/queue_seeds.txt GATED=launch/run_h100_gated_seed.sh NUM_WORKERS=2 NEED_MIB=25600 GPUS="0 1" START_GAP=240 \
    setsid nohup bash launch/queue_worker_seeded.sh > logs/worker_seeds_ml_$i.log 2>&1 < /dev/null &
  ML="$ML $!"
done
WAIT_PIDS="$ML" SEEDS="43 44" GATED=launch/run_h100_gated_seed.sh NUM_WORKERS=2 NEED_MIB=25600 GPUS="0 1" START_GAP=240 \
  setsid nohup bash launch/final_pass_seeds.sh > logs/final_pass_seeds.log 2>&1 < /dev/null &
cd /root/beauty14
BE=""
for i in 1 2 3 4 5; do
  QUEUE=launch/queue_seeds.txt GATED=launch/run_h100_gated.sh NUM_WORKERS=0 NEED_MIB=7168 GPUS="1" START_GAP=60 \
    EXTRA_ARGS="--dataset_preset amazon_beauty_2014" \
    setsid nohup bash launch/queue_worker_seeded.sh > logs/worker_seeds_be_$i.log 2>&1 < /dev/null &
  BE="$BE $!"
done
WAIT_PIDS="$BE" SEEDS="43 44" GATED=launch/run_h100_gated.sh NUM_WORKERS=0 NEED_MIB=7168 GPUS="0 1" START_GAP=60 \
  EXTRA_ARGS="--dataset_preset amazon_beauty_2014" \
  setsid nohup bash launch/final_pass_seeds.sh > logs/final_pass_seeds.log 2>&1 < /dev/null &
setsid nohup python3 -u /root/early_stop/early_stop_auto.py launch/queue_seeds.txt 43 44 > /root/early_stop/watcher_seeds.log 2>&1 < /dev/null &
echo "ml workers:$ML  beauty workers:$BE" | tee /root/early_stop/launch_seeds.pids
