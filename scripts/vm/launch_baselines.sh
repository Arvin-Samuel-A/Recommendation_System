#!/bin/bash
# Study X (external baselines) arms, one process each, from the frozen run dir.
#   DATASET=beauty|ml  SEEDS="43 44"  ARMS="sasrec ..."  GPU=1  bash launch_baselines.sh
# Re-running the same command after a container restart resumes every unfinished arm from
# its last.pt (finished arms are reused from their arm JSON).
cd /root/baselines_run || exit 1
mkdir -p logs
GPU=${GPU:-1}
DATASET=${DATASET:-beauty}
case "$DATASET" in
  beauty) EXTRA="--dataset_preset amazon_beauty_2014"; TAG=beauty ;;
  ml)     EXTRA=""; TAG=ml ;;
  *) echo "DATASET must be beauty or ml"; exit 1 ;;
esac
for SEED in ${SEEDS:-${SEED:-42}}; do
  for ARM in ${ARMS:-sasrec bert4rec gru4rec duorec bsarec}; do
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup /root/run13/venv/bin/python -u scripts/run_with_retry.py --wait-seconds 120 -- \
      /root/run13/venv/bin/python -u main.py --stage ablation $EXTRA \
      --study X --arms $ARM --seed $SEED --num_workers 0 \
      >> logs/${TAG}_X_${ARM}_seed${SEED}.log 2>&1 < /dev/null &
    echo "$TAG seed $SEED $ARM pid $!"
    sleep 20
  done
done
