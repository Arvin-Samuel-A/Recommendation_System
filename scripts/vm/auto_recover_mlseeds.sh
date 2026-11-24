#!/usr/bin/env bash
# Restart the MovieLens seed 43/44 runs after a container restart (2026-11-10).
# Started from /root/.bashrc when the container's PID-1 shell starts (guarded by $$ == 1), and
# by the Mac-side watcher. Safe to run any number of times: one run per boot (flock + marker),
# and it does nothing while any MovieLens trainer, scheduler or final pass is alive.
# Requeues every seed 43/44 arm without a result JSON (interrupted arms resume from last.pt),
# then starts ml_scheduler2.py (no reserve, 2 GB margin) and the final pass.
# DRY_RUN=1: print what it would do, change nothing (IGNORE_ALIVE=1 with it to test the queue).
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root
ROOT=/root/run13; ES=/root/early_stop; LOG=$ES/auto_recover.log
DRY=${DRY_RUN:-0}
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
exec 9>"$ES/auto_recover.lock"; flock -n 9 || exit 0
BOOT=$(ps -o lstart= -p 1 | tr -s ' ' '_')
MARK="$ES/recovered_$BOOT"
[ -e "$MARK" ] && [ "$DRY" = 0 ] && exit 0
if [ "${IGNORE_ALIVE:-0}" = 0 ] && { pgrep -f ml_scheduler2.py >/dev/null || pgrep -f final_pass_seeds.sh >/dev/null \
   || pgrep -f "main.py --stage ablation" >/dev/null; }; then
  say "boot $BOOT: runs already alive -- nothing to do"; [ "$DRY" = 0 ] && touch "$MARK"; exit 0
fi
# wait for the GPUs (driver may lag the container start)
for i in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi >/dev/null 2>&1 || { say "boot $BOOT: nvidia-smi not ready after 10 min -- giving up"; exit 1; }
cd "$ROOT"
ARMS="A:arv_mmoe_gradnorm A:single_obj_baseline A:conv_mmoe A:arv_mmoe A:4obj_no_routing B:input_only
C:mmoe D:none_gradnorm F:no_masked F:no_multistep F:no_contrastive G:K2 G:K6 G:K8
H:H2_seq25 H:H2_seq50 H:H2_seq100 H:H1_no_metadata"
Q=""
for sa in $ARMS; do for s in 43 44; do
  st=${sa%%:*}; arm=${sa#*:}
  [ -e "outputs/ablations/seed_$s/arms/${st}_t_base/$arm.json" ] || Q+="$s $sa"$'\n'
done; done
say "boot $BOOT: requeue $(printf %s "$Q" | grep -c .) runs: $(printf %s "$Q" | tr '\n' ';')"
if [ "$DRY" != 0 ]; then say "dry run -- stopping here"; exit 0; fi
touch "$MARK"
printf %s "$Q" > launch/queue_sized.txt.new && mv launch/queue_sized.txt.new launch/queue_sized.txt
rm -f launch/.start_gate
setsid nohup python3 -u $ES/ml_scheduler2.py launch/queue_sized.txt >> logs/ml_scheduler.log 2>&1 < /dev/null &
S=$!
WAIT_PIDS="$S" SEEDS="43 44" GATED=launch/run_h100_gated_seed.sh NUM_WORKERS=2 NEED_MIB=25600 GPUS="0 1" START_GAP=240 \
  setsid nohup bash launch/final_pass_seeds.sh >> logs/final_pass_seeds.log 2>&1 < /dev/null &
say "boot $BOOT: started scheduler2 $S, final pass $!"
echo "$(date -Is) auto_recover: scheduler2 $S final pass $!" >> $ES/launch_seeds.pids
