#!/usr/bin/env bash
# Restart the unfinished gradient-conflict runs after a container restart (2026-11-11).
# Started from /root/.bashrc (container PID-1 shell) and /etc/default/ssh (sshd start). Safe to
# run any number of times: one run per boot (flock + marker), and it does nothing while a
# gradient runner is alive. The MovieLens seed 43/44 runs it used to restart are all finished;
# that version is kept as auto_recover_mlseeds.sh.
# DRY_RUN=1: print what it would do, change nothing.
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root
ES=/root/early_stop; LOG=$ES/auto_recover.log
DRY=${DRY_RUN:-0}
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
exec 9>"$ES/auto_recover.lock"; flock -n 9 || exit 0
BOOT=$(ps -o lstart= -p 1 | tr -s ' ' '_')
MARK="$ES/recovered_gc_$BOOT"
[ -e "$MARK" ] && [ "$DRY" = 0 ] && exit 0
if pgrep -f "[r]un_gc_" >/dev/null || pgrep -f "[g]rad_conflict.py" >/dev/null; then
  say "boot $BOOT: gradient runs already alive -- nothing to do"; [ "$DRY" = 0 ] && touch "$MARK"; exit 0
fi
for i in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi >/dev/null 2>&1 || { say "boot $BOOT: nvidia-smi not ready after 10 min -- giving up"; exit 1; }
[ "$DRY" = 0 ] && touch "$MARK"
for pair in "ml:gc_ml.log" "beauty:gc_beauty.log"; do
  name=${pair%%:*}; log=$ES/${pair#*:}
  if grep -q "ALL DONE" "$log" 2>/dev/null; then say "boot $BOOT: $name gradient runs already finished"; continue; fi
  if [ "$DRY" != 0 ]; then say "dry run: would start run_gc_${name}_resume.sh"; continue; fi
  (cd $ES && setsid nohup bash run_gc_${name}_resume.sh >> "$log" 2>&1 < /dev/null &)
  say "boot $BOOT: restarted $name gradient runs (run_gc_${name}_resume.sh)"
done
pgrep -f "[b]aseline_scheduler.py" >/dev/null || (cd /root/early_stop && RESERVE_MIB= MAX_RUNNING=15 setsid nohup python3 -u baseline_scheduler.py >> baseline_scheduler.log 2>&1 < /dev/null &)
