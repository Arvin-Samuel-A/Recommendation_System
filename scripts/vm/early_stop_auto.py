#!/usr/bin/env python3
"""Early stopping for every running Beauty arm of the given seeds (patience 50 epochs).

Like early_stop.py, but finds the arms itself: any `main.py --stage ablation --study S
--arms A --seed N` process in ~/beauty14 with N in SEEDS, mapped to its fingerprint through
the per-epoch log it holds open. Exits once QUEUE is empty and no such trainer has run for
IDLE_POLLS polls.

usage: early_stop_auto.py QUEUE SEED [SEED ...]
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import early_stop as es  # noqa: E402  (stop(), val_curve(), best_of(), log())

QUEUE = sys.argv[1]
SEEDS = {int(s) for s in sys.argv[2:]}
IDLE_POLLS = 15
FP_RE = re.compile(r"/by_fingerprint/([0-9a-f]{12})/logs/pretrain_t_base\.log$")


def arg(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def trainers():
    """(fingerprint, seed, study, arm) for each running single-arm trainer of SEEDS."""
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                args = f.read().decode().split("\0")
            if os.readlink(f"/proc/{pid}/cwd") != es.ROOT or "main.py" not in args:
                continue
            seed, study, arms = arg(args, "--seed"), arg(args, "--study"), arg(args, "--arms")
            if seed is None or int(seed) not in SEEDS or not arms or "," in arms:
                continue
            fp = None
            for fd in os.listdir(f"/proc/{pid}/fd"):
                m = FP_RE.search(os.readlink(f"/proc/{pid}/fd/{fd}"))
                if m:
                    fp = m.group(1)
                    break
        except (OSError, ValueError):
            continue
        if fp:
            found.append((fp, int(seed), study, arms))
    return found


def queue_empty():
    try:
        with open(os.path.join(es.ROOT, QUEUE)) as f:
            return not f.read().strip()
    except FileNotFoundError:
        return True


def main():
    os.chdir(es.ROOT)
    es.log(f"watching seeds {sorted(SEEDS)}, patience {es.PATIENCE}, queue {QUEUE}")
    idle = 0
    while True:
        running = trainers()
        for fp, seed, study, arm in running:
            if os.path.exists(f"{es.STORE}/{fp}/backbone.json"):
                continue
            curve = es.val_curve(fp)
            if not curve:
                continue
            best_e, _ = es.best_of(curve)
            if max(curve) - best_e >= es.PATIENCE:
                es.stop(fp, seed, study, arm, max(curve))
        idle = idle + 1 if (not running and queue_empty()) else 0
        if idle >= IDLE_POLLS:
            es.log("queue empty and no trainers -- exiting")
            return
        time.sleep(es.POLL)


if __name__ == "__main__":
    main()
