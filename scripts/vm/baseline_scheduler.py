#!/usr/bin/env python3
"""Gated launcher for the study-X baseline 3-seed runs (2026-11-13, approved by the owner).

Runs from the frozen run dir /root/baselines_run. Each queue entry is one arm process
(main.py --stage ablation --study X --arms ARM --seed SEED, through run_with_retry.py).
Every POLL seconds it starts the next pending entry on a GPU where

    actual free MiB - reserve(gpu) - MARGIN_MIB >= peak(entry)

and no arm was started on that GPU in the last SETTLE seconds (so a new process's memory is
visible before the next decision). An entry is skipped when its arm JSON (status ok) exists
or a process for it is already running, so re-running this script after a container restart
resumes everything: unfinished arms continue from last.pt. Exits once every entry is done
or running.

env: GPUS="0 1"  RESERVE_MIB="1=16384"  MARGIN_MIB=2048  MAX_RUNNING=10  POLL=60  SETTLE=150
     DRY_RUN=1 (print the plan only)
"""
import datetime
import json
import os
import subprocess
import time

ROOT = "/root/baselines_run"
PY = "/root/run13/venv/bin/python"
ARMS = ["duorec", "sasrec", "bert4rec", "gru4rec", "bsarec"]
DATASETS = {
    "beauty": ("--dataset_preset amazon_beauty_2014", "outputs/amazon_beauty_2014/ablations"),
    "ml": ("", "outputs/ablations"),
}
#: Peak nvidia-smi MiB per process (CUDA context included), measured 2026-11-13 plus headroom.
PEAK = {("beauty", None): 1800, ("ml", "duorec"): 3000, ("ml", None): 2400}

QUEUE = (
    [("beauty", s, a) for s in (43, 44) for a in ARMS]
    + [("ml", s, "duorec") for s in (42, 43, 44)]
    + [("ml", s, a) for s in (42, 43, 44) for a in ARMS if a != "duorec"]
)

GPUS = [int(g) for g in os.environ.get("GPUS", "0 1").split()]
RESERVE = {int(g): int(m) for g, m in (kv.split("=") for kv in
           os.environ.get("RESERVE_MIB", "1=16384").split(",") if kv)}
MARGIN = int(os.environ.get("MARGIN_MIB", "2048"))
MAX_RUNNING = int(os.environ.get("MAX_RUNNING", "10"))
POLL = int(os.environ.get("POLL", "60"))
SETTLE = int(os.environ.get("SETTLE", "150"))
DRY = os.environ.get("DRY_RUN", "0") != "0"


def log(msg):
    print(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def peak(ds, arm):
    return PEAK.get((ds, arm), PEAK[(ds, None)])


def done(ds, seed, arm):
    path = os.path.join(ROOT, DATASETS[ds][1], f"seed_{seed}", "arms", "X_t_base", f"{arm}.json")
    try:
        return json.load(open(path)).get("status") == "ok"
    except (OSError, ValueError):
        return False


def running_entries():
    """(ds, seed, arm) of every live main.py study-X process in ROOT."""
    out = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            args = open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\0")
            if os.readlink(f"/proc/{pid}/cwd") != ROOT:
                continue
        except OSError:
            continue
        if "main.py" not in args or "X" not in args or "--arms" not in args:
            continue
        arm = args[args.index("--arms") + 1]
        seed = int(args[args.index("--seed") + 1])
        ds = "beauty" if "amazon_beauty_2014" in args else "ml"
        out.add((ds, seed, arm))
    return out


def free_mib():
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return {int(i): int(f) for i, f in (r.split(",") for r in rows)}


def launch(ds, seed, arm, gpu):
    extra = DATASETS[ds][0].split()
    cmd = [PY, "-u", "scripts/run_with_retry.py", "--wait-seconds", "120", "--",
           PY, "-u", "main.py", "--stage", "ablation", *extra, "--study", "X",
           "--arms", arm, "--seed", str(seed), "--num_workers", "0"]
    logf = open(os.path.join(ROOT, "logs", f"{ds}_X_{arm}_seed{seed}.log"), "a")
    p = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                         stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    log(f"started {ds} seed {seed} {arm} on GPU {gpu} (pid {p.pid})")


def main():
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log(f"queue {len(QUEUE)} entries; GPUs {GPUS} reserve {RESERVE} margin {MARGIN} "
        f"max_running {MAX_RUNNING}")
    last_start = {g: 0.0 for g in GPUS}
    while True:
        running = running_entries()
        pending = [e for e in QUEUE if e not in running and not done(*e)]
        if DRY:
            log(f"dry run: running {sorted(running)}; pending {pending}; free {free_mib()}")
            return
        if not pending:
            log(f"nothing left to start ({len(running)} running) -- exiting")
            return
        if len(running) < MAX_RUNNING:
            free = free_mib()
            entry = pending[0]
            for g in GPUS:
                if time.time() - last_start[g] < SETTLE:
                    continue
                if free.get(g, 0) - RESERVE.get(g, 0) - MARGIN >= peak(entry[0], entry[2]):
                    launch(*entry, g)
                    last_start[g] = time.time()
                    break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
