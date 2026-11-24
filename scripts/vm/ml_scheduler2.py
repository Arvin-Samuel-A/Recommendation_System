#!/usr/bin/env python3
"""Peak-budget launcher for the remaining MovieLens seed runs (no settle wait).

Each queue line is "SEED STUDY:ARM". Every MovieLens trainer running in ROOT counts as its
arm's full peak (PEAK_MIB) on its GPU from the moment it starts, whether it has reached that
peak yet or not, and so does every launcher this scheduler started whose trainer is not up
yet. An arm starts on a GPU as soon as

    total - reserve - margin - (sum of peaks budgeted on that GPU)  >=  arm peak

and the GPU's actual free memory minus reserve also covers peak + margin (a guard against
memory used by anything outside the budget). So arms start back to back without waiting
for the previous one to reach its peak, and the GPU never holds more than its budget even
when every arm peaks at once. Each start runs launch/run_h100_gated_seed.sh pinned to that
GPU (it also handles OOM retries). Exits when the queue is empty and every run it started
has finished.

usage: ml_scheduler2.py QUEUE_FILE
env:   RESERVE_MIB="1=16384"   memory per GPU kept for other users' jobs
"""
import datetime
import os
import subprocess
import sys
import time

ROOT = "/root/run13"
QUEUE = sys.argv[1]
MARGIN_MIB = 2048
POLL_S = 20
DEFAULT_PEAK = 23500
PEAK_MIB = {
    "G:K2": 19500,           # measured 19.0 GB
    "G:K6": 27500,           # between K4 (22.8 GB measured) and K8 (30.1 GB measured)
    "G:K8": 30200,           # measured
    "H:H2_seq25": 5500,      # seq50 measured 6.8 GB
    "H:H2_seq50": 6900,      # measured
    "H:H2_seq100": 13500,
}
RESERVE = {int(g): int(m) for g, m in
           (kv.split("=") for kv in os.environ.get("RESERVE_MIB", "").split(",") if kv)}


def log(msg):
    print(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def gpu_mem():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = [line.split(",") for line in out.strip().splitlines()]
    return {int(i): (int(t), int(f)) for i, t, f in rows}


def peak_of(study_arm):
    return PEAK_MIB.get(study_arm, DEFAULT_PEAK)


def arg(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def trainers():
    """{pid: (ppid, gpu, "STUDY:ARM")} for each MovieLens trainer running in ROOT."""
    found = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                args = f.read().decode().split("\0")
            if "main.py" not in args or os.readlink(f"/proc/{pid}/cwd") != ROOT:
                continue
            with open(f"/proc/{pid}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            with open(f"/proc/{ppid}/cmdline", "rb") as f:
                if b"main.py" in f.read():      # DataLoader worker of a trainer
                    continue
            with open(f"/proc/{pid}/environ", "rb") as f:
                env = dict(kv.split("=", 1) for kv in f.read().decode().split("\0") if "=" in kv)
            gpu = int(env["CUDA_VISIBLE_DEVICES"])
            study, arms = arg(args, "--study"), arg(args, "--arms")
        except (OSError, ValueError, KeyError, IndexError):
            continue
        found[int(pid)] = (ppid, gpu, f"{study}:{arms}" if arms else study)
    return found


def read_queue():
    with open(QUEUE) as f:
        return [line.strip() for line in f if line.strip()]


def write_queue(entries):
    tmp = QUEUE + ".tmp"
    with open(tmp, "w") as f:
        f.write("".join(e + "\n" for e in entries))
    os.replace(tmp, QUEUE)


def launch(entry, gpu, need):
    seed, study = entry.split()
    env = dict(os.environ, SEED=seed, STUDIES=study, GPUS=str(gpu), NEED_MIB=str(need),
               NUM_WORKERS="2", START_GAP="60")
    name = study.replace(":", "_")
    out = open(f"{ROOT}/logs/sched_{name}_seed{seed}.log", "a")
    return subprocess.Popen(["bash", "launch/run_h100_gated_seed.sh"], cwd=ROOT, env=env,
                            stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)


def main():
    os.chdir(ROOT)
    running = []   # (entry, Popen, gpu, peak)
    log(f"peak budget, margin {MARGIN_MIB} MiB, reserve {RESERVE}, peaks {PEAK_MIB}")
    while True:
        running = [r for r in running if r[1].poll() is None]
        queue = read_queue()
        if not queue and not running:
            log("queue empty and all started runs finished -- exiting")
            return
        if queue:
            mem = gpu_mem()
            budget = {g: 0 for g in mem}
            ours = {r[1].pid: r for r in running}
            seen = set()
            for pid, (ppid, gpu, sa) in trainers().items():
                budget[gpu] = budget.get(gpu, 0) + peak_of(sa)
                seen.add(ppid)
            for pid, (entry, proc, gpu, peak) in ours.items():
                if pid not in seen:          # launcher still waiting / loading data
                    budget[gpu] += peak
            for gpu in sorted(mem):
                total, free = mem[gpu]
                reserve = RESERVE.get(gpu, 0)
                room = total - reserve - MARGIN_MIB - budget[gpu]
                for entry in queue:
                    peak = peak_of(entry.split()[1])
                    if room >= peak and free - reserve >= peak + MARGIN_MIB:
                        queue.remove(entry)
                        write_queue(queue)
                        proc = launch(entry, gpu, peak + MARGIN_MIB)
                        running.append((entry, proc, gpu, peak))
                        log(f"GPU {gpu}: budget {budget[gpu]} MiB, room {room} MiB, "
                            f"free {free} MiB -- started {entry} (peak {peak})")
                        break
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
