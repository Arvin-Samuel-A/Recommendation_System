#!/usr/bin/env python3
"""Size-aware launcher for the remaining MovieLens seed runs.

Each queue line is "SEED STUDY:ARM". An arm starts on a GPU only when that GPU's free memory
is at least the arm's peak (PEAK_MIB) plus MARGIN_MIB, and at most one arm starts per GPU per
SETTLE_S seconds, so the previous start has reached its peak (first validation included)
before the next decision. Any queued arm that fits may start (first fit), so small arms fill
gaps a big one cannot. Each start runs launch/run_h100_gated_seed.sh pinned to that GPU, which
also handles OOM retries. Exits when the queue is empty and every run it started has finished.

usage: ml_scheduler.py QUEUE_FILE
"""
import datetime
import os
import subprocess
import sys
import time

ROOT = "/root/run13"
QUEUE = sys.argv[1]
MARGIN_MIB = 2048
SETTLE_S = int(os.environ.get("SETTLE_MIN", "25")) * 60
# RESERVE_MIB="1=16384": memory per GPU kept free for other users' jobs (subtracted from free).
RESERVE = {int(g): int(m) for g, m in
           (kv.split("=") for kv in os.environ.get("RESERVE_MIB", "").split(",") if kv)}
POLL_S = 30
DEFAULT_PEAK = 23500
# Peak MiB per arm label (measured on seeds 43/44 where noted, otherwise estimated + ~1 GB).
PEAK_MIB = {
    "G:K2": int(os.environ.get("K2_PEAK", "20000")),
    "G:K6": 27500,           # between K4 (22.8 GB measured) and K8 (30.1 GB measured)
    "G:K8": 30200,           # measured
    "H:H2_seq25": 5500,      # seq50 measured 6.8 GB
    "H:H2_seq50": 6900,      # measured
    "H:H2_seq100": 13500,
}


def log(msg):
    print(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def free_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {int(i): int(f) - RESERVE.get(int(i), 0)
            for i, f in (line.split(",") for line in out.strip().splitlines())}


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
    # LAST_START="0=<unix ts>,1=<unix ts>": starts made outside this scheduler that are still ramping.
    last_start = {int(g): float(t) for g, t in
                  (kv.split("=") for kv in os.environ.get("LAST_START", "").split(",") if kv)}
    running = []
    log(f"margin {MARGIN_MIB} MiB, settle {SETTLE_S} s, reserve {RESERVE}, peaks {PEAK_MIB}")
    while True:
        running = [(e, p) for e, p in running if p.poll() is None]
        queue = read_queue()
        if not queue and not running:
            log("queue empty and all started runs finished -- exiting")
            return
        if queue:
            free = free_mib()
            now = time.time()
            for gpu in sorted(free):
                if now - last_start.get(gpu, 0) < SETTLE_S:
                    continue
                for entry in queue:
                    need = PEAK_MIB.get(entry.split()[1], DEFAULT_PEAK) + MARGIN_MIB
                    if free[gpu] >= need:
                        queue.remove(entry)
                        write_queue(queue)
                        running.append((entry, launch(entry, gpu, need)))
                        last_start[gpu] = now
                        log(f"GPU {gpu}: {free[gpu]} MiB free -- started {entry} (needs {need})")
                        break
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
