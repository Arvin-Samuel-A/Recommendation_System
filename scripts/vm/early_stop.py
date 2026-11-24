#!/usr/bin/env python3
"""Early-stop running Beauty arms after PATIENCE epochs without a new best val NDCG@10.

Kept outside ~/beauty14 so the repository's source hash is untouched. For each watched arm:
SIGTERM its trainer, register its backbone (backbone.json) from best.pt, and launch an
eval-only re-run through launch/run_h100_gated.sh (the runner sees the finished backbone
and only evaluates best.pt).
"""
import datetime
import json
import os
import re
import signal
import subprocess
import time

ROOT = "/root/beauty14"
STORE = "outputs/amazon_beauty_2014/checkpoints/ablations/by_fingerprint"
PATIENCE = 50
POLL = 60
STOP_LOG = os.path.expanduser("~/early_stop/stopped.jsonl")

# fingerprint -> (seed, study, arm)
ARMS = {
    "8499c501a6e4": (42, "F", "no_masked"),
    "f193a00de0eb": (42, "F", "no_contrastive"),
    "bfd1c727f99c": (42, "F", "no_multistep"),
    "dfaa94e1a3e6": (42, "H", "H2_seq10"),
    "7f237b057727": (42, "H", "H2_seq15"),
    "e03760df0052": (42, "H", "H2_seq5"),
    "4eb186248b77": (42, "H", "H1_no_metadata"),
    "58aa552ee385": (42, "D", "none_gradnorm"),
    "f713a25e6307": (42, "C", "mmoe"),
    "5a07da8fecff": (42, "B", "input_only"),
    "472830d6ff19": (43, "F", "full_reference"),
    "06f20969f135": (43, "F", "no_masked"),
    "5e0b3d02a615": (43, "F", "no_multistep"),
    "5eef8d81b5b0": (43, "F", "no_contrastive"),
    "f8abb4665804": (44, "F", "full_reference"),
    "486bbc090558": (44, "F", "no_masked"),
    "d415ba0d997b": (44, "F", "no_multistep"),
    "355719a34325": (44, "F", "no_contrastive"),
}
EPOCH_RE = re.compile(r"epoch=(\d+) \|.*?val_ndcg@10=([0-9.]+)")


def log(msg):
    print(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def val_curve(fp):
    with open(f"{STORE}/{fp}/logs/pretrain_t_base.log") as f:
        return {int(e): float(v) for e, v in EPOCH_RE.findall(f.read())}


def best_of(curve):
    best_e, best_v = None, -1.0
    for e in sorted(curve):
        if curve[e] > best_v:  # strict, as in run_pretrain
            best_e, best_v = e, curve[e]
    return best_e, best_v


def trainer_pid(seed, study, arm):
    want = ["--stage", "ablation", "--study", study, "--arms", arm, "--seed", str(seed)]
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                args = f.read().decode().split("\0")
            if os.readlink(f"/proc/{pid}/cwd") != ROOT:
                continue
        except OSError:
            continue
        if "main.py" not in args:
            continue
        if all(any(args[i:i + 2] == want[j:j + 2] for i in range(len(args))) for j in range(0, len(want), 2)):
            return int(pid)
    return None


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().split()[2] != "Z"


def read_best_pt(path):
    code = (
        "import json,sys,torch;c=torch.load(sys.argv[1],map_location='cpu',weights_only=False);"
        "print(json.dumps({'best_epoch':int(c['best_epoch']),'best_val_ndcg':float(c['best_val_ndcg'])}))"
    )
    out = subprocess.run([f"{ROOT}/venv/bin/python", "-c", code, path], cwd="/", check=True,
                         capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def stop(fp, seed, study, arm, last_epoch):
    pid = trainer_pid(seed, study, arm)
    if pid is None:
        log(f"{fp} seed{seed} {study}:{arm}: no trainer process found -- skipping")
        return False
    log(f"{fp} seed{seed} {study}:{arm}: SIGTERM pid {pid} at epoch {last_epoch}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(120):
        if not alive(pid):
            break
        time.sleep(1)
    else:
        log(f"{fp}: pid {pid} still alive after 120 s -- SIGKILL")
        os.kill(pid, signal.SIGKILL)
        time.sleep(5)

    store = f"{STORE}/{fp}"
    best_pt = f"{store}/t_base/best.pt"
    best = read_best_pt(f"{ROOT}/{best_pt}")
    curve = val_curve(fp)
    log_best_e, log_best_v = best_of(curve)
    if best["best_epoch"] != log_best_e:
        log(f"{fp}: best.pt epoch {best['best_epoch']} != log best {log_best_e} -- NOT registering")
        return True
    stopped_after = max(curve)
    entry = {
        "fingerprint": fp,
        "checkpoint_dir": store,
        "checkpoint_path": best_pt,
        "trained_by": f"{study}_{arm}",
        "best_val_ndcg": best["best_val_ndcg"],
        "best_epoch": best["best_epoch"],
        "early_stopped": {
            "patience": PATIENCE,
            "last_completed_epoch": stopped_after,
            "stopped_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    }
    tmp = f"{store}/backbone.json.tmp"
    with open(tmp, "w") as f:
        json.dump(entry, f, indent=2)
    os.replace(tmp, f"{store}/backbone.json")
    with open(STOP_LOG, "a") as f:
        f.write(json.dumps({"seed": seed, "study": study, "arm": arm, **entry}) + "\n")
    log(f"{fp}: registered best epoch {best['best_epoch']} val {best['best_val_ndcg']:.6f} "
        f"(last completed epoch {stopped_after})")

    env = dict(os.environ, SEED=str(seed), STUDIES=f"{study}:{arm}", NUM_WORKERS="0",
               NEED_MIB="7168", GPUS="0 1", START_GAP="60",
               EXTRA_ARGS="--dataset_preset amazon_beauty_2014")
    out = open(f"{ROOT}/logs/eval_{study}_{arm}_seed{seed}.log", "a")
    subprocess.Popen(["bash", "launch/run_h100_gated.sh"], cwd=ROOT, env=env, stdout=out,
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log(f"{fp}: launched eval-only re-run {study}:{arm} seed {seed}")
    return True


def main():
    os.chdir(ROOT)
    pending = dict(ARMS)
    while pending:
        for fp, (seed, study, arm) in list(pending.items()):
            if os.path.exists(f"{STORE}/{fp}/backbone.json"):
                log(f"{fp} seed{seed} {study}:{arm}: already registered -- done")
                del pending[fp]
                continue
            curve = val_curve(fp)
            if not curve:
                continue
            best_e, _ = best_of(curve)
            last = max(curve)
            if last - best_e >= PATIENCE:
                if stop(fp, seed, study, arm, last):
                    del pending[fp]
        if pending:
            time.sleep(POLL)
    log("all watched arms registered -- exiting")


if __name__ == "__main__":
    main()
