"""Run a command; when it exits with status 75 (out of memory), wait and run it again.

specs/ablation-run FR-9 / design D9. On a shared GPU another user's job can push a long run
out of memory. `main.py` exits with `utils.OOM_EXIT_CODE` (75) on an OOM error, and every
stage it runs resumes from its persisted state (`last.pt`, completed ablation arms), so
re-running the SAME command continues the work. This wrapper is that loop:

  exit 0          -> stop, exit 0
  exit 75         -> wait --wait-seconds, re-run (at most --max-retries times), then exit 75
  any other exit  -> stop immediately with that status (no retry)
                     (70 = ablation arm(s) failed: a code/data error, never retried)
  SIGTERM/SIGINT  -> terminate the child, exit 143/130 without retrying

The child inherits stdout/stderr, so a `nohup ... > log` around the wrapper captures both.

Usage:
    python scripts/run_with_retry.py [--max-retries 20] [--wait-seconds 300] -- CMD [ARGS...]
"""
import argparse
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

OOM_EXIT_CODE = 75  # mirrors utils.OOM_EXIT_CODE; kept import-free so it runs anywhere
ARM_FAILED_EXIT_CODE = 70  # mirrors utils.ARM_FAILED_EXIT_CODE; never retried


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[run_with_retry {stamp}] {message}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--max-retries", type=int, default=20,
                   help="Re-runs after an exit-75 attempt (total attempts = this + 1).")
    p.add_argument("--wait-seconds", type=float, default=300.0,
                   help="Pause before each re-run.")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- CMD [ARGS...]")
    args = p.parse_args(argv)
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        p.error("no command given (use: run_with_retry.py [options] -- CMD ...)")
    if args.max_retries < 0:
        p.error("--max-retries must be >= 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    state = {"child": None, "signal": None}

    def on_signal(signum, _frame):
        state["signal"] = signum
        child = state["child"]
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def stopped_status() -> int:
        signum = state["signal"]
        _log(f"received signal {signum}; stopping without retrying")
        return 128 + int(signum)

    total = args.max_retries + 1
    status = 0
    for attempt in range(1, total + 1):
        if state["signal"] is not None:
            return stopped_status()
        _log(f"attempt {attempt}/{total}: {' '.join(args.cmd)}")
        child = subprocess.Popen(args.cmd)
        state["child"] = child
        while True:
            try:
                status = child.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if state["signal"] is not None and child.poll() is None:
                    try:
                        status = child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        status = child.wait()
                    break
        state["child"] = None
        _log(f"attempt {attempt}/{total} exited with status {status}")

        if state["signal"] is not None:
            return stopped_status()
        if status == 0:
            return 0
        if status != OOM_EXIT_CODE:
            if status == ARM_FAILED_EXIT_CODE:
                _log(f"status {status}: ablation arm(s) failed (see the command's log); not retrying")
            else:
                _log(f"status {status} is not {OOM_EXIT_CODE} (OOM); not retrying")
            return status
        if attempt == total:
            break
        _log(f"out of memory (status {OOM_EXIT_CODE}); retrying in {args.wait_seconds:g} s")
        deadline = time.monotonic() + args.wait_seconds
        while time.monotonic() < deadline:
            if state["signal"] is not None:
                return stopped_status()
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    _log(f"giving up after {total} attempts; exiting {OOM_EXIT_CODE}")
    return OOM_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
