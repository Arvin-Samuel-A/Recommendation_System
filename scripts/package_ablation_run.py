"""Package the 10%-user ablation run for the H100 and AWS machines and emit its launch commands.

specs/ablation-run FR-12 / design D12. Run on the Mac after
`python main.py --stage preprocess --user_sample_fraction 0.1`:

  1. refuse unless the sampled processed directory exists and its manifest matches the
     requested fraction/seed with stored_seq_length >= max(sequence_length_grid);
  2. render launch/run_h100.sh, launch/run_aws.sh and launch/LAUNCH.md from ONE command
     builder (scripts and docs cannot drift) and `bash -n` both scripts;
  3. write <out>/code.tar.gz: every git-tracked file (working-tree contents) + launch/;
  4. write <out>/<processed dir>.tar.gz with members under data/<processed dir>/ (unless
     --skip-data);
  5. print the detached launch one-liners and the tarball sizes.

No host name, address or key is written anywhere unless given with --h100-ssh/--aws-ssh
(an ssh destination or ~/.ssh/config alias); LAUNCH.md otherwise carries placeholders.

Example:
    uv run python scripts/package_ablation_run.py
    uv run python scripts/package_ablation_run.py --skip-data --out /tmp/dist
"""
import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.config import Config  # noqa: E402

#: Files the generated launch scripts execute; they must be in the code tarball.
REQUIRED_TRACKED = (
    "main.py",
    "scripts/run_with_retry.py",
    "scripts/verify_attention_backend.py",
    "scripts/install_fa3.sh",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--user-sample-fraction", type=float, default=0.1)
    p.add_argument("--user-sample-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="t-base", choices=["t-small", "t-base", "t-large"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--h100-studies", default="A,D,E,H")
    p.add_argument("--aws-studies", default="B,C,F,G")
    p.add_argument("--h100-backend", default="flash_attention_3", choices=["flash_attention_3", "sdpa"])
    p.add_argument("--aws-backend", default="sdpa", choices=["flash_attention_3", "sdpa"])
    p.add_argument("--h100-python", default="python3")
    p.add_argument("--aws-venv", default="/opt/pytorch")
    p.add_argument("--max-retries", type=int, default=20)
    p.add_argument("--retry-wait", type=int, default=300)
    p.add_argument("--out", default="dist")
    p.add_argument("--processed-dir", default=None,
                   help="Default: derived exactly as main.py derives it from the fraction/seed.")
    p.add_argument("--h100-ssh", default=None, help="ssh destination or config alias")
    p.add_argument("--aws-ssh", default=None, help="ssh destination or config alias")
    p.add_argument("--skip-data", action="store_true")
    p.add_argument(
        "--allow-dirty", action="store_true",
        help="Package even if launch-time files or covered sources are untracked or have "
             "uncommitted changes (ships the working tree; untracked launch scripts are "
             "added). Without it the packager refuses (review REV-001).",
    )
    return p.parse_args(argv)


def run_config(args) -> Config:
    """The Config the launched runs will resolve, as main.py would build it."""
    cfg = Config()
    cfg.user_sample_fraction = args.user_sample_fraction
    cfg.user_sample_seed = args.user_sample_seed
    cfg.seed = args.seed
    return cfg


def target_processed_dir(cfg: Config) -> str:
    """Where the runs will look for the data on the target machines (shared with main.py)."""
    from main import derive_processed_dir

    return derive_processed_dir(cfg) or cfg.processed_data_dir


def validate_manifest(processed: str, cfg: Config) -> Dict:
    """Refuse a processed directory that is not the one the runs will need."""
    path = os.path.join(processed, "preprocessing_manifest.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"refusing: {path} not found. Run `python main.py --stage preprocess "
            f"--user_sample_fraction {cfg.user_sample_fraction:g}` first."
        )
    with open(path) as f:
        manifest = json.load(f)
    problems = []
    fraction = float(manifest.get("user_sample_fraction", 1.0))
    if abs(fraction - cfg.user_sample_fraction) > 1e-12:
        problems.append(f"user_sample_fraction {fraction:g} != {cfg.user_sample_fraction:g}")
    if fraction < 1.0 and manifest.get("user_sample_seed") != cfg.user_sample_seed:
        problems.append(f"user_sample_seed {manifest.get('user_sample_seed')} != {cfg.user_sample_seed}")
    need = max([cfg.max_seq_length] + list(cfg.sequence_length_grid))
    stored = manifest.get("stored_seq_length")
    if stored is None or int(stored) < need:
        problems.append(
            f"stored_seq_length {stored} < {need} = max(sequence_length_grid + [max_seq_length]); "
            "re-run --stage preprocess with this code"
        )
    if problems:
        raise SystemExit(f"refusing: {path}: " + "; ".join(problems))
    return manifest


# ---------------------------------------------------------------- command builder


def study_command(args, backend: str, py: str, study_var: str = '"${STUDY}"') -> str:
    """The ONE ablation command both scripts and LAUNCH.md use (FR-12 flag set)."""
    parts = [
        py, "-u", "main.py", "--stage", "ablation", "--study", study_var,
        "--model", args.model, "--seed", str(args.seed),
        "--user_sample_fraction", f"{args.user_sample_fraction:g}",
        "--user_sample_seed", str(args.user_sample_seed),
        "--dtype", "bf16", "--mixed_precision",
        "--batch_size", str(args.batch_size),
        "--attention_backend", backend,
        "--num_workers", str(args.num_workers),
    ]
    return " ".join(parts)


def retry_prefix(args, py: str) -> str:
    return (
        f"{py} -u scripts/run_with_retry.py --max-retries {args.max_retries} "
        f"--wait-seconds {args.retry_wait} --"
    )


def _studies(text: str) -> List[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def render_script(args, machine: str) -> str:
    mn = args.model.replace("-", "_")
    if machine == "h100":
        studies, backend = _studies(args.h100_studies), args.h100_backend
        setup = f'PY="${{PYTHON:-{args.h100_python}}}"'
    else:
        studies, backend = _studies(args.aws_studies), args.aws_backend
        venv = shlex.quote(args.aws_venv)
        # Review REV-005: activate scripts may expand unset variables, which `set -u`
        # turns into a fatal error; and a silent activation failure must not let every
        # study fail later with an interpreter that has no torch.
        setup = (
            "set +u\n"
            f"# shellcheck disable=SC1091\nsource {venv}/bin/activate\n"
            "set -u\n"
            'PY="${PYTHON:-python}"\n'
            '"$PY" -c "import torch" || {\n'
            f'    echo "$(date -Is) torch is not importable with $PY after activating {venv}"; exit 1;\n'
            "}"
        )
    gate = ""
    if backend == "flash_attention_3":
        gate = (
            '"$PY" -u scripts/verify_attention_backend.py --backend flash_attention_3 '
            f"--model {args.model} \\\n"
            "    --report logs/verify_fa3.json > logs/verify_fa3.log 2>&1 || {\n"
            '    echo "$(date -Is) FA3 verification failed -- see logs/verify_fa3.log"; exit 1;\n'
            "}\n"
            'echo "$(date -Is) FA3 verification passed"\n'
        )
    return (
        "#!/usr/bin/env bash\n"
        f"# Generated by scripts/package_ablation_run.py for the {machine} machine -- do not edit.\n"
        f"# Studies {', '.join(studies)} run SEQUENTIALLY on one GPU with one attention backend\n"
        f"# ({backend}); a failed study does not stop the next one. Each study goes through\n"
        "# scripts/run_with_retry.py, which re-runs it after an out-of-memory exit (75);\n"
        "# completed arms and the last finished epoch are reused on the re-run.\n"
        "set -uo pipefail\n"
        'cd "$(dirname "$0")/.."\n'
        "mkdir -p logs\n"
        f"{setup}\n"
        f"{gate}"
        f"for STUDY in {' '.join(studies)}; do\n"
        '  echo "$(date -Is) study ${STUDY} start"\n'
        f'  {retry_prefix(args, chr(34) + "$PY" + chr(34))} \\\n'
        f'    {study_command(args, backend, chr(34) + "$PY" + chr(34))} \\\n'
        f'    > "logs/ablation_${{STUDY}}_{mn}_seed{args.seed}.log" 2>&1\n'
        "  STATUS=$?\n"
        '  echo "$(date -Is) study ${STUDY} exit ${STATUS}"\n'
        "done\n"
        'echo "$(date -Is) all studies finished"\n'
    )


def launch_one_liner(machine: str) -> str:
    return (
        f"mkdir -p logs && setsid nohup bash launch/run_{machine}.sh "
        f"> logs/run_{machine}.log 2>&1 < /dev/null &"
    )


def render_launch_md(args, data_name: str, manifest: Dict) -> str:
    h100 = args.h100_ssh or "<H100_SSH>"
    aws = args.aws_ssh or "<AWS_SSH>"
    mn = args.model.replace("-", "_")
    data_tar = f"{data_name}.tar.gz"
    lines = [
        "# Ablation run launch (generated -- do not edit)",
        "",
        f"Generated by `scripts/package_ablation_run.py` (specs/ablation-run FR-12). Model "
        f"{args.model}, seed {args.seed}, {args.user_sample_fraction:g} of users "
        f"(sample seed {args.user_sample_seed}; {manifest.get('num_users_after_sampling')} users, "
        f"stored width {manifest.get('stored_seq_length')}), mixed bf16, batch size "
        f"{args.batch_size}.",
        "",
        f"- H100 (`{h100}`): studies {args.h100_studies}, backend `{args.h100_backend}`.",
        f"- AWS (`{aws}`): studies {args.aws_studies}, backend `{args.aws_backend}`.",
        "",
        "`<H100_SSH>` / `<AWS_SSH>` are ssh destinations; put ports and keys in an "
        "`~/.ssh/config` alias. `<DIR>` is the run directory on the remote machine.",
        "",
        "Every ablation command (from `launch/run_<machine>.sh`):",
        "",
        "```bash",
        f"{retry_prefix(args, 'python3')} \\",
        f"  {study_command(args, args.h100_backend, 'python3', 'A')}",
        "```",
        "",
        "## H100",
        "",
        "On the Mac, once (needs network): `bash scripts/fetch_fa3_source.sh <tag> dist`.",
        "",
        "```bash",
        f"scp dist/code.tar.gz dist/{data_tar} dist/flash-attention-src.tar.gz {h100}:<DIR>/",
        f"ssh {h100}",
        "cd <DIR> && tar xzf code.tar.gz && tar xzf " + data_tar,
        "# FlashAttention-3 (skips the build if it already imports; ends with the verification):",
        "mkdir -p logs && setsid nohup bash scripts/install_fa3.sh flash-attention-src.tar.gz "
        "> logs/fa3_install.log 2>&1 < /dev/null &",
        "tail -f logs/fa3_install.log   # wait for 'verification exit status 0'",
        "# install_fa3.sh exit status: 0 built+verified, 1 parity failure (FA3 runs but disagrees",
        "# with SDPA), 2 preflight failed / FA3 unavailable, 3 build or extraction failed",
        "# (see logs/fa3_build.log). run_h100.sh refuses to start unless the verification passes.",
        "# launch (detached):",
        launch_one_liner("h100"),
        "```",
        "",
        "## AWS",
        "",
        "```bash",
        f"scp dist/code.tar.gz dist/{data_tar} {aws}:<DIR>/",
        f"ssh {aws}",
        "cd <DIR> && tar xzf code.tar.gz && tar xzf " + data_tar,
        launch_one_liner("aws"),
        "```",
        "",
        "## Monitor",
        "",
        "```bash",
        "tail -f logs/run_<machine>.log                       # study start/exit lines",
        f"tail -f logs/ablation_<STUDY>_{mn}_seed{args.seed}.log  # one study (attempt lines from run_with_retry)",
        "pgrep -af 'main.py --stage ablation'; nvidia-smi",
        "```",
        "",
        "An out-of-memory exit (75) is retried after the wait; arms already finished are "
        "reused and an interrupted backbone resumes from its last finished epoch "
        "(`last.pt`). To stop everything, end the launch script first (so it does not "
        "start the next study), then the wrapper, which terminates its child: "
        "`pkill -TERM -f 'launch/run_'; pkill -TERM -f run_with_retry.py`.",
        "",
        "## Collect results",
        "",
        "```bash",
        f"ssh <SSH> 'cd <DIR> && tar czf results_<machine>.tar.gz outputs/ablations/seed_{args.seed} "
        "outputs/plots logs outputs/checkpoints/ablations/by_fingerprint/*/logs'",
        "# (per-arm training logs live next to each backbone, under by_fingerprint/<fp>/logs)",
        "scp <SSH>:<DIR>/results_<machine>.tar.gz .",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- archives


def git_tracked_files(root: str = ROOT) -> List[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True,
    ).stdout.decode()
    return [p for p in out.split("\0") if p]


def write_code_tarball(
    path: str, launch_files: Dict[str, str], extra_files: Sequence[str] = (),
) -> List[str]:
    """Tracked files (working-tree contents), `extra_files`, and the generated launch files."""
    members = []
    missing = []
    with tarfile.open(path, "w:gz") as tar:
        for rel in list(git_tracked_files()) + list(extra_files):
            full = os.path.join(ROOT, rel)
            if not os.path.isfile(full):
                missing.append(rel)
                continue
            tar.add(full, arcname=rel, recursive=False)
            members.append(rel)
        for rel, text in launch_files.items():
            data = text.encode()
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mode = 0o755 if rel.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
            members.append(rel)
    if missing:
        print(f"WARNING: tracked but missing from the working tree (skipped): {missing}")
    return members


def write_data_tarball(path: str, processed: str, data_name: str) -> None:
    with tarfile.open(path, "w:gz") as tar:
        tar.add(processed, arcname=f"data/{data_name}")


def git_status_entries(root: str = ROOT) -> List[tuple]:
    """(XY status, path) for every changed or untracked path, rename targets included."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
        capture_output=True, text=True, check=True,
    ).stdout
    entries = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        code, rest = line[:2], line[3:]
        for part in rest.split(" -> "):
            part = part.strip()
            if len(part) >= 2 and part[0] == '"' and part[-1] == '"':
                part = part[1:-1]
            entries.append((code, part))
    return entries


def launch_readiness_problems(root: str = ROOT) -> List[str]:
    """Why the code tarball would not be exactly a commit (review REV-001).

    The launch scripts execute `main.py` (and through it every first-party module),
    `scripts/run_with_retry.py`, `scripts/verify_attention_backend.py` and
    `scripts/install_fa3.sh`. Each must be tracked, and none of them -- nor any source the
    provenance rule covers -- may be modified or untracked, so the shipped code is a commit
    and its `source_revision` names it.
    """
    from utils import _is_covered_source

    tracked = set(git_tracked_files(root))
    problems = [
        f"{p}: not tracked by git (the code tarball would not contain it)"
        for p in REQUIRED_TRACKED if p not in tracked
    ]
    for code, path in git_status_entries(root):
        if path in REQUIRED_TRACKED and path not in tracked:
            continue  # already reported above
        if path in REQUIRED_TRACKED or _is_covered_source(path):
            kind = "untracked" if code == "??" else f"uncommitted change ({code.strip()})"
            problems.append(f"{path}: {kind}")
    return problems


def _warn_dirty_sources() -> Optional[dict]:
    from utils import get_source_revision

    revision = get_source_revision()
    if revision.get("dirty"):
        print(
            "WARNING: covered sources are dirty; the tarball carries the working tree and "
            f"its source_hash {revision.get('source_hash')} identifies it."
        )
    return revision


def _human(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n} B"


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = run_config(args)
    target = target_processed_dir(cfg)
    processed = args.processed_dir or os.path.join(ROOT, target)
    data_name = os.path.basename(os.path.normpath(target))
    manifest = validate_manifest(processed, cfg)

    # Review REV-001: refuse BEFORE writing anything when the shipped code would not be a
    # commit -- e.g. the launch scripts are still untracked and would be missing on the
    # target, or a covered source has uncommitted edits.
    problems = launch_readiness_problems()
    extra_files: List[str] = []
    if problems:
        listing = "\n  ".join(problems)
        if not args.allow_dirty:
            raise SystemExit(
                "refusing: the code tarball would not match a commit:\n  " + listing
                + "\nCommit these files first, or pass --allow-dirty to ship the working "
                "tree as-is."
            )
        print("WARNING (--allow-dirty): shipping the working tree despite:\n  " + listing)
        tracked = set(git_tracked_files())
        extra_files = [
            p for p in REQUIRED_TRACKED
            if p not in tracked and os.path.isfile(os.path.join(ROOT, p))
        ]

    launch = {
        "launch/run_h100.sh": render_script(args, "h100"),
        "launch/run_aws.sh": render_script(args, "aws"),
        "launch/LAUNCH.md": render_launch_md(args, data_name, manifest),
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "launch"), exist_ok=True)
    for rel, text in launch.items():
        dest = os.path.join(out, rel)
        with open(dest, "w") as f:
            f.write(text)
        if rel.endswith(".sh"):
            os.chmod(dest, 0o755)
            check = subprocess.run(["bash", "-n", dest], capture_output=True, text=True)
            if check.returncode != 0:
                raise SystemExit(f"generated {rel} fails `bash -n`: {check.stderr}")

    _warn_dirty_sources()
    code_tar = os.path.join(out, "code.tar.gz")
    members = write_code_tarball(code_tar, launch, extra_files)
    print(f"wrote {code_tar} ({_human(os.path.getsize(code_tar))}, {len(members)} files)")
    if not args.skip_data:
        data_tar = os.path.join(out, f"{data_name}.tar.gz")
        write_data_tarball(data_tar, processed, data_name)
        print(f"wrote {data_tar} ({_human(os.path.getsize(data_tar))}) from {processed}")
    print(f"launch files: {os.path.join(out, 'launch')}")
    print("detached launch one-liners (run in the extracted directory):")
    print(f"  H100: {launch_one_liner('h100')}")
    print(f"  AWS:  {launch_one_liner('aws')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
