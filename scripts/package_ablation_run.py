"""Package the ablation run for the H100 and AWS machines and emit its launch commands.

specs/ablation-run FR-12 / design D12. Run on the Mac after `python main.py --stage
preprocess` (the full dataset by default; pass --user-sample-fraction for a user sample,
and preprocess with the same `--user_sample_fraction`):

  1. refuse unless the processed directory exists and its manifest matches the
     requested fraction/seed, is leave-one-out, and has
     stored_seq_length >= max(sequence_length_grid);
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
from data.preprocessing import SPLIT_STRATEGY  # noqa: E402

#: Files the generated launch scripts execute; they must be in the code tarball.
REQUIRED_TRACKED = (
    "main.py",
    "scripts/run_with_retry.py",
    "scripts/verify_attention_backend.py",
    "scripts/install_fa3.sh",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--user-sample-fraction", type=float, default=1.0)
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
    # Training options passed through to every main.py command when given (absent = not
    # rendered, so the configured defaults apply and the commands are unchanged).
    p.add_argument("--logit-temperature", type=float, default=None)
    p.add_argument("--learnable-logit-scale", action="store_true")
    p.add_argument("--logit-scale-max", type=float, default=None)
    p.add_argument("--shared-logit-temperature", action="store_true")
    p.add_argument("--gradnorm-renorm-scope", default=None, choices=["active", "enabled"])
    p.add_argument("--gradnorm-primary-objective", default=None,
                   choices=["next_item", "masked_item", "multi_step", "contrastive"])
    p.add_argument("--gradnorm-aux-budget", type=float, default=None)
    p.add_argument("--contrastive-interval", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None,
                   help="pretrain_epochs for every arm (main.py --epochs). An --estimate-json "
                        "measured at a different epoch count must be rescaled to match.")
    p.add_argument("--sequence-length-grid", type=int, nargs="+", default=None)
    p.add_argument("--processed-dir", default=None,
                   help="Default: derived exactly as main.py derives it from the fraction/seed.")
    p.add_argument("--h100-ssh", default=None, help="ssh destination or config alias")
    p.add_argument("--aws-ssh", default=None, help="ssh destination or config alias")
    p.add_argument("--skip-data", action="store_true")
    # specs/ablation-run FR-18: plan a machine's arms over several GPUs.
    p.add_argument("--gpus", type=int, default=1,
                   help="GPUs per machine (default 1 = one sequential loop, the FR-12 launch).")
    p.add_argument("--h100-gpus", type=int, default=None, help="Override --gpus for the H100.")
    p.add_argument("--aws-gpus", type=int, default=None, help="Override --gpus for AWS.")
    p.add_argument("--h100-gpu-ids", default=None,
                   help="Explicit CUDA_VISIBLE_DEVICES ids for the H100, e.g. 0,1 (count = N).")
    p.add_argument("--aws-gpu-ids", default=None,
                   help="Explicit CUDA_VISIBLE_DEVICES ids for AWS, e.g. 0,1,2,3.")
    p.add_argument("--estimate-json", default=None,
                   help="scripts/estimate_training_time.py --out file: training hours per arm.")
    # specs/eval-integrity FR-9: opt-in so the run-only package (and its file set) is
    # unchanged; pass it when shipping code to rescore a finished run.
    p.add_argument(
        "--rescore-scripts", action="store_true",
        help="Also write launch/rescore_{h100,aws}.sh: --stage ablation_rescore for each of "
             "the machine's studies with the run's exact base flags (eval-integrity FR-9).",
    )
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
    if manifest.get("split_strategy") != SPLIT_STRATEGY:
        problems.append(
            f"split_strategy {manifest.get('split_strategy')!r} != {SPLIT_STRATEGY!r}; "
            "re-run --stage preprocess with this code"
        )
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


def study_command(
    args, backend: str, py: str, study_var: str = '"${STUDY}"',
    study: Optional[str] = None, arms: Optional[Sequence[str]] = None,
    stage: str = "ablation",
) -> str:
    """The ONE ablation command both scripts and LAUNCH.md use (FR-12 flag set).

    FR-18: `study`/`arms` name a fixed study and an `--arms` selection; with neither the
    legacy string is rendered unchanged.
    """
    parts = [
        py, "-u", "main.py", "--stage", stage, "--study", study if study is not None else study_var,
        "--model", args.model, "--seed", str(args.seed),
        "--user_sample_fraction", f"{args.user_sample_fraction:g}",
        "--user_sample_seed", str(args.user_sample_seed),
        "--dtype", "bf16", "--mixed_precision",
        "--batch_size", str(args.batch_size),
        "--attention_backend", backend,
        "--num_workers", str(args.num_workers),
    ]
    parts += training_option_flags(args)
    if arms:
        parts += ["--arms", ",".join(arms)]
    return " ".join(parts)


def training_option_flags(args) -> List[str]:
    """The optional training flags (logit scale, GradNorm scope and primary objective,
    contrastive interval, epochs) the package was asked to pass through; empty when none
    was given."""
    out: List[str] = []
    for name in ("logit_temperature", "logit_scale_max", "gradnorm_aux_budget"):
        value = getattr(args, name, None)
        if value is not None:
            out += [f"--{name}", repr(float(value))]
    for name in ("gradnorm_renorm_scope", "gradnorm_primary_objective", "contrastive_interval",
                 "epochs"):
        value = getattr(args, name, None)
        if value is not None:
            out += [f"--{name}", str(value)]
    for name in ("learnable_logit_scale", "shared_logit_temperature"):
        if getattr(args, name, False):
            out.append(f"--{name}")
    # Narrows study H's sequence-length sweep (e.g. drop seq200). Rendered on every
    # command, collect included, so the planner and ablation_collect expect the same arms.
    grid = getattr(args, "sequence_length_grid", None)
    if grid:
        out += ["--sequence_length_grid"] + [str(int(v)) for v in grid]
    return out


def study_argv(args, backend: str, study: str) -> List[str]:
    """`main.py` argv of a study command (no interpreter), for resolving it like the run."""
    return shlex.split(study_command(args, backend, "PY", study=study))[3:]


def retry_prefix(args, py: str) -> str:
    return (
        f"{py} -u scripts/run_with_retry.py --max-retries {args.max_retries} "
        f"--wait-seconds {args.retry_wait} --"
    )


def _studies(text: str) -> List[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def machine_studies(args, machine: str) -> List[str]:
    return _studies(args.h100_studies if machine == "h100" else args.aws_studies)


def machine_backend(args, machine: str) -> str:
    return args.h100_backend if machine == "h100" else args.aws_backend


def _machine_setup(args, machine: str) -> str:
    """Interpreter selection (and, on AWS, venv activation) shared by every script."""
    if machine == "h100":
        return f'PY="${{PYTHON:-{args.h100_python}}}"'
    venv = shlex.quote(args.aws_venv)
    # Review REV-005: activate scripts may expand unset variables, which `set -u`
    # turns into a fatal error; and a silent activation failure must not let every
    # study fail later with an interpreter that has no torch.
    return (
        "set +u\n"
        f"# shellcheck disable=SC1091\nsource {venv}/bin/activate\n"
        "set -u\n"
        'PY="${PYTHON:-python}"\n'
        '"$PY" -c "import torch" || {\n'
        f'    echo "$(date -Is) torch is not importable with $PY after activating {venv}"; exit 1;\n'
        "}"
    )


def _fa3_gate(args, report: str = "logs/verify_fa3.json", log: str = "logs/verify_fa3.log") -> str:
    return (
        '"$PY" -u scripts/verify_attention_backend.py --backend flash_attention_3 '
        f"--model {args.model} \\\n"
        f"    --report {report} > {log} 2>&1 || {{\n"
        f'    echo "$(date -Is) FA3 verification failed -- see {log}"; exit 1;\n'
        "}\n"
        'echo "$(date -Is) FA3 verification passed"\n'
    )


def render_script(args, machine: str) -> str:
    """The single-GPU launch (FR-12): every study of the machine, sequentially."""
    mn = args.model.replace("-", "_")
    studies, backend = machine_studies(args, machine), machine_backend(args, machine)
    setup = _machine_setup(args, machine)
    gate = _fa3_gate(args) if backend == "flash_attention_3" else ""
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


def render_rescore_script(args, machine: str) -> str:
    """`launch/rescore_{machine}.sh` (specs/eval-integrity FR-9, design D11).

    Re-evaluates the machine's finished run with `--stage ablation_rescore`, one study at a
    time, with exactly the base flags of the machine's run commands (same sample, seed,
    precision and backend), through the retry launcher. Then `--study all` assembles the
    rescored summary; it rescores nothing (every arm already has its rescored record) and
    exits 1 when studies run on the OTHER machine are absent, which is expected.
    """
    mn = args.model.replace("-", "_")
    studies, backend = machine_studies(args, machine), machine_backend(args, machine)
    py = chr(34) + "$PY" + chr(34)
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by scripts/package_ablation_run.py for the {machine} machine -- do not edit.",
        "# Rescores the finished ablation run in this directory (specs/eval-integrity FR-8):",
        "# every saved epoch checkpoint is re-evaluated with fp32 scores and tie-averaged",
        "# metrics, nothing is trained, and no checkpoint or original result is modified.",
        "# Outputs: outputs/ablations/rescored/seed_*/, by_fingerprint/<fp>/rescore.json,",
        "# outputs/plots/rescored/. Resumable: a re-run continues where it stopped.",
        "set -uo pipefail",
        'cd "$(dirname "$0")/.."',
        "mkdir -p logs",
        _machine_setup(args, machine),
        f"for STUDY in {' '.join(studies)}; do",
        '  echo "$(date -Is) rescore ${STUDY} start"',
        f"  {retry_prefix(args, py)} \\",
        f"    {study_command(args, backend, py, stage='ablation_rescore')} \\",
        f'    > "logs/rescore_${{STUDY}}_{mn}_seed{args.seed}.log" 2>&1',
        "  STATUS=$?",
        '  echo "$(date -Is) rescore ${STUDY} exit ${STATUS}"',
        "done",
        "# The rescored summary (collects only; exit 1 if the other machine's studies are absent).",
        f"{study_command(args, backend, py, study='all', stage='ablation_rescore')} \\",
        f'  > "logs/rescore_all_{mn}_seed{args.seed}.log" 2>&1',
        'echo "$(date -Is) rescore summary exit $?"',
        'echo "$(date -Is) rescore finished"',
    ]
    return "\n".join(lines) + "\n"


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
        "Exit statuses in the `exit` lines: 0 ok; 75 out of memory after the last retry; "
        "70 at least one arm ended `failed` (every other arm still ran and every record "
        "was written; the arm's error is in its log and arm JSON, and a re-run retries "
        "only the failed arms); 1 any other error, or for `ablation_collect` / "
        "`ablation_rescore` a study that could not be assembled. A non-zero status never "
        "stops the next command.",
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


# ---------------------------------------------------------------- planner (FR-18, D20)

#: Training hours of one arm on one L4 at 10% users, 20 epochs (handoff measurements).
FALLBACK_HOURS_SEQ_LE_50 = 1.81
FALLBACK_HOURS_PER_100_POSITIONS = 4.5

#: Rough post-training hours per arm (fixed user/client counts, so not scaled by fraction).
OVERHEAD_BASE = 0.05
OVERHEAD_PERSONALIZATION = 0.25
OVERHEAD_COMMUNICATION = 0.5
OVERHEAD_GRADIENT = 0.1
OVERHEAD_EFFICIENCY = 0.05


def gpu_ids(args, machine: str) -> Optional[List[str]]:
    """The CUDA_VISIBLE_DEVICES ids a machine's arms are planned over, or None.

    None = one GPU without explicit ids = the legacy single-loop launch (byte-identical,
    AC-43). Explicit ids win over counts; they must be unique integers.
    """
    explicit = getattr(args, f"{machine}_gpu_ids", None)
    if explicit is not None:
        ids = [x.strip() for x in str(explicit).split(",")]
        if not ids or any(not x.isdigit() for x in ids):
            raise SystemExit(f"refusing: --{machine}-gpu-ids must be comma-separated integers (got {explicit!r})")
        if len(set(ids)) != len(ids):
            raise SystemExit(f"refusing: --{machine}-gpu-ids has duplicates ({explicit!r})")
        return ids
    if args.gpus < 1:
        raise SystemExit(f"refusing: --gpus must be >= 1 (got {args.gpus})")
    per_machine = getattr(args, f"{machine}_gpus", None)
    n = args.gpus if per_machine is None else per_machine
    if machine == "h100" and per_machine is None and n > 1:
        # REV-014: the shared H100 box is usually one GPU; scripts for GPUs it lacks would
        # fail their FA3 check and never run their arms.
        print(f"WARNING: --gpus {n} also applies to the H100; pass --h100-gpus 1 (or "
              "--aws-gpus instead of --gpus) if that box has a single GPU.")
    if n < 1:
        raise SystemExit(f"refusing: --{machine}-gpus must be >= 1 (got {n})")
    if n == 1:
        return None
    return [str(i) for i in range(n)]


def check_study_overlap(args) -> None:
    both = sorted(set(_studies(args.h100_studies)) & set(_studies(args.aws_studies)))
    if both:
        raise SystemExit(
            f"refusing: studies {both} are listed for both machines; a study's arms must "
            "all run on one machine (one backend, one device type)"
        )


def base_config_from_argv(argv: Sequence[str], config_factory=Config) -> Config:
    """The base Config `main.py` builds from a study command's argv (D20)."""
    import main as cli

    parsed = cli.parse_args(list(argv))
    cfg = config_factory()
    cli.apply_cli_overrides(cfg, parsed)
    cfg.apply_model_config(parsed.model)
    return cfg


def resolve_machine_arms(base: Config, studies: Sequence[str], stats_dir: str) -> List[Dict]:
    """Every arm of `studies` with its training fingerprint, resolved like the run does.

    Mirrors `main` (`_resolve_before_snapshot` resolves the dataset fields on the base
    config) and `_run_single_config` (`resolve_dataset_fields` per arm, then
    `_training_fingerprint`), reading the statistics from the LOCAL processed directory
    while the configs keep the remote's relative `processed_data_dir`.
    """
    import copy

    from training.ablation_runner import (
        _training_fingerprint, resolve_dataset_fields, resolve_study_configs,
    )

    base = copy.deepcopy(base)
    resolve_dataset_fields(base, stats_dir=stats_dir)
    arms = []
    for s_index, study in enumerate(studies):
        for a_index, cfg in enumerate(resolve_study_configs(study, base)):
            resolve_dataset_fields(cfg, stats_dir=stats_dir)
            arms.append({
                "study": study,
                "label": cfg._study_label,
                "key": f"{study}:{cfg._study_label}",
                "order": (s_index, a_index),
                "fingerprint": _training_fingerprint(cfg),
                "config": cfg,
            })
    return arms


def fallback_training_hours(cfg: Config) -> float:
    """Handoff L4 figures: 1.81 h for S <= 50, 4.5 h x S/100 above; x fraction/0.1, x epochs/20."""
    S = cfg.max_seq_length
    base = FALLBACK_HOURS_SEQ_LE_50 if S <= 50 else FALLBACK_HOURS_PER_100_POSITIONS * S / 100.0
    return base * (cfg.user_sample_fraction / 0.1) * (cfg.pretrain_epochs / 20.0)


def load_estimate_hours(path: Optional[str], model: str) -> Dict[str, float]:
    """`"study:label" -> run_hours` from `estimate_training_time.py --out` (rows without
    `run_hours`, e.g. OOM rows, are skipped)."""
    if not path:
        return {}
    with open(path) as f:
        estimate = json.load(f)
    rows = (estimate.get("models", {}).get(model, {}) or {}).get("configs", [])
    hours: Dict[str, float] = {}
    for row in rows:
        if row.get("run_hours") is None:
            continue
        for arm in row.get("arms", []):
            hours[arm] = float(row["run_hours"])
    return hours


def arm_overhead_hours(study: str, cfg: Config) -> float:
    from training.ablation_runner import STUDY_EXTRAS

    extras = STUDY_EXTRAS.get(study, ())
    hours = OVERHEAD_BASE
    if cfg.personalization_mode == "lora" and "personalization" in extras:
        hours += OVERHEAD_PERSONALIZATION
    if cfg.system_eval == "communication":
        hours += OVERHEAD_COMMUNICATION
    if "gradient" in extras:
        hours += OVERHEAD_GRADIENT
    if "efficiency" in extras or cfg.system_eval == "latency":
        hours += OVERHEAD_EFFICIENCY
    return hours


def build_groups(arms: Sequence[Dict], estimate: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Arms sharing a training fingerprint form one group, costed as one training run plus
    each arm's post-training overhead. Groups are listed in first-arm order."""
    estimate = estimate or {}
    groups: Dict[str, Dict] = {}
    for arm in arms:
        group = groups.setdefault(arm["fingerprint"], {"fingerprint": arm["fingerprint"], "arms": []})
        group["arms"].append(arm)
    out = []
    for group in groups.values():
        matched = [estimate[a["key"]] for a in group["arms"] if a["key"] in estimate]
        if matched:
            training, source = max(matched), "estimate-json"
        else:
            training, source = fallback_training_hours(group["arms"][0]["config"]), "fallback"
        overhead = sum(arm_overhead_hours(a["study"], a["config"]) for a in group["arms"])
        out.append({
            "fingerprint": group["fingerprint"],
            "arms": group["arms"],
            "training_hours": round(training, 4),
            "overhead_hours": round(overhead, 4),
            "hours": round(training + overhead, 4),
            "hours_source": source,
            "sort_key": min(a["order"] for a in group["arms"]),
        })
    return sorted(out, key=lambda g: g["sort_key"])


def lpt_assign(groups: Sequence[Dict], n_gpus: int) -> List[List[Dict]]:
    """Longest-processing-time-first: by hours descending (ties: first arm's study order),
    each group to the GPU with the least assigned hours (ties: lowest index)."""
    loads = [0.0] * n_gpus
    bins: List[List[Dict]] = [[] for _ in range(n_gpus)]
    for group in sorted(groups, key=lambda g: (-g["hours"], g["sort_key"])):
        target = min(range(n_gpus), key=lambda i: (loads[i], i))
        bins[target].append(group)
        loads[target] += group["hours"]
    return bins


def gpu_commands(bin_groups: Sequence[Dict]) -> List[Dict]:
    """A GPU's arms in (machine study order, study order); consecutive arms of one study
    become one `--study X --arms ...` command."""
    arms = sorted((a for g in bin_groups for a in g["arms"]), key=lambda a: a["order"])
    commands: List[Dict] = []
    for arm in arms:
        if commands and commands[-1]["study"] == arm["study"]:
            commands[-1]["arms"].append(arm["label"])
        else:
            commands.append({"study": arm["study"], "arms": [arm["label"]]})
    return commands


def plan_machine(args, machine: str, ids: List[str], stats_dir: str) -> Dict:
    """The multi-GPU plan for one machine (written to launch/plan_{machine}.json)."""
    studies = machine_studies(args, machine)
    backend = machine_backend(args, machine)
    arms: List[Dict] = []
    for study in studies:
        base = base_config_from_argv(study_argv(args, backend, study))
        study_arms = resolve_machine_arms(base, [study], stats_dir)
        s_index = studies.index(study)
        for arm in study_arms:
            arm["order"] = (s_index, arm["order"][1])
        arms.extend(study_arms)
    groups = build_groups(arms, load_estimate_hours(args.estimate_json, args.model))
    bins = lpt_assign(groups, len(ids))
    gpus = {}
    for gpu, groups_on_gpu in zip(ids, bins):
        for group in groups_on_gpu:
            group["gpu"] = gpu
        gpus[gpu] = {
            "arms": [a["key"] for a in sorted(
                (a for g in groups_on_gpu for a in g["arms"]), key=lambda a: a["order"])],
            "hours": round(sum(g["hours"] for g in groups_on_gpu), 4),
            "commands": gpu_commands(groups_on_gpu),
        }
    sources = sorted({g["hours_source"] for g in groups})
    return {
        "machine": machine,
        "gpu_ids": ids,
        "backend": backend,
        "studies": studies,
        "hours_source": sources[0] if len(sources) == 1 else "mixed",
        "groups": [
            {k: (([a["key"] for a in v]) if k == "arms" else v)
             for k, v in g.items() if k not in ("sort_key",)}
            for g in groups
        ],
        "gpus": gpus,
        "makespan_hours": max(v["hours"] for v in gpus.values()) if gpus else 0.0,
    }


def gpu_log_name(args, gpu: str, index: int, study: str) -> str:
    mn = args.model.replace("-", "_")
    return f"logs/ablation_{mn}_seed{args.seed}_gpu{gpu}_{index:02d}_{study}.log"


def render_gpu_script(args, machine: str, gpu: str, commands: Sequence[Dict]) -> str:
    """One GPU's share of a machine's arms (FR-18): its commands, sequentially."""
    backend = machine_backend(args, machine)
    gate = ""
    if backend == "flash_attention_3":
        gate = _fa3_gate(args, f"logs/verify_fa3_gpu{gpu}.json", f"logs/verify_fa3_gpu{gpu}.log")
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by scripts/package_ablation_run.py: GPU {gpu} of the {machine} machine "
        "-- do not edit.",
        f"# Runs its planned arms sequentially (see launch/plan_{machine}.json); each command",
        "# goes through scripts/run_with_retry.py and a failed command does not stop the next.",
        "set -uo pipefail",
        'cd "$(dirname "$0")/.."',
        "mkdir -p logs",
        _machine_setup(args, machine),
        f"export CUDA_VISIBLE_DEVICES={gpu}",
    ]
    text = "\n".join(lines) + "\n" + gate
    py = chr(34) + "$PY" + chr(34)
    for index, command in enumerate(commands, start=1):
        what = f"{command['study']}: {','.join(command['arms'])}"
        text += (
            f'echo "$(date -Is) gpu{gpu} command {index} ({what}) start"\n'
            f"{retry_prefix(args, py)} \\\n"
            f"  {study_command(args, backend, py, study=command['study'], arms=command['arms'])} \\\n"
            f'  > "{gpu_log_name(args, gpu, index, command["study"])}" 2>&1\n'
            "STATUS=$?\n"
            f'echo "$(date -Is) gpu{gpu} command {index} ({what}) exit ${{STATUS}}"\n'
        )
    return text + f'echo "$(date -Is) gpu{gpu} finished"\n'


def collect_command(args, backend: str, py: str, study: str) -> str:
    return study_command(args, backend, py, study=study, stage="ablation_collect")


def render_driver_script(args, machine: str, plan: Dict) -> str:
    """`launch/run_{machine}.sh` for N > 1: every GPU script in the background, wait, then
    `ablation_collect` for each of the machine's studies (FR-18)."""
    mn = args.model.replace("-", "_")
    backend = machine_backend(args, machine)
    busy = [g for g in plan["gpu_ids"] if plan["gpus"].get(g, {}).get("commands")]
    studies = plan["studies"]
    return (
        "#!/usr/bin/env bash\n"
        f"# Generated by scripts/package_ablation_run.py for the {machine} machine -- do not edit.\n"
        f"# {len(busy)} GPU(s) run their planned arms in parallel (launch/plan_{machine}.json);\n"
        "# when all have finished, every study's JSON and plots are assembled from the\n"
        "# persisted arms with --stage ablation_collect.\n"
        "set -uo pipefail\n"
        'cd "$(dirname "$0")/.."\n'
        "mkdir -p logs\n"
        f"{_machine_setup(args, machine)}\n"
        f"for G in {' '.join(busy)}; do\n"
        f'  echo "$(date -Is) gpu${{G}} start"\n'
        f'  bash "launch/run_{machine}_gpu${{G}}.sh" > "logs/run_{machine}_gpu${{G}}.log" 2>&1 &\n'
        "done\n"
        "wait\n"
        'echo "$(date -Is) all GPUs finished"\n'
        f"for STUDY in {' '.join(studies)}; do\n"
        f'  {collect_command(args, backend, chr(34) + "$PY" + chr(34), chr(34) + "${STUDY}" + chr(34))} \\\n'
        f'    > "logs/collect_${{STUDY}}_{mn}_seed{args.seed}.log" 2>&1\n'
        "  STATUS=$?\n"
        '  echo "$(date -Is) collect ${STUDY} exit ${STATUS}"\n'
        "done\n"
        'echo "$(date -Is) all studies collected"\n'
    )


def gpu_one_liner(machine: str, gpu: str) -> str:
    return (
        f"mkdir -p logs && setsid nohup bash launch/run_{machine}_gpu{gpu}.sh "
        f"> logs/run_{machine}_gpu{gpu}.log 2>&1 < /dev/null &"
    )


def render_multi_gpu_md(args, plans: Dict[str, Dict]) -> str:
    """Appended to LAUNCH.md when a machine runs on several GPUs (FR-18)."""
    lines = ["", "## Multi-GPU plan", ""]
    for machine, plan in plans.items():
        backend = machine_backend(args, machine)
        lines += [
            f"### {machine.upper()} -- {len(plan['gpu_ids'])} GPUs "
            f"(hours: {plan['hours_source']}; makespan ~{plan['makespan_hours']:.1f} h)",
            "",
            f"`launch/run_{machine}.sh` (the one-liner above) starts every GPU script, waits, "
            f"then collects. Plan: `launch/plan_{machine}.json`.",
            "",
            "| GPU | hours | arms |",
            "|---|---|---|",
        ]
        for gpu in plan["gpu_ids"]:
            info = plan["gpus"].get(gpu, {"hours": 0.0, "arms": []})
            lines.append(f"| {gpu} | {info['hours']:.2f} | {', '.join(info['arms']) or '-'} |")
        lines += ["", "Per-GPU detached one-liners (only for running a GPU's share by hand):", "", "```bash"]
        lines += [gpu_one_liner(machine, g) for g in plan["gpu_ids"] if plan["gpus"].get(g, {}).get("commands")]
        lines += ["```", "", "Collect (after every GPU script has finished; exit 1 lists missing arms):",
                  "", "```bash"]
        lines += [collect_command(args, backend, "python3", s) for s in plan["studies"]]
        lines += ["```", ""]
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

    # FR-18: machines with several GPUs get a plan, per-GPU scripts and a driver; a
    # single-GPU machine keeps exactly the FR-12 script.
    check_study_overlap(args)
    plans: Dict[str, Dict] = {}
    for machine in ("h100", "aws"):
        ids = gpu_ids(args, machine)
        if ids is not None:
            plans[machine] = plan_machine(args, machine, ids, processed)

    launch = {}
    for machine in ("h100", "aws"):
        if machine in plans:
            launch[f"launch/run_{machine}.sh"] = render_driver_script(args, machine, plans[machine])
            for gpu, info in plans[machine]["gpus"].items():
                if info["commands"]:
                    launch[f"launch/run_{machine}_gpu{gpu}.sh"] = render_gpu_script(
                        args, machine, gpu, info["commands"],
                    )
            launch[f"launch/plan_{machine}.json"] = json.dumps(plans[machine], indent=2) + "\n"
        else:
            launch[f"launch/run_{machine}.sh"] = render_script(args, machine)
        if args.rescore_scripts:
            # specs/eval-integrity FR-9: rescoring a finished run, one process per machine.
            launch[f"launch/rescore_{machine}.sh"] = render_rescore_script(args, machine)
    launch["launch/LAUNCH.md"] = render_launch_md(args, data_name, manifest) + (
        render_multi_gpu_md(args, plans) if plans else ""
    )
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
