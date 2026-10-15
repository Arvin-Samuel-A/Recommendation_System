"""specs/ablation-run: the operational scripts (FR-3, FR-9, FR-12), run as subprocesses."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_script(args, timeout=600, **kwargs):
    return subprocess.run(
        [sys.executable] + list(args), cwd=ROOT, capture_output=True, text=True,
        timeout=timeout, **kwargs,
    )


class TestVerifyAttentionBackend:
    """T4.4 / AC-08."""

    REPORT_KEYS = (
        "torch_version", "torch_cuda", "device", "device_name", "capability", "backend",
        "backend_module", "head_dims", "seq_lens", "cases", "passed", "tolerance",
    )
    CASE_KEYS = (
        "head_dim", "seq_len", "padded", "max_abs_out_diff", "max_abs_grad_diff", "passed",
    )

    def test_sdpa_passes_and_writes_a_full_report(self, tmp_path):
        report = tmp_path / "verify.json"
        out = run_script([
            "scripts/verify_attention_backend.py", "--backend", "sdpa", "--model", "t-base",
            "--seq-lens", "50", "--report", str(report),
        ])
        assert out.returncode == 0, out.stdout + out.stderr
        data = json.loads(report.read_text())
        for key in self.REPORT_KEYS:
            assert key in data, key
        assert data["passed"] is True
        assert data["backend"] == "sdpa"
        assert data["head_dims"] == [64, 80] and data["seq_lens"] == [50]
        assert len(data["cases"]) == 4  # 2 head dims x 1 length x {padded, unpadded}
        assert {c["padded"] for c in data["cases"]} == {True, False}
        for case in data["cases"]:
            for key in self.CASE_KEYS:
                assert key in case, key
            assert case["passed"] is True

    def test_fa3_on_a_machine_without_it_exits_2(self, tmp_path):
        report = tmp_path / "verify_fa3.json"
        out = run_script([
            "scripts/verify_attention_backend.py", "--backend", "flash_attention_3",
            "--model", "t-base", "--seq-lens", "50", "--report", str(report),
        ])
        if out.returncode == 0:
            pytest.skip("FlashAttention-3 is installed here; the unavailable path is moot")
        assert out.returncode == 2, out.stdout + out.stderr
        assert "FlashAttention-3 is unavailable" in out.stderr
        data = json.loads(report.read_text())
        assert data["passed"] is False and data["exit_code"] == 2


class TestFA3InstallScripts:
    """T4.5 / AC-09. `fetch_fa3_source.sh` is never executed beyond `--help` (network)."""

    SCRIPTS = ("scripts/fetch_fa3_source.sh", "scripts/install_fa3.sh")

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_bash_syntax_and_executable_bit(self, script):
        out = subprocess.run(["bash", "-n", script], cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert os.access(os.path.join(ROOT, script), os.X_OK)
        with open(os.path.join(ROOT, script)) as f:
            assert "set -euo pipefail" in f.read()

    def test_fetch_help_exits_zero(self):
        out = subprocess.run(
            ["bash", "scripts/fetch_fa3_source.sh", "--help"], cwd=ROOT,
            capture_output=True, text=True, timeout=30,
        )
        assert out.returncode == 0
        assert "REF" in out.stdout and "OUT_DIR" in out.stdout

    def test_dry_run_fails_preflight_without_creating_a_build_dir(self, tmp_path):
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
            pytest.skip("this host is Hopper; the preflight would pass")
        build_dir = tmp_path / "fa3_build"
        out = subprocess.run(
            ["bash", "scripts/install_fa3.sh", str(tmp_path / "missing.tar.gz"),
             "--dry-run", "--build-dir", str(build_dir)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHON": sys.executable},
        )
        assert out.returncode != 0
        assert "PREFLIGHT FAILED" in out.stderr
        assert "CUDA" in out.stderr or "Hopper" in out.stderr, out.stderr
        assert not build_dir.exists()

    def test_install_ends_by_running_the_fa3_verification(self):
        with open(os.path.join(ROOT, "scripts/install_fa3.sh")) as f:
            text = f.read()
        assert "scripts/verify_attention_backend.py --backend flash_attention_3" in text
        assert "--no-build-isolation --no-deps" in text
        tail = text[text.rindex("# ---------------------------------------------------------------- 5. verify"):]
        assert 'exit "$STATUS"' in tail


class TestRunWithRetry:
    """T7.2 / AC-25."""

    @staticmethod
    def _counter_child(counter, statuses):
        """A child that exits with statuses[n] on its n-th run (last one repeats)."""
        code = (
            "import os, sys\n"
            f"p = {str(counter)!r}\n"
            "n = int(open(p).read()) if os.path.exists(p) else 0\n"
            "open(p, 'w').write(str(n + 1))\n"
            f"s = {list(statuses)!r}\n"
            "sys.exit(s[min(n, len(s) - 1)])\n"
        )
        return [sys.executable, "-c", code]

    def _wrap(self, child, *opts):
        return run_script(
            ["scripts/run_with_retry.py", "--wait-seconds", "0", *opts, "--", *child],
            timeout=120,
        )

    def test_retries_oom_until_success(self, tmp_path):
        counter = tmp_path / "n"
        out = self._wrap(self._counter_child(counter, [75, 75, 0]))
        assert out.returncode == 0, out.stdout + out.stderr
        assert counter.read_text() == "3"
        assert "attempt 1/21" in out.stdout and "attempt 3/21" in out.stdout
        assert "attempt 4/21" not in out.stdout

    def test_gives_up_after_max_retries_with_75(self, tmp_path):
        counter = tmp_path / "n"
        out = self._wrap(self._counter_child(counter, [75]), "--max-retries", "2")
        assert out.returncode == 75
        assert counter.read_text() == "3"
        assert "attempt 3/3" in out.stdout

    def test_other_failures_are_not_retried(self, tmp_path):
        counter = tmp_path / "n"
        out = self._wrap(self._counter_child(counter, [3]))
        assert out.returncode == 3
        assert counter.read_text() == "1"

    def test_sigterm_stops_the_child_and_the_loop(self, tmp_path):
        import signal
        import time

        pid_file = tmp_path / "child.pid"
        child = [
            sys.executable, "-c",
            f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(120)",
        ]
        proc = subprocess.Popen(
            [sys.executable, "scripts/run_with_retry.py", "--wait-seconds", "0", "--", *child],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 60
            while not pid_file.exists() or not pid_file.read_text():
                assert time.monotonic() < deadline, "child never started"
                time.sleep(0.1)
            child_pid = int(pid_file.read_text())
            proc.send_signal(signal.SIGTERM)
            stdout, _ = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == 128 + signal.SIGTERM
        assert "attempt 2/" not in stdout, "the wrapper retried after SIGTERM"
        time.sleep(0.2)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)


class TestPackageAblationRun:
    """T8.1 / AC-01, AC-28."""

    FRACTION = "0.5"
    FLAG_SET = (
        "--model t-base --seed 42 --user_sample_fraction 0.5 --user_sample_seed 42 "
        "--dtype bf16 --mixed_precision --batch_size 256"
    )

    @pytest.fixture
    def run_ready_dir(self, tiny_processed_dir):
        # The run config's default grid needs stored width 200; the tiny data is built
        # with that grid so the manifest check can pass (still < 20 MB).
        return tiny_processed_dir(
            user_sample_fraction=0.5, sequence_length_grid=[25, 50, 100, 200],
        )

    def _package(self, processed, out, *extra):
        return run_script([
            "scripts/package_ablation_run.py", "--user-sample-fraction", self.FRACTION,
            "--processed-dir", processed, "--out", str(out), *extra,
        ])

    @staticmethod
    def _ablation_commands(text):
        # Commands span continuation lines; join them before matching.
        joined = text.replace("\\\n", " ")
        return [line for line in joined.splitlines() if "--stage ablation --study" in line]

    def test_builds_tarballs_and_launch_files(self, run_ready_dir, tmp_path):
        import subprocess as sp
        import tarfile

        out = tmp_path / "dist"
        # In-repo run: the working tree may hold uncommitted feature code, so this test opts
        # in with --allow-dirty; the refusal itself is TestPackagerRefusesUncommittedCode.
        result = self._package(run_ready_dir, out, "--allow-dirty")
        assert result.returncode == 0, result.stdout + result.stderr

        tracked = [
            p for p in sp.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.split("\n")
            if p and os.path.isfile(os.path.join(ROOT, p))
        ]
        launch = {"launch/run_h100.sh", "launch/run_aws.sh", "launch/LAUNCH.md"}
        with tarfile.open(out / "code.tar.gz") as tar:
            members = set(tar.getnames())
        required = {
            "main.py", "scripts/run_with_retry.py", "scripts/verify_attention_backend.py",
            "scripts/install_fa3.sh",
        }
        # --allow-dirty adds launch-time scripts that are not tracked yet.
        assert members == set(tracked) | launch | required
        for bad in ("data/raw/", "data/processed", "outputs/", ".venv/", "dist/"):
            assert not any(m.startswith(bad) for m in members), bad

        with tarfile.open(out / "processed_users50pct.tar.gz") as tar:
            names = tar.getnames()
        assert "data/processed_users50pct/preprocessing_manifest.json" in names

        h100 = (out / "launch" / "run_h100.sh").read_text()
        aws = (out / "launch" / "run_aws.sh").read_text()
        md = (out / "launch" / "LAUNCH.md").read_text()

        assert h100.index("verify_attention_backend.py --backend flash_attention_3") < h100.index("for STUDY")
        assert "for STUDY in A D E H; do" in h100
        # Review REV-005: activation runs with `set -u` suspended, then a torch check.
        assert "set +u\n# shellcheck disable=SC1091\nsource /opt/pytorch/bin/activate\nset -u\n" in aws
        assert aws.index("set -u\n") < aws.index('"$PY" -c "import torch" ||') < aws.index("for STUDY")
        assert "for STUDY in B C F G; do" in aws
        assert "verify_attention_backend" not in aws

        for text, backend in ((h100, "flash_attention_3"), (aws, "sdpa")):
            commands = self._ablation_commands(text)
            assert commands
            for command in commands:
                assert "scripts/run_with_retry.py" in command
                assert self.FLAG_SET in command
                assert f"--attention_backend {backend} --num_workers 8" in command
        md_commands = self._ablation_commands(md)
        assert md_commands and all(
            "--dtype bf16 --mixed_precision" in c and "--batch_size 256" in c for c in md_commands
        )
        for machine in ("h100", "aws"):
            assert (
                f"setsid nohup bash launch/run_{machine}.sh > logs/run_{machine}.log "
                "2>&1 < /dev/null &" in md
            )
        assert "<H100_SSH>" in md and "<AWS_SSH>" in md
        for script in ("run_h100.sh", "run_aws.sh"):
            check = sp.run(["bash", "-n", str(out / "launch" / script)], capture_output=True)
            assert check.returncode == 0

    def test_refuses_a_different_fraction(self, run_ready_dir, tmp_path):
        result = run_script([
            "scripts/package_ablation_run.py", "--user-sample-fraction", "0.25",
            "--processed-dir", run_ready_dir, "--out", str(tmp_path / "dist"), "--skip-data",
        ])
        assert result.returncode != 0
        assert "user_sample_fraction" in result.stdout + result.stderr

    def test_refuses_a_narrow_stored_width(self, tiny_processed_dir, tmp_path):
        result = self._package(
            tiny_processed_dir(user_sample_fraction=0.5), tmp_path / "dist", "--skip-data",
        )
        assert result.returncode != 0
        assert "stored_seq_length 16 < 200" in result.stdout + result.stderr


class TestFA3InstallExitCodes:
    """Review REV-006: a build failure exits 3, distinct from a parity failure (1) and from
    a preflight/unavailable failure (2). Runs the real script against stub `python`, `nvcc`
    and `ninja` executables that impersonate a Hopper box whose build fails."""

    FAKE_PYTHON = r"""#!/usr/bin/env bash
case "$*" in
  *"sys.version_info"*) exit 0 ;;
  *"torch.cuda.is_available"*) exit 0 ;;
  *"get_device_capability"*) echo 9.0 ;;
  *"torch.version.cuda or"*) echo 13 ;;
  *"print(torch.__version__"*) echo "2.14.1 13.0" ;;
  *"import flash_attn_interface"*) exit 1 ;;
  *"-m pip install"*) echo "stub: compilation terminated."; exit 1 ;;
  *"verify_attention_backend.py"*) echo "stub verify must not run"; exit 0 ;;
  *) exit 0 ;;
esac
"""

    def _env(self, tmp_path):
        import stat
        import tarfile

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        stubs = {
            "python": self.FAKE_PYTHON,
            "nvcc": "#!/usr/bin/env bash\necho 'Cuda compilation tools, release 13.0, V13.0.48'\n",
            "ninja": "#!/usr/bin/env bash\nexit 0\n",
        }
        for name, text in stubs.items():
            path = bin_dir / name
            path.write_text(text)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        src = tmp_path / "src" / "flash-attention" / "hopper"
        src.mkdir(parents=True)
        (src / "setup.py").write_text(
            "\n".join(f"FLASH_ATTENTION_DISABLE_{f}" for f in (
                "FP8", "FP16", "PAGEDKV", "APPENDKV", "SOFTCAP", "LOCAL", "SPLIT", "SM80",
                "HDIM192", "HDIM256",
            ))
        )
        tarball = tmp_path / "fa3.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(tmp_path / "src" / "flash-attention", arcname="flash-attention")
        env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
               "PYTHON": str(bin_dir / "python"), "FA3_MIN_FREE_GB": "0"}
        env.pop("CUDA_HOME", None)
        return env, tarball

    def _install(self, tmp_path, env, tarball, *extra):
        return subprocess.run(
            ["bash", "scripts/install_fa3.sh", str(tarball), "--build-dir",
             str(tmp_path / "build"), "--log-dir", str(tmp_path / "logs"), *extra],
            cwd=ROOT, capture_output=True, text=True, timeout=120, env=env,
        )

    def test_failed_build_exits_3(self, tmp_path):
        env, tarball = self._env(tmp_path)
        out = self._install(tmp_path, env, tarball)
        assert out.returncode == 3, out.stdout + out.stderr
        assert "BUILD FAILED" in out.stderr
        assert "compilation terminated" in (tmp_path / "logs" / "fa3_build.log").read_text()
        assert "verifying" not in out.stdout, "verification must not run after a failed build"

    def test_dry_run_passes_preflight_on_the_stub_hopper(self, tmp_path):
        env, tarball = self._env(tmp_path)
        out = self._install(tmp_path, env, tarball, "--dry-run")
        assert out.returncode == 0, out.stdout + out.stderr
        assert "preflight passed" in out.stdout
        assert not (tmp_path / "build").exists()


class TestPackagerRefusesUncommittedCode:
    """Review REV-001: the packager refuses unless the shipped code is exactly a commit.

    Runs the real packager inside a throwaway git repository holding a copy of this
    working tree, so the result does not depend on whether the feature is committed yet.
    """

    REQUIRED = (
        "main.py", "scripts/run_with_retry.py", "scripts/verify_attention_backend.py",
        "scripts/install_fa3.sh",
    )

    @pytest.fixture
    def clean_repo(self, tmp_path):
        import shutil

        files = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.split("\n")
        repo = tmp_path / "repo"
        for rel in files:
            if not rel or rel.startswith(".claude/") or not os.path.isfile(os.path.join(ROOT, rel)):
                continue
            dest = repo / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(os.path.join(ROOT, rel), dest)
        git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid"]
        for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "snapshot"]):
            subprocess.run(git + cmd, cwd=repo, check=True, capture_output=True)
        return repo

    def _package(self, repo, processed, out, *extra):
        return subprocess.run(
            [sys.executable, str(repo / "scripts" / "package_ablation_run.py"),
             "--user-sample-fraction", "0.5", "--processed-dir", processed,
             "--out", str(out), "--skip-data", *extra],
            cwd=repo, capture_output=True, text=True, timeout=300,
        )

    @staticmethod
    def _members(out):
        import tarfile

        with tarfile.open(out / "code.tar.gz") as tar:
            return set(tar.getnames())

    def test_committed_tree_packages_without_override(self, clean_repo, tiny_processed_dir, tmp_path):
        processed = tiny_processed_dir(user_sample_fraction=0.5, sequence_length_grid=[25, 50, 100, 200])
        result = self._package(clean_repo, processed, tmp_path / "dist")
        assert result.returncode == 0, result.stdout + result.stderr
        assert set(self.REQUIRED) <= self._members(tmp_path / "dist")

    def test_modified_or_untracked_launch_file_is_refused(
        self, clean_repo, tiny_processed_dir, tmp_path,
    ):
        processed = tiny_processed_dir(user_sample_fraction=0.5, sequence_length_grid=[25, 50, 100, 200])
        with open(clean_repo / "scripts" / "run_with_retry.py", "a") as f:
            f.write("\n# local edit\n")
        subprocess.run(
            ["git", "rm", "-q", "--cached", "scripts/install_fa3.sh"], cwd=clean_repo, check=True,
        )
        with open(clean_repo / "training" / "pretrain.py", "a") as f:
            f.write("\n# local edit\n")

        refused = self._package(clean_repo, processed, tmp_path / "dist")
        text = refused.stdout + refused.stderr
        assert refused.returncode != 0
        assert "scripts/run_with_retry.py: uncommitted change" in text
        assert "scripts/install_fa3.sh: not tracked by git" in text
        assert "training/pretrain.py: uncommitted change" in text
        assert "--allow-dirty" in text
        assert not (tmp_path / "dist" / "code.tar.gz").exists(), "nothing may be written"

        allowed = self._package(clean_repo, processed, tmp_path / "dist2", "--allow-dirty")
        assert allowed.returncode == 0, allowed.stdout + allowed.stderr
        assert "WARNING (--allow-dirty)" in allowed.stdout
        assert "scripts/install_fa3.sh" in self._members(tmp_path / "dist2")
