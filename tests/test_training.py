import math
import os

import torch
import pytest

from configs.config import Config, OBJECTIVE_ORDER
from models.transformer import TransformerModel
from models.moe import MoELayer
from training.pretrain import (
    build_scheduler,
    build_model,
    save_checkpoint,
    load_checkpoint,
)
from training.gradnorm import GradNormController
from utils import get_device, resolve_dtype


@pytest.fixture
def base_config():
    cfg = Config()
    cfg.apply_model_config("t-small")
    cfg.num_items = 50
    cfg.feature_dim = 0
    cfg.use_metadata = False
    cfg.dropout = 0.0
    return cfg


class TestLRScheduler:
    def test_warmup_increases(self, base_config):
        model = TransformerModel(base_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, base_config, total_steps=100)

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1] - 1e-8

    def test_cosine_reaches_near_zero(self, base_config):
        model = TransformerModel(base_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, base_config, total_steps=100)

        for _ in range(100):
            optimizer.step()
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < 1e-5

    def test_per_batch_stepping(self, base_config):
        model = TransformerModel(base_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, base_config, total_steps=50)

        lr_at_0 = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        lr_at_1 = optimizer.param_groups[0]["lr"]
        assert lr_at_1 != lr_at_0


class TestCheckpointRoundtrip:
    def test_save_load_identical_output(self, base_config, tmp_path):
        device = get_device(base_config)
        model, moe = build_model(base_config)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(moe.parameters()), lr=1e-3,
        )
        scheduler = build_scheduler(optimizer, base_config, total_steps=100)
        gn = GradNormController(alpha=1.5, lr=1e-2)
        gn.initial_losses = {"next_item": 3.5, "masked_item": 2.1}

        B, S = 4, base_config.max_seq_length
        seq = torch.randint(1, 30, (B, S), device=device)

        model.eval()
        with torch.no_grad():
            out_before = model(seq, return_all_hidden=False).clone()

        path = str(tmp_path / "test_ckpt.pt")
        val_metrics = {"ndcg": {"mean": 0.5}, "mrr": {"mean": 0.3}}
        save_checkpoint(path, 5, model, moe, optimizer, scheduler, gn, val_metrics, base_config)

        model2, moe2 = build_model(base_config)
        optimizer2 = torch.optim.AdamW(
            list(model2.parameters()) + list(moe2.parameters()), lr=1e-3,
        )
        scheduler2 = build_scheduler(optimizer2, base_config, total_steps=100)
        gn2 = GradNormController(alpha=1.5, lr=1e-2)

        ckpt = load_checkpoint(path, model2, moe2, optimizer2, scheduler2, gn2)

        model2.eval()
        with torch.no_grad():
            out_after = model2(seq, return_all_hidden=False)

        assert torch.allclose(out_before, out_after, atol=1e-6)
        assert ckpt["epoch"] == 5
        assert abs(ckpt["val_ndcg"] - 0.5) < 1e-6
        assert gn2.initial_losses["next_item"] == 3.5


def _build_subset_processed_dir(dest, n_train=512, n_val=128, seed=0, n_test=128):
    """Materialise a small, self-consistent copy of `data/processed` under `dest`.

    REV-005: the T6.3 gate must run a FULL epoch. Subsetting the dataset is acceptable;
    subsetting the epoch is not. Everything `run_pretrain` reads is written here --
    tensorised splits, stats, negative pool, both user histories and the fixed validation
    candidate sets -- so the orchestrator runs unmodified against real MovieLens rows.
    """
    import json
    import pickle
    import shutil

    import numpy as np
    import torch

    src = os.path.join("data", "processed")
    os.makedirs(dest, exist_ok=True)

    def _subset_split(name, n):
        data = torch.load(os.path.join(src, f"{name}_tensorised.pt"), weights_only=True, mmap=True)
        n = min(n, data["sequences"].shape[0])
        out = {k: v[:n].clone() for k, v in data.items() if v is not None}
        torch.save(out, os.path.join(dest, f"{name}_tensorised.pt"))
        return out, n

    train_out, n_train = _subset_split("train", n_train)
    val_out, n_val = _subset_split("val", n_val)
    test_out, n_test = _subset_split("test", n_test)

    shutil.copy(os.path.join(src, "dataset_stats.json"), os.path.join(dest, "dataset_stats.json"))
    shutil.copy(os.path.join(src, "item_metadata.npy"), os.path.join(dest, "item_metadata.npy"))
    for fname in ("global_negative_pool.npy", "global_negative_pool.npy.meta.json"):
        shutil.copy(os.path.join(src, fname), os.path.join(dest, fname))

    # Histories are kept as distinct objects, exactly as run_pretrain asserts.
    for fname in ("train_user_history.pkl", "eval_user_history.pkl"):
        with open(os.path.join(src, fname), "rb") as f:
            hist = pickle.load(f)
        with open(os.path.join(dest, fname), "wb") as f:
            pickle.dump(hist, f)

    # The stored validation candidate sets, truncated to the validation subset and
    # re-fingerprinted so load_eval_candidates' integrity check still passes.
    from data.sequence_generator import save_eval_candidates
    for split, count in (("val", n_val), ("test", n_test)):
        cand = np.load(os.path.join(src, f"{split}_candidates.npy"), mmap_mode="r")
        save_eval_candidates(
            [np.asarray(row) for row in cand[:count]],
            os.path.join(dest, f"{split}_candidates.npy"),
        )

    # The *_examples.pkl artifacts the cold-start / long-tail / personalization stages read.
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        src_ex = os.path.join(src, f"{split}_examples.pkl")
        if not os.path.exists(src_ex):
            continue
        with open(src_ex, "rb") as f:
            examples = pickle.load(f)
        subset_ex = {
            k: (v[:count] if isinstance(v, list) else v) for k, v in examples.items()
        }
        with open(os.path.join(dest, f"{split}_examples.pkl"), "wb") as f:
            pickle.dump(subset_ex, f)

    for fname in ("cold_items.pkl", "warm_items.pkl", "popularity_buckets.pkl"):
        src_f = os.path.join(src, fname)
        if os.path.exists(src_f):
            shutil.copy(src_f, os.path.join(dest, fname))

    return n_train, n_val


MAX_SMOKE_STEPS = 12


class TestFewStepSmokeViaRunPretrain:
    """T6.3 / BUILD_ORDER Phase 11 dev-host gate: a capped-step `run_pretrain` on real
    MovieLens data.

    REV-005: the gate was previously carried by a test that never called `run_pretrain`,
    never ran validation, never wrote a log and never checked that the loss decreases.
    All four of T6.3's verifications are made here, through the real orchestrator.

    The step budget is capped via `config.max_steps_per_epoch` and the dataset is a
    subset of the real `data/processed`, so this runs inside the suite. Owner decision
    (2026-09-17): a few hundred steps is what Phase 17 gates on, so the CLI form of this
    same check (`--stage pretrain --model t-small --epochs 1 --max_steps N` against the
    FULL processed dataset) is the gate BUILD runs by hand; the uncapped full epoch is no
    longer required by BUILD_ORDER Phase 11.
    """

    @pytest.mark.slow
    def test_few_steps_via_run_pretrain(self, tmp_path):
        import json

        import numpy as np

        from training.pretrain import run_pretrain

        processed = tmp_path / "processed"
        n_train, n_val = _build_subset_processed_dir(str(processed))

        cfg = Config()
        cfg.apply_model_config("t-small")
        # REV-062 (resolved, T5.2): the smoke gate that authorises full training must not
        # certify a kernel other than the one the manifest records.
        # `steering/code-quality.md`: "MPS smoke test must fail fast if SDPA cannot
        # execute -- no silent fallback." SDPA is now the sanctioned MPS/CPU kernel
        # (FR-4, design D2b); the arming is unchanged and the gate passes because the
        # preferred backend genuinely executes.
        cfg.require_attention_backend = True
        cfg.pretrain_epochs = 1
        cfg.batch_size = 64
        cfg.max_steps_per_epoch = MAX_SMOKE_STEPS
        cfg.num_workers = 0
        cfg.processed_data_dir = str(processed)
        cfg.output_dir = str(tmp_path / "outputs")
        cfg.checkpoint_dir = str(tmp_path / "outputs" / "checkpoints")
        cfg.log_dir = str(tmp_path / "outputs" / "logs")
        cfg.plot_dir = str(tmp_path / "outputs" / "plots")
        assert cfg.processed_data_dir == str(processed), (
            f"config.processed_data_dir resolved to {cfg.processed_data_dir}, "
            f"expected the subset dir {processed}"
        )

        result = run_pretrain(cfg, "t-small")

        # (1) the capped run completed its full step budget
        windows = result["epoch_loss_windows"]
        assert len(windows) == 1, f"Expected 1 epoch, recorded {len(windows)}"
        expected_batches = min(MAX_SMOKE_STEPS, math.ceil(n_train / cfg.batch_size))
        assert windows[0]["num_batches"] == expected_batches, (
            f"Run took {windows[0]['num_batches']} steps, expected {expected_batches}"
        )

        # (2) training loss decreases across the run
        first, last = windows[0]["first_window_loss"], windows[0]["last_window_loss"]
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first, (
            f"Training loss did not decrease: first-window mean {first:.4f}, "
            f"last-window mean {last:.4f}"
        )

        # (3) validation metrics were computed and are finite
        ndcg = result["best_val_ndcg"]
        assert np.isfinite(ndcg), f"val NDCG is not finite: {ndcg}"
        assert 0.0 <= ndcg <= 1.0, f"val NDCG out of range: {ndcg}"
        assert result["best_epoch"] == 0

        # (4) checkpoints were saved
        ckpt_dir = os.path.join(cfg.checkpoint_dir, "t_small")
        best_path = os.path.join(ckpt_dir, "best.pt")
        assert os.path.exists(best_path), f"best.pt not written to {ckpt_dir}"
        assert os.path.exists(os.path.join(ckpt_dir, "epoch_000.pt"))

        ckpt = torch.load(best_path, weights_only=False)
        for key in ("best_epoch", "best_val_ndcg", "best_val_mrr", "best_val_recall",
                    "best_val_precision", "seed", "model_name", "moe_variant",
                    "use_gradnorm", "num_experts"):
            assert key in ckpt, f"best checkpoint missing FR-3 metadata field '{key}'"

        # (5) a per-epoch log line was written with every AC-07 field
        log_path = os.path.join(cfg.log_dir, "pretrain_t_small.log")
        assert os.path.exists(log_path), f"no epoch log at {log_path}"
        log_text = open(log_path).read()
        assert "epoch=0" in log_text, f"log has no epoch=0 line:\n{log_text}"
        k = cfg.eval_k
        for field in (
            "train_loss_total=", f"val_ndcg@{k}=", f"val_mrr@{k}=",
            f"val_recall@{k}=", f"val_precision@{k}=", "lr=",
            "lambda_next_item=",
        ):
            assert field in log_text, f"epoch log missing '{field}':\n{log_text}"

        # (6) AC-09 / REV-136: the checkpoint records `source_revision`, and a real
        # artifact derived from it echoes that revision VERBATIM. Both halves are read
        # back out of the files -- a test that only asserted the writing code exists
        # would not have caught the two dead stages REV-135 found in `ablations-cli`.
        assert "source_revision" in ckpt, (
            "FR-3 (amended 2026-09-18): every checkpoint must record source_revision. "
            "`config` does not substitute -- the multi-step objective change at the "
            "centre of REV-129 left `config` byte-identical."
        )
        revision = ckpt["source_revision"]
        assert set(revision) >= {"commit", "dirty", "source_hash"}, revision
        assert revision["source_hash"], "source_hash must identify the running source"

        routing_path = os.path.join(
            cfg.log_dir, "routing_diagnostics_t_small.json",
        )
        assert os.path.exists(routing_path), (
            f"no routing diagnostics artifact at {routing_path}"
        )
        with open(routing_path) as f:
            routing_report = json.load(f)
        assert routing_report.get("source_revision") == revision, (
            "AC-09: a derived artifact must echo its checkpoint's source_revision "
            f"verbatim. checkpoint={revision} "
            f"artifact={routing_report.get('source_revision')}"
        )

        # The lambda round log carries it too (steering: "metrics, diagnostics, reports,
        # round logs").
        jsonl_path = os.path.join(cfg.log_dir, "gradnorm_t_small.jsonl")
        if os.path.exists(jsonl_path):
            first = json.loads(open(jsonl_path).readline())
            assert first.get("source_revision") == revision, first
        # And the per-epoch log, which is the artifact REV-129 had to date by hand.
        assert f"source_hash={revision['source_hash']}" in log_text


class TestArtifactProvenance:
    """AC-09 / REV-136 (owner, 2026-09-18) -- `steering/code-quality.md`, "Artifact
    Provenance". Every artifact must be able to say which code produced it, from the
    file itself and without reference to prose.
    """

    def test_source_revision_identifies_the_running_source(self):
        from utils import get_source_revision

        revision = get_source_revision(refresh=True)
        assert set(revision) >= {"commit", "dirty", "source_hash"}
        assert revision["source_hash"], "the content hash is what a dirty tree is by"
        assert isinstance(revision["dirty"], bool)

    def test_dirty_tree_is_never_attributed_to_its_nearest_commit(self):
        """The inference that fails in REV-129: commit time is not edit time.

        A dirty tree must be RECORDED as dirty, and the source hash must move when the
        sources move -- otherwise two different runs share one identifier, which is
        exactly the condition that made the multi-step dispute unresolvable.

        The probe edits `evaluation/longtail.py`, a covered file under the widened rule
        (REV-138) and the one the allowlist silently missed. It is restored byte-for-byte.
        """
        import utils

        root = os.path.dirname(os.path.abspath(utils.__file__))
        original = utils._hash_source_tree(root)
        probe = os.path.join(root, "evaluation", "longtail.py")
        with open(probe, "rb") as f:
            before = f.read()
        try:
            with open(probe, "ab") as f:
                f.write(b"\n# temporary provenance probe\n")
            changed = utils._hash_source_tree(root)
        finally:
            with open(probe, "wb") as f:
                f.write(before)
        assert changed != original, (
            "the source hash must change when a covered source changes; otherwise it "
            "cannot distinguish two runs from one dirty tree"
        )
        assert utils._hash_source_tree(root) == original, (
            "hash must be stable once the probe is removed"
        )

    def test_hashed_set_equals_the_set_the_coverage_rule_defines(self, monkeypatch):
        """Coverage completeness, per `steering/code-quality.md` (amended 2026-09-18).

        Asserting that *some* change moves the hash proves the mechanism works, not that
        it covers what it must -- which is precisely how the allowlist shipped green
        while omitting `evaluation/` (REV-138). So this observes the files the hash
        ACTUALLY reads and compares them to the set the rule defines, derived
        independently here from `git ls-files`.
        """
        import builtins
        import subprocess

        import utils

        root = os.path.dirname(os.path.abspath(utils.__file__))

        listed = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.py"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
        excluded = ("tests/", ".venv/", "__pycache__/", "data/raw/", "data/processed/")
        expected = {
            p for p in listed.split("\0") if p and not p.startswith(excluded)
        }
        assert expected, "git ls-files returned nothing; the comparison would be vacuous"

        real_open = builtins.open
        read: set = set()

        def recording_open(file, *args, **kwargs):
            try:
                rel = os.path.relpath(os.fspath(file), root)
            except (TypeError, ValueError):
                rel = None
            if rel and rel.endswith(".py") and not rel.startswith(".."):
                read.add(rel)
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", recording_open)
        try:
            utils._hash_source_tree(root)
        finally:
            monkeypatch.undo()

        assert read == expected, (
            "the set of files actually hashed must equal the set the coverage rule "
            f"defines. Missing: {sorted(expected - read)}; "
            f"unexpected: {sorted(read - expected)}"
        )

        # The specific omission REV-138 was filed for: every derived artifact this
        # project publishes is produced by code in `evaluation/`.
        for must_be_covered in (
            "evaluation/longtail.py",
            "evaluation/cold_start.py",
            "evaluation/metrics.py",
            "evaluation/flops.py",
            "evaluation/gradient_diagnostics.py",
            "evaluation/routing_diagnostics.py",
            "main.py",
            "utils.py",
            "training/pretrain.py",
            "models/moe.py",
            "configs/config.py",
            "data/preprocessing.py",
        ):
            assert must_be_covered in read, f"{must_be_covered} is not hashed"
        assert not any(p.startswith("tests/") for p in read), (
            "a test cannot change a published artifact and must not be covered"
        )

    def test_dirtiness_and_hash_share_one_coverage_rule(self, monkeypatch):
        """Both halves of a revision must ask the same question (steering, REV-139).

        The old dirty predicate was `path.startswith(SOURCE_REVISION_DIRS)`, a parallel
        string test under which an edit confined to `evaluation/` or `main.py` moved
        NEITHER the hash nor the flag -- lost silently, not loudly.
        """
        import utils

        root = os.path.dirname(os.path.abspath(utils.__file__))
        real_git = utils._git

        def with_status(text):
            def _fake(*args, cwd):
                if args[0] == "status":
                    return text
                if args[0] == "rev-parse":
                    return "0" * 40
                return real_git(*args, cwd=cwd)
            return _fake

        cases = {
            " M evaluation/longtail.py": True,
            " M main.py": True,
            " M utils.py": True,
            " M training/pretrain.py": True,
            "R  models/old.py -> models/new.py": True,
            " D configs/config.py": True,
            " M tests/test_training.py": False,
            " M specs/training-pipeline/tasks.md": False,
            "?? scratch_untracked.py": False,
            "": False,
        }
        try:
            for status, expected_dirty in cases.items():
                monkeypatch.setattr(utils, "_git", with_status(status))
                revision = utils.get_source_revision(refresh=True)
                assert revision["dirty"] is expected_dirty, (
                    f"status {status!r} should report dirty={expected_dirty}"
                )
        finally:
            monkeypatch.undo()
            utils.get_source_revision(refresh=True)

        # Every file the hash covers is a file the dirty predicate accepts, and the
        # predicate is the same object -- not a second rule that happens to agree today.
        for rel in utils._covered_source_files(root):
            assert utils._is_covered_source(rel)

    def test_checkpoint_without_revision_is_recorded_as_unestablished(self):
        """A pre-AC-09 checkpoint must NOT be silently attributed to the running code.

        Substituting the current revision would assert a provenance that does not hold
        -- the silent-substitution failure `steering/code-quality.md` forbids, and the
        precise inference that produced REV-129.
        """
        from utils import artifact_source_revision, get_source_revision

        current = get_source_revision()
        degraded = artifact_source_revision({"config": {}, "epoch": 0})
        assert degraded["commit"] is None
        assert degraded["source_hash"] is None
        assert "NOT established" in degraded["note"]
        assert degraded != current

        echoed = artifact_source_revision({"source_revision": current})
        assert echoed == current, "a real revision must be echoed verbatim"

    def test_every_derived_artifact_writer_emits_the_field(self, tmp_path):
        """Each writer named by REV-136 item 3 stamps the artifact it produces.

        Checked by writing a real file with each writer and reading the field back out,
        not by inspecting the writing code.
        """
        import json

        import numpy as np
        import torch as _torch

        from evaluation.cold_start import _save_report as save_cold_start
        from evaluation.longtail import _save_report as save_longtail
        from evaluation.flops import save_efficiency_report
        from evaluation.gradient_diagnostics import save_gradient_diagnostics
        from evaluation.routing_diagnostics import save_routing_diagnostics
        from training.local_train import _save_report as save_personalization

        revision = {"commit": "deadbeef", "dirty": False, "source_hash": "0123456789abcdef"}

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.output_dir = str(tmp_path)
        cfg.log_dir = str(tmp_path / "logs")
        cfg.plot_dir = str(tmp_path / "plots")
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.plot_dir, exist_ok=True)

        metric = {"ndcg": {"mean": 0.1, "std": 0.0}}
        written = {}

        save_cold_start(cfg, "t-small", {"cold_meta_on": metric}, source_revision=revision)
        written["cold_start_t_small.json"] = None

        save_longtail(cfg, "t-small", {"head": metric}, None, source_revision=revision)
        written["longtail_t_small.json"] = None

        save_efficiency_report(
            cfg, "t-small",
            {"ms_per_recommendation": 1.0},
            {"total_parameters": 10, "model_size_bf16_mb": 1.0},
            source_revision=revision,
        )
        written["efficiency_t_small.json"] = None

        save_gradient_diagnostics(
            _torch.eye(4), {"next_item": {"mean": 1.0}}, OBJECTIVE_ORDER,
            cfg, "t-small", source_revision=revision,
        )
        written["gradient_diagnostics_t_small.json"] = None

        routing = {
            name: _torch.full((2, cfg.num_experts), 1.0 / cfg.num_experts)
            for name in OBJECTIVE_ORDER
        }
        save_routing_diagnostics(routing, cfg, "t-small", source_revision=revision)
        written["routing_diagnostics_t_small.json"] = None

        save_personalization(
            cfg, "t-small",
            {"num_users_evaluated": 0, "num_users_skipped": 0},
            source_revision=revision,
        )
        written["personalization_gain_t_small.json"] = None

        for name in written:
            path = os.path.join(cfg.log_dir, name)
            assert os.path.exists(path), f"{name} was not written"
            with open(path) as f:
                payload = json.load(f)
            assert payload.get("source_revision") == revision, (
                f"{name} does not echo source_revision: "
                f"{payload.get('source_revision')}"
            )


class TestFewStepsSmoke:
    """T6.2: one-batch / few-step T-Small gate on real MovieLens data.

    REV-005: this test was previously mislabelled as the T6.3 gate. It runs 4 optimizer
    steps over a 256-example subset with a flat LR -- a legitimate one-batch smoke test,
    and nothing more. The T6.3 gate is `TestFewStepSmokeViaRunPretrain`, which goes
    through `run_pretrain` itself.
    """

    @pytest.mark.slow
    def test_few_steps_t_small(self, tmp_path):
        import json
        import numpy as np
        import random
        from training.pretrain import (
            build_model, _build_loss_fns, train_one_step, save_checkpoint,
        )
        from training.gradnorm import GradNormController
        from data.sequence_generator import load_tensorised, build_dataloader, GlobalNegativePool
        from utils import get_device, resolve_dtype

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.batch_size = 64
        cfg.output_dir = str(tmp_path)
        cfg.checkpoint_dir = str(tmp_path / "checkpoints")
        cfg.log_dir = str(tmp_path / "logs")
        cfg.plot_dir = str(tmp_path / "plots")
        os.makedirs(cfg.log_dir, exist_ok=True)

        processed = cfg.processed_data_dir
        with open(os.path.join(processed, "dataset_stats.json")) as f:
            stats = json.load(f)
        cfg.num_items = stats["num_items"]
        cfg.num_users = stats["num_users"]
        cfg.vocab_size = stats["VOCAB_SIZE"]

        meta_path = os.path.join(processed, "item_metadata.npy")
        metadata_matrix = np.load(meta_path) if os.path.exists(meta_path) else None
        if metadata_matrix is not None:
            cfg.feature_dim = metadata_matrix.shape[-1]

        train_dataset = load_tensorised(os.path.join(processed, "train_tensorised.pt"), metadata_matrix)

        # Subset to first 256 examples to save memory
        from data.sequence_generator import RecDataset
        n = min(256, len(train_dataset))
        small_dataset = RecDataset(
            train_dataset.sequences[:n],
            train_dataset.targets[:n],
            train_dataset.user_ids[:n],
            train_dataset.lengths[:n],
            train_dataset.targets_k[:n] if train_dataset.targets_k is not None else None,
            train_dataset.metadata_matrix,
        )
        del train_dataset

        loader = build_dataloader(small_dataset, cfg, shuffle=False)
        device = get_device(cfg)
        dtype = resolve_dtype(cfg)
        rng_obj = random.Random(cfg.seed)

        model, moe = build_model(cfg)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(moe.parameters()),
            lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        gradnorm = GradNormController(cfg.gradnorm_alpha, cfg.gradnorm_lr)
        gradnorm.log_lambdas = torch.nn.Parameter(gradnorm.log_lambdas.data.to(device=device))
        gradnorm.optimizer = torch.optim.Adam([gradnorm.log_lambdas], lr=cfg.gradnorm_lr)

        loss_fns = _build_loss_fns(cfg, device)

        neg_pool = GlobalNegativePool.load(os.path.join(processed, "global_negative_pool.npy"), set())
        neg_array = neg_pool.pool

        losses_over_steps = []
        num_steps = 4
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= num_steps:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            B = batch["sequences"].shape[0]
            neg_start = (batch_idx * cfg.num_sampled_negatives) % max(1, len(neg_array) - cfg.num_sampled_negatives)
            neg_slice = neg_array[neg_start:neg_start + B * cfg.num_sampled_negatives]
            if len(neg_slice) < B * cfg.num_sampled_negatives:
                neg_slice = np.resize(neg_slice, B * cfg.num_sampled_negatives)
            batch["neg_ids"] = torch.tensor(
                neg_slice.reshape(B, cfg.num_sampled_negatives), dtype=torch.long, device=device,
            )

            loss_values, snapshots = train_one_step(
                model, moe, batch, cfg, loss_fns, gradnorm,
                optimizer, scheduler, batch_idx, rng_obj,
            )
            losses_over_steps.append(loss_values.get("next_item", 0.0))

        assert len(losses_over_steps) == num_steps, f"Expected {num_steps} steps, got {len(losses_over_steps)}"
        assert all(l > 0 for l in losses_over_steps), "All losses should be positive"
        assert all(np.isfinite(l) for l in losses_over_steps), "All losses should be finite"

        ckpt_dir = os.path.join(cfg.checkpoint_dir, "t_small")
        os.makedirs(ckpt_dir, exist_ok=True)
        save_checkpoint(
            os.path.join(ckpt_dir, "best.pt"), 0, model, moe,
            optimizer, scheduler, gradnorm, {"ndcg": {"mean": 0.1}}, cfg,
        )
        assert os.path.exists(os.path.join(ckpt_dir, "best.pt"))


class TestNegativePoolReuse:
    """T6.4 / AC-05: the pool is loaded from disk and never regenerated.

    REV-027: the previous tests could not fail if regeneration were introduced --
    `test_pool_fingerprint_stable` asserted that `np.load` is deterministic, and
    `test_pool_loaded_matches_saved` asserted `pool.fingerprint == pool._fingerprint`,
    an attribute compared against the property that returns it. Neither invoked a code
    path that would regenerate. These do.
    """

    def test_round_trip_matches_recorded_fingerprint(self, tmp_path):
        """Build with a known seed, save, reload -- the reloaded pool must carry the
        recorded fingerprint, recomputed from the bytes rather than trusted."""
        import json

        import numpy as np
        from data.sequence_generator import GlobalNegativePool

        warm = set(range(1, 500))
        built = GlobalNegativePool(warm, seed=1234)
        built.build_once(pool_size=2000)
        recorded = built.fingerprint

        path = str(tmp_path / "pool.npy")
        built.save(path)

        reloaded = GlobalNegativePool.load(path, warm)
        assert reloaded.fingerprint == recorded
        assert np.array_equal(reloaded.pool, built.pool)
        assert reloaded.seed == 1234

        with open(path + ".meta.json") as f:
            assert json.load(f)["fingerprint"] == recorded

    def test_tampered_pool_is_rejected(self, tmp_path):
        """A pool whose bytes no longer match its meta fingerprint must raise, not load
        silently -- this is what makes the fingerprint check load-bearing."""
        import numpy as np
        from data.sequence_generator import GlobalNegativePool

        warm = set(range(1, 500))
        built = GlobalNegativePool(warm, seed=1234)
        built.build_once(pool_size=2000)
        path = str(tmp_path / "pool.npy")
        built.save(path)

        tampered = np.load(path)
        tampered[0] = tampered[0] + 1
        np.save(path, tampered)

        with pytest.raises(RuntimeError, match="fingerprint mismatch"):
            GlobalNegativePool.load(path, warm)

    def test_run_pretrain_never_calls_build_once(self, tmp_path, monkeypatch):
        """AC-05 and the steering anti-pattern 'No per-epoch negative pool regeneration':
        run the real orchestrator for two epochs with `build_once` booby-trapped."""
        from data.sequence_generator import GlobalNegativePool
        from training.pretrain import run_pretrain

        def _boom(self, pool_size):
            raise AssertionError(
                "GlobalNegativePool.build_once was called during training -- the pool "
                "must be loaded from disk once, never regenerated."
            )

        monkeypatch.setattr(GlobalNegativePool, "build_once", _boom)

        processed = tmp_path / "processed"
        _build_subset_processed_dir(str(processed), n_train=128, n_val=32)

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.pretrain_epochs = 2
        cfg.batch_size = 64
        cfg.max_steps_per_epoch = 2
        cfg.num_workers = 0
        cfg.processed_data_dir = str(processed)
        cfg.output_dir = str(tmp_path / "outputs")
        cfg.checkpoint_dir = str(tmp_path / "outputs" / "checkpoints")
        cfg.log_dir = str(tmp_path / "outputs" / "logs")
        cfg.plot_dir = str(tmp_path / "outputs" / "plots")

        result = run_pretrain(cfg, "t-small")
        assert len(result["epoch_loss_windows"]) == 2


class TestMoECacheLifetime:
    """REV-071: the expert cache outlived the step that created it.

    `_cached_expert_outputs` is a (B, S, K, D) tensor still attached to the step's
    autograd graph -- ~131 MB of activations plus the graph at batch_size=256, S=50,
    K=4, D=320 in bf16. `clear_cache()` was called only at the START of a step, so the
    previous step's activations survived from `optimizer.step()` until the next step
    began, and both allocations coexisted during the next forward. The cache can never
    be hit across steps anyway: every step brings a fresh `all_hidden`.
    """

    @staticmethod
    def _setup():
        import random

        import torch
        from torch.optim.lr_scheduler import LambdaLR

        from configs.config import Config, OBJECTIVE_ORDER
        from training.pretrain import build_model, _build_loss_fns

        config = Config()
        config.apply_model_config("t-small")
        config.num_items = 200
        config.vocab_size = 202
        config.use_metadata = False
        config.feature_dim = 0
        config.device = "cpu"
        config.moe_variant = "arv_mmoe"
        config.batch_size = 4

        torch.manual_seed(0)
        model, moe = build_model(config)
        loss_fns = _build_loss_fns(config, torch.device("cpu"))
        params = list(model.parameters()) + list(moe.parameters())
        for fn in loss_fns.values():
            params += list(fn.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        return config, model, moe, loss_fns, optimizer, scheduler, random.Random(0)

    @staticmethod
    def _batch(config):
        import torch

        b = config.batch_size
        return {
            "sequences": torch.randint(1, config.num_items + 1, (b, config.max_seq_length)),
            "targets": torch.randint(1, config.num_items + 1, (b,)),
            "neg_ids": torch.randint(
                1, config.num_items + 1, (b, config.num_sampled_negatives),
            ),
            "targets_k": torch.randint(1, config.num_items + 1, (b, config.multi_step_k)),
        }

    def test_cache_is_released_when_the_step_returns(self):
        from training.pretrain import train_one_step

        config, model, moe, loss_fns, optimizer, scheduler, rng = self._setup()
        loss_values, _ = train_one_step(
            model, moe, self._batch(config), config, loss_fns, None,
            optimizer, scheduler, 0, rng,
        )

        assert loss_values["total"] is not None
        assert moe._cached_expert_outputs is None, (
            "the step's expert activations are still retained after it returned"
        )
        assert moe._cache_input is None, (
            "the retained input reference keeps the previous graph alive"
        )
        assert moe._cache_key is None

    def test_repeated_steps_stay_released(self):
        from training.pretrain import train_one_step

        config, model, moe, loss_fns, optimizer, scheduler, rng = self._setup()
        for step in range(3):
            train_one_step(
                model, moe, self._batch(config), config, loss_fns, None,
                optimizer, scheduler, step, rng,
            )
            assert moe._cached_expert_outputs is None, f"cache retained after step {step}"

    def test_cache_still_serves_the_objectives_within_one_step(self):
        """Releasing at the end must not break the within-step sharing FR-1 requires.

        The three objectives that score the ORIGINAL sequence share one `all_hidden`, so
        the experts must run once for all of them. The contrastive objective forwards an
        augmented view, which is a genuinely different input and correctly misses the
        cache, so it is disabled here to isolate the sharing being tested.
        """
        from training.pretrain import train_one_step

        config, model, moe, loss_fns, optimizer, scheduler, rng = self._setup()
        config.use_contrastive = False

        calls = []
        handle = moe.experts.register_forward_pre_hook(
            lambda module, args: calls.append(args[0].shape)
        )
        try:
            train_one_step(
                model, moe, self._batch(config), config, loss_fns, None,
                optimizer, scheduler, 0, rng,
            )
        finally:
            handle.remove()

        assert len(calls) == 1, (
            f"experts ran {len(calls)} times in one step; the shared-expert cache "
            f"should compute them once for every objective scoring the same sequence"
        )
