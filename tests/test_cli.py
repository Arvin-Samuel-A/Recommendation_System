import copy
import json
import inspect
import os
import textwrap
import pytest
from unittest.mock import patch, MagicMock

from configs.config import Config
from evaluation.metrics import build_full_ranking
from main import parse_args, apply_cli_overrides, STAGE_MAP, run_all_pipeline
from training.ablation_runner import resolve_study_configs, STUDY_CONFIGS
from models.transformer import TransformerModel


class TestCLIParsing:
    def test_all_stages_accepted(self):
        for stage in ["preprocess", "pretrain",
                       "evaluate", "cold_start", "longtail", "latency",
                       "routing_diagnostics", "gradient_diagnostics",
                       "ablation", "plots", "all"]:
            args = parse_args(["--stage", stage])
            assert args.stage == stage

    def test_removed_lora_and_federated_stages_are_rejected(self):
        for stage in ("personalize", "federated", "lora_rank_sweep"):
            with pytest.raises(SystemExit):
                parse_args(["--stage", stage])

    def test_model_choices(self):
        for model in ["t-small", "t-base", "t-large"]:
            args = parse_args(["--stage", "pretrain", "--model", model])
            assert args.model == model

    def test_overrides_applied(self):
        args = parse_args(["--stage", "pretrain", "--epochs", "5", "--batch_size", "32"])
        config = Config()
        apply_cli_overrides(config, args)
        assert config.pretrain_epochs == 5
        assert config.batch_size == 32

    def test_max_steps_override(self):
        """The Phase 17 few-step gate must be runnable without editing source.

        Owner decision (2026-09-17): the smoke gate is a capped run of `run_pretrain` on
        real data, so `max_steps_per_epoch` needs a CLI surface -- `conventions.md`
        ("Config-Driven Design") forbids requiring a source edit to run an experiment.
        """
        config = Config()
        apply_cli_overrides(config, parse_args(["--stage", "pretrain"]))
        assert config.max_steps_per_epoch == 0, "uncapped must stay the default"

        args = parse_args(
            ["--stage", "pretrain", "--model", "t-small", "--epochs", "1",
             "--max_steps", "200"]
        )
        config = Config()
        apply_cli_overrides(config, args)
        assert config.max_steps_per_epoch == 200
        assert config.pretrain_epochs == 1

    def test_gradnorm_override(self):
        args = parse_args(["--stage", "pretrain", "--no_gradnorm"])
        config = Config()
        config.use_gradnorm = True
        apply_cli_overrides(config, args)
        assert config.use_gradnorm is False


class TestAblationConfigResolution:
    def test_study_A_produces_5_configs(self):
        config = Config()
        configs = resolve_study_configs("A", config)
        assert len(configs) == 5

    def test_all_studies_produce_expected_count(self):
        config = Config()
        expected = {
            "A": 5, "B": 2, "C": 2, "D": 4, "F": 4,
            # REV-067: G and H2 are sweeps over config grids, so their arm counts are
            # derived from the config rather than asserted as literals.
            "G": len(config.expert_count_grid),
            "H": 4 + len(config.sequence_length_grid),
        }
        assert set(expected) == set(STUDY_CONFIGS), "study E was removed with LoRA"
        for study, count in expected.items():
            configs = resolve_study_configs(study, config)
            assert len(configs) == count, f"Study {study}: expected {count}, got {len(configs)}"

    def test_grid_studies_track_config_not_source(self):
        """REV-067: changing a sweep must be a config change, never a source edit.

        `expert_count_grid` and `sequence_length_grid` existed in `configs/config.py` and
        were read by nothing -- G hardcoded `[2, 4, 6, 8]` and H2 was a single `seq25`
        arm where FR-13 specifies a four-point sweep.
        """
        config = Config()

        g_arms = resolve_study_configs("G", config)
        assert [c.num_experts for c in g_arms] == config.expert_count_grid
        assert [c._study_label for c in g_arms] == [
            f"K{k}" for k in config.expert_count_grid
        ]

        h2_arms = [
            c for c in resolve_study_configs("H", config)
            if c._study_label.startswith("H2_")
        ]
        assert [c.max_seq_length for c in h2_arms] == config.sequence_length_grid

        # An overridden grid must flow through with no change to ablation_runner.py.
        custom = Config()
        custom.expert_count_grid = [3, 16]
        custom.sequence_length_grid = [10, 20, 30]
        assert [c.num_experts for c in resolve_study_configs("G", custom)] == [3, 16]
        custom_h2 = [
            c for c in resolve_study_configs("H", custom)
            if c._study_label.startswith("H2_")
        ]
        assert [c.max_seq_length for c in custom_h2] == [10, 20, 30]

    def test_grid_marker_never_leaks_into_recorded_overrides(self):
        """The `_grid` marker is expansion machinery, not a config field.

        `_study_overrides` is what makes an arm self-describing in its result JSON
        (AC-06), so it must carry the resolved value, not the marker.
        """
        config = Config()
        for study in ("G", "H"):
            for cfg in resolve_study_configs(study, config):
                assert "_grid" not in cfg._study_overrides

        g_first = resolve_study_configs("G", config)[0]
        assert g_first._study_overrides == {"num_experts": config.expert_count_grid[0]}

    def test_empty_grid_raises_rather_than_running_nothing(self):
        config = Config()
        config.expert_count_grid = []
        with pytest.raises(ValueError, match="expert_count_grid"):
            resolve_study_configs("G", config)

    def test_study_A_correct_fields(self):
        config = Config()
        configs = resolve_study_configs("A", config)
        assert configs[0].moe_variant == "none"
        assert configs[0].use_masked_lm is False
        assert configs[3].moe_variant == "arv_mmoe"
        assert configs[4].use_gradnorm is True

class TestCoreTradeoffPlotAxes:
    """REV-101: FR-14: "`accuracy_vs_model_size.png`: Size MB vs NDCG@10" -- the code
    plotted `total_parameters` under an axis labelled "Total Parameters"."""

    @staticmethod
    def _write_inputs(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "test_metrics_t_small.json"), "w") as f:
            json.dump({"ndcg": {"mean": 0.21}}, f)
        with open(os.path.join(log_dir, "efficiency_t_small.json"), "w") as f:
            json.dump({
                "total_parameters": 12_000_000,
                "model_size_bf16_mb": 22.5,
                "model_size_fp32_mb": 45.0,
                "flops_per_inference": 1.5e9,
            }, f)

    def _run(self, tmp_path):
        from main import run_plots_stage

        config = Config()
        config.log_dir = str(tmp_path / "logs")
        config.plot_dir = str(tmp_path / "plots")
        self._write_inputs(config.log_dir)
        run_plots_stage(config, MagicMock(model="t-small"))
        return config

    def test_model_size_axis_is_megabytes(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        captured = []
        real_subplots = plt.subplots

        def spy(*args, **kwargs):
            fig, ax = real_subplots(*args, **kwargs)
            captured.append(fig)
            return fig, ax

        plt.subplots = spy
        try:
            self._run(tmp_path)
        finally:
            plt.subplots = real_subplots

        size_ax = captured[0].axes[0]
        assert "MB" in size_ax.get_xlabel(), (
            f"FR-14 specifies size in MB; axis reads {size_ax.get_xlabel()!r}"
        )
        xs = size_ax.collections[0].get_offsets()[:, 0].tolist()
        assert xs == [22.5], f"expected model_size_bf16_mb on the x axis, got {xs}"

class TestAblationNoTrainFlag:
    """REV-115: `run_ablation(train=...)` had no CLI surface.

    `conventions.md` ("Config-Driven Design") requires that no ablation need source
    edits, and `train=False` -- which records `status: "skipped_no_checkpoint"` for arms
    with no checkpoint and evaluates the rest -- was reachable only by editing
    `run_ablation_stage`.
    """

    def test_defaults_to_training(self):
        args = parse_args(["--stage", "ablation", "--study", "A"])
        assert args.no_train is False

    def test_flag_is_plumbed_through_to_run_ablation(self):
        from main import run_ablation_stage

        config = Config()
        for flag, expected_train in ((["--no_train"], False), ([], True)):
            args = parse_args(["--stage", "ablation", "--study", "C"] + flag)
            with patch("training.ablation_runner.run_ablation") as fake:
                run_ablation_stage(config, args)
            assert fake.call_args.kwargs["train"] is expected_train, (
                f"--no_train={bool(flag)} must reach run_ablation as "
                f"train={expected_train}"
            )


class TestMissingCheckpoint:
    def test_evaluate_skips_missing_checkpoint(self, tmp_path, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        config = Config()
        config.processed_data_dir = str(tmp_path)
        config.checkpoint_dir = str(tmp_path / "ckpts")

        args = MagicMock()
        args.model = "t-small"

        from main import run_evaluate_stage
        run_evaluate_stage(config, args)

        assert any("Checkpoint not found" in r.message for r in caplog.records)


class TestLoadTrainedModelResolvesItsNames:
    """`_load_trained_model` must not reference a name it never imported.

    It called `load_metadata_matrix` while importing only `build_model`,
    `load_checkpoint`, `_build_loss_fns` and `get_device`. Both call sites
    (`run_routing_diagnostics_stage`, `run_gradient_diagnostics_stage`) import
    `load_metadata_matrix` into THEIR OWN function scope, which does not reach the
    helper -- so `--stage gradient_diagnostics` and `--stage routing_diagnostics` died
    with `NameError: name 'load_metadata_matrix' is not defined` before touching a model.
    Found while running the REV-129 / REV-128 verification, which needs these stages.

    No existing test reached this line: `TestMissingCheckpoint` returns at the checkpoint
    guard ABOVE it, and every other CLI test stops at argument parsing.
    """

    def test_reaches_model_construction_without_a_name_error(self, tmp_path):
        import json as _json

        from main import _load_trained_model

        config = Config()
        config.processed_data_dir = str(tmp_path / "processed")
        config.checkpoint_dir = str(tmp_path / "ckpt")
        os.makedirs(config.processed_data_dir, exist_ok=True)
        with open(os.path.join(config.processed_data_dir, "dataset_stats.json"), "w") as f:
            _json.dump({"num_items": 100, "num_users": 10, "MASK_ID": 101}, f)

        args = MagicMock()
        args.model = "t-small"
        config.apply_model_config("t-small")
        os.makedirs(config.model_checkpoint_dir(), exist_ok=True)
        with open(os.path.join(config.model_checkpoint_dir(), "best.pt"), "wb") as f:
            f.write(b"checkpoint")

        sentinel_model, sentinel_moe = MagicMock(), MagicMock()
        with patch("training.pretrain.build_model",
                   return_value=(sentinel_model, sentinel_moe)), \
                patch("training.pretrain._build_loss_fns", return_value={}), \
                patch("training.pretrain.load_checkpoint"), \
                patch("data.sequence_generator.load_metadata_matrix", return_value=None):
            loaded = _load_trained_model(config, args)

        assert loaded is not None, "the checkpoint exists, so this must not skip"
        model, moe, _, metadata_matrix, _, processed, source_revision = loaded
        assert model is sentinel_model and moe is sentinel_moe
        assert metadata_matrix is None
        assert processed == config.processed_data_dir
        # AC-09: the helper also resolves the provenance the diagnostics artifacts echo.
        # `load_checkpoint` is patched here, so it returns a MagicMock rather than a real
        # checkpoint dict -- the helper must degrade to the "not established" record
        # instead of inventing one.
        assert isinstance(source_revision, dict)
        assert "commit" in source_revision


class TestEvaluationStagesResolveTheirNames:
    """The REV-135 defect class, generalised: a stage whose body references a name it
    never imported raises `NameError` only when the stage is actually RUN, and every
    existing CLI test stops at argument parsing or returns at the checkpoint guard
    above the faulty line.

    This recurred while implementing AC-09: `run_longtail_stage` gained a call to
    `artifact_source_revision` without the matching import, and `--stage longtail` died
    on a NameError that the whole green suite did not see. One test per stage, driving
    each past its checkpoint guard with the heavy collaborators mocked, so the names in
    each body are resolved.
    """

    STAGES = [
        ("run_evaluate_stage", "evaluate"),
        ("run_cold_start_stage", "cold_start"),
        ("run_longtail_stage", "longtail"),
        ("run_latency_stage", "latency"),
    ]

    @pytest.mark.parametrize("func_name,stage", STAGES)
    def test_stage_body_resolves_every_name(self, func_name, stage, tmp_path):
        """Compile-level name resolution for each stage body.

        Every global load in the function must resolve against `main`'s module
        namespace or the function's own imports -- which is exactly what a missing
        `from utils import ...` breaks.
        """
        import ast
        import builtins

        import main as main_module

        source = inspect.getsource(getattr(main_module, func_name))
        fn = ast.parse(textwrap.dedent(source)).body[0]

        bound = {a.arg for a in fn.args.args}
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom):
                bound |= {a.asname or a.name for a in node.names}
            elif isinstance(node, ast.Import):
                bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node is not fn:
                bound.add(node.name)
            elif isinstance(node, (ast.comprehension,)):
                for t in ast.walk(node.target):
                    if isinstance(t, ast.Name):
                        bound.add(t.id)

        module_ns = vars(main_module)
        unresolved = sorted({
            node.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in bound
            and node.id not in module_ns
            and not hasattr(builtins, node.id)
        })
        assert not unresolved, (
            f"--stage {stage} references names it never imports: {unresolved}. "
            "This raises NameError at runtime and no other test reaches the line "
            "(REV-135)."
        )


class TestOverrideLogging:
    def test_override_logged(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        args = parse_args(["--stage", "pretrain", "--epochs", "5", "--batch_size", "32"])
        config = Config()
        apply_cli_overrides(config, args)

        log_text = " ".join(r.message for r in caplog.records)
        assert "pretrain_epochs" in log_text
        assert "batch_size" in log_text


class TestStageAllExcludesAblations:
    """REV-033: assert against the resolved stage list, not the function's source text."""

    def test_all_does_not_dispatch_ablation(self):
        from main import _all_pipeline_stages, run_ablation_stage

        names = [name for name, _ in _all_pipeline_stages()]
        fns = [fn for _, fn in _all_pipeline_stages()]

        assert "ablation" not in names, f"'all' dispatches the ablation stage: {names}"
        assert run_ablation_stage not in fns, (
            "'all' dispatches run_ablation_stage directly"
        )

    def test_all_stages_are_real_registered_stages(self):
        """Every stage 'all' runs must be a genuine, registered pipeline stage."""
        from main import _all_pipeline_stages, STAGE_MAP, VALID_STAGES

        for name, fn in _all_pipeline_stages():
            assert name in VALID_STAGES, f"'{name}' is not a valid stage"
            assert name in STAGE_MAP, f"'{name}' is not in STAGE_MAP"
            assert STAGE_MAP[name] is fn, (
                f"'all' runs a different function for '{name}' than STAGE_MAP does"
            )

    def test_ablation_is_still_reachable_on_its_own(self):
        """Excluded from 'all' must not mean unreachable."""
        from main import STAGE_MAP, VALID_STAGES, run_ablation_stage

        assert "ablation" in VALID_STAGES
        assert STAGE_MAP["ablation"] is run_ablation_stage


class TestSeeding:
    """REV-083: `main.setup_seed` must be the canonical `utils.set_seed`, not a copy.

    The duplicate dropped `torch.cuda.manual_seed_all` -- leaving every CUDA generator
    beyond the default device unseeded on a multi-GPU host -- and left
    `cudnn.benchmark` enabled, so autotuning could pick different algorithms per run.
    """

    def test_setup_seed_delegates_to_utils_set_seed(self):
        from unittest.mock import patch
        from configs.config import Config
        import main

        config = Config()
        config.seed = 4242
        with patch("main.set_seed") as mock_set_seed:
            main.setup_seed(config)
        mock_set_seed.assert_called_once_with(4242)

    def test_setup_seed_disables_cudnn_benchmark(self):
        import torch
        from configs.config import Config
        import main

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        config = Config()
        config.seed = 7
        main.setup_seed(config)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_setup_seed_is_reproducible_across_all_generators(self):
        import random
        import numpy as np
        import torch
        from configs.config import Config
        import main

        config = Config()
        config.seed = 99

        def draw():
            main.setup_seed(config)
            return (random.random(), float(np.random.rand()), float(torch.rand(1)))

        assert draw() == draw()


class TestNamedAblationFigures:
    """REV-065: four requirements name specific figures that were never produced.

    `_plot_study` rendered one generic grouped bar chart per study, so Ablation G's
    latency-vs-K tradeoff -- the entire point of that study -- had no figure, and G did
    not even gather latency.
    """

    @staticmethod
    def _metrics(base):
        return {
            m: {"mean": base + i * 0.01, "std": 0.002}
            for i, m in enumerate(["ndcg", "mrr", "recall", "precision"])
        }

    @staticmethod
    def _real_parameters(moe_variant):
        """A real `count_parameters` dict, exactly as `_run_single_config` stores it.

        REV-099: this fixture used to hand-feed `{"total": ...}`, a key
        `evaluation/flops.py::count_parameters` has never returned. The plotting code
        read that invented key, so it silently plotted zeros on a green suite. Driving
        the fixture from the production function is what makes a future rename fail here.
        """
        from evaluation.flops import count_parameters
        from models.moe import MoELayer

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.use_metadata = False
        cfg.num_items = 50
        cfg.moe_variant = moe_variant
        model = TransformerModel(cfg)
        moe = MoELayer(cfg) if moe_variant != "none" else None
        return {k: float(v) for k, v in count_parameters(model, moe).items()}

    def test_G_gathers_latency_for_the_twin_axis_chart(self):
        from training.ablation_runner import STUDY_EXTRAS

        assert "efficiency" in STUDY_EXTRAS["G"], (
            "FR-12's figure is 'NDCG left axis, latency right axis', so G must collect "
            "efficiency extras"
        )
        assert "routing" in STUDY_EXTRAS["G"], "G's routing diagnostics must be kept"

    def test_every_named_figure_is_written(self, tmp_path):
        from training.ablation_runner import _STUDY_PLOTS

        config = Config()
        config.plot_dir = str(tmp_path)

        study_results = {
            "A": [
                {"label": label, "metrics": self._metrics(0.10 + 0.02 * i)}
                for i, label in enumerate([
                    "single_obj_baseline", "4obj_no_routing", "conv_mmoe",
                    "arv_mmoe", "arv_mmoe_gradnorm",
                ])
            ],
            "C": [
                {
                    "label": label,
                    "metrics": self._metrics(0.12 + 0.01 * i),
                    "parameters": self._real_parameters(label),
                    "latency": {"mean_ms": 11.0 + i * 2},
                }
                for i, label in enumerate(["mmoe", "arv_mmoe"])
            ],
            "G": [
                {
                    "label": f"K{k}",
                    "metrics": self._metrics(0.11 + 0.005 * i),
                    "resolved_config": {"num_experts": k},
                    "latency": {"mean_ms": 8.0 + 1.7 * k},
                }
                for i, k in enumerate([2, 4, 6, 8])
            ],
        }

        for study, results in study_results.items():
            _STUDY_PLOTS[study](config, "t-base", results)

        # The specs name these literally, with no model suffix.
        for filename in (
            "architecture_progression.png",       # FR-6  / T2.2
            "mmoe_vs_arvmmoe_comparison.png",     # FR-8  / T2.4
            "expert_count_ablation.png",          # FR-12 / T2.8
        ):
            path = tmp_path / filename
            assert path.exists(), f"{filename} was not written"
            assert path.stat().st_size > 1000, f"{filename} is empty"

    def test_expert_count_figure_has_ndcg_left_and_latency_right(self, tmp_path):
        """FR-12 states the axis assignment explicitly."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from training.ablation_runner import _plot_expert_count

        config = Config()
        config.plot_dir = str(tmp_path)
        results = [
            {
                "label": f"K{k}",
                "metrics": self._metrics(0.11 + 0.005 * i),
                "resolved_config": {"num_experts": k},
                "latency": {"mean_ms": 8.0 + 1.7 * k},
            }
            # Deliberately out of order: arms must be plotted by K, not by label order.
            for i, k in enumerate([6, 2, 8, 4])
        ]

        captured = {}
        real_subplots = plt.subplots

        def spy(*args, **kwargs):
            fig, ax = real_subplots(*args, **kwargs)
            captured["fig"], captured["ax"] = fig, ax
            return fig, ax

        plt.subplots = spy
        try:
            _plot_expert_count(config, "t-base", results)
        finally:
            plt.subplots = real_subplots

        fig = captured["fig"]
        left, right = fig.axes[0], fig.axes[1]
        assert "NDCG" in left.get_ylabel()
        assert "latency" in right.get_ylabel().lower()
        # The NDCG series must ascend in K regardless of the input arm order.
        xdata = list(left.lines[0].get_xdata())
        assert xdata == sorted(xdata), f"arms not ordered by K: {xdata}"

    def test_mmoe_comparison_plots_real_parameter_values(self, tmp_path):
        """REV-099: FR-8's cost panel must plot the parameter counts that exist.

        The panel read `parameters["total"]`, which `count_parameters` never returns, so
        `.get("total", 0.0)` made BOTH arms' bars zero-height -- in the figure whose job
        is to substantiate moe-routing FR-6's parameter-matched claim ("Never claim
        'parameter-matched' without the report backing it"). FR-8 names three parameter
        quantities; all three are asserted here against the real dict.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from training.ablation_runner import _plot_mmoe_comparison

        config = Config()
        config.plot_dir = str(tmp_path)
        results = [
            {
                "label": label,
                "metrics": self._metrics(0.12 + 0.01 * i),
                "parameters": self._real_parameters(label),
                "latency": {"mean_ms": 11.0 + i * 2},
            }
            for i, label in enumerate(["mmoe", "arv_mmoe"])
        ]
        for r in results:
            for key in ("total_parameters", "expert_parameters", "router_parameters"):
                assert r["parameters"].get(key, 0.0) > 0, (
                    f"fixture precondition: {key} must be non-zero for {r['label']}, "
                    f"or this test cannot detect a zeroed bar"
                )

        captured = {}
        real_subplots = plt.subplots

        def spy(*args, **kwargs):
            fig, axes = real_subplots(*args, **kwargs)
            captured["fig"] = fig
            return fig, axes

        plt.subplots = spy
        try:
            _plot_mmoe_comparison(config, "t-base", results)
        finally:
            plt.subplots = real_subplots

        cost_ax = captured["fig"].axes[1]
        heights = [patch.get_height() for patch in cost_ax.patches]
        assert heights, "the cost panel drew no bars at all"
        assert all(h > 0 for h in heights), (
            f"the cost panel plotted a zero-height parameter bar: {heights}"
        )

        # Each of FR-8's three parameter quantities is plotted, in millions, for
        # every arm -- so a key rename cannot silently zero one series.
        for r in results:
            for key in ("total_parameters", "expert_parameters", "router_parameters"):
                expected = r["parameters"][key] / 1e6
                assert any(abs(h - expected) < 1e-9 for h in heights), (
                    f"{key} for {r['label']} ({expected:.6f}M) is not among the "
                    f"plotted bar heights {heights}"
                )

    def test_named_figures_are_empty_safe(self, tmp_path):
        """A study with no usable arms must warn and skip, never abort the run."""
        from training.ablation_runner import _STUDY_PLOTS

        config = Config()
        config.plot_dir = str(tmp_path)
        for study, plot_fn in _STUDY_PLOTS.items():
            plot_fn(config, "t-base", [])
            plot_fn(config, "t-base", [{"label": "x", "status": "failed"}])

class TestSharedBackboneReuse:
    """REV-084: paired arms differing only in post-training fields trained twice.

    Arms that differ only in fields `run_pretrain` never reads (H3-H5's `system_eval`)
    trained bit-identical models under separate checkpoint namespaces -- a full redundant
    pretraining run per arm.
    """

    def test_paired_arms_share_a_training_fingerprint(self):
        from training.ablation_runner import _training_fingerprint

        base = Config()
        a_fps = [_training_fingerprint(c) for c in resolve_study_configs("A", base)]
        assert len(set(a_fps)) == 5, "genuinely different arms must not share"

        h = {c._study_label: c for c in resolve_study_configs("H", base)}
        h_fps = {_training_fingerprint(h[k]) for k in ("H3_cold_start", "H4_longtail", "H5_latency")}
        assert len(h_fps) == 1, "H3-H5 differ only in system_eval and must share a backbone"
        assert _training_fingerprint(h["H1_no_metadata"]) not in h_fps

    def test_only_verified_post_training_fields_are_excluded(self):
        """A wrong exclusion would silently make two different arms share one model."""
        from training.ablation_runner import (
            _training_fingerprint, POST_TRAINING_ONLY_FIELDS,
        )

        baseline = _training_fingerprint(Config())

        for field in POST_TRAINING_ONLY_FIELDS | {"checkpoint_dir"}:
            cfg = Config()
            current = getattr(cfg, field)
            setattr(
                cfg, field,
                17 if isinstance(current, int) and not isinstance(current, bool)
                else "changed",
            )
            assert _training_fingerprint(cfg) == baseline, (
                f"'{field}' is excluded, so changing it must not change the fingerprint"
            )

        for field in (
            "num_experts", "use_gradnorm", "moe_variant", "max_seq_length",
            "seed", "use_masked_lm", "use_contrastive",
            "router_conditioning", "use_metadata",
        ):
            cfg = Config()
            current = getattr(cfg, field)
            if isinstance(current, bool):
                setattr(cfg, field, not current)
            elif isinstance(current, int):
                setattr(cfg, field, current + 1)
            else:
                setattr(cfg, field, "changed")
            assert _training_fingerprint(cfg) != baseline, (
                f"'{field}' changes the trained model and must be in the fingerprint"
            )

    def test_second_paired_arm_reuses_instead_of_retraining(self, tmp_path):
        import json
        import os
        from unittest.mock import patch, MagicMock

        import numpy as np
        import torch

        import training.ablation_runner as ar

        class FakeDataset:
            def __init__(self):
                self.sequences = torch.zeros(2, 50, dtype=torch.long)
                self.targets = torch.tensor([1, 2])
                self.user_ids = torch.tensor([0, 1])
                self.metadata_matrix = None

            def __len__(self):
                return 2

        base = Config()
        base.checkpoint_dir = str(tmp_path / "ckpt")
        base.processed_data_dir = str(tmp_path / "processed")
        os.makedirs(base.processed_data_dir, exist_ok=True)
        with open(os.path.join(base.processed_data_dir, "dataset_stats.json"), "w") as f:
            json.dump({"num_items": 100, "num_users": 10, "MASK_ID": 101}, f)

        first = resolve_study_configs("A", base)[4]
        # The same arm with only a post-training field changed.
        second = copy.deepcopy(first)
        second._study_label = "arv_mmoe_gradnorm_eval_seed1"
        second.ablation_eval_seed = 1
        arms = [first, second]
        trained = []

        def fake_pretrain(cfg, model_name):
            trained.append(cfg._study_label)
            os.makedirs(cfg.model_checkpoint_dir(), exist_ok=True)
            with open(os.path.join(cfg.model_checkpoint_dir(), "best.pt"), "wb") as f:
                f.write(b"checkpoint")
            return {"best_val_ndcg": 0.42, "best_epoch": 3}

        with patch("training.pretrain.run_pretrain", side_effect=fake_pretrain), \
                patch("training.pretrain.build_model", return_value=(MagicMock(), MagicMock())), \
                patch("training.pretrain._build_loss_fns", return_value={}), \
                patch("training.pretrain.load_checkpoint"), \
                patch("data.sequence_generator.load_metadata_matrix", return_value=None), \
                patch("data.sequence_generator.load_tensorised", return_value=FakeDataset()), \
                patch("evaluation.metrics.load_full_ranking",
                      return_value=build_full_ranking([1, 2], [0, 1], {})), \
                patch("evaluation.metrics.load_item_popularity",
                      return_value=torch.zeros(102)), \
                patch("evaluation.metrics.evaluate_model",
                      return_value={"ndcg": {"mean": 0.1, "std": 0.0}}):
            results = [
                ar._run_single_config(cfg, "t-base", "A", base, True)
                for cfg in arms
            ]

        assert trained == ["arv_mmoe_gradnorm"], (
            f"the paired arm retrained an identical model: {trained}"
        )
        assert results[1]["shared_backbone_with"] == "A_arv_mmoe_gradnorm", (
            "a reused backbone must be recorded, so identical metrics are not misread "
            "as a null result"
        )
        # The reusing arm still reports the training outcome of the model it evaluates.
        assert results[1]["best_val_ndcg"] == 0.42
        assert results[1]["best_epoch"] == 3
        # And the pair stays distinguishable on the dimension it exists to isolate.
        assert results[0]["resolved_config"]["ablation_eval_seed"] == 0
        assert results[1]["resolved_config"]["ablation_eval_seed"] == 1


class TestBackboneStore:
    """Backbones are stored by training fingerprint and shared across studies.

    Reuse used to be an in-memory dict scoped to ONE study, so the base config trained
    separately for B, C, D, F, G and H, separate processes (one per GPU) could not see
    each other's models, and a re-run retrained everything. The multi-seed stage's
    per-seed checkpoints, which ARE the base config, were never consulted either.
    """

    @staticmethod
    def _base(tmp_path, seed=42):
        base = Config()
        base.apply_model_config("t-base")
        base.seed = seed
        base.checkpoint_dir = str(tmp_path / "ckpt")
        base.output_dir = str(tmp_path / "out")
        base.processed_data_dir = str(tmp_path / "processed")
        os.makedirs(base.processed_data_dir, exist_ok=True)
        with open(os.path.join(base.processed_data_dir, "dataset_stats.json"), "w") as f:
            json.dump({"num_items": 100, "num_users": 10, "MASK_ID": 101}, f)
        return base

    @staticmethod
    def _fake_pretrain(trained):
        """Mutates the config exactly as `run_pretrain` does, then writes `best.pt`."""
        def fake(cfg, model_name):
            cfg.apply_model_config(model_name)
            with open(os.path.join(cfg.processed_data_dir, "dataset_stats.json")) as f:
                stats = json.load(f)
            cfg.num_items, cfg.num_users = stats["num_items"], stats["num_users"]
            cfg.vocab_size = cfg.num_items + 2
            trained.append(getattr(cfg, "_study_label", f"seed_{cfg.seed}"))
            os.makedirs(cfg.model_checkpoint_dir(), exist_ok=True)
            with open(os.path.join(cfg.model_checkpoint_dir(), "best.pt"), "wb") as f:
                f.write(b"checkpoint")
            return {"best_val_ndcg": 0.42, "best_epoch": 3}
        return fake

    @staticmethod
    def _run_arms(base, study_arms, trained, train=True):
        import numpy as np
        import torch

        import training.ablation_runner as ar

        class FakeDataset:
            sequences = torch.zeros(2, 50, dtype=torch.long)
            targets = torch.tensor([1, 2])
            user_ids = torch.tensor([0, 1])
            metadata_matrix = None

            def __len__(self):
                return 2

        loaded = []
        with patch("training.pretrain.run_pretrain",
                   side_effect=TestBackboneStore._fake_pretrain(trained)), \
                patch("training.pretrain.build_model", return_value=(MagicMock(), MagicMock())), \
                patch("training.pretrain._build_loss_fns", return_value={}), \
                patch("training.pretrain.load_checkpoint",
                      side_effect=lambda path, *a, **k: loaded.append(path)), \
                patch("data.sequence_generator.load_metadata_matrix", return_value=None), \
                patch("data.sequence_generator.load_tensorised", return_value=FakeDataset()), \
                patch("evaluation.metrics.load_full_ranking",
                      return_value=build_full_ranking([1, 2], [0, 1], {})), \
                patch("evaluation.metrics.load_item_popularity",
                      return_value=torch.zeros(102)), \
                patch("evaluation.metrics.evaluate_model",
                      return_value={"ndcg": {"mean": 0.1, "std": 0.0}}), \
                patch("training.pretrain.collect_validation_routing", return_value={}):
            results = []
            for study, cfg in study_arms:
                results.append(ar._run_single_config(cfg, "t-base", study, base, train))
        return results, loaded

    def test_identical_arms_in_different_studies_train_once(self, tmp_path):
        base = self._base(tmp_path)
        b_arm = resolve_study_configs("B", base)[0]   # objective_conditioned == base
        f_arm = resolve_study_configs("F", base)[0]   # full_reference == base
        trained = []
        _, loaded = self._run_arms(base, [("B", b_arm), ("F", f_arm)], trained)

        assert trained == ["objective_conditioned"], f"base config retrained: {trained}"
        assert len(loaded) == 2 and loaded[0] == loaded[1], (
            "both arms must evaluate the same stored checkpoint"
        )

    def test_reuse_survives_a_new_process(self, tmp_path):
        """The store is on disk, so a second run (or another GPU's process) reuses it."""
        base = self._base(tmp_path)
        trained = []
        self._run_arms(base, [("B", resolve_study_configs("B", base)[0])], trained)
        self._run_arms(base, [("F", resolve_study_configs("F", base)[0])], trained)
        assert trained == ["objective_conditioned"], trained

    def test_unfinished_run_is_retrained_not_reused(self, tmp_path):
        """`best.pt` exists after epoch 0; without the marker it is not a trained model."""
        from training.ablation_runner import _training_fingerprint, backbone_store_dir

        base = self._base(tmp_path)
        arm = resolve_study_configs("B", base)[0]
        probe = resolve_study_configs("B", base)[0]
        probe.num_items, probe.num_users, probe.vocab_size = 100, 10, 102
        crashed = os.path.join(
            backbone_store_dir(base.checkpoint_dir, _training_fingerprint(probe)), "t_base",
        )
        os.makedirs(crashed)
        with open(os.path.join(crashed, "best.pt"), "wb") as f:
            f.write(b"partial")

        trained = []
        self._run_arms(base, [("B", arm)], trained)
        assert trained == ["objective_conditioned"], "a partial checkpoint was reused"

        no_train = []
        results, _ = self._run_arms(
            self._base(tmp_path / "other"),
            [("B", resolve_study_configs("B", self._base(tmp_path / "other"))[0])],
            no_train, train=False,
        )
        assert no_train == [] and results[0]["status"] == "skipped_no_checkpoint"

    def test_multi_seed_backbones_are_reused_by_base_config_arms(self, tmp_path):
        import argparse

        from main import run_multi_seed_stage

        base = self._base(tmp_path)
        base.final_seeds = [42, 123]
        trained = []
        args = argparse.Namespace(model="t-base")
        with patch("training.pretrain.run_pretrain",
                   side_effect=self._fake_pretrain(trained)), \
                patch("main.run_evaluate_stage"):
            run_multi_seed_stage(base, args)
        assert trained == ["seed_42", "seed_123"]

        # The ablation for seed 123 (`--stage ablation --seed 123`) reuses that seed's
        # backbone; an arm that changes training (B's input_only) still trains.
        abl = self._base(tmp_path, seed=123)
        b_arms = resolve_study_configs("B", abl)
        trained.clear()
        results, loaded = self._run_arms(abl, [("B", b_arms[0]), ("B", b_arms[1])], trained)

        assert trained == ["input_only"], trained
        assert loaded[0] == os.path.join(
            str(tmp_path / "out"), "seed_123", "checkpoints", "t_base", "best.pt",
        )
        assert results[0]["shared_backbone_with"] == "multi_seed_123"


class TestAblationRunConfig:
    """specs/ablation-run T1.1: new config fields, their defaults and validation."""

    def test_new_field_defaults(self):
        cfg = Config()
        assert cfg.min_lr_ratio == 0.1
        assert cfg.user_sample_fraction == 1.0
        assert cfg.user_sample_seed == 42
        assert cfg.resume_training is True

    def test_project_default_precision_is_unchanged(self):
        # AC-01 (default half): the ablation commands pass mixed bf16; the default stays.
        cfg = Config()
        assert cfg.dtype == "bf16"
        assert cfg.mixed_precision is False

    @pytest.mark.parametrize("field,value", [
        ("min_lr_ratio", -0.1),
        ("min_lr_ratio", 1.1),
        ("user_sample_fraction", 0.0),
        ("user_sample_fraction", 1.5),
    ])
    def test_out_of_range_values_are_rejected(self, field, value):
        cfg = Config()
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match=field):
            cfg.validate()

    @pytest.mark.parametrize("field,value", [
        ("min_lr_ratio", 0.0), ("min_lr_ratio", 1.0), ("user_sample_fraction", 1.0),
        ("user_sample_fraction", 0.1),
    ])
    def test_boundary_values_are_accepted(self, field, value):
        cfg = Config()
        setattr(cfg, field, value)
        cfg.validate()


class TestAblationRunCLI:
    """specs/ablation-run T1.2: CLI flags, override logging, derived processed dir."""

    FLAGS = [
        "--user_sample_fraction", "0.5", "--user_sample_seed", "7",
        "--sequence_length_grid", "25", "50", "--attention_backend", "sdpa",
        "--num_workers", "3", "--min_lr_ratio", "0.2", "--no_resume",
    ]
    EXPECTED = {
        "user_sample_fraction": 0.5,
        "user_sample_seed": 7,
        "sequence_length_grid": [25, 50],
        "attention_backend": "sdpa",
        "num_workers": 3,
        "min_lr_ratio": 0.2,
        "resume_training": False,
        "processed_data_dir": "data/processed_users50pct_seed7",
    }

    def test_each_flag_is_applied_and_logged(self, caplog):
        import logging

        cfg = Config()
        with caplog.at_level(logging.INFO, logger="main"):
            apply_cli_overrides(cfg, parse_args(["--stage", "pretrain"] + self.FLAGS))
        default = Config()
        for field, value in self.EXPECTED.items():
            assert getattr(cfg, field) == value, field
            assert (
                f"CLI override: {field} {getattr(default, field)} -> {value}" in caplog.text
            ), field

    def test_absent_flags_leave_the_config_untouched(self):
        from dataclasses import asdict

        cfg = Config()
        apply_cli_overrides(cfg, parse_args(["--stage", "pretrain"]))
        assert asdict(cfg) == asdict(Config())

    def test_fraction_and_seed_name_the_directory(self):
        cfg = Config()
        apply_cli_overrides(cfg, parse_args([
            "--stage", "preprocess", "--user_sample_fraction", "0.25",
            "--user_sample_seed", "7",
        ]))
        assert cfg.processed_data_dir == "data/processed_users25pct_seed7"

    def test_default_seed_has_no_suffix(self):
        from main import derive_processed_dir

        cfg = Config()
        cfg.user_sample_fraction = 0.1
        assert derive_processed_dir(cfg) == "data/processed_users10pct"
        cfg.user_sample_fraction = 1.0
        assert derive_processed_dir(cfg) is None

    def test_explicit_processed_dir_is_never_rewritten(self):
        cfg = Config()
        cfg.processed_data_dir = "/somewhere/else"
        apply_cli_overrides(cfg, parse_args([
            "--stage", "preprocess", "--user_sample_fraction", "0.1",
        ]))
        assert cfg.processed_data_dir == "/somewhere/else"

    def test_no_resume_turns_resume_off(self):
        cfg = Config()
        apply_cli_overrides(cfg, parse_args(["--stage", "pretrain", "--no_resume"]))
        assert cfg.resume_training is False

    def test_backend_choices_come_from_the_model_module(self):
        from models.transformer import VALID_BACKENDS

        with pytest.raises(SystemExit):
            parse_args(["--stage", "pretrain", "--attention_backend", "flex_attention"])
        for backend in VALID_BACKENDS:
            assert parse_args(
                ["--stage", "pretrain", "--attention_backend", backend]
            ).attention_backend == backend
