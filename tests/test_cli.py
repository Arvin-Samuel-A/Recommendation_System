import json
import os
import pytest
from unittest.mock import patch, MagicMock

from configs.config import Config
from main import parse_args, apply_cli_overrides, STAGE_MAP, run_all_pipeline
from training.ablation_runner import resolve_study_configs, STUDY_CONFIGS
from models.transformer import TransformerModel


class TestCLIParsing:
    def test_all_stages_accepted(self):
        for stage in ["preprocess", "pretrain", "personalize", "federated",
                       "evaluate", "cold_start", "longtail", "latency",
                       "routing_diagnostics", "gradient_diagnostics",
                       "ablation", "plots", "all"]:
            args = parse_args(["--stage", stage])
            assert args.stage == stage

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
    def test_study_A_produces_6_configs(self):
        config = Config()
        configs = resolve_study_configs("A", config)
        assert len(configs) == 6

    def test_all_studies_produce_expected_count(self):
        config = Config()
        expected = {
            "A": 6, "B": 2, "C": 2, "D": 4, "E": 3, "F": 4,
            # REV-067: G and H2 are sweeps over config grids, so their arm counts are
            # derived from the config rather than asserted as literals.
            "G": len(config.expert_count_grid),
            "H": 5 + len(config.sequence_length_grid),
        }
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

    def test_study_A_final_arm_is_federated(self):
        """REV-100: FR-6 ends A's progression at "+federated"; T2.2 names arm 6
        "federated ARV-MMoE+GradNorm+LoRA".

        The arm was `personalization_mode="lora"` with no `system_eval`, so nothing in
        study A ever reached `run_federated` -- the progression's final point was
        labelled as the federated system while measuring single-machine LoRA
        personalization. Those are different experiments: federated adds FedAvg across
        `clients_per_round` clients and the communication cost RQ4 is about.
        """
        arm = resolve_study_configs("A", Config())[5]
        assert arm.system_eval == "communication", (
            "study A's final arm must route into run_federated"
        )
        assert arm.personalization_mode == "lora", "T2.2 names LoRA as well"
        assert arm.moe_variant == "arv_mmoe"
        assert arm.use_gradnorm is True

    def test_federated_arm_reuses_the_gradnorm_arm_backbone(self):
        """`system_eval` is post-training-only, so REV-100's fix must not add a
        28th-plus training run to `--study all`."""
        from training.ablation_runner import _training_fingerprint

        arms = resolve_study_configs("A", Config())
        assert _training_fingerprint(arms[4]) == _training_fingerprint(arms[5]), (
            "arms 5 and 6 differ only in post-training fields, so they must share "
            "one trained backbone"
        )


class TestCoreTradeoffPlotAxes:
    """REV-101: two of FR-14's four figures plotted a different quantity from the one
    the requirement names.

    (a) FR-14: "`accuracy_vs_model_size.png`: Size MB vs NDCG@10" -- the code plotted
        `total_parameters` under an axis labelled "Total Parameters".
    (b) FR-14: "`accuracy_vs_communication.png`: Comm cost/round vs NDCG@10" -- the code
        plotted the CUMULATIVE cost summed over every round, so the ratio between two
        models changed with how many rounds each happened to run.
    These are paper figures: a reader taking the x-axis at face value against the
    requirement's wording would read a per-round cost off a cumulative axis.
    """

    @staticmethod
    def _write_inputs(log_dir, per_round_kb=64.0, n_rounds=4):
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
        rounds = [
            {"round": i, "val_ndcg": 0.10 + 0.01 * i, "val_examples_scored": 5000,
             "total_round_kb": per_round_kb, "upload_kb": per_round_kb / 8}
            for i in range(n_rounds)
        ]
        rounds[-1]["val_ndcg_full"] = 0.17
        rounds[-1]["val_examples_scored_full"] = 993_372
        with open(os.path.join(log_dir, "federated_rounds_t_small.json"), "w") as f:
            json.dump(rounds, f)

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

    def test_communication_axis_is_per_round(self, tmp_path):
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

        comm_ax = captured[1].axes[0]
        label = comm_ax.get_xlabel().lower()
        assert "round" in label and "cumulative" not in label, (
            f"FR-14 specifies comm cost PER ROUND; axis reads {comm_ax.get_xlabel()!r}"
        )
        offsets = comm_ax.collections[0].get_offsets()
        xs = offsets[:, 0].tolist()
        assert xs == [64.0], (
            f"expected the per-round cost (64.0 KB), got {xs} -- 256.0 would be the "
            f"cumulative sum across 4 rounds"
        )
        # REV-117: the headline y value is the final round's FULL-SPLIT score, not the
        # bounded prefix the convergence curve plots.
        ys = offsets[:, 1].tolist()
        assert ys == [0.17], f"expected val_ndcg_full on the y axis, got {ys}"


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
                    "arv_mmoe", "arv_mmoe_gradnorm", "arv_mmoe_gradnorm_lora",
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
            "E": [
                {"label": "global_only", "metrics": self._metrics(0.10)},
                {
                    "label": "global_lora",
                    "metrics": self._metrics(0.12),
                    "personalization": {"per_user_scores": [
                        {"user_id": u, "ndcg_without_lora": 0.10, "ndcg_with_lora": 0.11}
                        for u in range(20)
                    ]},
                },
                {
                    "label": "global_lora_arv",
                    "metrics": self._metrics(0.13),
                    "personalization": {"per_user_scores": [
                        {"user_id": u, "ndcg_without_lora": 0.10, "ndcg_with_lora": 0.125}
                        for u in range(20)
                    ]},
                },
            ],
        }

        for study, results in study_results.items():
            _STUDY_PLOTS[study](config, "t-base", results)

        # The specs name these literally, with no model suffix.
        for filename in (
            "architecture_progression.png",       # FR-6  / T2.2
            "mmoe_vs_arvmmoe_comparison.png",     # FR-8  / T2.4
            "expert_count_ablation.png",          # FR-12 / T2.8
            "global_vs_personalized_ablation_E.png",  # FR-10 / T2.6
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

    def test_personalization_report_carries_paired_scores(self):
        """FR-10's third series needs the paired NDCGs, not just the gain."""
        from training.local_train import _build_report

        results = [
            {"user_id": 1, "ndcg_with_lora": 0.2, "ndcg_without_lora": 0.1,
             "gain": 0.1, "is_cold": False},
            {"user_id": 2, "ndcg_with_lora": 0.3, "ndcg_without_lora": 0.25,
             "gain": 0.05, "is_cold": True},
        ]
        report = _build_report(results, num_users_skipped=0)

        assert "per_user_gains" in report, "the spec-mandated array must be kept"
        scores = report["per_user_scores"]
        assert [s["user_id"] for s in scores] == [1, 2]
        assert scores[0]["ndcg_without_lora"] == 0.1
        assert scores[0]["ndcg_with_lora"] == 0.2

    def test_scatter_draws_every_series(self, tmp_path):
        from training.local_train import plot_personalization_scatter

        config = Config()
        config.plot_dir = str(tmp_path)
        series = [
            {"label": name, "results": [
                {"ndcg_without_lora": 0.1, "ndcg_with_lora": 0.1 + bump}
                for _ in range(10)
            ]}
            for name, bump in (("global_lora", 0.01), ("global_lora_arv", 0.03))
        ]
        plot_personalization_scatter(
            config, "t-base", series, filename="three_series.png",
        )
        assert (tmp_path / "three_series.png").exists()

        # A series with no users must not suppress the ones that have data.
        plot_personalization_scatter(
            config, "t-base",
            [{"label": "empty", "results": []}] + series,
            filename="with_empty.png",
        )
        assert (tmp_path / "with_empty.png").exists()


class TestSharedBackboneReuse:
    """REV-084: paired arms differing only in post-training fields trained twice.

    Ablation A's arms 5/6 and E's `global_only`/`global_lora` differ only in
    `personalization_mode`, which `run_pretrain` never reads, so each pair trained two
    bit-identical models under two checkpoint namespaces -- a full redundant pretraining
    run per pair, and identical `metrics` that read as a null result for personalization.
    """

    def test_paired_arms_share_a_training_fingerprint(self):
        from training.ablation_runner import _training_fingerprint

        base = Config()
        a = resolve_study_configs("A", base)
        a_fps = [_training_fingerprint(c) for c in a]
        assert a_fps[4] == a_fps[5], (
            "A5/A6 differ only in personalization_mode and must share a backbone"
        )
        assert len(set(a_fps[:5])) == 5, "genuinely different arms must not share"

        e = resolve_study_configs("E", base)
        e_fps = [_training_fingerprint(c) for c in e]
        assert e_fps[0] == e_fps[1], "E global_only/global_lora must share a backbone"
        assert e_fps[2] != e_fps[0], "global_lora_arv changes moe_variant, so it retrains"

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
            "lora_rank", "seed", "use_masked_lm", "use_contrastive",
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
                self.metadata_matrix = None

            def __len__(self):
                return 2

        base = Config()
        base.checkpoint_dir = str(tmp_path / "ckpt")
        base.processed_data_dir = str(tmp_path / "processed")
        os.makedirs(base.processed_data_dir, exist_ok=True)
        with open(os.path.join(base.processed_data_dir, "dataset_stats.json"), "w") as f:
            json.dump({"num_items": 100, "num_users": 10, "MASK_ID": 101}, f)

        arms = resolve_study_configs("A", base)[4:6]
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
                patch("data.sequence_generator.load_eval_candidates",
                      return_value=np.zeros((2, 11), dtype=np.int64)), \
                patch("evaluation.metrics.evaluate_model",
                      return_value={"ndcg": {"mean": 0.1, "std": 0.0}}), \
                patch("training.local_train.run_local_personalization", return_value={
                    "num_users_evaluated": 1, "with_lora": {}, "without_lora": {},
                    "gain": {}, "per_user_scores": []}):
            registry = {}
            results = [
                ar._run_single_config(
                    cfg, "t-base", "A", base, True, trained_backbones=registry,
                )
                for cfg in arms
            ]

        assert trained == ["arv_mmoe_gradnorm"], (
            f"the paired arm retrained an identical model: {trained}"
        )
        assert results[1]["shared_backbone_with"] == "arv_mmoe_gradnorm", (
            "a reused backbone must be recorded, so identical metrics are not misread "
            "as a null result"
        )
        # The reusing arm still reports the training outcome of the model it evaluates.
        assert results[1]["best_val_ndcg"] == 0.42
        assert results[1]["best_epoch"] == 3
        # And the pair stays distinguishable on the dimension it exists to isolate.
        assert "personalization" not in results[0]
        assert "personalization" in results[1]
