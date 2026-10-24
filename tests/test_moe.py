import torch
import pytest

from configs.config import Config, OBJECTIVE_ORDER
from models.moe import (
    Expert,
    SharedExperts,
    ARVRouter,
    ConventionalMMoEGates,
    MoELayer,
    route_objective,
    count_router_parameters,
)
from utils import get_device, resolve_dtype


@pytest.fixture
def base_config():
    cfg = Config()
    cfg.apply_model_config("t-small")
    cfg.num_items = 100
    cfg.dropout = 0.0
    return cfg


@pytest.fixture
def device(base_config):
    return get_device(base_config)


@pytest.fixture
def dtype(base_config):
    return resolve_dtype(base_config)


class TestShapes:
    def test_arv_mmoe_shapes(self, base_config, device, dtype):
        base_config.moe_variant = "arv_mmoe"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)
        B, S, D = 4, 10, base_config.hidden_dim
        K = base_config.num_experts

        all_hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        pooled = torch.randn(B, D, device=device, dtype=dtype)

        Z_ut, r_ut = moe(all_hidden, pooled, 0)
        assert Z_ut.shape == (B, S, D)
        assert r_ut.shape == (B, K)
        moe.clear_cache()

    def test_mmoe_shapes(self, base_config, device, dtype):
        base_config.moe_variant = "mmoe"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)
        B, S, D = 4, 10, base_config.hidden_dim
        K = base_config.num_experts

        all_hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        pooled = torch.randn(B, D, device=device, dtype=dtype)

        Z_ut, r_ut = moe(all_hidden, pooled, 0)
        assert Z_ut.shape == (B, S, D)
        assert r_ut.shape == (B, K)
        moe.clear_cache()

    def test_none_passthrough_shapes(self, base_config, device, dtype):
        base_config.moe_variant = "none"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)
        B, S, D = 4, 10, base_config.hidden_dim

        all_hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        pooled = torch.randn(B, D, device=device, dtype=dtype)

        Z_ut, r_ut = moe(all_hidden, pooled, 0)
        assert Z_ut.shape == (B, S, D)
        assert r_ut is None


class TestPerPositionRouting:
    """Objectives that predict at every position route each position from its own
    hidden state, so routing never carries information from later items."""

    @pytest.mark.parametrize("variant", ["arv_mmoe", "mmoe"])
    def test_shapes_and_last_position_equals_pooled_routing(self, base_config, variant):
        base_config.moe_variant = variant
        torch.manual_seed(0)
        moe = MoELayer(base_config).eval()
        B, S, D = 3, 7, base_config.hidden_dim
        all_hidden = torch.randn(B, S, D)

        Z_pos, r_pos = moe(all_hidden, all_hidden, 0)
        assert Z_pos.shape == (B, S, D)
        assert r_pos.shape == (B, S, base_config.num_experts)

        Z_pool, r_pool = moe(all_hidden, all_hidden[:, -1, :], 0)
        assert torch.allclose(r_pos[:, -1], r_pool)
        assert torch.allclose(Z_pos[:, -1], Z_pool[:, -1])

    def test_a_position_does_not_depend_on_later_positions(self, base_config):
        base_config.moe_variant = "arv_mmoe"
        torch.manual_seed(0)
        moe = MoELayer(base_config).eval()
        B, S, D = 2, 6, base_config.hidden_dim
        all_hidden = torch.randn(B, S, D)
        Z, _ = moe(all_hidden, all_hidden, 0)
        changed = all_hidden.clone()
        changed[:, 3:] += 2.0
        moe.clear_cache()
        Z_changed, _ = moe(changed, changed, 0)
        assert torch.allclose(Z[:, :3], Z_changed[:, :3])


class TestRoutingDifferentiation:
    def test_different_objectives_different_routing(self, base_config, device, dtype):
        base_config.moe_variant = "arv_mmoe"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)
        B, S, D = 4, 10, base_config.hidden_dim

        all_hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        pooled = torch.randn(B, D, device=device, dtype=dtype)

        _, r0 = moe(all_hidden, pooled, 0)
        _, r1 = moe(all_hidden, pooled, 1)
        assert not torch.allclose(r0, r1, atol=1e-5)
        moe.clear_cache()

    def test_different_users_different_routing(self, base_config, device, dtype):
        base_config.moe_variant = "arv_mmoe"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)
        B, S, D = 4, 10, base_config.hidden_dim

        all_hidden1 = torch.randn(B, S, D, device=device, dtype=dtype)
        pooled1 = torch.randn(B, D, device=device, dtype=dtype)

        all_hidden2 = torch.randn(B, S, D, device=device, dtype=dtype)
        pooled2 = torch.randn(B, D, device=device, dtype=dtype)

        _, r1 = moe(all_hidden1, pooled1, 0)
        moe.clear_cache()
        _, r2 = moe(all_hidden2, pooled2, 0)
        moe.clear_cache()

        assert not torch.allclose(r1, r2, atol=1e-5)


class TestExpertSharing:
    def test_expert_outputs_cached(self, base_config, device, dtype):
        base_config.moe_variant = "arv_mmoe"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)
        B, S, D = 4, 10, base_config.hidden_dim

        all_hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        pooled = torch.randn(B, D, device=device, dtype=dtype)

        moe(all_hidden, pooled, 0)
        cached_ref = moe._cached_expert_outputs
        assert cached_ref is not None

        moe(all_hidden, pooled, 1)
        assert moe._cached_expert_outputs is cached_ref
        moe.clear_cache()

    def test_cache_invalidates_on_new_input_without_clear(self, base_config, device, dtype):
        """REV-053 / AC-03: the SAFETY direction of the cache contract.

        `test_expert_outputs_cached` verifies the performance property (experts computed
        once per batch) by passing the same tensor twice -- it cannot fail. This asserts
        what actually broke in production (REV-045): a caller that forgets `clear_cache()`
        between different batches must still get correct results. `score_candidates`
        never called `clear_cache`, so before the fix every example after the first was
        scored with example 0's expert outputs.
        """
        base_config.moe_variant = "arv_mmoe"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)
        moe.eval()  # dropout would make the two computations differ for unrelated reasons
        B, S, D = 2, 10, base_config.hidden_dim

        A = torch.randn(B, S, D, device=device, dtype=dtype)
        Bt = torch.randn(B, S, D, device=device, dtype=dtype) * 5.0

        with torch.no_grad():
            moe.clear_cache()
            z_ref, _ = moe(Bt, Bt[:, -1, :], 0)
            z_ref = z_ref.clone()

            # A then B with NO clear_cache in between
            moe.clear_cache()
            moe(A, A[:, -1, :], 0)
            z_dirty, _ = moe(Bt, Bt[:, -1, :], 0)

        assert torch.allclose(z_ref.float(), z_dirty.float(), atol=1e-3), (
            "MoE reused expert outputs from a previous batch: forward(B) after "
            "forward(A) without clear_cache() did not match a freshly-cleared forward(B)"
        )
        moe.clear_cache()

    def test_cache_invalidates_across_a_scoring_loop(self, base_config, device, dtype):
        """REV-053: the same property in the shape `score_candidates` actually runs --
        one fresh single-example tensor per iteration, the previous one freed, and no
        `clear_cache()` anywhere. Tensor ids and storage addresses are recycled by the
        allocator here, so this also guards the cache key against reuse collisions."""
        base_config.moe_variant = "arv_mmoe"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)
        moe.eval()
        S, D = 10, base_config.hidden_dim

        wrong = 0
        moe.clear_cache()
        with torch.no_grad():
            for i in range(25):
                h = torch.randn(1, S, D, device=device, dtype=dtype) * (i + 1)
                z_loop, _ = moe(h, h[:, -1, :], 0)
                z_loop = z_loop.clone()

                moe.clear_cache()
                z_clean, _ = moe(h, h[:, -1, :], 0)
                if not torch.allclose(z_loop.float(), z_clean.float(), atol=1e-3):
                    wrong += 1
                del h

        assert wrong == 0, f"{wrong}/25 examples scored with stale cached expert outputs"
        moe.clear_cache()


class TestPassthrough:
    def test_none_returns_input_unchanged(self, base_config, device, dtype):
        base_config.moe_variant = "none"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)
        B, S, D = 4, 10, base_config.hidden_dim

        all_hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        pooled = torch.randn(B, D, device=device, dtype=dtype)

        Z_ut, r_ut = moe(all_hidden, pooled, 0)
        assert torch.equal(Z_ut, all_hidden)
        assert r_ut is None


class TestParameterAccounting:
    def test_count_matches_manual(self, base_config, tmp_path):
        base_config.log_dir = str(tmp_path)
        report = count_router_parameters(base_config)

        D = base_config.hidden_dim
        K = base_config.num_experts
        E = base_config.objective_embedding_dim
        T = len(OBJECTIVE_ORDER)

        expected_mmoe = T * (D * K + K)
        expected_arv_router = (D + E) * K + K
        expected_arv_emb = T * E
        expected_arv_total = expected_arv_router + expected_arv_emb

        assert report["mmoe_gate_params"] == expected_mmoe
        assert report["arv_router_params"] == expected_arv_router
        assert report["arv_objective_embedding_params"] == expected_arv_emb
        assert report["arv_total_router_params"] == expected_arv_total
        assert report["difference"] == expected_mmoe - expected_arv_total


class TestInferenceIdentity:
    def test_e_rec_is_e_next(self, base_config, device, dtype):
        base_config.moe_variant = "arv_mmoe"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)

        assert OBJECTIVE_ORDER.index("next_item") == 0
        assert not hasattr(moe, "e_rec")

        B, S, D = 4, 10, base_config.hidden_dim
        all_hidden = torch.randn(B, S, D, device=device, dtype=dtype)
        pooled = torch.randn(B, D, device=device, dtype=dtype)
        _, r_inference = moe(all_hidden, pooled, 0)
        assert r_inference is not None
        moe.clear_cache()


class TestConventionalMMoEIndependence:
    def test_gates_are_independent(self, base_config, device, dtype):
        base_config.moe_variant = "mmoe"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)

        gate_params = []
        for gate in moe.gates.gates:
            gate_params.append(gate.weight.data_ptr())

        assert len(set(gate_params)) == len(gate_params)


class TestAblationBRouterConditioning:
    def test_input_only_dimension(self, base_config, device, dtype):
        base_config.moe_variant = "arv_mmoe"
        base_config.router_conditioning = "input_only"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)

        in_features = moe.router.linear.in_features
        assert in_features == base_config.hidden_dim

    def test_objective_conditioned_dimension(self, base_config, device, dtype):
        base_config.moe_variant = "arv_mmoe"
        base_config.router_conditioning = "objective_conditioned"
        moe = MoELayer(base_config).to(device=device, dtype=dtype)

        in_features = moe.router.linear.in_features
        assert in_features == base_config.hidden_dim + base_config.objective_embedding_dim


class TestParameterReportNamespacing:
    """REV-078: FR-6 requires the report for every `moe_variant` run.

    The filename depended only on the model SIZE, but the contents vary with
    `moe_variant`, `num_experts`, `router_conditioning` and `match_mmoe_expert_count`.
    Ablation C runs two variants at one size and G runs four expert counts at one size,
    so every arm wrote to the same path and only the last survived -- including the
    report that is supposed to back C's parameter-matching claim.
    """

    def test_ablation_C_arms_do_not_overwrite_each_other(self, tmp_path):
        from configs.config import Config
        from models.moe import count_router_parameters

        for variant in ("mmoe", "arv_mmoe"):
            config = Config()
            config.log_dir = str(tmp_path)
            config.moe_variant = variant
            count_router_parameters(config, run_tag=f"C_{variant}")

        written = sorted(p.name for p in tmp_path.glob("moe_parameter_report_*.json"))
        assert len(written) == 2, f"C's two arms collided: {written}"

    def test_ablation_G_expert_counts_do_not_overwrite_each_other(self, tmp_path):
        from configs.config import Config
        from models.moe import count_router_parameters

        for k in (2, 4, 6, 8):
            config = Config()
            config.log_dir = str(tmp_path)
            config.moe_variant = "arv_mmoe"
            config.num_experts = k
            count_router_parameters(config, run_tag=f"G_K{k}")

        written = sorted(p.name for p in tmp_path.glob("moe_parameter_report_*.json"))
        assert len(written) == 4, f"G's four arms collided: {written}"

    def test_distinguishing_fields_separate_arms_without_a_run_tag(self, tmp_path):
        from configs.config import Config
        from models.moe import count_router_parameters

        for variant in ("mmoe", "arv_mmoe"):
            config = Config()
            config.log_dir = str(tmp_path)
            config.moe_variant = variant
            count_router_parameters(config)
        for k in (2, 8):
            config = Config()
            config.log_dir = str(tmp_path)
            config.moe_variant = "arv_mmoe"
            config.num_experts = k
            count_router_parameters(config)

        written = sorted(p.name for p in tmp_path.glob("moe_parameter_report_*.json"))
        assert len(written) == 4, f"untagged arms collided: {written}"

    def test_report_carries_the_running_source_revision(self, tmp_path):
        """`steering/code-quality.md` "Artifact Provenance"; REV-136 closing note.

        This report is a pure function of `config` and is NOT derived from a checkpoint,
        so the honest provenance is the RUNNING revision. `artifact_source_revision`
        would be the wrong call: there is no model here whose provenance this could be
        reporting. Read back off disk rather than off the returned dict, since the file
        is the artifact.
        """
        import json

        from configs.config import Config
        from models.moe import count_router_parameters
        from utils import get_source_revision

        config = Config()
        config.log_dir = str(tmp_path)
        report = count_router_parameters(config, run_tag="C_arv_mmoe")

        with open(tmp_path / f"moe_parameter_report_{report['report_suffix']}.json") as f:
            on_disk = json.load(f)

        revision = on_disk.get("source_revision")
        assert revision, "the report must say which code produced it"
        assert set(revision) >= {"commit", "dirty", "source_hash"}, revision
        assert revision == get_source_revision(), (
            "a config-derived report echoes the running revision verbatim"
        )

    def test_report_contents_are_unchanged_by_the_naming(self, tmp_path):
        """The parameter accounting itself must be identical to the untagged call."""
        import json

        from configs.config import Config
        from models.moe import count_router_parameters

        config = Config()
        config.log_dir = str(tmp_path)
        report = count_router_parameters(config, run_tag="C_mmoe")

        with open(tmp_path / f"moe_parameter_report_{report['report_suffix']}.json") as f:
            on_disk = json.load(f)

        assert on_disk == report
        assert on_disk["run_tag"] == "C_mmoe"
        for key in (
            "mmoe_gate_params", "arv_router_params",
            "arv_objective_embedding_params", "arv_total_router_params", "difference",
        ):
            assert key in on_disk, f"FR-6 field '{key}' missing from the report"
