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
