import os

import numpy as np
import torch
import pytest

from configs.config import Config
from models.transformer import TransformerModel
from models.moe import MoELayer
from models.lora import (
    inject_lora,
    get_lora_state_dict,
    load_lora_state_dict,
    count_lora_parameters,
)
from training.federated import weighted_fedavg
from training.local_train import compute_frozen_checksum


@pytest.fixture
def base_config():
    cfg = Config()
    cfg.apply_model_config("t-small")
    cfg.num_items = 50
    cfg.use_metadata = False
    cfg.dropout = 0.0
    cfg.vocab_size = 52
    return cfg


class TestWeightedFedAvg:
    def test_weighted_average(self):
        sd1 = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
        sd2 = {"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([7.0])}
        counts = [3, 1]

        result = weighted_fedavg([sd1, sd2], counts, use_weighted=True)
        expected_a = (3/4) * torch.tensor([1.0, 2.0]) + (1/4) * torch.tensor([5.0, 6.0])
        expected_b = (3/4) * torch.tensor([3.0]) + (1/4) * torch.tensor([7.0])

        assert torch.allclose(result["a"], expected_a, atol=1e-5)
        assert torch.allclose(result["b"], expected_b, atol=1e-5)

    def test_simple_mean(self):
        sd1 = {"a": torch.tensor([2.0, 4.0])}
        sd2 = {"a": torch.tensor([6.0, 8.0])}
        counts = [10, 1]

        result = weighted_fedavg([sd1, sd2], counts, use_weighted=False)
        expected = torch.tensor([4.0, 6.0])
        assert torch.allclose(result["a"], expected, atol=1e-5)


class TestCheckpointIntegrity:
    def test_frozen_checksum_unchanged_after_lora(self, base_config):
        from utils import get_device
        device = get_device(base_config)
        model = TransformerModel(base_config)
        moe = MoELayer(base_config).to(device=device, dtype=torch.bfloat16)

        cksum_before = compute_frozen_checksum(model, moe)
        inject_lora(model, rank=8)

        lora_params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(lora_params, lr=1e-2)
        B, S = 2, base_config.max_seq_length
        seq = torch.randint(1, 30, (B, S), device=device)
        out = model(seq, return_all_hidden=True)
        loss = out[:, -1, :].sum()
        loss.backward()
        opt.step()

        cksum_after = compute_frozen_checksum(model, moe)
        assert cksum_before == cksum_after


def _build_federated_fixture(dest, n_users=6, n_val=32):
    """A tiny but REAL processed dir plus a trained best.pt, so the federated and
    personalization entry points can be called for real rather than reimplemented."""
    import json
    import pickle

    import numpy as np

    from configs.config import Config
    from tests.test_training import _build_subset_processed_dir
    from training.pretrain import run_pretrain

    processed = os.path.join(dest, "processed")
    _build_subset_processed_dir(processed, n_train=128, n_val=n_val)

    # `test_examples.pkl` drives client/user selection in both stages.
    src_test = os.path.join("data", "processed", "test_examples.pkl")
    with open(src_test, "rb") as f:
        test_examples = pickle.load(f)

    keep_users, idx = [], []
    seen = {}
    for i, uid in enumerate(test_examples["user_ids"]):
        seen.setdefault(uid, []).append(i)
        if len(seen) > n_users:
            break
    for uid, rows in list(seen.items())[:n_users]:
        keep_users.append(uid)
        idx.extend(rows[:4])

    # REV-086: the example splits are numpy arrays now (they were lists of Python
    # ints). Subset either representation.
    n_rows = len(test_examples["user_ids"])
    subset = {}
    for k, v in test_examples.items():
        if v is None or len(v) != n_rows:
            continue
        subset[k] = v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]
    with open(os.path.join(processed, "test_examples.pkl"), "wb") as f:
        pickle.dump(subset, f)

    # REV-111: this rewrites `test_examples.pkl` with a SCATTERED set of original rows,
    # while `_build_subset_processed_dir` wrote `test_candidates.npy` as the first
    # n_test rows -- so example i and candidate row i no longer described the same
    # example. Nothing cross-checked them while personalization built its own
    # candidates; now that it scores the stored rows by index, the fixture has to be
    # self-consistent the way a real `data/processed` is. Take the SAME rows from the
    # source matrix and re-fingerprint.
    from data.sequence_generator import save_eval_candidates

    src_cand = np.load(os.path.join("data", "processed", "test_candidates.npy"), mmap_mode="r")
    save_eval_candidates(
        [np.asarray(src_cand[i]) for i in idx],
        os.path.join(processed, "test_candidates.npy"),
    )

    cfg = Config()
    cfg.apply_model_config("t-small")
    cfg.pretrain_epochs = 1
    cfg.batch_size = 64
    cfg.max_steps_per_epoch = 2
    cfg.num_workers = 0
    cfg.processed_data_dir = processed
    cfg.output_dir = os.path.join(dest, "outputs")
    cfg.checkpoint_dir = os.path.join(dest, "outputs", "checkpoints")
    cfg.log_dir = os.path.join(dest, "outputs", "logs")
    cfg.plot_dir = os.path.join(dest, "outputs", "plots")
    run_pretrain(cfg, "t-small")
    return cfg, keep_users


class TestMoEExclusionGuard:
    """T5.4 / REV-026b,c: the guard must catch a deliberately-injected MoE key.

    The previous test implemented only the positive half -- it asserted that a clean
    LoRA dict contains no MoE keys. The negative case was absent, and the guard it
    exercised was structurally unreachable, so it would still have passed if the guard
    were deleted.
    """

    def test_injected_moe_key_is_rejected(self, base_config):
        from training.federated import assert_no_moe_keys

        base_config.moe_variant = "arv_mmoe"
        model = TransformerModel(base_config)
        moe = MoELayer(base_config).to(dtype=torch.bfloat16)
        inject_lora(model, rank=8)

        tampered = dict(get_lora_state_dict(model))
        tampered["moe.router.linear.weight"] = torch.zeros(4, 4)

        with pytest.raises(AssertionError, match="MoE parameters present"):
            assert_no_moe_keys(tampered, moe)

    def test_real_moe_state_key_is_rejected(self, base_config):
        """Not just the hardcoded prefixes: a real key from the MoE's own state dict."""
        from training.federated import assert_no_moe_keys

        base_config.moe_variant = "arv_mmoe"
        model = TransformerModel(base_config)
        moe = MoELayer(base_config).to(dtype=torch.bfloat16)
        inject_lora(model, rank=8)

        real_moe_key = next(iter(moe.state_dict().keys()))
        tampered = dict(get_lora_state_dict(model))
        tampered[real_moe_key] = torch.zeros(1)

        with pytest.raises(AssertionError, match="MoE parameters present"):
            assert_no_moe_keys(tampered, moe)

    def test_clean_update_passes(self, base_config):
        from training.federated import assert_no_moe_keys

        base_config.moe_variant = "arv_mmoe"
        model = TransformerModel(base_config)
        moe = MoELayer(base_config).to(dtype=torch.bfloat16)
        inject_lora(model, rank=8)

        sd = get_lora_state_dict(model)
        assert sd, "LoRA state dict should not be empty"
        assert_no_moe_keys(sd, moe)  # must not raise
        for key in sd:
            assert "lora_" in key


class TestFederatedConvergence:
    """T5.3 / REV-016: exercise `run_federated` itself.

    The previous test never imported `run_federated`. It reimplemented a toy loop using
    `loss = out.sum()`, never computed NDCG or communication cost, reused one model
    across all clients, and set `federated_clients_per_round` / `lora_local_epochs` --
    neither of which exists on Config, so they silently created dead attributes.
    """

    @pytest.mark.slow
    def test_run_federated_two_rounds(self, tmp_path):
        from training.federated import run_federated
        from training.pretrain import build_model, load_checkpoint
        from models.lora import inject_lora, count_lora_parameters

        cfg, _ = _build_federated_fixture(str(tmp_path))
        cfg.federated_rounds = 2
        cfg.clients_per_round = 3
        cfg.local_epochs = 1

        result = run_federated(cfg, "t-small")
        logs = result["round_logs"]

        assert len(logs) == 2, f"expected 2 rounds, got {len(logs)}"

        # AC-06: NDCG present and finite per round
        for r in logs:
            assert "val_ndcg" in r, f"round {r['round']} has no val_ndcg"
            assert np.isfinite(r["val_ndcg"]), f"round {r['round']} NDCG not finite"
            assert 0.0 <= r["val_ndcg"] <= 1.0

        # AC-07: communication cost matches the spec formula
        model, moe = build_model(cfg)
        load_checkpoint(os.path.join(cfg.model_checkpoint_dir(), "best.pt"), model, moe)
        inject_lora(model, cfg.lora_rank)
        expected_upload_kb = count_lora_parameters(model) * 2 / 1024

        for r in logs:
            assert abs(r["upload_kb"] - expected_upload_kb) < 1e-6, (
                f"upload_kb={r['upload_kb']}, expected {expected_upload_kb}"
            )
            assert abs(r["download_kb"] - expected_upload_kb) < 1e-6
            # REV-036: the multiplier is clients_per_round, not the contributing count
            expected_total = (r["upload_kb"] + r["download_kb"]) * cfg.clients_per_round
            assert abs(r["total_round_kb"] - expected_total) < 1e-6, (
                f"total_round_kb={r['total_round_kb']}, expected {expected_total}"
            )
            assert r["clients_per_round"] == cfg.clients_per_round

        # the round log and convergence plot are written
        mn = "t_small"
        assert os.path.exists(os.path.join(cfg.log_dir, f"federated_rounds_{mn}.json"))
        assert os.path.exists(
            os.path.join(cfg.plot_dir, f"federated_convergence_{mn}.png")
        )

    @pytest.mark.slow
    def test_frozen_backbone_survives_run_federated(self, tmp_path):
        """AC-02 via the real entry point: run_federated asserts this internally, so
        reaching the end without raising is the evidence."""
        from training.federated import run_federated

        cfg, _ = _build_federated_fixture(str(tmp_path))
        cfg.federated_rounds = 2
        cfg.clients_per_round = 2
        cfg.local_epochs = 1

        result = run_federated(cfg, "t-small")
        assert result["round_logs"]


class TestFederatedValidationProtocol:
    """REV-112: the federated validation pass must reuse the STORED candidate sets and
    must not score the whole validation split on every round.

    (a) `run_federated` rebuilt the entire validation candidate matrix on each call via
        `generate_eval_candidates` with no `seed=`, so it silently used that function's
        default of 42 -- a `--seed 123` run evaluated against seed-42 candidates while
        every other stage used 123 -- and did ~1M sequential `filter_negatives` calls to
        reproduce a matrix `val_candidates.npy` already holds, fingerprinted.
    (b) `evaluate_model` was called on the full split inside the round loop, i.e. once
        per `federated_rounds`, which dominated the stage's cost.
    """

    @staticmethod
    def _spy_evaluate(monkeypatch):
        """Record the sequences/candidates handed to each per-round evaluation."""
        import training.federated as fed_mod

        seen = []
        real = fed_mod.evaluate_model

        def spy(model, sequences, candidates, config, **kwargs):
            seen.append((sequences.detach().cpu().clone(), candidates.detach().cpu().clone()))
            return real(model, sequences, candidates, config, **kwargs)

        monkeypatch.setattr(fed_mod, "evaluate_model", spy)
        return seen

    @pytest.mark.slow
    def test_scores_stored_candidates_and_bounds_each_round(self, tmp_path, monkeypatch):
        import numpy as np

        from data.sequence_generator import load_eval_candidates
        from training.federated import run_federated

        cfg, _ = _build_federated_fixture(str(tmp_path), n_val=32)
        cfg.federated_rounds = 2
        cfg.clients_per_round = 2
        cfg.local_epochs = 1
        cfg.federated_eval_examples = 8

        stored = load_eval_candidates(
            os.path.join(cfg.processed_data_dir, "val_candidates.npy")
        )
        assert cfg.federated_eval_examples < stored.shape[0], (
            "fixture must have more validation examples than the per-round bound, "
            "or the bounding half of this test proves nothing"
        )

        seen = self._spy_evaluate(monkeypatch)
        result = run_federated(cfg, "t-small")
        logs = result["round_logs"]

        # REV-117: two bounded per-round passes, plus ONE extra full-split pass on the
        # final round for the headline number.
        assert len(seen) == 3, (
            f"expected one bounded evaluation per round plus one final full-split pass, "
            f"got {len(seen)}"
        )

        # (b) EVERY round -- the last one included -- scores exactly the bounded prefix.
        # REV-117: the final round used to be exempted, so rounds 0..N-2 and round N-1
        # were computed over different user populations and plotted as one series.
        per_round = seen[:-1]
        for round_idx, (_, cands) in enumerate(per_round):
            assert len(cands) == cfg.federated_eval_examples, (
                f"round {round_idx} scored {len(cands)} examples, expected the bounded "
                f"{cfg.federated_eval_examples}"
            )
            assert np.array_equal(
                cands.numpy(), stored[: cfg.federated_eval_examples]
            ), (
                f"round {round_idx} must score a PREFIX of the stored rows, so the "
                f"round-over-round curve stays comparable"
            )

        # ... and the final round additionally scores the full split, once.
        full_seqs, full_cands = seen[-1]
        assert len(full_cands) == stored.shape[0], (
            f"the final full-split pass scored {len(full_cands)} examples, expected the "
            f"full {stored.shape[0]}-example split"
        )

        # (a) the rows scored are the STORED rows, not a regenerated matrix.
        assert np.array_equal(full_cands.numpy(), stored), (
            "run_federated scored candidates that are not the stored val_candidates.npy "
            "rows -- federated NDCG is on a different candidate protocol from every "
            "other stage"
        )

        # REV-117: every point the convergence figure plots comes from the same number of
        # examples. This is the assertion that makes the curve readable.
        scored = {log["val_examples_scored"] for log in logs}
        assert scored == {cfg.federated_eval_examples}, (
            f"the plotted series mixes evaluation populations: {scored}"
        )
        assert all(log["is_final_round"] is False for log in logs[:-1])
        assert logs[-1]["is_final_round"] is True

        # The headline number is the final round's full-split score, reported separately.
        assert logs[-1]["val_examples_scored_full"] == stored.shape[0]
        assert "val_ndcg_full" in logs[-1]
        assert "val_ndcg_full" not in logs[0], (
            "only the final round runs the full split"
        )
        assert result["final_val_ndcg"] == logs[-1]["val_ndcg_full"], (
            "the headline federated NDCG must be the full-split score, not the "
            "0.5%-user prefix the curve plots"
        )

    @pytest.mark.slow
    def test_zero_means_full_split_every_round(self, tmp_path, monkeypatch):
        from training.federated import run_federated

        cfg, _ = _build_federated_fixture(str(tmp_path), n_val=32)
        cfg.federated_rounds = 2
        cfg.clients_per_round = 2
        cfg.local_epochs = 1
        cfg.federated_eval_examples = 0

        seen = self._spy_evaluate(monkeypatch)
        run_federated(cfg, "t-small")

        sizes = {len(c) for _, c in seen}
        assert len(sizes) == 1, (
            f"federated_eval_examples=0 must score the same full split every round, "
            f"got sizes {sizes}"
        )
        # REV-117: when the prefix already IS the full split there is nothing extra to
        # run, so the final round must not pay for a second identical pass.
        assert len(seen) == 2, (
            f"federated_eval_examples=0 must not add a redundant final-round pass; "
            f"got {len(seen)} evaluations for 2 rounds"
        )


class TestPersonalizationEntryPoint:
    """T5.2 / REV-026a: call `run_local_personalization` rather than performing one
    manual Adam step on a `.sum()` loss."""

    @pytest.mark.slow
    def test_run_local_personalization_preserves_frozen_backbone(self, tmp_path):
        from training.local_train import run_local_personalization

        cfg, _ = _build_federated_fixture(str(tmp_path))
        cfg.personalization_eval_users = 2
        cfg.local_epochs = 1

        # run_local_personalization asserts the frozen checksum per user internally
        report = run_local_personalization(cfg, "t-small")

        assert report["num_users_evaluated"] >= 1, "no users were evaluated"
        for key in ("with_lora", "without_lora", "gain"):
            assert key in report, f"report missing '{key}'"
            assert np.isfinite(report[key]["mean"]), f"{key} mean is not finite"


class TestPersonalizationNegativeSampling:
    """REV-111: negatives must come from a run-scoped RNG, and the per-user evaluation
    must score the stored candidate rows.

    Both `filter_negatives` call sites in `local_train.py` omitted `rng=`, so the
    function fell back to a RandomState seeded from the TARGET ITEM. In `train_user_lora`
    that made the adapter see the identical negatives for a given target on every one of
    `local_epochs` passes -- the contrastive signal on passes 2 and 3 duplicated pass 1 --
    and in the candidate loop it put personalization NDCG on a different candidate
    protocol from `test_metrics_{model}.json`.
    """

    @pytest.mark.slow
    def test_training_negatives_are_run_scoped_and_evaluation_uses_stored_rows(
        self, tmp_path, monkeypatch
    ):
        import numpy as np

        import training.local_train as lt
        from data.sequence_generator import load_eval_candidates

        cfg, _ = _build_federated_fixture(str(tmp_path))
        cfg.personalization_eval_users = 3
        cfg.local_epochs = 3

        stored = load_eval_candidates(
            os.path.join(cfg.processed_data_dir, "test_candidates.npy")
        )

        # --- record every training negative draw ---
        real_filter = lt.filter_negatives
        draws = []

        def filter_spy(pool, target, history_set, num_negatives, *args, **kwargs):
            out = real_filter(pool, target, history_set, num_negatives, *args, **kwargs)
            draws.append((int(target), kwargs.get("rng"), tuple(int(x) for x in out)))
            return out

        monkeypatch.setattr(lt, "filter_negatives", filter_spy)

        # --- record the candidate sets each user is actually scored against ---
        real_eval_user = lt.evaluate_user
        scored = []

        def eval_spy(model, moe, seqs, targets, candidate_sets, config, **kwargs):
            scored.append([c.detach().cpu().numpy().copy() for c in candidate_sets])
            return real_eval_user(model, moe, seqs, targets, candidate_sets, config, **kwargs)

        monkeypatch.setattr(lt, "evaluate_user", eval_spy)

        lt.run_local_personalization(cfg, "t-small")

        assert draws, "no negatives were drawn -- the fixture trained no user"
        assert all(rng is not None for _, rng, _ in draws), (
            "a filter_negatives call site still omits rng=, so its draw is seeded from "
            "the target item rather than the run seed"
        )

        # A target seen on several passes must not receive byte-identical negatives.
        by_target = {}
        for target, _, negs in draws:
            by_target.setdefault(target, []).append(negs)
        repeated = {t: v for t, v in by_target.items() if len(v) > 1}
        assert repeated, (
            "fixture drew every target once; raise local_epochs so the repeat-draw "
            "property is actually exercised"
        )
        assert any(len(set(v)) > 1 for v in repeated.values()), (
            "every repeated target received the identical negative set -- the draw is "
            "still a function of the target, not of the run RNG"
        )

        # Every scored candidate row is one of the stored rows.
        assert scored, "no user was evaluated"
        stored_rows = {tuple(int(x) for x in row) for row in stored}
        for user_sets in scored:
            for row in user_sets:
                assert tuple(int(x) for x in row) in stored_rows, (
                    "personalization scored a candidate row that is not in the stored "
                    "test_candidates.npy -- its NDCG is on a different protocol from "
                    "the headline test metrics"
                )


class TestLocalTrainingSourceIsNotTheEvaluationSplit:
    """REV-003 / REV-004: local adaptation must never be fitted to the test split.

    Personalization used to train each user's LoRA on the first half of that user's TEST
    examples and then report NDCG over ALL of them, so the headline gain was inflated by
    memorisation of the scored examples. The federated path drew client training data
    from the same place, putting test-split interactions into the global adapter state
    that every later round and every downstream test evaluation inherits.
    """

    def _config(self):
        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.max_seq_length = 8
        cfg.local_train_max_examples = 4
        return cfg

    def test_config_rejects_the_test_split_outright(self):
        cfg = self._config()
        cfg.local_train_source = "test"
        with pytest.raises(ValueError, match="test split is never a training source"):
            cfg.validate()

    def test_train_source_examples_come_from_training_history(self):
        from training.local_train import build_local_training_examples

        cfg = self._config()
        cfg.local_train_source = "train"
        history = {7: [11, 12, 13, 14, 15, 16]}

        seqs, tgts = build_local_training_examples(7, cfg, history)

        assert seqs, "the user has 6 training interactions; examples must be built"
        assert len(seqs) <= cfg.local_train_max_examples
        assert set(tgts).issubset(set(history[7])), (
            "every local training target must be a TRAINING interaction"
        )
        for s in seqs:
            assert set(int(x) for x in s.tolist()) <= set(history[7]) | {0}

    def test_training_and_evaluation_indices_are_disjoint(self):
        """The property the defect violated: nothing the adapter is fitted to is scored."""
        from training.local_train import build_local_training_examples

        cfg = self._config()
        cfg.local_train_source = "train"
        train_history = {7: [11, 12, 13, 14]}
        # The user's TEST examples -- what `evaluate_user` scores.
        test_targets = {21, 22, 23}

        _, train_tgts = build_local_training_examples(7, cfg, train_history)
        assert set(train_tgts).isdisjoint(test_targets)

    def test_val_source_reads_the_validation_examples(self):
        from training.local_train import build_local_training_examples

        cfg = self._config()
        cfg.local_train_source = "val"
        cfg.validate()
        val_examples = {
            "sequences": [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
            "targets": [31, 32, 33],
            "user_ids": [7, 7, 9],
        }
        user_val_map = {7: [0, 1], 9: [2]}

        seqs, tgts = build_local_training_examples(7, cfg, {}, val_examples, user_val_map)
        assert tgts == [31, 32]
        assert len(seqs) == 2

    def test_user_with_too_little_history_yields_nothing(self):
        """Rather than falling back to the evaluation examples, as the defect did."""
        from training.local_train import build_local_training_examples

        cfg = self._config()
        cfg.local_train_source = "train"
        seqs, tgts = build_local_training_examples(7, cfg, {7: [11]})
        assert seqs == [] and tgts == []


class TestLocalLossNormalizationFlag:
    """REV-061: `SampledSoftmaxLoss(config.num_sampled_negatives)` bound the negative
    COUNT to the `normalize_embeddings` boolean. It only behaved correctly because 100 is
    truthy and the config default is True; with `normalize_embeddings=False` the adapter
    would train under a different scoring geometry than it is evaluated in."""

    @pytest.mark.parametrize("normalize", [True, False])
    def test_loss_flag_tracks_config(self, normalize, monkeypatch):
        import training.local_train as lt

        cfg = Config()
        cfg.apply_model_config("t-small")
        cfg.num_items = 20
        cfg.vocab_size = 22
        cfg.use_metadata = False
        cfg.normalize_embeddings = normalize
        cfg.local_epochs = 0  # construct the loss, run no steps

        captured = {}
        real_loss_cls = lt.SampledSoftmaxLoss

        def spy(*args, **kwargs):
            obj = real_loss_cls(*args, **kwargs)
            captured["normalize_embeddings"] = obj.normalize_embeddings
            return obj

        monkeypatch.setattr(lt, "SampledSoftmaxLoss", spy)

        model = TransformerModel(cfg)
        for p in model.parameters():
            p.requires_grad_(False)
        inject_lora(model, cfg.lora_rank)
        moe = MoELayer(cfg)

        lt.train_user_lora(
            model, moe, [], [], np.arange(1, 21), set(), cfg, torch.device("cpu"),
        )

        assert captured["normalize_embeddings"] is normalize, (
            "the loss must be constructed with config.normalize_embeddings, not with "
            "the negative count"
        )
