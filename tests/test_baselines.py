"""External baselines (models/baselines, training/baseline_train.py, study X)."""
import copy
import json
import os

import pytest
import torch
import torch.nn.functional as F

from configs.config import Config
from models.baselines import BASELINE_CLASSES, build_baseline, count_baseline_parameters
from models.baselines.hparams import BASELINE_NAMES, published_params
from tests.conftest import tiny_config

NUM_ITEMS = 30


def _small_params(name, **over):
    p = published_params(name, "movielens")
    p.update(hidden_size=16, max_seq_length=8, num_heads=2)
    p.update(over)
    return p


def _batch(model, B=4, S=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(1, NUM_ITEMS + 1, (B, S), generator=g)
    seq[0, :5] = 0          # a left-padded row
    batch = {"input": seq, "target": torch.randint(1, NUM_ITEMS + 1, (B,), generator=g)}
    if model.needs_negatives:
        batch["negative"] = torch.randint(1, NUM_ITEMS + 1, (B,), generator=g)
    if model.needs_same_target:
        batch["same_target"] = torch.randint(1, NUM_ITEMS + 1, (B, S), generator=g)
    return batch


class TestModels:
    @pytest.mark.parametrize("name", BASELINE_NAMES)
    def test_shapes_loss_and_gradients(self, name):
        torch.manual_seed(0)
        model = BASELINE_CLASSES[name](NUM_ITEMS, _small_params(name))
        batch = _batch(model)
        assert model.encode(batch["input"]).shape == (4, 8, 16)
        loss, parts = model.training_loss(batch, torch.Generator().manual_seed(1))
        assert loss.dim() == 0 and torch.isfinite(loss)
        assert parts and all(torch.isfinite(v) for v in parts.values())
        loss.backward()
        assert model.item_embeddings.weight.grad is not None
        assert torch.isfinite(model.item_embeddings.weight.grad).all()
        model.eval()
        scores = model.full_scores(batch["input"])
        assert scores.shape == (4, NUM_ITEMS + 2) and scores.dtype == torch.float32
        if name != "bert4rec":   # no MASK row: its column is padded with -inf
            assert torch.isinf(scores[:, -1]).all()

    def test_vocab_rows_match_the_reference(self):
        assert BASELINE_CLASSES["sasrec"](NUM_ITEMS, _small_params("sasrec")).item_embeddings.num_embeddings == NUM_ITEMS + 1
        assert BASELINE_CLASSES["bert4rec"](NUM_ITEMS, _small_params("bert4rec")).item_embeddings.num_embeddings == NUM_ITEMS + 2

    @pytest.mark.parametrize("name", ["sasrec", "duorec", "gru4rec"])
    def test_causal_models_ignore_the_future(self, name):
        torch.manual_seed(0)
        model = BASELINE_CLASSES[name](NUM_ITEMS, _small_params(name)).eval()
        seq = torch.randint(1, NUM_ITEMS + 1, (2, 8))
        changed = seq.clone()
        changed[:, -1] = (changed[:, -1] % NUM_ITEMS) + 1
        a, b = model.encode(seq), model.encode(changed)
        assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)

    def test_bsarec_filter_is_identity_when_every_frequency_is_kept(self):
        from models.baselines.models import FrequencyLayer

        layer = FrequencyLayer(16, c=40, dropout=0.0).eval()   # keeps all 5 rfft bins of S=8
        x = torch.randn(2, 8, 16)
        assert torch.allclose(layer(x), layer.norm(2 * x), atol=1e-5)

    def test_bsarec_mixes_filter_and_attention_by_alpha(self):
        torch.manual_seed(0)
        model = BASELINE_CLASSES["bsarec"](NUM_ITEMS, _small_params("bsarec", alpha=1.0)).eval()
        seq = torch.randint(1, NUM_ITEMS + 1, (2, 8))
        layer = model.encoder.layers[0]
        x = model.embed(seq)
        from models.baselines.common import additive_mask
        assert torch.allclose(layer(x, additive_mask(seq, True)),
                              layer.feed_forward(layer.filter(x)), atol=1e-6)

    def test_bert4rec_masks_int_s_times_ratio_positions(self):
        model = BASELINE_CLASSES["bert4rec"](NUM_ITEMS, _small_params("bert4rec", max_seq_length=50))
        seq = torch.randint(1, NUM_ITEMS + 1, (6, 50))
        masked = model.mask_positions(seq, torch.Generator().manual_seed(0))
        assert ((masked == model.mask_token).sum(1) == 10).all()     # int(50 * 0.2)

    def test_bert4rec_inference_appends_mask(self):
        torch.manual_seed(0)
        model = BASELINE_CLASSES["bert4rec"](NUM_ITEMS, _small_params("bert4rec")).eval()
        seq = torch.randint(1, NUM_ITEMS + 1, (2, 8))
        shifted = torch.cat([seq[:, 1:], torch.full((2, 1), model.mask_token)], dim=1)
        assert torch.allclose(model.user_embedding(seq), model.encode(shifted)[:, -1])

    def test_duorec_info_nce_matches_the_reference_formulation(self):
        torch.manual_seed(0)
        model = BASELINE_CLASSES["duorec"](NUM_ITEMS, _small_params("duorec", tau=0.7))
        B = 5
        zi, zj = torch.randn(B, 16), torch.randn(B, 16)
        z = torch.cat([zi, zj])
        sim = z @ z.T / 0.7
        pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)])[:, None]
        mask = torch.ones(2 * B, 2 * B, dtype=torch.bool).fill_diagonal_(False)
        for i in range(B):
            mask[i, B + i] = mask[B + i, i] = False
        logits = torch.cat([pos, sim[mask].reshape(2 * B, -1)], dim=1)
        ref = F.cross_entropy(logits, torch.zeros(2 * B, dtype=torch.long))
        assert torch.allclose(model.info_nce(zi, zj), ref, atol=1e-5)

    def test_parameter_count(self):
        model = BASELINE_CLASSES["sasrec"](NUM_ITEMS, _small_params("sasrec"))
        counts = count_baseline_parameters(model)
        assert counts["total_parameters"] == sum(p.numel() for p in model.parameters())
        assert counts["embedding_parameters"] == (NUM_ITEMS + 1) * 16 + 8 * 16


class TestPublishedParams:
    def test_bsarec_table6(self):
        ml = published_params("bsarec", "movielens")
        beauty = published_params("bsarec", "amazon2014", "Beauty")
        assert (ml["alpha"], ml["c"], ml["num_heads"], ml["learning_rate"]) == (0.3, 9, 4, 5e-4)
        assert (beauty["alpha"], beauty["c"], beauty["num_heads"], beauty["learning_rate"]) == (0.7, 5, 1, 5e-4)

    def test_shared_values(self):
        for name in BASELINE_NAMES:
            p = published_params(name, "movielens")
            assert (p["hidden_size"], p["num_layers"], p["max_seq_length"], p["batch_size"]) == (64, 2, 50, 256)

    def test_unknown_dataset_is_refused(self):
        with pytest.raises(ValueError):
            published_params("sasrec", "amazon2014", "Toys")

    def test_study_x_arms(self):
        from training.ablation_runner import _training_fingerprint, resolve_study_configs

        ml, beauty = Config(), Config()
        beauty.apply_dataset_preset("amazon_beauty_2014")
        for base, patience in ((ml, 0), (beauty, 50)):
            arms = resolve_study_configs("X", base)
            assert [a._study_label for a in arms] == list(BASELINE_NAMES)
            for a in arms:
                a.validate()
                assert a.baseline_model == a._study_label
                assert a.max_seq_length == 50 and a.hidden_dim == 64 and a.batch_size == 256
                assert a.early_stop_patience == patience
                assert a.moe_variant == "none" and not a.use_metadata
            assert len({_training_fingerprint(a) for a in arms}) == len(arms)

    def test_new_fields_leave_default_fingerprints_alone(self):
        from training.ablation_runner import _training_fingerprint

        cfg = Config()
        before = _training_fingerprint(cfg)
        cfg.early_stop_patience = 0
        cfg.baseline_model = ""
        cfg.baseline_params = {}
        assert _training_fingerprint(cfg) == before
        cfg.early_stop_patience = 50
        assert _training_fingerprint(cfg) != before

    def test_all_does_not_include_x(self):
        from training.ablation_runner import DEFAULT_STUDIES

        assert "X" not in DEFAULT_STUDIES and set(DEFAULT_STUDIES) == set("ABCDFGH")


class TestPrefixFeed:
    def _feed(self, **kw):
        from training.baseline_train import PrefixFeed

        seqs = torch.tensor([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8], [0, 0, 0, 9, 1]])
        tgts = torch.tensor([[0, 0, 2, 3, 4], [5, 6, 7, 8, 9], [0, 0, 0, 1, 7]])
        return PrefixFeed(seqs, tgts, num_items=12, device=torch.device("cpu"), seed=0, **kw), seqs, tgts

    def test_one_example_per_target(self):
        feed, _, tgts = self._feed()
        assert len(feed) == int((tgts > 0).sum())

    def test_prefixes_are_right_aligned(self):
        feed, _, _ = self._feed()
        got = feed.prefixes(torch.arange(len(feed)))
        want = torch.tensor([
            [0, 0, 0, 0, 1], [0, 0, 0, 1, 2], [0, 0, 1, 2, 3],
            [0, 0, 0, 0, 4], [0, 0, 0, 4, 5], [0, 0, 4, 5, 6], [0, 4, 5, 6, 7], [4, 5, 6, 7, 8],
            [0, 0, 0, 0, 9], [0, 0, 0, 9, 1],
        ])
        assert torch.equal(got, want)
        assert feed.targets.tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 1, 7]

    def test_negatives_avoid_prefix_and_target(self):
        feed, _, _ = self._feed(negatives=True)
        ex = torch.arange(len(feed)).repeat(50)
        b = feed.batch(ex)
        assert not (b["input"] == b["negative"][:, None]).any()
        assert not (b["negative"] == b["target"]).any()
        assert ((b["negative"] >= 1) & (b["negative"] <= 12)).all()

    def test_same_target_has_the_same_target_and_differs_when_possible(self):
        feed, _, _ = self._feed(same_target=True)
        # Target 7 has two examples (row 1 pos 2, row 2 pos 4); every other target one.
        ex = torch.arange(len(feed)).repeat(20)
        b = feed.batch(ex)
        sem = b["same_target"]
        sevens = feed.targets[ex] == 7
        assert sevens.any()
        assert not (sem[sevens] == b["input"][sevens]).all(1).any()
        assert torch.equal(sem[~sevens], b["input"][~sevens])   # singletons use themselves
        # Each supervised view is a training prefix whose target is the same.
        all_prefixes = feed.prefixes(torch.arange(len(feed)))
        for row, t in zip(sem[sevens], feed.targets[ex][sevens]):
            match = (all_prefixes == row).all(1)
            assert (feed.targets[match] == t).any()


class TestEvalSplit:
    def test_rebuilt_wider_split_agrees_with_stored(self, tiny_processed_dir):
        from data.sequence_generator import load_tensorised
        from training.baseline_train import load_eval_split

        processed = tiny_processed_dir()
        for split in ("val", "test"):
            stored = load_tensorised(os.path.join(processed, f"{split}_tensorised.pt"))
            width = stored.sequences.shape[1]
            seqs, targets, users = load_eval_split(processed, split, width + 10)
            assert seqs.shape[1] == width + 10
            assert torch.equal(seqs[:, -width:], stored.sequences.long())
            assert torch.equal(targets, stored.targets.long())
            narrow, _, _ = load_eval_split(processed, split, 4)
            assert torch.equal(narrow, stored.sequences[:, -4:].long())


def _baseline_cfg(processed, out_dir, name, **over):
    from training.ablation_runner import baseline_arm

    base = tiny_config(processed, out_dir)
    cfg = copy.deepcopy(base)
    for k, v in baseline_arm(name, base).items():
        if k != "label":
            setattr(cfg, k, v)
    cfg.pretrain_epochs = 2
    cfg.max_steps_per_epoch = 3
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class TestTraining:
    def test_study_x_runs_every_baseline(self, tiny_processed_dir, tmp_path):
        from training.ablation_runner import run_ablation

        cfg = tiny_config(tiny_processed_dir(), str(tmp_path), pretrain_epochs=2,
                          max_steps_per_epoch=3)
        results = run_ablation(cfg, "t-small", "X")["X"]
        assert [r["label"] for r in results] == list(BASELINE_NAMES)
        for r in results:
            assert r["status"] == "ok", (r["label"], r.get("error"))
            assert 0.0 <= r["metrics"]["ndcg"]["mean"] <= 1.0
            assert r["parameters"]["total_parameters"] > 0
            assert r["baseline_model"] == r["label"]
            assert r["eval_protocol"]["candidates"] == "full_catalogue"
            assert r["best_epoch"] in (0, 1)
        arm_json = os.path.join(cfg.ablation_dir, "seed_42", "arms", "X_t_small", "sasrec.json")
        assert json.load(open(arm_json))["status"] == "ok"
        assert os.path.exists(os.path.join(cfg.ablation_dir, "seed_42", "X_t_small.json"))

    def test_loss_decreases_on_fixed_batches(self, tiny_processed_dir, tmp_path):
        """A few optimizer steps on one repeated batch lower each model's loss."""
        from data.sequence_generator import load_train_dataset
        from training.baseline_train import PrefixFeed

        processed = tiny_processed_dir()
        for name in BASELINE_NAMES:
            cfg = _baseline_cfg(processed, str(tmp_path), name)
            cfg.num_items = json.load(open(os.path.join(processed, "dataset_stats.json")))["num_items"]
            torch.manual_seed(0)
            model = build_baseline(cfg)
            train = load_train_dataset(processed, cfg.max_seq_length)
            feed = PrefixFeed(train.sequences, train.targets, cfg.num_items, torch.device("cpu"), 0,
                              negatives=model.needs_negatives, same_target=model.needs_same_target)
            batch = feed.batch(torch.arange(64))
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            model.eval()      # no dropout, so the loss is a function of the weights only
            first = None
            for _ in range(15):
                gen = torch.Generator().manual_seed(3)
                loss, _ = model.training_loss(batch, gen)
                first = loss.item() if first is None else first
                opt.zero_grad()
                loss.backward()
                opt.step()
            gen = torch.Generator().manual_seed(3)
            assert model.training_loss(batch, gen)[0].item() < first, name

    def test_early_stopping(self, tiny_processed_dir, tmp_path, monkeypatch):
        import training.baseline_train as BT
        from utils import set_seed

        scores = iter([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

        def fake_eval(*a, **k):
            return {m: {"mean": next(scores) if m == "ndcg" else 0.0, "std": 0.0}
                    for m in ("ndcg", "mrr", "recall", "precision")}

        monkeypatch.setattr(BT, "evaluate_baseline", fake_eval)
        cfg = _baseline_cfg(tiny_processed_dir(), str(tmp_path), "sasrec",
                            pretrain_epochs=6, early_stop_patience=2)
        set_seed(cfg.seed)
        result = BT.run_baseline_pretrain(cfg, "t-small")
        assert result["best_epoch"] == 0
        assert result["early_stopped"] == {"patience": 2, "last_completed_epoch": 2}
        assert len(result["epoch_mean_losses"]) == 3

    def test_resume_is_exact(self, tiny_processed_dir, tmp_path, monkeypatch):
        import training.baseline_train as BT
        from utils import set_seed

        processed = tiny_processed_dir()
        ref_cfg = _baseline_cfg(processed, str(tmp_path / "ref"), "duorec", pretrain_epochs=3)
        set_seed(42)
        ref = BT.run_baseline_pretrain(ref_cfg, "t-small")

        cfg = _baseline_cfg(processed, str(tmp_path / "cut"), "duorec", pretrain_epochs=3)
        real_eval = BT.evaluate_baseline
        calls = {"n": 0}

        def crash_on_second(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt("simulated kill")
            return real_eval(*a, **k)

        monkeypatch.setattr(BT, "evaluate_baseline", crash_on_second)
        set_seed(42)
        with pytest.raises(KeyboardInterrupt):
            BT.run_baseline_pretrain(cfg, "t-small")
        monkeypatch.setattr(BT, "evaluate_baseline", real_eval)
        set_seed(42)
        resumed = BT.run_baseline_pretrain(cfg, "t-small")
        assert resumed["epoch_mean_losses"] == ref["epoch_mean_losses"]
        assert resumed["best_val_ndcg"] == ref["best_val_ndcg"]
        a = torch.load(os.path.join(ref_cfg.model_checkpoint_dir(), "last.pt"), weights_only=False)
        b = torch.load(os.path.join(cfg.model_checkpoint_dir(), "last.pt"), weights_only=False)
        for k, v in a["model_state_dict"].items():
            assert torch.equal(v, b["model_state_dict"][k]), k
