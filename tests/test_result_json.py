"""Result payloads are plain JSON, so an arm that trained to completion is never lost at
write time (AWS ablation on c094b2c: arms trained, then failed in `atomic_write_json`
with `TypeError: Object of type int32 is not JSON serializable`)."""
import json
import logging
import os

import numpy as np
import torch

from tests.conftest import tiny_config
from utils import atomic_write_json, to_plain_json


class TestToPlainJson:
    def test_converts_numpy_and_torch_and_names_each_path(self):
        payload = {
            "user_id": np.int32(7),
            "is_cold": np.bool_(True),
            "gain": np.float32(0.25),
            "mean": np.float64(0.5),
            "util": np.array([1, 2]),
            "t": torch.tensor([0.5, 1.5]),
            "scalar_t": torch.tensor(3),
            "rows": [{"id": np.int64(1)}, (np.int16(2), "x")],
            np.int32(4): "numpy key",
            "plain": {"a": 1, "b": 1.5, "c": True, "d": None, "e": "s"},
        }
        out, leaks = to_plain_json(payload)

        assert out == {
            "user_id": 7, "is_cold": True, "gain": 0.25, "mean": 0.5, "util": [1, 2],
            "t": [0.5, 1.5], "scalar_t": 3, "rows": [{"id": 1}, [2, "x"]], 4: "numpy key",
            "plain": {"a": 1, "b": 1.5, "c": True, "d": None, "e": "s"},
        }
        assert type(out["user_id"]) is int and type(out["is_cold"]) is bool
        assert type(out["mean"]) is float
        assert [type(k) for k in out if not isinstance(k, str)] == [int]
        assert len(leaks) == 10, leaks
        assert "$.user_id (numpy.int32)" in leaks
        assert "$.rows[1][0] (numpy.int16)" in leaks
        assert not [p for p in leaks if p.startswith("$.plain")]

    def test_unknown_objects_still_fail(self):
        out, leaks = to_plain_json({"x": object()})
        assert leaks == []
        try:
            json.dumps(out)
        except TypeError:
            return
        raise AssertionError("a non-numpy object must not be silently serialised")


def test_atomic_write_json_writes_numpy_payload_and_warns(tmp_path, caplog):
    path = str(tmp_path / "arm.json")
    with caplog.at_level(logging.WARNING, logger="utils"):
        atomic_write_json(path, {"status": "ok", "user_id": np.int32(3)})

    assert json.load(open(path)) == {"status": "ok", "user_id": 3}
    assert "$.user_id (numpy.int32)" in caplog.text
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f]


def test_arm_with_extras_is_written_ok(tiny_processed_dir, tmp_path):
    """End to end: an arm with post-training extras (B's routing diagnostics), run
    through `run_ablation`, persists a plain-JSON arm record with status "ok"."""
    from training.ablation_runner import _arm_path, run_ablation

    cfg = tiny_config(
        tiny_processed_dir(), str(tmp_path),
        pretrain_epochs=1, max_steps_per_epoch=2, ablation_eval_examples=32,
    )
    results = run_ablation(cfg, "t-small", "B", arms=["input_only"])

    (record,) = results["B"]
    assert record["status"] == "ok", record.get("error")
    # The source emits plain types; the writer's conversion is only a safety net.
    _, leaks = to_plain_json(record)
    assert leaks == [], leaks

    with open(_arm_path(cfg, "B", "t-small", "input_only")) as f:
        stored = json.load(f)
    assert stored["status"] == "ok"
