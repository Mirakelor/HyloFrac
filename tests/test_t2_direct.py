"""Tests for the T2 "predicted directly" submission channel."""

import json
import os

import numpy as np

from hylofrac.eval.evaluate import evaluate_submission
from hylofrac.eval.loader import SceneSample
from hylofrac.eval.metrics import scene_metrics
from hylofrac.eval.submit import (read_adjacency, validate_submission,
                                  write_submission)


def test_submission_adjacency_roundtrip(tmp_path):
    path = os.path.join(str(tmp_path), "s.json")
    fragments = {"frag_00": {"R": np.eye(3), "t": np.zeros(3)},
                 "frag_01": {"R": np.eye(3), "t": np.ones(3)}}
    write_submission(path, "000102166_v00", "phformer", fragments,
                     adjacency=[("frag_00", "frag_01")])
    ok, err = validate_submission(path)
    assert ok, err
    assert read_adjacency(path) == [("frag_00", "frag_01")]
    data = json.load(open(path, encoding="utf-8"))
    assert data["adjacency"] == [["frag_00", "frag_01"]]
    # absent adjacency round-trips as None
    write_submission(path, "000102166_v00", "dgl", fragments)
    assert read_adjacency(path) is None


def test_validate_rejects_bad_adjacency(tmp_path):
    path = os.path.join(str(tmp_path), "bad.json")
    payload = {"scene": "x", "fragments": {"frag_00": {"R": np.eye(3).tolist(),
                                                       "t": [0, 0, 0]}},
               "adjacency": [["frag_00"]]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    ok, err = validate_submission(path)
    assert not ok and "adjacency" in err


def test_scene_metrics_uses_explicit_adjacency(packaged_scene):
    sample = SceneSample(packaged_scene, "000102166_v00", n_points=300)
    gt = sample.adjacency
    # perfect poses with the GT graph passed directly: F1 = 1
    m = scene_metrics(sample, sample.gt_R, sample.gt_t, pred_adj=gt)
    assert m["adj_f1"] > 0.99
    # wrong explicit graph: F1 = 0 even though the poses are perfect
    wrong = {"frag_00": set(), "frag_01": set()}
    m2 = scene_metrics(sample, sample.gt_R, sample.gt_t, pred_adj=wrong)
    assert m2["adj_f1"] == 0.0
    assert m2["pa"] > 0.99          # pose metrics are unaffected


def test_evaluate_direct_channel(packaged_scene, tmp_path):
    scene_id = "000102166_v00"
    sample = SceneSample(packaged_scene, scene_id, n_points=300)
    fragments = {name: {"R": sample.gt_R[i], "t": sample.gt_t[i]}
                 for i, name in enumerate(sample.names)}
    write_submission(os.path.join(str(tmp_path), f"{scene_id}.json"),
                     scene_id, "phformer", fragments,
                     adjacency=[(sample.names[0], sample.names[1])])
    data_root = os.path.abspath(os.path.join(packaged_scene, "..", "..", ".."))
    result = evaluate_submission(os.path.join(str(tmp_path), f"{scene_id}.json"),
                                 data_root, ["val"], 300, 0)
    assert result is not None
    assert result["adj_f1"] > 0.99
    assert result["degrade"] == "none"
