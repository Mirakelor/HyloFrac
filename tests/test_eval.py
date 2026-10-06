"""Unit tests for the evaluation framework (metrics, submission, report)."""

import json
import os

import numpy as np
import pytest


def test_loader_protocol(packaged_scene):
    from hylofrac.eval.loader import SceneSample

    sample = SceneSample(packaged_scene, "000102166_v00", n_points=500)
    assert sample.n_parts == 2
    assert sample.pcs.shape == (2, 500, 3)
    assert sample.gt_t.shape == (2, 3)
    assert sample.gt_R.shape == (2, 3, 3)
    # rotations are valid
    for i in range(2):
        assert np.allclose(sample.gt_R[i] @ sample.gt_R[i].T, np.eye(3), atol=1e-9)
    # adjacency from the annotation: fragments 0 and 1 touch
    assert "frag_01" in sample.adjacency["frag_00"]
    assert sample.part == "cranium"
    # deterministic sampling
    again = SceneSample(packaged_scene, "000102166_v00", n_points=500)
    assert np.allclose(sample.pcs, again.pcs)
    assert np.allclose(sample.gt_R, again.gt_R)


def test_perfect_prediction_metrics(packaged_scene):
    from hylofrac.eval.loader import SceneSample
    from hylofrac.eval.metrics import scene_metrics

    sample = SceneSample(packaged_scene, "000102166_v00", n_points=500)
    metrics = scene_metrics(sample, sample.gt_R, sample.gt_t)
    assert metrics["pa"] > 0.99
    assert metrics["rmse_r_deg"] < 0.01
    assert metrics["rmse_t"] < 1e-6
    assert metrics["sym_cd"] < 1e-6
    assert metrics["qpos"] > 0.95
    assert metrics["adj_f1"] > 0.99


def test_identity_prediction_degraded(packaged_scene):
    from hylofrac.eval.loader import SceneSample
    from hylofrac.eval.metrics import scene_metrics

    sample = SceneSample(packaged_scene, "000102166_v00", n_points=500)
    n = sample.n_parts
    identity_r = np.tile(np.eye(3), (n, 1, 1))
    zero_t = np.zeros((n, 3))
    metrics = scene_metrics(sample, identity_r, zero_t)
    # GT poses live in the anchor frame, so the identity prediction is
    # exactly right on the anchor fragment and wrong on the others: PA
    # cannot exceed the anchor's share (here 1 of 2 fragments).
    assert np.allclose(sample.gt_R[sample.anchor], np.eye(3), atol=1e-9)
    assert metrics["pa"] <= 0.5 + 1e-9
    assert metrics["rmse_r_deg"] > 1.0
    # anchor alignment credits the (largest) anchor fragment, which carries
    # at most its volume share; a wrong second fragment keeps Qpos well below 1
    assert metrics["qpos"] < 0.75


def test_gt_poses_live_in_the_anchor_frame(packaged_scene):
    """The default GT frame is the anchor's observation frame.

    The anchor's GT pose is the identity, every other fragment carries its
    pose relative to the anchor, and switching to the old absolute frame
    only changes the global frame: the assembly (the relative poses) and
    the anchor-aligned Qpos metric are identical, while PA / RMSE are only
    well posed in the anchor frame.
    """
    from hylofrac.eval.loader import SceneSample
    from hylofrac.eval.metrics import scene_metrics

    anchor = SceneSample(packaged_scene, "000102166_v00", n_points=256)
    absolute = SceneSample(packaged_scene, "000102166_v00", n_points=256,
                           gt_frame="absolute")
    n = anchor.n_parts
    a = anchor.anchor
    assert np.allclose(anchor.gt_R[a], np.eye(3), atol=1e-9)
    assert np.allclose(anchor.gt_t[a], 0.0, atol=1e-9)
    # observations are unchanged by the frame choice
    assert np.allclose(anchor.pcs, absolute.pcs)
    # relative poses (the assembly) are identical in both frames
    rot_a = absolute.gt_R[a].T          # anchor's observation operator
    for i in range(n):
        r_abs, t_abs = absolute.gt_R[i], absolute.gt_t[i]
        r_anc = anchor.gt_R[i]
        np.testing.assert_allclose(r_anc, rot_a @ r_abs, atol=1e-6)
        np.testing.assert_allclose(anchor.gt_t[i],
                                   rot_a @ (t_abs - absolute.gt_t[a]),
                                   atol=1e-5)
    # the identity prediction now matches the anchor exactly, so the
    # canonical frame is what pinned every method at chance level
    m_anchor = scene_metrics(anchor, anchor.gt_R, anchor.gt_t)
    assert m_anchor["pa"] > 0.99 and m_anchor["rmse_r_deg"] < 1e-4
    m_absolute = scene_metrics(absolute, absolute.gt_R, absolute.gt_t)
    assert m_absolute["pa"] > 0.99


def test_geodesic_rotation_metric():
    from hylofrac.eval.metrics import rot_geodesic_deg

    assert rot_geodesic_deg(np.eye(3), np.eye(3)) == 0.0
    z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert abs(rot_geodesic_deg(np.eye(3), z90) - 90.0) < 1e-6
    z180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert abs(rot_geodesic_deg(np.eye(3), z180) - 180.0) < 1e-6


def test_submission_validation(packaged_scene, tmp_path):
    from hylofrac.eval.submit import (SubmissionError, load_submission,
                                      write_submission)

    good = {"scene": "000102166_v00", "method": "test",
            "fragments": {"frag_00": {"R": np.eye(3).tolist(), "t": [0, 0, 0]}}}
    path = os.path.join(str(tmp_path), "000102166_v00.json")
    write_submission(path, "000102166_v00", "test",
                     {"frag_00": {"R": np.eye(3), "t": np.zeros(3)}})
    assert os.path.exists(path)
    scene, frags = load_submission(path)
    assert scene == "000102166_v00"
    assert "frag_00" in frags

    bad = dict(good)
    bad["fragments"] = {"frag_00": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 2]], "t": [0, 0, 0]}}
    with pytest.raises(SubmissionError):
        from hylofrac.eval.submit import load_submission as _load
        bad_path = os.path.join(str(tmp_path), "bad.json")
        with open(bad_path, "w", encoding="utf-8") as fh:
            json.dump(bad, fh)
        _load(bad_path)


def test_report_subsets(packaged_scene, tmp_path):
    from hylofrac.eval.loader import SceneSample
    from hylofrac.eval.metrics import scene_metrics
    from hylofrac.eval.report import aggregate, write_report

    sample = SceneSample(packaged_scene, "000102166_v00", n_points=500)
    metrics = scene_metrics(sample, sample.gt_R, sample.gt_t)
    metrics["part"] = "cranium"
    metrics["scene"] = "000102166_v00"
    rows = aggregate([metrics])
    subsets = {r["subset"] for r in rows}
    # overall + cranium + cranium|easy_2_5
    assert "overall" in subsets
    assert "cranium" in subsets
    assert "cranium|easy_2_5" in subsets
    out = os.path.join(str(tmp_path), "report.md")
    write_report(rows, out)
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as fh:
        text = fh.read()
    assert "| subset |" in text and "overall" in text


def test_loader_points_on_posed_surface(packaged_scene):
    """SceneSample works on the assembled layout without an assembled/
    folder: its observation points, posed back to world coordinates, lie on
    the fragment surfaces posed by gt/transforms.json."""
    import trimesh

    from hylofrac.eval.loader import SceneSample

    sample = SceneSample(packaged_scene, "000102166_v00", n_points=256,
                         gt_frame="absolute")
    assert not os.path.isdir(os.path.join(packaged_scene, "assembled"))
    with open(os.path.join(packaged_scene, "gt", "transforms.json"),
              encoding="utf-8") as fh:
        gt = json.load(fh)
    for i, name in enumerate(sample.names):
        rmat = np.asarray(gt["fragments"][name]["R"])
        trans = np.asarray(gt["fragments"][name]["t"])
        mesh = trimesh.load(os.path.join(packaged_scene, "fragments",
                                         f"{name}.obj"), process=False)
        posed = trimesh.Trimesh(vertices=mesh.vertices @ rmat.T + trans,
                                faces=mesh.faces, process=False)
        world = sample.pcs[i] @ sample.gt_R[i].T + sample.gt_t[i]
        dist = trimesh.proximity.closest_point(posed, world)[1]
        assert dist.max() < 1e-7
        assert abs(float(mesh.volume) - sample.vols[i]) < 1e-9


def test_anchor_frame_points_are_the_same_assembly(packaged_scene):
    """In the default (anchor) frame the posed points are the same assembly
    as in the absolute frame, rigidly moved by the anchor's observation
    rotation; only the frame differs, the assembly does not."""
    from hylofrac.eval.loader import SceneSample

    anchor = SceneSample(packaged_scene, "000102166_v00", n_points=256)
    absolute = SceneSample(packaged_scene, "000102166_v00", n_points=256,
                           gt_frame="absolute")
    rot_a = absolute.gt_R[anchor.anchor].T          # anchor operator
    c_a = absolute.gt_t[anchor.anchor]             # anchor centroid
    for i in range(anchor.n_parts):
        world_anchor = anchor.pcs[i] @ anchor.gt_R[i].T + anchor.gt_t[i]
        world_absolute = (absolute.pcs[i] @ absolute.gt_R[i].T
                          + absolute.gt_t[i])
        # the anchor frame is the absolute assembly rotated by the anchor's
        # observation operator and translated so the anchor sits at the
        # origin (its own centred observation cloud)
        np.testing.assert_allclose(
            world_anchor, (world_absolute - c_a) @ rot_a.T, atol=1e-5)
