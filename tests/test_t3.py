"""Tests for the T3 robustness degradations in the loader and evaluator."""

import json
import os

import numpy as np
import pytest
import trimesh

from hylofrac.dataset.export import export_scene
from hylofrac.eval.loader import SceneSample, select_foreign_fragments


def _quat_from_matrix(rmat):
    trace = np.trace(rmat)
    s = np.sqrt(trace + 1.0) * 2
    w = 0.25 * s
    x = (rmat[2, 1] - rmat[1, 2]) / s
    y = (rmat[0, 2] - rmat[2, 0]) / s
    z = (rmat[1, 0] - rmat[0, 1]) / s
    return [x, y, z, w]


@pytest.fixture()
def packaged_three(tmp_path):
    """A fake packaged scene with three fragments of distinct sizes."""
    scene = tmp_path / "batchX" / "000103999_v00"
    (scene / "input").mkdir(parents=True)
    (scene / "output").mkdir(parents=True)

    intact = trimesh.creation.box(extents=[1.6, 0.6, 0.6])
    intact.export(str(scene / "input" / "specimen.obj"))

    extents = [[1.0, 0.6, 0.6], [0.4, 0.6, 0.6], [0.2, 0.6, 0.6]]
    xs = [-0.3, 0.3, 0.7]
    for i, (ext, x) in enumerate(zip(extents, xs)):
        box = trimesh.creation.box(extents=ext)
        box.apply_translation([x, 0, 0])
        box.export(str(scene / "output" / f"specimen_C{i}.obj"))

    frames = {"frames": [
        {"objects": [{"name": "specimen", "position": [0, 0, 0],
                      "rotation_quat": [0, 0, 0, 1]}]},
        {"objects": [
            {"name": f"specimen_C{i}",
             "position": [x, 0, 0],
             "rotation_quat": [0, 0, 0, 1]}
            for i, x in enumerate(xs)]},
    ]}
    (scene / "output" / "transforms.json").write_text(json.dumps(frames))
    (scene / "result.json").write_text(json.dumps(
        {"id": "000103999_v00", "fragments": 3, "done": True}))

    out_root = str(tmp_path / "dataset")
    export_scene(str(scene), out_root, split="val")
    return os.path.join(out_root, "val", "000103999", "v00")


def test_missing_removes_smallest_first(packaged_three):
    s = SceneSample(packaged_three, "000103999_v00", n_points=200,
                    missing_frac=0.5)
    assert s.n_parts == 2
    assert "frag_02" not in s.names          # smallest (0.2-wide box) removed
    assert s.pcs.shape == (2, 200, 3)
    assert s.gt_R.shape == (2, 3, 3)
    assert s.gt_t.shape == (2, 3)
    assert s.adjacency.keys() == set(s.names)


def test_missing_keeps_at_least_two(packaged_three):
    s = SceneSample(packaged_three, "000103999_v00", n_points=100,
                    missing_frac=0.9)
    assert s.n_parts == 2                      # clamped to n - 2
    assert len(s.names) == 2


def test_erosion_keeps_gt_changes_points(packaged_three):
    clean = SceneSample(packaged_three, "000103999_v00", n_points=300)
    ero = SceneSample(packaged_three, "000103999_v00", n_points=300,
                      erosion_depth=0.05)
    assert ero.n_parts == clean.n_parts
    assert ero.pcs.shape == clean.pcs.shape == (3, 300, 3)
    np.testing.assert_allclose(ero.gt_R, clean.gt_R)   # observation rotation
    # unchanged: erosion draws its own noise stream
    assert not np.allclose(ero.pcs, clean.pcs)   # surface actually moved
    # intact per-fragment clouds of the remaining fragments are unchanged
    # when only the other degradation is applied
    miss = SceneSample(packaged_three, "000103999_v00", n_points=300,
                       missing_frac=0.4)
    assert miss.names == ["frag_00", "frag_01"]
    idx = [clean.names.index(n) for n in miss.names]
    np.testing.assert_allclose(miss.pcs, clean.pcs[idx])
    np.testing.assert_allclose(miss.gt_R, clean.gt_R[idx])
    np.testing.assert_allclose(miss.gt_t, clean.gt_t[idx])


def test_foreign_injection(packaged_three):
    other = os.path.join(packaged_three, "fragments", "frag_01.obj")
    s = SceneSample(packaged_three, "000103999_v00", n_points=150,
                    foreign_fragments=[(other, "foreign_00")])
    assert s.n_foreign == 1
    assert s.n_parts == 3
    assert s.all_pcs.shape == (4, 150, 3)
    assert s.pcs.shape == (3, 150, 3)
    assert s.foreign_pcs.shape == (1, 150, 3)
    assert set(s.names) == {"frag_00", "frag_01", "frag_02"}
    assert s.gt_R.shape == (3, 3, 3)


def test_degradation_reproducible(packaged_three):
    a = SceneSample(packaged_three, "000103999_v00", n_points=200,
                    erosion_depth=0.05, missing_frac=0.4)
    b = SceneSample(packaged_three, "000103999_v00", n_points=200,
                    erosion_depth=0.05, missing_frac=0.4)
    np.testing.assert_allclose(a.pcs, b.pcs)
    np.testing.assert_allclose(a.gt_R, b.gt_R)


def test_select_foreign_excludes_own_specimen(packaged_three, tmp_path):
    # only one mandible scene exists: the query itself, which is excluded
    meta_path = os.path.join(packaged_three, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["part"] = "mandible"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    data_root = os.path.abspath(os.path.join(packaged_three, "..", "..", ".."))
    rng = np.random.default_rng(0)
    picked = select_foreign_fragments(data_root, "val", "mandible",
                                      "000103999_v00", 2, rng)
    assert picked == []                        # no other mandible specimen
