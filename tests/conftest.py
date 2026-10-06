"""Shared pytest fixtures."""

import json
import os

import numpy as np
import pytest
import trimesh


def _quat_from_matrix(rmat):
    trace = np.trace(rmat)
    s = np.sqrt(trace + 1.0) * 2
    w = 0.25 * s
    x = (rmat[2, 1] - rmat[1, 2]) / s
    y = (rmat[0, 2] - rmat[2, 0]) / s
    z = (rmat[1, 0] - rmat[0, 1]) / s
    return [x, y, z, w]


@pytest.fixture()
def fake_scene(tmp_path):
    """A fake raw simulator scene: two half-boxes assembled at x = -1..1.

    Returns (scene_dir, tmp_root). The scene contains input/specimen.obj,
    output/specimen_C0.obj + specimen_C1.obj (local coordinates),
    output/transforms.json (last frame poses) and result.json.
    """
    scene = tmp_path / "batch0" / "000102166_v00"
    (scene / "input").mkdir(parents=True)
    (scene / "output").mkdir(parents=True)

    intact = trimesh.creation.box(extents=[2.0, 0.5, 0.5])
    intact.export(str(scene / "input" / "specimen.obj"))

    def make_frag(extents, rmat, trans):
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(trans)
        local = (np.asarray(box.vertices) - trans) @ rmat
        return trimesh.Trimesh(vertices=local, faces=box.faces, process=False)

    r0, t0 = np.eye(3), np.array([-0.5, 0.0, 0.0])
    r1, t1 = np.eye(3), np.array([0.5, 0.0, 0.0])
    make_frag([1.0, 0.5, 0.5], r0, t0).export(str(scene / "output" / "specimen_C0.obj"))
    make_frag([1.0, 0.5, 0.5], r1, t1).export(str(scene / "output" / "specimen_C1.obj"))

    frames = {"frames": [
        {"objects": [{"name": "specimen", "position": [0, 0, 5],
                      "rotation_quat": [0, 0, 0, 1]}]},
        {"objects": [
            {"name": "specimen_C0", "position": list(t0),
             "rotation_quat": _quat_from_matrix(r0)},
            {"name": "specimen_C1", "position": list(t1),
             "rotation_quat": _quat_from_matrix(r1)},
        ]},
    ]}
    (scene / "output" / "transforms.json").write_text(json.dumps(frames))
    (scene / "result.json").write_text(
        json.dumps({"id": "000102166_v00", "fragments": 2, "done": True}))
    return scene, tmp_path


@pytest.fixture()
def packaged_scene(fake_scene):
    """A packaged scene (assembled pose + GT files) ready for evaluation."""
    from hylofrac.dataset.annotate import annotate_scene
    from hylofrac.dataset.export import export_scene
    from hylofrac.dataset.objio import write_obj

    scene, tmp = fake_scene
    out_root = str(tmp / "dataset")
    export_scene(str(scene), out_root, split="val")
    dest = os.path.join(out_root, "val", "000102166", "v00")

    # crack facet on the contact plane x = 0
    verts = np.array([[0.0, -0.25, -0.25], [0.0, 0.25, -0.25],
                      [0.0, 0.25, 0.25], [0.0, -0.25, 0.25]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    write_obj(os.path.join(str(scene), "output", "crackSurface_full_999.obj"),
              verts, faces)
    annotate_scene(str(scene), dest, eps=0.01)

    # metadata with morphology
    import json as _json
    with open(os.path.join(dest, "metadata.json"), "r", encoding="utf-8") as fh:
        meta = _json.load(fh)
    meta["part"] = "cranium"
    with open(os.path.join(dest, "metadata.json"), "w", encoding="utf-8") as fh:
        _json.dump(meta, fh, indent=2)
    return dest
