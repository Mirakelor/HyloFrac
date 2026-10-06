"""End-to-end tests for the dataset construction pipeline."""

import json
import os

import numpy as np


def test_export_layout_and_pose(fake_scene):
    """Fragments are stored once in local coordinates (no assembled/
    directory); the GT SE(3) reproduces the assembled vertex positions."""
    from hylofrac.dataset.export import export_scene
    from hylofrac.dataset.objio import read_obj

    scene, tmp = fake_scene
    out_root = str(tmp / "out")
    summary = export_scene(str(scene), out_root, split="val")
    assert summary["n_fragments"] == 2

    dest = summary["dir"]
    assert not os.path.isdir(os.path.join(dest, "assembled"))
    with open(os.path.join(dest, "gt", "transforms.json"), encoding="utf-8") as fh:
        gt = json.load(fh)

    for fname, xmin, xmax in (("frag_00", -1.0, 0.0), ("frag_01", 0.0, 1.0)):
        verts, faces = read_obj(os.path.join(dest, "fragments", f"{fname}.obj"))
        assert len(faces) > 0
        assert np.allclose(verts.mean(axis=0), 0.0, atol=1e-6)
        entry = gt["fragments"][fname]
        world = verts @ np.asarray(entry["R"]).T + np.asarray(entry["t"])
        assert np.allclose(world.min(axis=0), [xmin, -0.25, -0.25], atol=1e-6)
        assert np.allclose(world.max(axis=0), [xmax, 0.25, 0.25], atol=1e-6)


def test_export_gt_and_metadata(fake_scene):
    from hylofrac.dataset.export import export_scene

    scene, tmp = fake_scene
    summary = export_scene(str(scene), str(tmp / "out"), split="val")
    gt_path = os.path.join(summary["dir"], "gt", "transforms.json")
    meta_path = os.path.join(summary["dir"], "metadata.json")
    with open(gt_path, encoding="utf-8") as fh:
        gt = json.load(fh)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert gt["n_fragments"] == 2
    assert set(gt["fragments"]) == {"frag_00", "frag_01"}
    assert meta["scene"] == "000102166_v00"
    assert meta["fragments"] == 2
    intact_path = os.path.join(summary["dir"], "intact.obj")
    assert os.path.exists(intact_path)


def test_export_metadata_provenance(fake_scene):
    """The per-object provenance ends up in metadata.json, and only when a
    provenance map is supplied (the field is optional)."""
    from hylofrac.dataset.export import export_scene

    scene, tmp = fake_scene
    record = {
        "source": "MorphoSource",
        "media_id": "000102166",
        "physical_object_title": "USNM:MAMM:USNM 123456",
        "taxonomy_name": "Hylobates lar",
        "rights_holder": "Smithsonian Institution National Museum of "
                         "Natural History",
        "copyright_statement": "http://rightsstatements.org/vocab/NKC/1.0/",
        "permits_commercial_use": "CommercialUsePermitted",
        "permits_3d_use": "3DPrintingLimited",
    }
    summary = export_scene(str(scene), str(tmp / "out"), split="val",
                           provenance_of={"000102166": record})
    with open(os.path.join(summary["dir"], "metadata.json"),
              encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["provenance"] == record

    plain = export_scene(str(scene), str(tmp / "out_plain"), split="val")
    with open(os.path.join(plain["dir"], "metadata.json"),
              encoding="utf-8") as fh:
        assert "provenance" not in json.load(fh)


def test_split_no_leakage():
    from hylofrac.dataset.split import make_splits

    specimens = [f"s{i:03d}" for i in range(100)]
    mapping = make_splits(specimens, 0.7, 0.15, seed=42)
    counts = {}
    for split in mapping.values():
        counts[split] = counts.get(split, 0) + 1
    assert counts["train"] == 70
    assert counts["val"] == 15
    assert counts["test"] == 15
    # deterministic under the same seed
    assert make_splits(specimens, 0.7, 0.15, seed=42) == mapping


def test_annotate_adjacency(fake_scene):
    from hylofrac.dataset.annotate import annotate_scene
    from hylofrac.dataset.export import export_scene
    from hylofrac.dataset.objio import write_obj

    scene, tmp = fake_scene
    out_root = str(tmp / "out")
    export_scene(str(scene), out_root, split="val")
    dest = os.path.join(out_root, "val", "000102166", "v00")

    verts = np.array([[0.0, -0.25, -0.25], [0.0, 0.25, -0.25],
                      [0.0, 0.25, 0.25], [0.0, -0.25, 0.25]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    write_obj(os.path.join(str(scene), "output", "crackSurface_full_999.obj"),
              verts, faces)

    adjacency = annotate_scene(str(scene), dest, eps=0.01)
    assert adjacency["n_fragments"] == 2
    edges = adjacency["edges"]
    assert len(edges) == 1
    assert {edges[0]["frag_i"], edges[0]["frag_j"]} == {0, 1}
    assert 0.2 < edges[0]["area"] < 0.3  # crack plane is 0.5 x 0.5 = 0.25
    assert os.path.exists(os.path.join(dest, "gt", "contact_surfaces",
                                      edges[0]["contact_mesh"]))
    with open(os.path.join(dest, "gt", "adjacency.json"), encoding="utf-8") as fh:
        adj_file = json.load(fh)
    assert len(adj_file["edges"]) == 1
