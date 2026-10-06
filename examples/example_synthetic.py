"""End-to-end example on synthetic scenes.

Builds two synthetic two-fragment scenes, runs the full dataset pipeline
(export, annotate), trains the Global baseline for one epoch, predicts and
evaluates. Requires the ``ml`` extra (torch) and Open3D for the geometric
baseline only.

Run from the repository root::

    python examples/example_synthetic.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_raw_scene(root: str, scene_id: str) -> str:
    """Raw simulator-style scene: two half-boxes assembled at x = -1..1."""
    scene = os.path.join(root, scene_id)
    os.makedirs(os.path.join(scene, "input"))
    os.makedirs(os.path.join(scene, "output"))
    trimesh.creation.box(extents=[2.0, 0.5, 0.5]).export(
        os.path.join(scene, "input", "specimen.obj"))

    def make_frag(extents, trans):
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(trans)
        local = np.asarray(box.vertices) - trans
        return trimesh.Trimesh(vertices=local, faces=box.faces, process=False)

    make_frag([1.0, 0.5, 0.5], np.array([-0.5, 0.0, 0.0])).export(
        os.path.join(scene, "output", "specimen_C0.obj"))
    make_frag([1.0, 0.5, 0.5], np.array([0.5, 0.0, 0.0])).export(
        os.path.join(scene, "output", "specimen_C1.obj"))
    frames = {"frames": [{"objects": [
        {"name": "specimen_C0", "position": [-0.5, 0.0, 0.0],
         "rotation_quat": [0.0, 0.0, 0.0, 1.0]},
        {"name": "specimen_C1", "position": [0.5, 0.0, 0.0],
         "rotation_quat": [0.0, 0.0, 0.0, 1.0]}]}]}
    with open(os.path.join(scene, "output", "transforms.json"), "w") as fh:
        json.dump(frames, fh)
    with open(os.path.join(scene, "result.json"), "w") as fh:
        json.dump({"id": scene_id, "fragments": 2, "done": True}, fh)
    return scene


def main() -> None:
    work = tempfile.mkdtemp(prefix="hylofrac_example_")
    print(f"work dir: {work}")

    from hylofrac.baselines.global_net import GlobalPoseNet
    from hylofrac.baselines.common import walk_scenes
    from hylofrac.baselines.predict import load_checkpoint
    from hylofrac.baselines.train import save_checkpoint, train
    from hylofrac.dataset.annotate import annotate_scene
    from hylofrac.dataset.export import export_scene
    from hylofrac.dataset.objio import write_obj
    from hylofrac.dataset.split import make_splits
    from hylofrac.eval.evaluate import evaluate_submission
    from hylofrac.eval.loader import SceneSample
    from hylofrac.eval.submit import write_submission

    import torch

    # raw scenes: specimen A (two variants, train) and specimen B (val)
    raw_root = os.path.join(work, "raw")
    os.makedirs(raw_root)
    scene_ids = ["000102166_v00", "000102166_v01", "000102167_v00"]
    for sid in scene_ids:
        _make_raw_scene(raw_root, sid)

    # splits by specimen (one specimen -> all variants in the same split)
    split_map = make_splits(["000102166", "000102167"], 0.55, 0.4, seed=0)
    print("split:", split_map)

    # export and annotate
    data_root = os.path.join(work, "dataset")
    for sid in scene_ids:
        raw_scene = os.path.join(raw_root, sid)
        specimen, variant = sid.split("_v", 1)
        export_scene(raw_scene, data_root, split=split_map[specimen],
                     part_of={"000102166": "cranium", "000102167": "cranium"})
        dest = os.path.join(data_root, split_map["000102166"], specimen, f"v{variant}")
        # crack facet on the contact plane x = 0
        verts = np.array([[0.0, -0.25, -0.25], [0.0, 0.25, -0.25],
                          [0.0, 0.25, 0.25], [0.0, -0.25, 0.25]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        write_obj(os.path.join(raw_scene, "output", "crackSurface_full_999.obj"),
                  verts, faces)
        adjacency = annotate_scene(raw_scene, dest, eps=0.01)
        print(f"{sid}: {len(adjacency['edges'])} adjacency edge(s)")

    # train one epoch of the Global baseline (tiny, CPU)
    runs = os.path.join(work, "runs")
    model = GlobalPoseNet()
    train(model, "global", data_root, runs, epochs=1, lr=1e-3,
          physical_batch=1, accum=32, n_points=128, seed=0,
          device=torch.device("cpu"))
    save_checkpoint(model, "global", 0, runs, name="model_final")

    # predict with the perfect loader poses (illustrates the predict path)
    model, meta = load_checkpoint(os.path.join(runs, "model_final.pt"),
                                  torch.device("cpu"))
    preds = os.path.join(work, "preds")
    os.makedirs(preds, exist_ok=True)
    for scene_id, scene_dir in walk_scenes(data_root, ["train"]):
        sample = SceneSample(scene_dir, scene_id, n_points=128)
        # note: submissions normally come from a trained model; here we
        # store the loader poses to make the example deterministic
        fragments = {name: {"R": sample.gt_R[i], "t": sample.gt_t[i]}
                     for i, name in enumerate(sample.names)}
        write_submission(os.path.join(preds, f"{scene_id}.json"),
                         scene_id, "perfect-example", fragments)

    metrics = evaluate_submission(os.path.join(preds, f"{scene_ids[0]}.json"),
                                  data_root, ["train"], 128, 0)
    print("evaluation (perfect poses):", {k: round(float(v), 3)
                                          for k, v in metrics.items()
                                          if k not in ("n_parts", "scene", "part")})
    print(f"example finished; outputs under {work}")


if __name__ == "__main__":
    main()
