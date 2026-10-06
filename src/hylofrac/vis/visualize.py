"""Visualization of assemblies: GT vs prediction point clouds per scene.

Writes two point-cloud PLY files per scene (GT assembly and predicted
assembly, colored) under an output directory, and optionally opens an
interactive Open3D window when ``--show`` is given on a display.

Usage::

    hylofrac-visualize --data dataset/hylofrac --pred preds \\
        --out vis_out [--scene 000102166_v00] [--show]
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from hylofrac.eval.loader import SceneSample
from hylofrac.eval.metrics import transform
from hylofrac.eval.submit import load_submission


def _write_pcd(path: str, points: np.ndarray, color: np.ndarray) -> None:
    """Write a colored point cloud as a PLY file."""
    n = len(points)
    header = [
        "ply", "format ascii 1.0",
        f"element vertex {n}", "property float x", "property float y",
        "property float z", "property uchar red", "property uchar green",
        "property uchar blue", "end_header",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        for p, c in zip(points, np.broadcast_to(color, (n, 3))):
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                     f"{int(c[0])} {int(c[1])} {int(c[2])}\n")


def _assembly_points(sample: SceneSample, rmat: np.ndarray,
                        trans: np.ndarray) -> np.ndarray:
    """Concatenated world points of every fragment under one pose set."""
    return np.concatenate([transform(sample.pcs[i], rmat[i], trans[i])
                           for i in range(sample.n_parts)], axis=0)


def visualize_scene(sample: SceneSample, pred_r: np.ndarray, pred_t: np.ndarray,
                    out_dir: str, scene_id: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    _write_pcd(os.path.join(out_dir, f"{scene_id}_gt.ply"),
               _assembly_points(sample, sample.gt_R, sample.gt_t),
               np.array([80, 200, 120]))
    _write_pcd(os.path.join(out_dir, f"{scene_id}_pred.ply"),
               _assembly_points(sample, pred_r, pred_t),
               np.array([220, 120, 80]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize GT vs predicted assemblies.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred", required=True, help="submission directory")
    parser.add_argument("--out", default="vis_out")
    parser.add_argument("--scene", default="", help="scene id filter (all if empty)")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    from hylofrac.eval.evaluate import find_scene

    scenes = []
    for fname in sorted(os.listdir(args.pred)):
        if not fname.endswith(".json"):
            continue
        scene_id = fname[:-5]
        if args.scene and scene_id != args.scene:
            continue
        scene_dir = find_scene(args.data, scene_id, args.splits)
        if scene_dir is None:
            print(f"[skip] {scene_id}: scene not found", flush=True)
            continue
        scenes.append((scene_id, scene_dir))

    windows = []
    for scene_id, scene_dir in scenes:
        sample = SceneSample(scene_dir, scene_id)
        _, fragments = load_submission(os.path.join(args.pred, f"{scene_id}.json"))
        pred_r = np.stack([fragments[n]["R"] for n in sample.names])
        pred_t = np.stack([fragments[n]["t"] for n in sample.names])
        visualize_scene(sample, pred_r, pred_t, args.out, scene_id)
        print(f"[ok] {scene_id} -> {args.out}", flush=True)
        if args.show:
            windows.append(_open_window(sample, pred_r, pred_t, scene_id))

    print(f"[done] scenes={len(scenes)} -> {args.out}", flush=True)
    if windows:
        import open3d as o3d
        o3d.visualization.draw_geometries(windows)


def _open_window(sample: SceneSample, pred_r: np.ndarray, pred_t: np.ndarray,
                 scene_id: str):
    import open3d as o3d

    gt_all = _assembly_points(sample, sample.gt_R, sample.gt_t)
    pred_all = _assembly_points(sample, pred_r, pred_t)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_all)
    gt_pcd.paint_uniform_color([0.3, 0.8, 0.5])
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred_all)
    pred_pcd.paint_uniform_color([0.9, 0.5, 0.3])
    pred_pcd.translate([0.0, 1.5, 0.0])
    return [gt_pcd, pred_pcd]


if __name__ == "__main__":
    main()
