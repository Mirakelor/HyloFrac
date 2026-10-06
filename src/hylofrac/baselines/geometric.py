"""Geometric baseline: FPFH + RANSAC global registration + ICP refinement.

The baseline works on the same observation point clouds as the learned
methods (docs/benchmark.md, section 3). Fragment 0 is the anchor; every
other fragment is registered against the growing assembly with Open3D
feature matching (FPFH), RANSAC global registration and point-to-plane ICP
refinement, adding the best-scoring pair each iteration. The anchor is
submitted with the identity pose; metrics that align on the anchor (Qpos)
are therefore meaningful, while absolute per-fragment errors (PA, RMSE) are
reported as-is.

Usage::

    hylofrac-predict --method geometric --data dataset/hylofrac --out preds
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np

from hylofrac.eval.submit import write_submission


def _fpfh(pc: np.ndarray, voxel: float = 0.01) -> Tuple[object, object]:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5,
                                                              max_nn=30))
    try:
        pcd.orient_normals_consistent_tangent_plane(20)
    except RuntimeError:
        # Open3D orients normals through a 4-D Delaunay triangulation, and
        # Qhull raises a precision error on near-degenerate clouds (e.g.
        # val/000103527/v01 fragment 36: QH7088/QH6297 "wide merge error").
        # The exception is per-cloud, so keep the normals from
        # estimate_normals and orient them towards the cloud centroid; the
        # feature stays usable and the rest of the scene is unaffected.
        pcd.orient_normals_towards_camera_location(pcd.get_center())
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return pcd, fpfh


def _register_with_feats(src_pcd: object, src_f: object, tgt_pcd: object,
                         tgt_f: object, voxel: float,
                         max_iteration: int
                         ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                                    float]:
    """RANSAC global registration + ICP on precomputed clouds and FPFH
    features; returns (R, t, fitness) with (None, None, fitness) when the
    RANSAC fitness is below the acceptance threshold."""
    import open3d as o3d

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd, tgt_pcd, src_f, tgt_f, mutual_filter=True,
        max_correspondence_distance=voxel * 5,
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.
                  CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.
                  CorrespondenceCheckerBasedOnDistance(voxel * 5)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    if result.fitness < 0.3:
        return None, None, float(result.fitness)
    icp = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, voxel * 2, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    T = icp.transformation
    return T[:3, :3], T[:3, 3], float(icp.fitness)


def register_pair(source: np.ndarray, target: np.ndarray, voxel: float = 0.01,
                  max_iteration: int = 100
                  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Register source onto target; returns (R, t, fitness)."""
    src, src_f = _fpfh(source, voxel)
    tgt, tgt_f = _fpfh(target, voxel)
    return _register_with_feats(src, src_f, tgt, tgt_f, voxel, max_iteration)


def assemble(pcs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy assembly from fragment 0; returns (R, t) per fragment in the
    anchor (observation) frame.

    FPFH features and per-pair registration results are cached so every
    pair is registered exactly once (the pairwise outcome is independent
    of the assembly order), instead of re-running RANSAC for the same
    pairs every round.
    """
    n = len(pcs)
    voxel, max_iteration = 0.01, 100
    pcds: List[Optional[object]] = [None] * n
    feats: List[Optional[object]] = [None] * n

    def ensure(k: int) -> None:
        if pcds[k] is None:
            pcds[k], feats[k] = _fpfh(pcs[k], voxel)

    r = np.tile(np.eye(3), (n, 1, 1))
    t = np.zeros((n, 3))
    placed = {0}
    pair_cache: dict = {}
    while len(placed) < n:
        best = None
        for i in placed:
            for j in range(n):
                if j in placed:
                    continue
                key = (i, j)
                if key not in pair_cache:
                    ensure(i)
                    ensure(j)
                    pair_cache[key] = _register_with_feats(
                        pcds[j], feats[j], pcds[i], feats[i], voxel,
                        max_iteration)
                r_ij, t_ij, fitness = pair_cache[key]
                if r_ij is None:
                    continue
                if best is None or fitness > best[0]:
                    best = (fitness, i, j, r_ij, t_ij)
        if best is None:
            break
        _, i, j, r_ij, t_ij = best
        # p_j -> p_i -> anchor frame
        r[j] = r[i] @ r_ij
        t[j] = r[i] @ t_ij + t[i]
        placed.add(j)
    # any fragment that failed to register keeps the identity pose
    return r, t


def predict_scene(pcs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """pcs [P, N, 3] observation clouds -> (R [P,3,3], t [P,3])."""
    return assemble([pcs[i] for i in range(len(pcs))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the geometric baseline.")
    parser.add_argument("--data", required=True, help="packaged dataset root")
    parser.add_argument("--out", required=True, help="submission directory")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()

    from hylofrac.baselines.common import walk_scenes
    from hylofrac.eval.loader import SceneSample

    n_scenes = 0
    for split in args.splits:
        for scene_id, scene_dir in walk_scenes(args.data, [split]):
            sample = SceneSample(scene_dir, scene_id)
            r, t = predict_scene(sample.pcs)
            fragments = {name: {"R": r[i], "t": t[i]}
                         for i, name in enumerate(sample.names)}
            write_submission(os.path.join(args.out, f"{scene_id}.json"),
                             scene_id, "geometric", fragments)
            n_scenes += 1
            print(f"[ok] {scene_id}: n={sample.n_parts}", flush=True)
    print(f"[done] scenes={n_scenes} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
