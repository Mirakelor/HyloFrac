"""Self-contained metric implementations for the HyloFrac benchmark.

All metrics follow docs/benchmark.md (section 2) and are computed in the
normalized scene scale (bounding-box diagonal = 1). Point-cloud distances use
scipy ``cKDTree`` nearest-neighbor queries, so the evaluator has no GPU or
CUDA dependency and is deterministic.

Metric conventions:

- PA@0.01: for each fragment, the mean *squared* bidirectional Chamfer
  between the prediction-transformed and the GT-transformed point cloud;
  a fragment is correct below threshold 0.01. Scene accuracy is the mean
  over valid fragments (per-shape mean).
- RMSE(R): root mean square of the geodesic rotation error in degrees.
- RMSE(T): root mean square translation error in normalized units.
- Sym Chamfer: bidirectional mean distance between predicted and GT
  assemblies (all fragments concatenated).
- Qpos: the predicted assembly is rigidly aligned to the GT frame through
  the largest fragment (anchor), then each fragment's overlap ratio
  |voxels(pred) ∩ voxels(gt)| / |voxels(gt)| is averaged with
  volume weights.
- Adjacency P/R/F1: predicted adjacency (derived from the predicted
  assembly by contact) is compared with the GT adjacency graph using
  volume weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from hylofrac.eval.loader import SceneSample, derive_adjacency


def chamfer_sq(a: np.ndarray, b: np.ndarray) -> float:
    """Bidirectional mean squared nearest-neighbor distance (a, b: [N, 3])."""
    dist_ab = cKDTree(a).query(b)[0]
    dist_ba = cKDTree(b).query(a)[0]
    return float(np.mean(dist_ab ** 2) + np.mean(dist_ba ** 2))


def chamfer_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Bidirectional mean (non-squared) nearest-neighbor distance."""
    dist_ab = cKDTree(a).query(b)[0]
    dist_ba = cKDTree(b).query(a)[0]
    return float(np.mean(dist_ab) + np.mean(dist_ba))


def rot_geodesic_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    """Geodesic rotation error in degrees between two 3x3 rotations."""
    cos_ang = np.clip((np.trace(r1.T @ r2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def transform(pc: np.ndarray, rmat: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to a point cloud: pc @ R.T + t."""
    return pc @ rmat.T + trans


def part_accuracy(pcs: np.ndarray, gt_r: np.ndarray, gt_t: np.ndarray,
                  pred_r: np.ndarray, pred_t: np.ndarray,
                  threshold: float = 0.01) -> Tuple[np.ndarray, float]:
    """Per-fragment squared Chamfer errors and the scene-level accuracy."""
    n = len(pcs)
    errors = np.zeros(n)
    for i in range(n):
        gt_pts = transform(pcs[i], gt_r[i], gt_t[i])
        pred_pts = transform(pcs[i], pred_r[i], pred_t[i])
        errors[i] = chamfer_sq(gt_pts, pred_pts)
    accuracy = float((errors < threshold).mean())
    return errors, accuracy


def rmse_rot_geodesic(gt_r: np.ndarray, pred_r: np.ndarray
                      ) -> Tuple[float, np.ndarray]:
    errors = np.array([rot_geodesic_deg(gt_r[i], pred_r[i])
                       for i in range(len(gt_r))])
    return float(np.sqrt(np.mean(errors ** 2))), errors


def rmse_trans(gt_t: np.ndarray, pred_t: np.ndarray) -> Tuple[float, np.ndarray]:
    errors = np.linalg.norm(gt_t - pred_t, axis=1)
    return float(np.sqrt(np.mean(errors ** 2))), errors


def sym_chamfer_assembly(pcs: np.ndarray, gt_r: np.ndarray, gt_t: np.ndarray,
                         pred_r: np.ndarray, pred_t: np.ndarray) -> float:
    gt_all = np.concatenate([transform(pcs[i], gt_r[i], gt_t[i])
                             for i in range(len(pcs))], axis=0)
    pred_all = np.concatenate([transform(pcs[i], pred_r[i], pred_t[i])
                               for i in range(len(pcs))], axis=0)
    return chamfer_mean(gt_all, pred_all)


def anchor_alignment(gt_r: np.ndarray, gt_t: np.ndarray, pred_r: np.ndarray,
                     pred_t: np.ndarray, anchor: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid transform that maps the predicted frame onto the GT frame,
    computed on the anchor fragment: p' = R_a p + t_a with
    R_a = R_gt R_pred^T and t_a = t_gt - R_a t_pred."""
    r_a = gt_r[anchor] @ pred_r[anchor].T
    t_a = gt_t[anchor] - r_a @ pred_t[anchor]
    return r_a, t_a


def _voxel_keys(points: np.ndarray, pitch: float) -> np.ndarray:
    """Quantize points to voxel keys (round to nearest voxel center).

    Rounding (rather than flooring) keeps the quantization robust to
    sub-pitch numerical noise such as the floating-point residue of the
    anchor alignment.
    """
    return np.unique(np.floor(points / pitch + 0.5).astype(np.int64), axis=0)


def qpos(pcs: np.ndarray, vols: np.ndarray, gt_r: np.ndarray, gt_t: np.ndarray,
         pred_r: np.ndarray, pred_t: np.ndarray, pitch: float = 0.005,
         align: bool = True) -> Tuple[float, np.ndarray]:
    """Volume-weighted mean fragment overlap after anchor alignment.

    Overlap of fragment i is the fraction of its GT voxels (occupied by the
    fragment's surface points at resolution ``pitch``) that are also occupied
    by the aligned prediction.
    """
    n = len(pcs)
    if align and n > 1:
        anchor = int(np.argmax(vols))
        r_a, t_a = anchor_alignment(gt_r, gt_t, pred_r, pred_t, anchor)
    else:
        r_a, t_a = np.eye(3), np.zeros(3)

    ratios = np.zeros(n)
    for i in range(n):
        gt_pts = transform(pcs[i], gt_r[i], gt_t[i])
        pred_pts = transform(pcs[i], pred_r[i], pred_t[i])
        if align:
            pred_pts = transform(pred_pts, r_a, t_a)
        gt_keys = _voxel_keys(gt_pts, pitch)
        pred_keys = _voxel_keys(pred_pts, pitch)
        if len(gt_keys) == 0:
            continue
        pred_set = set(map(tuple, pred_keys))
        overlap = sum(1 for key in map(tuple, gt_keys) if key in pred_set)
        ratios[i] = overlap / len(gt_keys)

    total_vol = vols.sum()
    weights = vols / total_vol if total_vol > 0 else np.ones(n) / n
    return float((ratios * weights).sum()), ratios


def adjacency_prf(pred_adj: Dict[str, set], gt_adj: Dict[str, set],
                  vols: np.ndarray, names: List[str]
                  ) -> Tuple[float, float, float]:
    """Volume-weighted precision / recall / F1 of a predicted adjacency."""
    total_tp = total_fp = total_fn = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            weight = (vols[i] + vols[j]) / 2.0
            has_gt = b in gt_adj.get(a, set())
            has_pred = b in pred_adj.get(a, set())
            if has_gt and has_pred:
                total_tp += weight
            elif has_pred and not has_gt:
                total_fp += weight
            elif has_gt and not has_pred:
                total_fn += weight
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return precision, recall, f1


def scene_metrics(sample: SceneSample, pred_r: np.ndarray, pred_t: np.ndarray,
                  pa_threshold: float = 0.01, qpos_pitch: float = 0.005,
                  adj_eps: float = 0.1, adj_min_frac: float = 0.05,
                  pred_adj: Optional[Dict[str, set]] = None
                  ) -> Dict[str, float]:
    """Evaluate one scene; returns the full metric dict.

    ``pred_r`` / ``pred_t`` must be indexed like ``sample.names``. When the
    submission carries an explicit adjacency graph (T2 direct channel),
    pass it as ``pred_adj`` (name -> set of names); otherwise the graph is
    derived from the predicted poses.
    """
    gt_r, gt_t = sample.gt_R, sample.gt_t
    pa_errors, pa = part_accuracy(sample.pcs, gt_r, gt_t, pred_r, pred_t,
                                  pa_threshold)
    rmse_r, _ = rmse_rot_geodesic(gt_r, pred_r)
    rmse_t, _ = rmse_trans(gt_t, pred_t)
    cd = sym_chamfer_assembly(sample.pcs, gt_r, gt_t, pred_r, pred_t)
    qp, _ = qpos(sample.pcs, sample.vols, gt_r, gt_t, pred_r, pred_t, qpos_pitch)
    if pred_adj is None:
        pred_adj = derive_adjacency(sample.pcs, pred_r, pred_t, sample.names,
                                    adj_eps, adj_min_frac)
    precision, recall, f1 = adjacency_prf(pred_adj, sample.adjacency,
                                          sample.vols, sample.names)
    return {
        "n_parts": sample.n_parts,
        "pa": pa,
        "rmse_r_deg": rmse_r,
        "rmse_t": rmse_t,
        "sym_cd": cd,
        "qpos": qp,
        "adj_prec": precision,
        "adj_rec": recall,
        "adj_f1": f1,
    }
