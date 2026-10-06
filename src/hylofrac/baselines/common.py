"""Shared components for the learned baselines.

Data protocol (docs/benchmark.md, section 3): for every fragment of a scene
we sample 1,000 surface points from the assembled mesh, subtract the
centroid (the GT translation), and rotate by a uniform random SO(3) to form
the observation point cloud; the GT rotation is the inverse of that random
rotation. During training both the sampling and the rotation are randomized
every epoch (data augmentation); during evaluation they are seeded per scene
(see hylofrac.eval.loader).

Losses follow docs/baselines.md: translation L2, rotation cosine on
scalar-first quaternions, per-point L2 of rotated part clouds, per-part
Chamfer and whole-shape Chamfer (the latter evaluated on a subsample of
points to keep memory bounded).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import trimesh
from torch.utils.data import Dataset

from hylofrac.eval.loader import (_scene_seed, anchor_index, gt_poses,
                                  uniform_rotation, world_points)

MAX_PARTS = 100


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Scalar-first quaternion [w, x, y, z] to rotation matrix (batch)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rotmat_to_quat_scalar_first(r: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) to scalar-first quaternion [w, x, y, z].

    The branch is chosen by the largest diagonal entry (the standard
    Shepperd selector). That choice is what keeps the result exact when the
    scalar part vanishes: the sqrt argument of branch ``k`` is
    ``2 (1 - cos) * a_k**2`` for rotation axis ``a``, so taking the largest
    diagonal - hence the largest ``a_k**2`` - guarantees a non-zero divisor.
    Selecting by a cyclic pairwise comparison instead
    (``r[ax2, ax2] > r[ax1, ax1]``, as this function did before) picks the
    wrong branch whenever the trace branch does not apply: a rotation about a
    coordinate axis by more than 120 degrees came back as a completely
    different rotation (max error 2.0), a half turn about a coordinate axis
    likewise, and a half turn about (1, 1, 1) produced the zero quaternion
    (nan after normalisation).
    """
    trace = np.trace(r, axis1=-2, axis2=-1)
    q = np.zeros(r.shape[:-2] + (4,))
    idx = trace > 0
    if idx.any():
        s = np.sqrt(np.maximum(trace[idx] + 1.0, 0.0)) * 2
        q[idx, 0] = 0.25 * s
        q[idx, 1] = (r[idx, 2, 1] - r[idx, 1, 2]) / s
        q[idx, 2] = (r[idx, 0, 2] - r[idx, 2, 0]) / s
        q[idx, 3] = (r[idx, 1, 0] - r[idx, 0, 1]) / s
    diag = np.stack([r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]], axis=-1)
    big = np.argmax(diag, axis=-1)   # ties go to the lowest component index
    for axis in range(3):
        ax = np.roll(np.arange(3), 1 - axis)
        cond = ~idx & (big == ax[0])   # ax[0] is the component solved here
        if cond.any():
            s = np.sqrt(np.maximum(r[cond, ax[0], ax[0]] - r[cond, ax[1], ax[1]]
                                   - r[cond, ax[2], ax[2]] + 1.0, 0.0)) * 2
            s = np.maximum(s, 1e-8)
            q[cond, ax[0] + 1] = 0.25 * s
            q[cond, 0] = (r[cond, ax[2], ax[1]] - r[cond, ax[1], ax[2]]) / s
            q[cond, ax[1] + 1] = (r[cond, ax[1], ax[0]] + r[cond, ax[0], ax[1]]) / s
            q[cond, ax[2] + 1] = (r[cond, ax[2], ax[0]] + r[cond, ax[0], ax[2]]) / s
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def matrix_to_quat(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices [..., 3, 3] to scalar-first quaternions [..., 4].

    Inverse of :func:`quat_to_rotmat`: the branch axis order mirrors the numpy
    twin :func:`rotmat_to_quat_scalar_first` (``roll(arange(3), 1 - axis)``).
    The branches use single boolean masks (``r[cond]`` / ``q[cond] = block``)
    instead of the mixed ``q[cond, 0]`` form, which torch 2.4/2.5 rejects
    ("the shape of the mask ... does not match the shape of the indexed
    tensor").

    The branch is chosen by the largest diagonal entry, like the numpy twin;
    see that docstring for why a cyclic pairwise comparison fails at a half
    turn and above 120 degrees about a coordinate axis.
    """
    trace = r.diagonal(dim1=-2, dim2=-1).sum(-1)
    q = torch.zeros(r.shape[:-2] + (4,), device=r.device, dtype=r.dtype)
    idx = trace > 0
    if idx.any():
        sel = r[idx]
        s = (trace[idx] + 1.0).sqrt() * 2
        q[idx] = torch.stack([
            0.25 * s,
            (sel[..., 2, 1] - sel[..., 1, 2]) / s,
            (sel[..., 0, 2] - sel[..., 2, 0]) / s,
            (sel[..., 1, 0] - sel[..., 0, 1]) / s,
        ], dim=-1)
    diag = torch.stack([r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]], dim=-1)
    big = diag.argmax(dim=-1)   # ties go to the lowest component index
    for axis in range(3):
        ax = [(axis + 2) % 3, axis, (axis + 1) % 3]
        cond = ~idx & (big == ax[0])   # ax[0] is the component solved here
        if cond.any():
            sel = r[cond]
            s = (sel[..., ax[0], ax[0]] - sel[..., ax[1], ax[1]]
                 - sel[..., ax[2], ax[2]] + 1.0).sqrt() * 2
            s = s.clamp(min=1e-8)
            block = torch.zeros(sel.shape[0], 4, device=r.device,
                                dtype=r.dtype)
            block[:, ax[0] + 1] = 0.25 * s
            block[:, 0] = (sel[..., ax[2], ax[1]] - sel[..., ax[1], ax[2]]) / s
            block[:, ax[1] + 1] = (sel[..., ax[1], ax[0]]
                                   + sel[..., ax[0], ax[1]]) / s
            block[:, ax[2] + 1] = (sel[..., ax[2], ax[0]]
                                   + sel[..., ax[0], ax[2]]) / s
            q[cond] = block
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def exp_map(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map of rotation vectors [..., 3]."""
    angle = v.norm(dim=-1)
    axis = v / angle.unsqueeze(-1).clamp(min=1e-8)
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle), torch.sin(angle)
    k = torch.stack([
        torch.stack([torch.zeros_like(c), -z, y], dim=-1),
        torch.stack([z, torch.zeros_like(c), -x], dim=-1),
        torch.stack([-y, x, torch.zeros_like(c)], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, device=v.device).expand_as(k)
    return eye + s.unsqueeze(-1).unsqueeze(-1) * k \
        + (1 - c).unsqueeze(-1).unsqueeze(-1) * (k @ k)


def log_rmat(r: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix logarithm of rotations [..., 3, 3].

    The angle is recovered with atan2 (stable near the identity); exact
    pi rotations fall back to the eigenvector axis. Follows the numerical
    recipe of the reference DiffAssemble implementation.
    """
    skew = r - r.transpose(-1, -2)
    v = torch.stack([skew[..., 2, 1], -skew[..., 2, 0], skew[..., 1, 0]],
                    dim=-1)
    s_angle = v.norm(dim=-1) / 2
    c_angle = (r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    angle = torch.atan2(s_angle, c_angle)
    scale = torch.where(angle == 0, torch.zeros_like(angle),
                        angle / (2 * s_angle))
    out = scale[..., None, None] * skew
    bad = torch.isnan(out[..., 0, 0])
    if bad.any():
        _, eigvec = torch.linalg.eigh(r[bad])
        # the last column carries the +1 eigenvalue axis of a pi rotation
        v = eigvec[..., :, -1] * angle[bad].unsqueeze(-1)
        x, y, z = v.unbind(-1)
        out[bad] = torch.stack([
            torch.stack([torch.zeros_like(x), -z, y], dim=-1),
            torch.stack([z, torch.zeros_like(x), -x], dim=-1),
            torch.stack([-y, x, torch.zeros_like(x)], dim=-1),
        ], dim=-2)
    return out


def so3_scale(rmat: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
    """Scale the magnitude of rotations: exp(s * log R) per element."""
    logs = log_rmat(rmat)
    return torch.matrix_exp(logs * scalars[..., None, None])


def walk_scenes(data_root: str, splits) -> List[Tuple[str, str]]:
    """(scene_id, scene_dir) of the packaged scenes (with a fragments/
    folder) under <root>/<split>/<specimen>/<variant>, sorted per split."""
    out: List[Tuple[str, str]] = []
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for specimen in sorted(os.listdir(split_dir)):
            spec_dir = os.path.join(split_dir, specimen)
            if not os.path.isdir(spec_dir):
                continue
            for variant in sorted(os.listdir(spec_dir)):
                scene_dir = os.path.join(spec_dir, variant)
                if os.path.isdir(os.path.join(scene_dir, "fragments")):
                    out.append((f"{specimen}_{variant}", scene_dir))
    return out


class AssemblyDataset(Dataset):
    """One scene per sample; fragments padded to MAX_PARTS.

    Each item provides:
        pcs:    [MAX_PARTS, N, 3] observation point clouds
        quat:   [MAX_PARTS, 4] scalar-first GT rotations
        trans:  [MAX_PARTS, 3] GT translations (centroids)
        valids: [MAX_PARTS] 1 for real fragments, 0 for padding
    """

    def __init__(self, data_root: str, split: str, n_points: int = 1000,
                 seed: Optional[int] = None, with_adjacency: bool = False
                 ) -> None:
        """``with_adjacency`` also provides GT adjacency and relative poses
        per ordered pair (only PHFormer supervises them)."""
        self.n_points = n_points
        self.seed = seed
        self.with_adjacency = with_adjacency
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"{split_dir} not found")
        self.scene_dirs = [d for _, d in walk_scenes(data_root, [split])]
        if not self.scene_dirs:
            raise ValueError(f"no scenes under {split_dir}")

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        scene_dir = self.scene_dirs[index]
        # Fresh randomness on every access: both the surface sampling and
        # the observation rotation are re-drawn each epoch (training data
        # augmentation, see the module docstring).  Only the evaluation
        # path is seeded per scene (hylofrac.eval.loader.SceneSample).
        rng = np.random.default_rng(self.seed) if self.seed is not None \
            else np.random.default_rng()
        frag_dir = os.path.join(scene_dir, "fragments")
        files = sorted(f for f in os.listdir(frag_dir)
                       if f.startswith("frag_") and f.endswith(".obj"))
        n_parts = len(files)
        poses = gt_poses(scene_dir)
        pcs = np.zeros((MAX_PARTS, self.n_points, 3), dtype=np.float32)
        quats = np.zeros((MAX_PARTS, 4), dtype=np.float32)
        trans = np.zeros((MAX_PARTS, 3), dtype=np.float32)
        gt_r = np.zeros((MAX_PARTS, 3, 3), dtype=np.float32)
        areas = np.zeros(n_parts, dtype=np.float32)
        rot_obs_all = []
        centroids = np.zeros((n_parts, 3), dtype=np.float64)
        for i, fname in enumerate(files):
            name = fname[:-4]
            mesh = trimesh.load(os.path.join(frag_dir, fname), process=False)
            rmat, trans_gt = poses[name]
            world = world_points(mesh, rmat, trans_gt, self.n_points, rng)
            centroid = world.mean(axis=0)
            rot_obs = uniform_rotation(rng)
            pcs[i] = (world - centroid) @ rot_obs.T
            rot_obs_all.append(rot_obs)
            centroids[i] = centroid
            areas[i] = float(mesh.area)

        # GT poses live in the anchor fragment's observation frame (see
        # hylofrac.eval.loader.SceneSample): the anchor's pose is the
        # identity and every other fragment carries its pose relative to
        # the anchor, which is the frame the reference geometric baseline
        # submits in and the frame every metric of docs/benchmark.md is
        # well posed in.  The anchor rule must match the evaluator's.
        # the anchor rule is fragment 0, so no per-fragment volume/area
        # statistics are needed here (mesh.is_watertight()/mesh.volume are
        # expensive and this runs once per fragment per epoch)
        anchor = anchor_index()
        rot_obs_stack = np.stack(rot_obs_all)
        r_a = rot_obs_stack[anchor]
        # HF_GT_FRAME=canonical switches the training target to the reference
        # implementations framing (per-fragment canonical rotation, anchor
        # relative translation); default keeps the HyloFrac anchor frame.
        if os.environ.get("HF_GT_FRAME", "anchor") == "canonical":
            gt_r[:n_parts] = np.transpose(rot_obs_stack, (0, 2, 1))
            trans[:n_parts] = centroids - centroids[anchor]
        else:
            gt_r[:n_parts] = np.einsum("ij,pkj->pik", r_a, rot_obs_stack)
            trans[:n_parts] = (centroids - centroids[anchor]) @ r_a.T
        quats[:n_parts] = rotmat_to_quat_scalar_first(gt_r[:n_parts])
        valids = np.zeros(MAX_PARTS, dtype=np.float32)
        valids[:n_parts] = 1.0

        item: Dict[str, torch.Tensor] = {
            "pcs": torch.from_numpy(pcs), "quat": torch.from_numpy(quats),
            "trans": torch.from_numpy(trans), "valids": torch.from_numpy(valids),
            "n_parts": torch.tensor(n_parts),
            # anchor = the fragment the GT frame is pinned to
            "anchor": torch.tensor(anchor),
        }
        if self.with_adjacency:
            item.update(self._adjacency_tensors(scene_dir, gt_r, trans))
        return item

    def _adjacency_tensors(self, scene_dir: str, gt_r: np.ndarray,
                           gt_t: np.ndarray) -> Dict[str, torch.Tensor]:
        """GT adjacency and relative poses of adjacent ordered pairs
        (PHFormer supervision), read from gt/adjacency.json."""
        adj = np.zeros((MAX_PARTS, MAX_PARTS), dtype=np.float32)
        rel_q = np.zeros((MAX_PARTS, MAX_PARTS, 4), dtype=np.float32)
        rel_t = np.zeros((MAX_PARTS, MAX_PARTS, 3), dtype=np.float32)
        adj_path = os.path.join(scene_dir, "gt", "adjacency.json")
        if os.path.exists(adj_path):
            with open(adj_path, "r", encoding="utf-8") as fh:
                annotation = json.load(fh)
            for edge in annotation.get("edges", []):
                i, j = edge["frag_i"], edge["frag_j"]
                if i >= MAX_PARTS or j >= MAX_PARTS:
                    continue
                adj[i, j] = adj[j, i] = 1.0
                r_i, t_i = gt_r[i], gt_t[i]
                r_j, t_j = gt_r[j], gt_t[j]
                r_rel = r_i @ r_j.T
                t_rel = t_i - r_rel @ t_j
                rel_q[i, j] = rotmat_to_quat_scalar_first(r_rel)
                rel_t[i, j] = t_rel
                rel_q[j, i] = rotmat_to_quat_scalar_first(r_rel.T)
                rel_t[j, i] = -r_rel.T @ t_rel
        return {"adj": torch.from_numpy(adj),
                "rel_quat": torch.from_numpy(rel_q),
                "rel_trans": torch.from_numpy(rel_t)}



class PointNetEncoder(nn.Module):
    """The reference PointNet feature extractor (Huang et al., NeurIPS
    2020; Breaking Bad): five 1x1 convolutions 64/64/64/128/feat_dim with
    batch norm on every layer, relu on the first four, global max
    pooling over the points."""

    def __init__(self, feat_dim: int = 128) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        channels = [64, 64, 64, 128, feat_dim]
        self.convs = nn.ModuleList([
            nn.Conv1d(3 if i == 0 else channels[i - 1], channels[i], 1,
                      bias=False)
            for i in range(len(channels))])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(c) for c in channels])

    def forward(self, pcs: torch.Tensor) -> torch.Tensor:
        """pcs: [B, N, 3] -> global codes [B, feat_dim]."""
        x = pcs.transpose(1, 2)
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = bn(conv(x))
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        return x.max(dim=-1).values


class _ConvMLP(nn.Module):
    """Shared MLP with batch norm and relu on every layer but the last
    (the reference's Conv1d MLP, applied to the feature axis)."""

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i]))
            if i < len(channels) - 1:
                layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        y = self.net(x.reshape(-1, x.shape[-1]))
        return y.reshape(*lead, -1)


class PoseRegressor(nn.Module):
    """MLP pose regressor: shared features, separate rotation and
    translation heads; the quaternion head output is normalized."""

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.fc = _ConvMLP([feat_dim, 256, 128])
        self.rot_head = nn.Linear(128, 4)
        self.trans_head = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.fc(x)
        rot = self.rot_head(f)
        rot = rot / rot.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return rot, self.trans_head(f)


def cosine_rotation_loss(pred: torch.Tensor, target: torch.Tensor,
                         valids: torch.Tensor) -> torch.Tensor:
    """1 - |dot| on scalar-first quaternions over valid fragments."""
    dot = (pred * target).sum(-1).abs()
    return (1.0 - dot).mul(valids).sum() / valids.sum().clamp(min=1.0)


def l2_loss(pred: torch.Tensor, target: torch.Tensor,
            valids: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).pow(2).sum(-1)
    return diff.mul(valids).sum() / valids.sum().clamp(min=1.0)


def per_part_chamfer(pred_pts: torch.Tensor, gt_pts: torch.Tensor,
                     valids: torch.Tensor | None = None) -> torch.Tensor:
    """Bidirectional Chamfer per part on [B, P, N, 3] point clouds.

    Nearest-neighbor distances are exact (torch.cdist). Memory per sample is
    O(P * N^2), which is about 1 GB for N = 1000 on a GPU. With ``valids``
    [B, P] only the real fragments enter the mean (the zero-filled padded
    slots would otherwise dominate it).
    """
    if valids is not None:
        pred_pts = pred_pts[valids > 0.5]
        gt_pts = gt_pts[valids > 0.5]
    m = pred_pts.reshape(-1, pred_pts.shape[-2], 3)
    n = gt_pts.reshape(-1, gt_pts.shape[-2], 3)
    d_ab = torch.cdist(m, n).min(-1).values.mean()
    d_ba = torch.cdist(n, m).min(-1).values.mean()
    return (d_ab + d_ba) / 2.0


class GeometricLoss(nn.Module):
    """Combined pose loss; weights follow docs/baselines.md.

    The per-point L2 and per-part Chamfer terms compare the rotated
    fragment clouds only (rotation-only, as in the reference geometric
    loss); the whole-shape Chamfer compares the fully posed assemblies.
    The whole-shape term evaluates on a per-part subsample of points to
    keep the memory bounded.
    """

    def __init__(self, w_trans: float = 1.0, w_rot: float = 0.2,
                 w_pt_l2: float = 1.0, w_pt_cd: float = 10.0,
                 w_shape_cd: float = 10.0) -> None:
        super().__init__()
        self.w_trans = w_trans
        self.w_rot = w_rot
        self.w_pt_l2 = w_pt_l2
        self.w_pt_cd = w_pt_cd
        self.w_shape_cd = w_shape_cd

    def forward(self, pcs: torch.Tensor, pred_quat: torch.Tensor,
                pred_trans: torch.Tensor, gt_quat: torch.Tensor,
                gt_trans: torch.Tensor, valids: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_r = quat_to_rotmat(pred_quat)
        gt_r = quat_to_rotmat(gt_quat)
        # p_w = R @ p_obs + t, implemented as p @ R^T + t
        pred_world = torch.einsum("bpnc,bpdc->bpnd", pcs, pred_r) + pred_trans[:, :, None, :]
        gt_world = torch.einsum("bpnc,bpdc->bpnd", pcs, gt_r) + gt_trans[:, :, None, :]
        pred_rot = torch.einsum("bpnc,bpdc->bpnd", pcs, pred_r)
        gt_rot = torch.einsum("bpnc,bpdc->bpnd", pcs, gt_r)

        loss_trans = l2_loss(pred_trans, gt_trans, valids)
        loss_rot = cosine_rotation_loss(pred_quat, gt_quat, valids)
        # per-point L2: sum over xyz, MEAN over points, then mean over valid
        # fragments -- the reference's rot_points_l2_loss is
        # ``(p1 - p2).pow(2).sum(-1).mean(-1)`` followed by a valid-part mean.
        # Without the ``mean(-1)`` the term is N (=1000) times too large and
        # dominates the total loss.
        loss_pt_l2 = ((pred_rot - gt_rot).pow(2).sum(-1).mean(-1)
                      .mul(valids).sum() / valids.sum().clamp(min=1.0))
        loss_pt_cd = per_part_chamfer(pred_rot, gt_rot, valids)

        # whole-shape Chamfer, as the reference computes it for training
        # (shape_cd_loss with training=True): squared per-part Chamfer between
        # the posed prediction and the GT over all points, weighted by valids
        # and divided by the fixed slot count MAX_PARTS -- the reference's
        # "automatic hard negative" weighting.
        _, p, _, _ = pred_world.shape
        d_ab = torch.cdist(pred_world, gt_world).min(-1).values   # [B, P, N]
        d_ba = torch.cdist(gt_world, pred_world).min(-1).values
        per_part = (d_ab + d_ba).mean(-1)                          # [B, P]
        loss_shape_cd = (per_part * valids).sum(-1).mean() / float(p)

        total = (self.w_trans * loss_trans + self.w_rot * loss_rot
                 + self.w_pt_l2 * loss_pt_l2 + self.w_pt_cd * loss_pt_cd
                 + self.w_shape_cd * loss_shape_cd)
        return {"total": total, "trans": loss_trans, "rot": loss_rot,
                "pt_l2": loss_pt_l2, "pt_cd": loss_pt_cd, "shape_cd": loss_shape_cd}


def rotate_point_clouds(pcs: torch.Tensor, quat: torch.Tensor,
                        trans: torch.Tensor) -> torch.Tensor:
    """World points of an observation cloud: p @ R^T + t."""
    r = quat_to_rotmat(quat)
    return torch.einsum("bpnc,bpdc->bpnd", pcs, r) + trans[:, :, None, :]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
