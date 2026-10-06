"""Numerical checks pinning the baselines to their reference semantics
(GARF / DiffAssemble / PHFormer official implementations; see
docs/baselines.md and the module docstrings). All checks run on CPU.
"""

import os
import types

import numpy as np
import pytest
import torch

torch.set_num_threads(2)


def _random_rotation(rng: np.random.Generator) -> torch.Tensor:
    u1, u2, u3 = rng.random(3)
    q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                  np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3),
                  np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    x, y, z, w = q
    return torch.from_numpy(np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])).float()


# ---------------------------------------------------------------- VN-DGCNN

def test_vn_knn_includes_self():
    """The reference kNN keeps the point itself among the k neighbors
    (its zero distance always ranks first)."""
    from hylofrac.baselines.vn_layers import _knn_indices

    x = torch.randn(2, 3, 40)
    idx = _knn_indices(x, 20)
    assert (idx[..., 0] == torch.arange(40).view(1, 40)).all()


def test_vn_code_duplicates_global_mean():
    """Mean pooling after concatenating the global point mean makes the
    second half of the 768-dim code duplicate the first, as in the
    reference's mean-pooling path."""
    from hylofrac.baselines.vn_layers import VNDGCNNEncoder

    enc = VNDGCNNEncoder()
    code = enc(torch.randn(1, 2, 64, 3))
    assert code.shape == (1, 2, 768)
    assert torch.allclose(code[..., :384], code[..., 384:], atol=1e-5)


def test_vn_edge_feature_layout():
    """The edge feature stacks the neighbor differences before the center
    vectors; the center block is constant across the neighbor axis."""
    from hylofrac.baselines.vn_layers import _edge_feature

    x = torch.randn(1, 2, 3, 16)
    e = _edge_feature(x, 8)                     # [1, 4, 3, 16, 8]
    diff, center = e[:, :2], e[:, 2:]
    center_ref = x.permute(0, 1, 2, 3).unsqueeze(-1).expand(1, 2, 3, 16, 8)
    assert torch.allclose(center, center_ref, atol=1e-6)
    # diff + center recovers the neighbor vectors
    from hylofrac.baselines.vn_layers import _knn_indices
    flat = x.reshape(1, -1, 16)
    idx = _knn_indices(flat, 8)
    n, k = 5, 3
    neighbor = flat[0, :, idx[0, n, k]]
    assert torch.allclose((diff + center)[0, :, :, n, k].reshape(-1),
                          neighbor, atol=1e-6)


# -------------------------------------------------------------- SO(3) tools

def test_so3_scale_power_law():
    from hylofrac.baselines.common import log_rmat, so3_scale

    rng = np.random.default_rng(3)
    r = _random_rotation(rng)
    assert torch.allclose(so3_scale(r, torch.tensor(1.0)), r, atol=1e-5)
    assert torch.allclose(so3_scale(r, torch.tensor(0.0)),
                          torch.eye(3), atol=1e-5)
    half = so3_scale(r, torch.tensor(0.5))
    assert torch.allclose(so3_scale(half, torch.tensor(2.0)), r, atol=1e-5)
    # exp(log R) roundtrip
    logs = log_rmat(r.unsqueeze(0))
    assert torch.allclose(torch.matrix_exp(logs[0]), r, atol=1e-5)


def test_log_rmat_identity_and_pi():
    from hylofrac.baselines.common import log_rmat, so3_scale

    assert torch.allclose(log_rmat(torch.eye(3).unsqueeze(0)),
                          torch.zeros(1, 3, 3), atol=1e-6)
    # 180-degree rotations take the eigenvector branch of the log
    rot_pi = torch.tensor([[-1.0, 0, 0], [0, -1, 0], [0, 0, 1]])
    out = so3_scale(rot_pi.unsqueeze(0), torch.tensor(0.5))
    assert torch.allclose(out @ out, rot_pi, atol=1e-5)


# ------------------------------------------------------------------ IGSO3

def test_igso3_zero_spread_is_identity():
    from hylofrac.baselines.diffassemble_net import igso3_sample

    rng = torch.Generator().manual_seed(0)
    rot = igso3_sample(torch.zeros(2, 3), 1, rng)
    assert rot.shape == (2, 3, 1, 3, 3)
    assert torch.allclose(rot.squeeze(2), torch.eye(3), atol=1e-6)


def test_igso3_samples_are_rotations_and_grow_with_eps():
    from hylofrac.baselines.diffassemble_net import igso3_sample
    from hylofrac.baselines.garf_net import axis_angle_of

    rng = torch.Generator().manual_seed(7)
    eps = torch.tensor([0.05, 0.2, 1.0])
    rot = igso3_sample(eps, 2000, rng)            # [3, 2000, 1, 3, 3]
    orth_err = (rot.transpose(-1, -2) @ rot
                - torch.eye(3)).abs().max()
    assert orth_err < 1e-5
    angle = axis_angle_of(rot.reshape(3, 2000, 3, 3)).norm(dim=-1)
    mean_angle = angle.mean(dim=1)
    # the mean rotation angle grows monotonically with the spread
    assert mean_angle[0] < mean_angle[1] < mean_angle[2]
    assert mean_angle[2] > 1.0


# ------------------------------------------------------------------ GARF

def test_garf_add_noise_reference_semantics():
    """The corrupted poses follow the reference path (per-scene sigma in
    {1/1000, ..., 1}, rotation input exp(sigma log R1) @ R0); anchored
    fragments keep the GT pose as input with a zero velocity target."""
    from hylofrac.baselines.garf_net import GarfPoseNet

    torch.manual_seed(0)
    model = GarfPoseNet()
    b, p = 2, 4
    quat_gt = torch.zeros(b, p, 4)
    quat_gt[..., 0] = 1.0
    trans_gt = torch.randn(b, p, 3)
    anchor = torch.zeros(b, p)
    anchor[0, 1] = 1.0
    rng = torch.Generator().manual_seed(3)
    pose_t, v_gt, timestep = model.add_noise(quat_gt, trans_gt, anchor, rng)

    assert timestep.shape == (b, p)
    # per-scene timestep shared by the fragments, in {1, ..., 1000}
    assert torch.allclose(timestep[:, 0], timestep[:, -1])
    assert 1 <= timestep.min() and timestep.max() <= 1000
    # anchor rows: GT input and zero target
    assert torch.allclose(pose_t[0, 1], torch.cat([quat_gt[0, 1],
                                                   trans_gt[0, 1]]))
    assert torch.allclose(v_gt[0, 1], torch.zeros(6), atol=1e-6)
    # non-anchor rows are corrupted (rotation and translation)
    assert (v_gt[0, 0] != 0).any()
    assert not torch.allclose(pose_t[0, 0, :4], quat_gt[0, 0])
    # translation consistency: x_t = (1 - sigma) GT + sigma noise with the
    # velocity target noise - GT, i.e. v_t = (x_t - GT) / sigma
    gt = trans_gt[0, 0]
    sigma = timestep[0, 0] / 1000.0
    x_t = pose_t[0, 0, 4:]
    assert torch.allclose(v_gt[0, 0, :3], (x_t - gt) / sigma, atol=1e-4)


def test_garf_integration_grid_reaches_zero():
    """Inference walks sigma from 1 to 0 in 20 Euler steps, querying the
    denoiser at timesteps 1000, 950, ..., 50 (never at 0)."""
    from hylofrac.baselines.garf_net import INTEGRATION_STEPS, GarfPoseNet

    torch.manual_seed(0)
    model = GarfPoseNet()
    model.eval()
    seen: list[float] = []

    def counting_forward(self, pcs, pose, timestep, anchor, valids,
                         code=None):
        seen.append(float(timestep[0, 0]))
        return torch.zeros(pcs.shape[0], pcs.shape[1], 6)

    model.forward = types.MethodType(counting_forward, model)
    pcs = torch.randn(1, 3, 16, 3)
    valids = torch.zeros(1, 3)
    valids[0, :2] = 1.0
    with torch.no_grad():
        quat, trans = model.predict(pcs, valids)
    assert len(seen) == INTEGRATION_STEPS
    assert seen[0] == 1000.0 and seen[-1] == 50.0
    assert seen == sorted(seen, reverse=True)
    assert torch.isfinite(quat).all() and torch.isfinite(trans).all()


def test_garf_anchor_loss_uses_zero_target_not_mask():
    """Anchored fragments stay in the loss with a zero target: corrupting
    the anchor prediction must move the loss."""
    from hylofrac.baselines.garf_net import GarfPoseNet

    torch.manual_seed(0)
    model = GarfPoseNet()
    b, p = 1, 2
    pcs = torch.randn(b, p, 16, 3)
    quat_gt = torch.zeros(b, p, 4)
    quat_gt[..., 0] = 1.0
    trans_gt = torch.zeros(b, p, 3)
    valids = torch.ones(b, p)
    anchor = torch.zeros(b, p)
    anchor[0, 0] = 1.0
    loss1 = model.loss_step(pcs, quat_gt, trans_gt, valids,
                            anchor=anchor)["total"]
    model.mlp_t[2].weight.data[0, 0] += 100.0  # perturb the anchor output
    loss2 = model.loss_step(pcs, quat_gt, trans_gt, valids,
                            anchor=anchor)["total"]
    assert float(loss1.detach()) != float(loss2.detach())


def test_garf_velocity_layout_trans_first():
    """The velocity is [trans(3), rot(3)] everywhere: the loss target and
    the integration step share the layout (regression: the integration
    used to read the rotation half for the translation)."""
    from hylofrac.baselines.common import exp_map, quat_to_rotmat
    from hylofrac.baselines.garf_net import GarfPoseNet, axis_angle_of

    torch.manual_seed(0)
    model = GarfPoseNet()
    b, p = 1, 2
    quat_gt = torch.zeros(b, p, 4)
    quat_gt[..., 0] = 1.0
    trans_gt = torch.randn(b, p, 3)
    anchor = torch.zeros(b, p)

    def fixed_noise(b, p, rng):
        # 90 degrees around z
        q = torch.zeros(b, p, 4)
        q[..., 0] = 2 ** -0.5
        q[..., 3] = 2 ** -0.5
        return q

    model._uniform_quat = fixed_noise
    rng = torch.Generator().manual_seed(1)
    pose_t, v_gt, timestep = model.add_noise(quat_gt, trans_gt, anchor, rng)
    sigma = (timestep / 1000.0).unsqueeze(-1)
    # translation half: x_t = (1 - sigma) GT + sigma noise and
    # v_t = noise - GT = (x_t - GT) / sigma
    v_t = (pose_t[..., 4:] - trans_gt) / sigma
    assert torch.allclose(v_t, v_gt[..., :3], atol=1e-4)
    # rotation half: v = log R1 (fixed noise rotation) and the corrupted
    # rotation is exp(sigma v) @ R0 with R0 = identity
    v_r_ref = axis_angle_of(quat_to_rotmat(fixed_noise(b, p, None)))
    assert torch.allclose(v_gt[..., 3:], v_r_ref, atol=1e-5)
    r_t = exp_map(sigma * v_gt[..., 3:])
    assert torch.allclose(quat_to_rotmat(pose_t[..., :4]), r_t, atol=1e-5)


# ---------------------------------------------------------------- PHFormer

def test_phformer_fps_starts_at_point_zero():
    from hylofrac.baselines.phformer_net import farthest_point_sample

    x = torch.randn(4, 50, 3)
    idx = farthest_point_sample(x, 16)
    assert (idx[:, 0] == 0).all()
    assert idx.max() < 50 and idx.min() >= 0
    assert (farthest_point_sample(x, 16) == idx).all()  # deterministic


def test_phformer_ball_query_scan_order_and_padding():
    """The ball query keeps the first inliers in scan order and pads
    undersized balls by repeating the first inlier (reference kernel
    semantics); radii are relative to the per-cloud max norm."""
    from hylofrac.baselines.phformer_net import ball_query

    x = torch.arange(9, dtype=torch.float32).view(1, 9, 1).repeat(1, 1, 3)
    centers = torch.tensor([[4]])
    # radius 0.4 of the max norm (8 * sqrt(3)): points within distance
    # 0.4 * 8 * sqrt(3) of point 4, i.e. |i - 4| < 3.2 (inliers 1..7,
    # boundaries kept well outside the float rounding)
    idx = ball_query(x, centers, 0.4, 5)
    assert idx.tolist() == [[[1, 2, 3, 4, 5]]]
    # scale invariance: rescaling leaves the indices unchanged
    assert (ball_query(x * 10.0, centers, 0.4, 5) == idx).all()
    # undersized ball keeps only the center, repeated
    idx = ball_query(x, centers, 0.05, 5)
    assert idx.tolist() == [[[4, 4, 4, 4, 4]]]


def test_phformer_voxel_pe_matches_reference_frequencies():
    """The per-axis frequencies equal the reference grid
    exp(-9.2103 * 2m / (feature_dim // 3)) up to the log(10000) constant
    truncation; the code is 126 dims zero-padded to 128."""
    from hylofrac.baselines.phformer_net import VolumetricPositionEncoding

    pe = VolumetricPositionEncoding(128)
    ours = pe.freq[0, 0]
    ref = torch.exp(-9.2103 * torch.arange(0, 128 // 3 - 1, 2,
                                           dtype=torch.float)
                    / (128 // 3))
    assert len(ref) == 21
    assert torch.allclose(ours, ref, rtol=2e-3)
    assert pe.pad_dim == 2
    code = pe(torch.rand(1, 8, 3) - 0.5)
    assert code.shape == (1, 8, 128)
    assert torch.allclose(code[..., 126:], torch.zeros(1, 8, 2), atol=1e-7)


# --------------------------------------- Breaking Bad baselines (BB)

def test_pointnet_reference_layers():
    """The shared encoder is the reference PointNet: five convolutions
    64/64/64/128/feat_dim, relu on the first four layers only."""
    from hylofrac.baselines.common import PointNetEncoder

    enc = PointNetEncoder(128)
    convs = [c.out_channels for c in enc.convs]
    assert convs == [64, 64, 64, 128, 128]
    assert all(c.bias is None for c in enc.convs)
    assert enc(torch.randn(2, 64, 3)).shape == (2, 128)


def test_global_uses_two_encoders():
    """B-Global: a shared per-fragment PointNet plus a second PointNet
    over the whole scene cloud."""
    from hylofrac.baselines.global_net import GlobalPoseNet

    model = GlobalPoseNet()
    assert model.encoder is not model.global_encoder
    pcs = torch.randn(1, 4, 32, 3)
    valids = torch.zeros(1, 4)
    valids[0, :2] = 1.0
    quat, trans = model(pcs, valids)
    assert quat.shape == (1, 4, 4) and trans.shape == (1, 4, 3)
    assert torch.allclose(quat.norm(dim=-1), torch.ones(1, 4), atol=1e-5)


def test_lstm_eval_is_deterministic_autoregressive():
    """At evaluation the decoder is autoregressive and reproducible (the
    reference noise and no teacher forcing)."""
    from hylofrac.baselines.lstm_net import LstmPoseNet

    torch.manual_seed(0)
    model = LstmPoseNet()
    model.eval()
    pcs = torch.randn(1, 4, 32, 3)
    valids = torch.zeros(1, 4)
    valids[0, :2] = 1.0
    with torch.no_grad():
        q1, t1 = model(pcs, valids)
        q2, t2 = model(pcs, valids)
    assert torch.allclose(q1, q2) and torch.allclose(t1, t2)
    assert torch.allclose(q1.norm(dim=-1), torch.ones(1, 4), atol=1e-5)


def test_dgl_three_rounds_with_per_round_parameters():
    """DGL runs three rounds with independent parameters and supervises
    every round; the forward output is the last round."""
    from hylofrac.baselines.dgl_net import DglPoseNet

    torch.manual_seed(0)
    model = DglPoseNet()
    assert model.iterations == 3
    assert len(model.edge_mlps) == len(model.node_mlps) \
        == len(model.pose_predictors) == 3
    pcs = torch.randn(1, 3, 32, 3)
    valids = torch.ones(1, 3)
    quat, trans = model(pcs, valids)
    assert len(model.rounds) == 3
    q_last, t_last = model.rounds[-1]
    assert torch.allclose(quat, q_last) and torch.allclose(trans, t_last)
    # per-round parameters are independent instances
    assert model.pose_predictors[0].fc is not model.pose_predictors[1].fc
    gt_q = torch.zeros(1, 3, 4)
    gt_q[..., 0] = 1.0
    gt_t = torch.zeros(1, 3, 3)
    losses = model.loss_step(pcs, gt_q, gt_t, valids)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_phformer_global_encoder_multi_sample_broadcast():
    """GlobalEncoder weighs per-fragment codes over samples and returns
    [B, 1, C]; the full pose net runs a padded multi-sample batch."""
    from hylofrac.baselines.phformer_net import GlobalEncoder, PhformerPoseNet

    ge = GlobalEncoder(128).eval()
    node = torch.randn(4, 100, 128)
    valids = torch.ones(4, 100)
    with torch.no_grad():
        out = ge(node, valids)
    assert out.shape == (4, 1, 128)
    net = PhformerPoseNet(feat_dim=128).eval()
    pcs = torch.randn(4, 100, 256, 3)
    with torch.no_grad():
        q, t = net(pcs, torch.ones(4, 100))
    assert q.shape == (4, 100, 4)
    assert t.shape == (4, 100, 3)


def test_garf_adaln_multi_sample_broadcast():
    """AdaLN scale/shift [B, C] broadcast over the fragment axis; the full
    denoiser runs a padded multi-sample batch."""
    from hylofrac.baselines.garf_net import GarfPoseNet

    net = GarfPoseNet().eval()
    b, p = 4, 100
    pcs = torch.randn(b, p, 256, 3)
    pose = torch.randn(b, p, 7)
    pose[..., 0] = 1.0
    ts = torch.full((b, 1), 500.0)
    anchor = torch.zeros(b, p)
    anchor[:, 0] = 1.0
    with torch.no_grad():
        out = net(pcs, pose, ts, anchor, torch.ones(b, p))
    assert out.shape == (b, p, 6)
    assert torch.isfinite(out).all()
