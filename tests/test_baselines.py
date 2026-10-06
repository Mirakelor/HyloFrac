"""Smoke tests for the learned baselines and the geometric baseline."""

import os

import numpy as np
import pytest
import torch

@pytest.mark.parametrize("method", ["global", "lstm", "dgl"])
def test_model_forward_backward(method):
    from hylofrac.baselines.common import GeometricLoss
    from hylofrac.baselines.dgl_net import DglPoseNet
    from hylofrac.baselines.global_net import GlobalPoseNet
    from hylofrac.baselines.lstm_net import LstmPoseNet

    models = {"global": GlobalPoseNet, "lstm": LstmPoseNet, "dgl": DglPoseNet}
    model = models[method]()
    pcs = torch.zeros(1, 4, 128, 3)
    valids = torch.zeros(1, 4)
    valids[0, :2] = 1.0
    quat, trans = model(pcs, valids)
    assert quat.shape == (1, 4, 4)
    assert trans.shape == (1, 4, 3)
    # rotations are unit quaternions
    assert torch.allclose(quat.norm(dim=-1), torch.ones(1, 4), atol=1e-5)
    loss_fn = GeometricLoss()
    gt_q = torch.zeros(1, 4, 4)
    gt_q[..., 0] = 1.0
    gt_t = torch.zeros(1, 4, 3)
    losses = loss_fn(pcs, quat, trans, gt_q, gt_t, valids)
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_quat_rotmat_roundtrip():
    from hylofrac.baselines.common import quat_to_rotmat, rotmat_to_quat_scalar_first

    rng = np.random.default_rng(0)
    for _ in range(5):
        u1, u2, u3 = rng.random(3)
        q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                      np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                      np.sqrt(u1) * np.sin(2 * np.pi * u3),
                      np.sqrt(u1) * np.cos(2 * np.pi * u3)])
        r = quat_to_rotmat(torch.from_numpy(q).float()).numpy()
        q2 = rotmat_to_quat_scalar_first(r)
        # quaternions are equal up to sign
        dot = abs(float(np.dot(q, q2)))
        assert dot > 1 - 1e-5


def test_torch_matrix_to_quat_roundtrip():
    """The torch twin must be the inverse of ``quat_to_rotmat`` too.

    Only the numpy twin used to be covered, which is how a swapped branch
    axis order survived in the torch version: it round-tripped fine for
    matrices with a positive trace and failed (error ~1.0) for the three
    branches that handle rotations of or near 180 degrees, exactly the
    quaternions GARF and DiffAssemble consume during training.
    """
    from hylofrac.baselines.common import matrix_to_quat, quat_to_rotmat

    rng = np.random.default_rng(0)

    # uniform random rotations: R -> q -> R is sign-invariant
    q = torch.from_numpy(rng.normal(size=(2048, 4))).float()
    q = q / q.norm(dim=-1, keepdim=True)
    r = quat_to_rotmat(q)
    err = (quat_to_rotmat(matrix_to_quat(r)) - r).abs().amax(dim=(-2, -1))
    # float32 round-off only; the worst of 4096 measured samples is 1.8e-4
    assert err.max() < 1e-3
    # ...and the quaternions themselves agree up to the double cover
    dot = (q * matrix_to_quat(r)).sum(-1).abs()
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-4)

    # rotations just next to 180 degrees hit the non-trace branches, where
    # the off-diagonal terms are divided by a small s
    axes = torch.from_numpy(rng.normal(size=(512, 3))).float()
    axes = axes / axes.norm(dim=-1, keepdim=True)
    for w in (1e-6, 1e-4, 1e-2):
        q_near = torch.cat([torch.full((512, 1), w), axes], dim=-1)
        q_near = q_near / q_near.norm(dim=-1, keepdim=True)
        r_near = quat_to_rotmat(q_near)
        err = (quat_to_rotmat(matrix_to_quat(r_near)) - r_near
               ).abs().amax(dim=(-2, -1))
        assert err.max() < 1e-3, f"w={w:g}"

    # in float64 the same round trip has to be exact (a fresh draw: promoting
    # the float32 quaternions above would just measure their own rounding)
    q64 = torch.from_numpy(rng.normal(size=(2048, 4))).double()
    q64 = q64 / q64.norm(dim=-1, keepdim=True)
    r64 = quat_to_rotmat(q64)
    assert torch.allclose(quat_to_rotmat(matrix_to_quat(r64)), r64, atol=1e-10)


def test_matrix_to_quat_exact_half_turn():
    """A half turn has a scalar part of exactly zero, which is where a cyclic
    branch selector picks a branch whose divisor vanishes.

    Both twins used to fail here (the torch one returned another rotation, the
    numpy one returned nan for a (1,1,1)-aligned half turn); the branch is now
    chosen by the largest diagonal entry.
    """
    from hylofrac.baselines.common import (matrix_to_quat, quat_to_rotmat,
                                           rotmat_to_quat_scalar_first)

    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0], [0.6, 0.8, 0.0], [0.3, 0.5, 0.81]):
        v = torch.tensor(axis)
        v = v / v.norm()
        r_pi = (2.0 * v[:, None] * v[None, :] - torch.eye(3)).double()
        assert torch.allclose(quat_to_rotmat(matrix_to_quat(r_pi[None]))[0],
                              r_pi, atol=1e-6), axis
        q_np = rotmat_to_quat_scalar_first(r_pi.numpy())
        assert np.isfinite(q_np).all(), axis
        assert np.allclose(quat_to_rotmat(torch.tensor(q_np[None]))[0],
                           r_pi.numpy(), atol=1e-6), axis


def test_matrix_to_quat_axis_aligned_large_angle():
    """Above 120 degrees about a coordinate axis the trace branch no longer
    applies, which is where the old selector silently returned another pose
    (max error 2.0) rather than losing a little precision."""
    from hylofrac.baselines.common import (matrix_to_quat, quat_to_rotmat,
                                           rotmat_to_quat_scalar_first)

    for axis in range(3):
        v = torch.zeros(3)
        v[axis] = 1.0
        skew = torch.tensor([[0.0, -v[2], v[1]],
                             [v[2], 0.0, -v[0]],
                             [-v[1], v[0], 0.0]])
        for angle in (2.1, 2.6, np.pi - 1e-3):
            r = (torch.eye(3) + np.sin(angle) * skew
                 + (1 - np.cos(angle)) * (skew @ skew)).double()
            assert torch.allclose(quat_to_rotmat(matrix_to_quat(r[None]))[0],
                                  r, atol=1e-6), (axis, angle)
            q_np = rotmat_to_quat_scalar_first(r.numpy())
            assert np.allclose(quat_to_rotmat(torch.tensor(q_np[None]))[0],
                               r.numpy(), atol=1e-6), (axis, angle)


def test_geometric_assemble_runs(packaged_scene):
    from hylofrac.baselines.geometric import predict_scene
    from hylofrac.eval.loader import SceneSample

    sample = SceneSample(packaged_scene, "000102166_v00", n_points=256)
    r, t = predict_scene(sample.pcs)
    assert r.shape == (2, 3, 3)
    assert t.shape == (2, 3)
    for i in range(2):
        assert np.allclose(r[i] @ r[i].T, np.eye(3), atol=1e-6)


def test_predict_write_and_evaluate(packaged_scene, tmp_path):
    """End-to-end: submission written from loader poses evaluates to ~1.0."""
    from hylofrac.eval.evaluate import evaluate_submission
    from hylofrac.eval.loader import SceneSample
    from hylofrac.eval.submit import write_submission

    scene_dir = packaged_scene
    scene_id = "000102166_v00"
    sample = SceneSample(scene_dir, scene_id, n_points=300)
    fragments = {name: {"R": sample.gt_R[i], "t": sample.gt_t[i]}
                 for i, name in enumerate(sample.names)}
    pred_path = os.path.join(str(tmp_path), f"{scene_id}.json")
    write_submission(pred_path, scene_id, "perfect", fragments)

    # packaged_scene = <data_root>/val/000102166/v00
    data_root = os.path.abspath(os.path.join(scene_dir, "..", "..", ".."))
    metrics = evaluate_submission(pred_path, data_root, ["val"], 300, 0)
    assert metrics is not None
    assert metrics["pa"] > 0.99
    assert metrics["qpos"] > 0.95


def test_vn_dgcnn_encoder_fidelity():
    """DiffAssemble uses the equivariant VN-DGCNN backbone: codes rotate
    with the input and the diffusion chain (noise, x0 prediction, short
    DDIM) is finite with gradients."""
    from hylofrac.baselines.diffassemble_net import DiffAssemblePoseNet

    torch.manual_seed(0)
    model = DiffAssemblePoseNet()
    pcs = torch.randn(1, 3, 64, 3)
    valids = torch.zeros(1, 3)
    valids[0, :2] = 1.0
    quat = torch.zeros(1, 3, 4)
    quat[..., 0] = 1.0
    trans = torch.zeros(1, 3, 3)
    losses = model.loss_step(pcs, quat, trans, valids)
    losses["total"].backward()
    grads = [g for p in model.parameters()
             if (g := p.grad) is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    model.eval()
    with torch.no_grad():
        quat2, trans2 = model.sample(pcs, valids, steps=3)
    assert torch.isfinite(quat2).all() and torch.isfinite(trans2).all()


def test_phformer_encoder_fidelity():
    """PHFormer runs the PointNet2-style encoder with ball-query grouping
    (incl. undersized balls) and voxel-sinusoidal PE on the self layers,
    over packed valid fragments with a padded scene."""
    from hylofrac.baselines.phformer_net import PhformerPoseNet

    torch.manual_seed(0)
    model = PhformerPoseNet()
    pcs = torch.rand(1, 3, 256, 3) - 0.5
    valids = torch.zeros(1, 3)
    valids[0, :2] = 1.0
    quat, trans = model(pcs, valids)
    assert quat.shape == (1, 3, 4) and trans.shape == (1, 3, 3)
    assert torch.allclose(quat.norm(dim=-1), torch.ones(1, 3), atol=1e-5)
    adj_gt = torch.zeros(1, 3, 3)
    adj_gt[0, 0, 1] = adj_gt[0, 1, 0] = 1.0
    rel_q = torch.zeros(1, 3, 3, 4)
    rel_q[..., 0] = 1.0
    rel_t = torch.zeros(1, 3, 3, 3)
    loss = model.aux_loss(adj_gt, rel_q, rel_t, valids)
    loss.backward()
    grads = [g for p in model.parameters()
             if (g := p.grad) is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_garf_encoder_fidelity():
    """GARF encodes fragments with the blocked (PTV3-style) point
    transformer; the flow-matching chain stays finite and the anchor
    fragment remains pinned during integration."""
    from hylofrac.baselines.garf_net import GarfPoseNet

    torch.manual_seed(0)
    model = GarfPoseNet()
    pcs = torch.rand(1, 3, 256, 3) - 0.5
    valids = torch.zeros(1, 3)
    valids[0, :2] = 1.0
    quat = torch.zeros(1, 3, 4)
    quat[..., 0] = 1.0
    trans = torch.zeros(1, 3, 3)
    losses = model.loss_step(pcs, quat, trans, valids)
    losses["total"].backward()
    grads = [g for p in model.parameters()
             if (g := p.grad) is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    model.eval()
    with torch.no_grad():
        quat2, trans2 = model.predict(pcs, valids)
    assert torch.isfinite(quat2).all() and torch.isfinite(trans2).all()
    assert torch.allclose(quat2[:, 0], torch.tensor([1.0, 0, 0, 0]),
                          atol=1e-5)


def test_rpf_flow_fidelity():
    """RPF transports the scene points: the rectified-flow step trains with
    finite gradients, prediction returns unit quaternions, and the anchor
    fragment stays pinned to the pose it is given."""
    from hylofrac.baselines.rpf_net import RpfPoseNet, u_shaped_timesteps

    torch.manual_seed(0)
    t = u_shaped_timesteps(4096, torch.device("cpu"))
    assert float(t.min()) >= 0.01 - 1e-6 and float(t.max()) <= 1.0
    assert float(t.mean()) < 0.6  # u-shaped: mass towards both ends

    model = RpfPoseNet()
    pcs = torch.rand(1, 3, 128, 3) - 0.5
    valids = torch.zeros(1, 3)
    valids[0, :2] = 1.0
    quat = torch.zeros(1, 3, 4)
    quat[..., 0] = 1.0
    trans = torch.rand(1, 3, 3) * 0.1
    anchor = torch.zeros(1, 3)
    anchor[0, 0] = 1.0
    losses = model.loss_step(pcs, quat, trans, valids, anchor=anchor)
    assert set(losses) == {"total", "flow_mse"}
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    model.eval()
    with torch.no_grad():
        quat2, trans2 = model.predict(pcs, valids)
    assert torch.isfinite(quat2).all() and torch.isfinite(trans2).all()
    assert torch.allclose(quat2.norm(dim=-1), torch.ones(1, 3), atol=1e-4)
    # the anchor is pinned to the identity pose, so its fitted pose is it
    assert torch.allclose(quat2[:, 0], torch.tensor([1.0, 0, 0, 0]),
                          atol=1e-4)
    assert torch.allclose(trans2[:, 0], torch.zeros(1, 3), atol=1e-4)


def test_rpf_procrustes_equivalence():
    """Pose recovery must be the reference's estimator: the batched Kabsch
    fit is compared element-wise with the reference's per-fragment routine
    (``procrustes.py``: H = source_c^T target_c, R = V U^T with the
    determinant correction, t = target mean - source mean @ R^T)."""
    from hylofrac.baselines.common import matrix_to_quat, quat_to_rotmat
    from hylofrac.baselines.rpf_net import solve_procrustes

    def reference(source, target):  # one fragment at a time, as the reference
        source_mean = source.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)
        source_c = source - source_mean
        target_c = target - target_mean
        h = source_c.t() @ target_c
        u, _s, vt = torch.linalg.svd(h)
        r = vt.t() @ u.t()
        if torch.det(r) < 0:
            vt = vt.clone()
            vt[-1, :] *= -1.0
            r = vt.t() @ u.t()
        return r, (target_mean - source_mean @ r.t()).squeeze(0)

    torch.manual_seed(0)
    b, n = 4, 64
    source = torch.randn(b, n, 3)
    quat = torch.zeros(b, 4)
    quat[:, 0] = 1.0
    quat[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])       # half turn
    quat[2] = torch.tensor([0.5, 0.5, 0.5, 0.5])        # longest path
    rot = quat_to_rotmat(quat)
    target = torch.einsum("bnc,bdc->bnd", source, rot) \
        + torch.randn(b, 1, 3) * 0.1

    r_batch, t_batch = solve_procrustes(source, target)
    for i in range(b):
        r_ref, t_ref = reference(source[i], target[i])
        assert torch.allclose(r_batch[i], r_ref, atol=1e-5)
        assert torch.allclose(t_batch[i], t_ref, atol=1e-5)
        # and the fit is the pose that generated the target
        back = torch.einsum("nc,dc->nd", source[i], r_batch[i]) + t_batch[i]
        assert torch.allclose(back, target[i], atol=1e-5)
        assert matrix_to_quat(r_batch[i]).norm().item() == pytest.approx(
            1.0, abs=1e-5)


def test_rpf_sampler_equivalence():
    """The sampler must be the reference's loop: start from the noise x1 and
    step ``x_t <- x_t - dt * v`` from t = 1 to 0, resetting the anchor points
    to the assembled ones after every step (``sampler.py`` /
    ``_reset_anchor``, 50 Euler steps).  Re-running that loop by hand with
    the same initial noise reproduces ``predict`` exactly."""
    from hylofrac.baselines.common import matrix_to_quat
    from hylofrac.baselines.rpf_net import (INTEGRATION_STEPS, RpfPoseNet,
                                            solve_procrustes)

    torch.manual_seed(0)
    model = RpfPoseNet().eval()
    pcs = torch.rand(1, 2, 64, 3) - 0.5
    valids = torch.ones(1, 2)
    with torch.no_grad():
        quat, trans = model.predict(pcs, valids)

        # hand-written reference loop over the same model and noise
        code = model.encoder(pcs[valids > 0.5])
        x1 = torch.randn(1, 2, 64, 3, generator=torch.Generator().manual_seed(0))
        x0_anchor = pcs.clone()               # anchor pose = identity
        anchor = torch.zeros(1, 2)
        anchor[0, 0] = 1.0
        a_mask = (anchor > 0.5)[..., None, None]
        x_t = torch.where(a_mask, x0_anchor, x1)
        d_t = 1.0 / INTEGRATION_STEPS
        for step in range(INTEGRATION_STEPS):
            t = torch.full((1, 1), 1.0 - step * d_t)
            x_t = x_t - d_t * model(pcs, x_t, t, anchor, valids, code=code)
            x_t = torch.where(a_mask, x0_anchor, x_t)
        r, t = solve_procrustes(pcs.reshape(2, 64, 3), x_t.reshape(2, 64, 3))
        assert torch.allclose(quat, matrix_to_quat(r).reshape(1, 2, 4),
                              atol=1e-6)
        assert torch.allclose(trans, t.reshape(1, 2, 3), atol=1e-6)


@pytest.mark.parametrize("method", ["global", "diffassemble", "garf", "rpf"])
def test_validate_full_metric_set(packaged_scene, method):
    """Training-time validation evaluates with the full HyloFrac metric set
    through the unified inference path, for every prediction interface
    (direct forward, DDIM sampling, anchored flow matching)."""
    from hylofrac.baselines.train import build_model, validate

    data_root = os.path.abspath(os.path.join(packaged_scene, "..", "..", ".."))
    model = build_model(method)
    out = validate(model, data_root, "val", n_points=200,
                   device=torch.device("cpu"), num_scenes=10)
    for key in ("val_pa", "val_rmse_r_deg", "val_rmse_t", "val_sym_cd",
                "val_qpos", "val_adj_f1"):
        assert key in out, f"{method} validation misses {key}"
        assert np.isfinite(out[key])


def test_validate_skips_non_finite_predictions(packaged_scene):
    """A diverged model must degrade the record, not kill the run.

    Predictions of inf/nan used to reach ``chamfer_sq`` and raise
    "'x' must be finite" out of ``cKDTree.query``, aborting a training run
    hours in. They are now counted in ``val_skipped`` and logged.
    """
    from hylofrac.baselines.train import validate

    class _NonFiniteModel(torch.nn.Module):
        def forward(self, pcs, valids):
            b, p = pcs.shape[0], pcs.shape[1]
            quat = torch.zeros(b, p, 4)
            quat[..., 0] = 1.0
            return quat, torch.full((b, p, 3), float("nan"))

    data_root = os.path.abspath(os.path.join(packaged_scene, "..", "..", ".."))
    out = validate(_NonFiniteModel(), data_root, "val", n_points=64,
                   device=torch.device("cpu"), num_scenes=1)
    # the fixture holds one scored scene and it was skipped, so there is no
    # metric to average
    assert out == {"val_skipped": 1.0}


def test_validate_loader_pool_is_capped(packaged_scene, capsys):
    """The validation loader must not scale its worker pool with --workers:
    a worker holds a whole scene, so an unbounded pool is an OOM waiting for
    the first validation."""
    from hylofrac.baselines.train import VAL_MAX_WORKERS, validate

    class _ZeroModel(torch.nn.Module):
        def forward(self, pcs, valids):
            b, p = pcs.shape[0], pcs.shape[1]
            quat = torch.zeros(b, p, 4)
            quat[..., 0] = 1.0
            return quat, torch.zeros(b, p, 3)

    data_root = os.path.abspath(os.path.join(packaged_scene, "..", "..", ".."))
    out = validate(_ZeroModel(), data_root, "val", n_points=64,
                   device=torch.device("cpu"), num_scenes=1, num_workers=16)
    printed = capsys.readouterr().out
    assert f"workers={VAL_MAX_WORKERS}" in printed
    assert "val_pa" in out


@pytest.mark.parametrize("method", ["garf", "rpf"])
def test_predict_anchor_reorder(packaged_scene, method):
    """predict_learned must map the anchor permutation back to fragment
    order when the anchor is not fragment 0 (GARF and RPF both pin their
    anchor fragment to its GT pose during inference)."""
    from hylofrac.baselines.predict import predict_learned
    from hylofrac.baselines.train import build_model
    from hylofrac.eval.loader import SceneSample

    sample = SceneSample(packaged_scene, "000102166_v00", n_points=64)
    sample.areas = np.array([1.0, 2.0])  # anchor = frag_01
    model = build_model(method)

    def fake_predict(pcs, valids, anchor_quat, anchor_trans):
        # per-slot marker translations reveal the permuted input order
        n = int(valids.sum().item())
        t = torch.zeros(1, n, 3)
        t[0, :, 0] = torch.arange(n, dtype=torch.float)
        q = torch.zeros(1, n, 4)
        q[..., 0] = 1.0
        return q, t

    model.predict = fake_predict
    _, t = predict_learned(model, sample, torch.device("cpu"))
    # frag_01 is the anchor: the sampler saw order [1, 0], so its slot
    # marker 0 must come back to frag_01 and marker 1 to frag_00
    assert t[sample.names.index("frag_01"), 0] == 0.0
    assert t[sample.names.index("frag_00"), 0] == 1.0


def test_chamfer_loss_ignores_padded_slots():
    """Chamfer terms must only see real fragments: a bogus translation on a
    padded slot must leave every loss term unchanged (regression test for
    the zero-filled padded slots polluting the Chamfer mean)."""
    from hylofrac.baselines.common import GeometricLoss

    torch.manual_seed(0)
    b, p, n = 1, 4, 128
    pcs = torch.randn(b, p, n, 3)
    pcs[:, 2:] = 0.0  # padded slots carry the zero cloud, as in training
    valids = torch.zeros(b, p)
    valids[0, :2] = 1.0
    quat = torch.zeros(b, p, 4)
    quat[..., 0] = 1.0
    gt_quat = quat.clone()
    trans = torch.randn(b, p, 3) * 0.1
    gt_trans = torch.zeros(b, p, 3)

    loss_fn = GeometricLoss()

    def evaluate(trans_in):
        # same shape-Camfer subsample in both calls
        torch.manual_seed(0)
        return loss_fn(pcs, quat, trans_in, gt_quat, gt_trans, valids)

    base = evaluate(trans)
    bogus = trans.clone()
    bogus[0, 2] = torch.tensor([5.0, -3.0, 2.0])  # padded slot only
    perturbed = evaluate(bogus)
    for key in ("total", "trans", "rot", "pt_l2", "pt_cd", "shape_cd"):
        assert abs(float(base[key]) - float(perturbed[key])) < 1e-6, key

    # sanity: the test is sensitive - counting the padded slot as valid
    # fragments changes the Chamfer terms
    valids_all = torch.ones(b, p)
    all_valid = loss_fn(pcs, quat, trans, gt_quat, gt_trans, valids_all)
    assert abs(float(all_valid["pt_cd"]) - float(base["pt_cd"])) > 1e-6


def test_predict_loads_training_checkpoint(tmp_path):
    """Regression: the prediction entry point used to hand the whole training
    payload (model + optimizer + scheduler + epoch) to ``load_state_dict``,
    which fails with "Unexpected key(s) ... 'model', 'optimizer'".  Both the
    payload and a legacy bare state_dict must load."""
    import json

    from hylofrac.baselines.global_net import GlobalPoseNet
    from hylofrac.baselines.predict import load_checkpoint
    from hylofrac.baselines.train import save_checkpoint

    model = GlobalPoseNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    path = save_checkpoint(model, "global", 3, str(tmp_path), optimizer=opt,
                           scheduler=sched)
    loaded, meta = load_checkpoint(path, torch.device("cpu"))
    assert meta["method"] == "global" and meta["epoch"] == 3
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key

    bare = tmp_path / "bare.pt"
    torch.save(model.state_dict(), bare)
    (tmp_path / "bare.meta.json").write_text(
        json.dumps({"method": "global", "epoch": 0}), encoding="utf-8")
    loaded_bare, _ = load_checkpoint(str(bare), torch.device("cpu"))
    first = next(iter(model.state_dict()))
    assert torch.equal(loaded_bare.state_dict()[first],
                       model.state_dict()[first])


def test_train_smoke_rpf(packaged_scene, tmp_path):
    """The unified training entry point runs the RPF objective end to end:
    the ``loss_step`` branch is reached with the scene's anchor and a
    checkpoint plus a training log with the validation metrics is written."""
    import json
    import shutil

    from hylofrac.baselines.train import build_model, train

    data_root = os.path.abspath(os.path.join(packaged_scene, "..", "..", ".."))
    # the fixture packages a val split only; training reads a train split
    train_root = os.path.join(data_root, "train")
    if not os.path.isdir(train_root):
        shutil.copytree(os.path.join(data_root, "val"), train_root)
    out = str(tmp_path / "runs" / "rpf")
    train(build_model("rpf"), "rpf", data_root, out, epochs=1, lr=1e-3,
          physical_batch=1, accum=1, n_points=64, seed=0,
          device=torch.device("cpu"))
    for name in ("model_ep0000.pt", "model_ep0000.meta.json",
                 "model_latest.pt", "train_log.jsonl"):
        assert os.path.exists(os.path.join(out, name)), name
    with open(os.path.join(out, "train_log.jsonl"), encoding="utf-8") as fh:
        record = json.loads(fh.readline())
    assert record["epoch"] == 0 and np.isfinite(record["train_loss"])
    assert "val_pa" in record and np.isfinite(record["val_pa"])
