"""DiffAssemble baseline: graph diffusion over fragment poses.

Fragment point clouds are encoded by the equivariant VN-DGCNN backbone of
the reference implementation (Scarpellini et al., CVPR 2024,
iit-pavis/diffassemble); noisy poses (quaternion + translation) and the
diffusion step are fused into the node features and a transformer GNN
over the fully connected fragment graph predicts the clean poses (x0
prediction). Forward diffusion corrupts the translation with Gaussian
noise and the rotation by right-multiplying an isotropic Gaussian
rotation on SO(3) (IGSO3) onto the GT rotation powered by the cumulative
alpha; inference runs deterministic DDIM over the 300 diffusion steps
with a skip ratio of 10, starting from the identity rotation and zero
translation.

The noise schedule, the SO(3) geometry and the sampling updates follow
the reference code: rotation noise is drawn by inverse transform sampling
of the numeric IGSO3 axis-angle CDF, and every DDIM step recovers the
noise rotation in the log domain and composes the previous state as
R(x0)^sqrt(alpha_prev) @ R(eps)^sqrt(1 - alpha_prev).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from hylofrac.baselines.common import (exp_map, matrix_to_quat,
                                       per_part_chamfer, quat_to_rotmat,
                                       so3_scale)
from hylofrac.baselines.vn_layers import VNDGCNNEncoder

BETA_MIN = 0.0001
BETA_MAX = 0.02
T_STEPS = 300

_IGSO3_GRID: torch.Tensor | None = None


def alpha_bar(t: torch.Tensor) -> torch.Tensor:
    """Cumulative product of (1 - beta) up to diffusion step t."""
    betas = torch.linspace(BETA_MIN, BETA_MAX, T_STEPS, device=t.device)
    ab = torch.cumprod(1.0 - betas, dim=0)
    return ab[t.long().clamp(0, T_STEPS - 1)]


def _igso3_grid(device: torch.device) -> torch.Tensor:
    """Angle grid of the inverse-transform table (denser near 0)."""
    global _IGSO3_GRID
    if _IGSO3_GRID is None:
        _IGSO3_GRID = math.pi * torch.linspace(0.0, 1.0, 1000) ** 3
    return _IGSO3_GRID.to(device)


def _igso3_pdf(eps: torch.Tensor) -> torch.Tensor:
    """IGSO3 axis-angle density on the angle grid, weighted by the
    (1 - cos) / pi factor of the axis-angle parameterization; returns
    [1000, n]. Formula and weighting follow the reference implementation.
    """
    var = eps.reshape(-1).double().square().unsqueeze(0)   # [1, n]
    grid = _igso3_grid(eps.device)
    t = grid.double().unsqueeze(-1)                        # [1000, 1]
    vals = (math.sqrt(math.pi) * var ** -1.5 * torch.exp(var / 4)
            * torch.exp(-(t / 2) ** 2 / var)
            * (t - torch.exp(-math.pi ** 2 / var)
               * ((t - 2 * math.pi) * torch.exp(math.pi * t / var)
                  + (t + 2 * math.pi) * torch.exp(-math.pi * t / var)))
            / (2 * torch.sin(t / 2)))
    vals = vals.float()
    vals[torch.isinf(vals) | torch.isnan(vals)] = 0.0
    return vals * ((1 - grid.cos()) / math.pi).unsqueeze(-1)


def _igso3_cdf(eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trapezoid-rule CDF [999, n] of the axis-angle magnitude on its grid
    [999], per element of eps."""
    grid = _igso3_grid(eps.device)
    pdf = _igso3_pdf(eps)                                # [1000, n]
    pdf[0] = 0.0
    steps = (grid[1:] - grid[:-1]).unsqueeze(-1)
    cdf = ((pdf[:-1] + pdf[1:]) / 2 * steps).cumsum(0)
    return grid[1:], cdf / cdf[-1].unsqueeze(0)


def igso3_sample(eps: torch.Tensor, count: int, rng: torch.Generator
                 ) -> torch.Tensor:
    """Independent rotations [n, count, 3, 3] from IGSO3(eps_i) (identity
    mean), by inverse transform sampling of the numeric axis-angle CDF.
    Zero-spread elements draw the identity (the limit of IGSO3)."""
    flat = eps.reshape(-1)
    n = flat.numel()
    grid, cdf = _igso3_cdf(flat)                    # [999], [999, n]
    cdf = cdf.repeat_interleave(count, dim=1)       # [999, n * count]
    m = n * count
    axis = torch.randn(m, 3, device=eps.device, generator=rng)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u = torch.rand(m, device=eps.device, generator=rng)
    idx_1 = (cdf <= u.unsqueeze(0)).sum(dim=0)
    idx_0 = (idx_1 - 1).clamp(min=0)
    cdf_0 = cdf.gather(0, idx_0.unsqueeze(0))[0]
    cdf_1 = cdf.gather(0, idx_1.unsqueeze(0))[0]
    weight = ((u - cdf_0) / (cdf_1 - cdf_0).clamp(min=1e-6)).clamp(0, 1)
    angle = torch.lerp(grid[idx_0], grid[idx_1], weight)
    angle = torch.where(flat.repeat_interleave(count) == 0,
                        torch.zeros_like(angle), angle)
    rot = exp_map(axis * angle.unsqueeze(-1))
    return rot.reshape(*eps.shape, count, 3, 3)


class TransformerGNN(nn.Module):
    """4 transformer blocks over the fully connected fragment graph."""

    def __init__(self, dim: int, heads: int = 8, layers: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                       activation="gelu", dropout=0.0)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None
                ) -> torch.Tensor:
        """x [B, P, D]; mask [B, P] True for padding (excluded)."""
        mask = mask if mask is not None and mask.any() else None
        for block in self.blocks:
            x = block(x, src_key_padding_mask=mask)
        return x


class DiffAssemblePoseNet(nn.Module):
    """Graph diffusion model; the forward pass predicts the clean pose of
    every fragment given noisy poses and the time step."""

    def __init__(self, gnn_dim: int = 832, time_embed_dim: int = 32) -> None:
        super().__init__()
        self.encoder = VNDGCNNEncoder(20)
        feat_dim = 768
        self.pos_mlp = nn.Sequential(nn.Linear(7, 16), nn.GELU(),
                                     nn.Linear(16, 32))
        self.time_embed = nn.Embedding(T_STEPS, time_embed_dim)
        self.fuse = nn.Sequential(nn.Linear(feat_dim + 32 + time_embed_dim,
                                            256),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Linear(256, gnn_dim),
                                  nn.LeakyReLU(0.2, inplace=True))
        self.gnn = TransformerGNN(gnn_dim)
        self.mlp_t = nn.Sequential(nn.Linear(gnn_dim, 256), nn.GELU(),
                                   nn.Linear(256, 3))
        self.mlp_r = nn.Sequential(nn.Linear(gnn_dim, 256), nn.GELU(),
                                   nn.Linear(256, 3))

    def forward(self, pcs: torch.Tensor, pose: torch.Tensor, t: torch.Tensor,
                valids: torch.Tensor, code: torch.Tensor | None = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """pose [B, P, 7] = [quat(wxyz), trans]; t [B] diffusion steps.

        Valid fragments are packed through the encoder and the fully
        connected pose graph, then scattered back to the padded layout.
        ``code`` may carry precomputed fragment codes [V, feat_dim] (the
        sampler reuses them across DDIM steps). The rotation head outputs
        an axis-angle vector mapped through the exponential map (the
        reference's skew-to-rotation head); the GNN output is added to
        the fused input features before the heads.
        """
        b, p, _, _ = pcs.shape
        valid = valids > 0.5
        if code is None:
            code = self.encoder(pcs[valid].unsqueeze(1)).squeeze(1)
        pos_feat = self.pos_mlp(pose[valid])
        t_feat = self.time_embed(t.long()).repeat_interleave(
            valid.sum(dim=1), dim=0)
        fused = self.fuse(torch.cat([code, pos_feat, t_feat], dim=-1))
        node = self.gnn(fused, None)  # fully connected over valid fragments
        out = node + fused
        quat = matrix_to_quat(exp_map(self.mlp_r(out)))
        trans = self.mlp_t(out)
        out_q = torch.zeros(b, p, 4, device=pcs.device)
        out_t = torch.zeros(b, p, 3, device=pcs.device)
        out_q[valid] = quat
        out_t[valid] = trans
        return out_q, out_t

    def add_noise(self, quat0: torch.Tensor, trans0: torch.Tensor,
                  t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion; returns (noisy_quat, noisy_trans)."""
        b = quat0.shape[0]
        device = quat0.device
        rng = torch.Generator(device=device).manual_seed(
            int(t.sum().detach().item()) % 2 ** 31)
        ab = alpha_bar(t).sqrt()                      # sqrt(alpha_bar)
        bt = (1.0 - alpha_bar(t)).sqrt()              # per-sample spread
        # torch.randn_like() only takes a generator on newer releases; the
        # equivalent torch.randn(...) works on every torch version
        noise_t = torch.randn(trans0.shape, dtype=trans0.dtype,
                              device=trans0.device, generator=rng)
        noisy_t = ab.view(b, 1, 1) * trans0 + bt.view(b, 1, 1) * noise_t
        r0 = quat_to_rotmat(quat0)
        # q_t = R0^sqrt(alpha_bar) @ IGSO3(sqrt(1 - alpha_bar)) (right
        # multiplication, GT rotation powered by the cumulative alpha)
        r_eps = igso3_sample(bt, quat0.shape[1], rng)
        noisy_q = matrix_to_quat(so3_scale(r0, ab.view(b, 1)) @ r_eps)
        return noisy_q, noisy_t

    def loss_step(self, pcs: torch.Tensor, quat_gt: torch.Tensor,
                  trans_gt: torch.Tensor, valids: torch.Tensor,
                  w_trans: float = 1.0, w_rot: float = 0.2,
                  w_shape: float = 10.0) -> dict:
        """Sample t, add noise, predict x0, and compute the diffusion loss
        on the predicted clean poses (translation L2, rotation cosine,
        assembly Chamfer)."""
        b = pcs.shape[0]
        t = torch.randint(0, T_STEPS, (b,), device=pcs.device)
        noisy_q, noisy_t = self.add_noise(quat_gt, trans_gt, t)
        pose = torch.cat([noisy_q, noisy_t], dim=-1)
        pred_q, pred_t = self(pcs, pose, t, valids)
        loss_trans = ((pred_t - trans_gt).pow(2).sum(-1)
                      * valids).sum() / valids.sum().clamp(min=1.0)
        dot = (pred_q * quat_gt).sum(-1).abs()
        loss_rot = ((1.0 - dot) * valids).sum() / valids.sum().clamp(min=1.0)
        pred_r = quat_to_rotmat(pred_q)
        gt_r = quat_to_rotmat(quat_gt)
        pred_world = torch.einsum("bpnc,bpdc->bpnd", pcs, pred_r) \
            + pred_t[:, :, None, :]
        gt_world = torch.einsum("bpnc,bpdc->bpnd", pcs, gt_r) \
            + trans_gt[:, :, None, :]
        loss_shape = per_part_chamfer(pred_world, gt_world, valids)
        total = (w_trans * loss_trans + w_rot * loss_rot
                 + w_shape * loss_shape)
        return {"total": total, "trans": loss_trans, "rot": loss_rot,
                "shape_cd": loss_shape}

    @torch.no_grad()
    def predict(self, pcs: torch.Tensor, valids: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias of the DDIM sampler used by the unified predict entry."""
        return self.sample(pcs, valids)

    @torch.no_grad()
    def sample(self, pcs: torch.Tensor, valids: torch.Tensor,
               steps: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic DDIM sampling starting from identity poses.

        One model call every ``T_STEPS // steps`` diffusion steps; each
        step recovers the noise from the current sample and the predicted
        x0 and moves to the previous level (the last step, t = 0, takes
        the prediction directly, which is the alpha_prev = 1 limit of the
        update).
        """
        b, p = pcs.shape[:2]
        device = pcs.device
        valid = valids > 0.5
        code = self.encoder(pcs[valid].unsqueeze(1)).squeeze(1)
        quat = torch.zeros(b, p, 4, device=device)
        quat[..., 0] = 1.0
        trans = torch.zeros(b, p, 3, device=device)
        stride = T_STEPS // steps
        ab_all = torch.cumprod(1.0 - torch.linspace(
            BETA_MIN, BETA_MAX, T_STEPS, device=device), dim=0)
        for i in reversed(range(0, T_STEPS, stride)):
            t = torch.full((b,), i, device=device)
            pred_q, pred_t = self(pcs, torch.cat([quat, trans], dim=-1),
                                  t, valids, code=code)
            if i == 0:
                quat, trans = pred_q, pred_t
                break
            a_t, b_t = ab_all[i].sqrt(), (1.0 - ab_all[i]).sqrt()
            j = max(i - stride, 0)
            a_p, b_p = ab_all[j].sqrt(), (1.0 - ab_all[j]).sqrt()
            # translation: eps = (x_t - sqrt(alpha_t) x0) / sqrt(1 - alpha_t)
            eps_tr = (trans - a_t * pred_t) / b_t
            trans = a_p * pred_t + b_p * eps_tr
            # rotation: eps recovered in the log domain as
            # R(x_t)^(1 / b_t) @ R(x0)^(-a_t / b_t); the previous state is
            # R(x0)^a_p @ R(eps)^b_p
            eps_r = so3_scale(quat_to_rotmat(quat), 1.0 / b_t) @ so3_scale(
                quat_to_rotmat(pred_q), -a_t / b_t)
            prev_r = so3_scale(quat_to_rotmat(pred_q), a_p) @ so3_scale(
                eps_r, b_p)
            quat = matrix_to_quat(prev_r)
        return quat, trans
