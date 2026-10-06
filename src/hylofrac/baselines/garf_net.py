"""GARF baseline: SE(3) flow matching with an anchored reference fragment.

The flow-matching machinery follows Li et al., ICCV 2025 and its
reference code (ai4ce/GARF): translation noise is Gaussian with the
schedule x_t = (1 - sigma) x0 + sigma z and the velocity target z - x0;
the rotation noise is a uniform random rotation R1 and the velocity
target is the axis-angle vector of R1 (the reference composes the path as
R_t = exp(sigma * v) @ R0, left multiplication). Training draws one sigma
per scene, uniformly from {1/1000, ..., 1}; the anchored fragment keeps
its GT pose as the input with a zero velocity target (the loss is
computed over all valid fragments). Inference integrates sigma from 1 to
0 over 20 Euler steps and pins the anchor to its GT pose after every
step; the model is conditioned on the timestep 1000 * sigma.

Per-fragment point codes are produced by a dense blocked point-transformer
encoder in the style of PointTransformerV3 (blocked window attention with
stride downsampling; see blocked_point_encoder.py). The denoiser follows
the reference architecture: Fourier pose embedding (10 log-spaced
frequencies, input pose included), a reference-fragment embedding, and
pre-LN transformer blocks whose attention is modulated by the sigma
embedding through (1 + scale) x + shift (zero-initialized, so the blocks
start unmodulated) with a GEGLU feed-forward network.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn

from hylofrac.baselines.blocked_point_encoder import BlockedPointEncoder
from hylofrac.baselines.common import (exp_map, matrix_to_quat,
                                       quat_to_rotmat)

INTEGRATION_STEPS = 20


def axis_angle_of(r: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector of a rotation matrix [..., 3, 3], as the reference
    computes it.

    The reference (ai4ce/GARF) calls pytorch3d's ``matrix_to_axis_angle``,
    i.e. ``quaternion_to_axis_angle(matrix_to_quaternion(R))``: going through
    the quaternion selects the branch by the largest component, which stays
    well conditioned at a half turn.  pytorch3d canonicalises the quaternion
    sign on the way: its ``matrix_to_quaternion`` never returns a negative
    scalar part, so the axis-angle magnitude stays at most pi.  Measured
    against pytorch3d 0.7.8 over 4,000 uniform rotations: 0/4000 come back
    with w < 0 and ``|log R|`` peaks at 3.1412 with mean 2.2127.  The sign is
    fixed here for the same reason: the branch selector below does *not*
    canonicalise it (it returns w < 0 for 36.9% of uniform rotations, giving
    |v| up to 4.665, i.e. above pi for the same 36.9%), and a target that
    flips between the short and the long way round for the same rotation
    makes the flow-matching supervision discontinuous -- the loss then
    diverges within the first optimiser steps.

    The naive acos + antisymmetric form this started from degenerates at an
    exact pi rotation (target 0 instead of pi * axis) and, in float32, loses
    ~3e-4 of relative accuracy within 0.01 rad of pi; the skew-based
    ``log_rmat`` twin is also ill conditioned for |theta - pi| < 1e-4.
    """
    q = _matrix_to_quaternion_p3d(r)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    half = torch.atan2(v.norm(dim=-1, keepdim=True), q[..., :1])
    # pytorch3d: quaternions[1:] / (0.5 * sinc(half / pi))
    ratio = 0.5 * torch.sinc(half / math.pi)
    return v / ratio.clamp(min=1e-12)


def _matrix_to_quaternion_p3d(matrix: torch.Tensor) -> torch.Tensor:
    """pytorch3d's ``matrix_to_quaternion`` (rotation_conversions.py).

    It builds all four branch candidates and picks the one with the largest
    ``sqrt(1 + ...)`` (§ the best-conditioned branch); unlike pytorch3d's own
    routine it does not canonicalise the sign, so w < 0 comes back for 36.9%
    of 4,000 uniform rotations (measured; the raw sampled quaternions are
    sign-symmetric, 49.3%).  ``axis_angle_of`` therefore flips the sign
    before taking the axis-angle, which keeps the target inside the
    principal branch.  Our shared
    ``matrix_to_quat`` prefers the trace branch whenever trace > 0, so the
    target uses this variant to follow the reference's branch selection.
    """
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1)
    stack = torch.stack([
        1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1)
    q_abs = torch.where(stack > 0, stack.sqrt(),
                        torch.zeros_like(stack))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20,
                     m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01,
                     m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2,
                     m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12,
                     q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    picked = torch.nn.functional.one_hot(q_abs.argmax(dim=-1), 4).bool()
    return quat_candidates[picked, :].reshape(*batch_dim, 4)


def fourier_embed(pos: torch.Tensor, n_freq: int = 10) -> torch.Tensor:
    """Fourier features of a 7-dim pose, input included: dims = 7 * (1 + 2 * n_freq)
    with the 2^i log-spaced frequencies of the reference."""
    freq = 2.0 ** torch.arange(n_freq, device=pos.device)
    arg = pos.unsqueeze(-1) * freq.view(1, 1, 1, -1)
    code = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1).flatten(-2)
    return torch.cat([pos, code], dim=-1)


class _AdaLNAttnBlock(nn.Module):
    """Pre-LN attention modulated by (1 + scale) x + shift, then an
    unmodulated GEGLU feed-forward network (reference denoiser block)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.gate = nn.Linear(dim, dim * 4)
        self.up = nn.Linear(dim, dim * 4)
        self.down = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.attn(h, h, h, key_padding_mask=mask,
                          need_weights=False)[0]
        h = self.norm2(x)
        return x + self.down(fn.gelu(self.gate(h)) * self.up(h))


class _TimeModulation(nn.Module):
    """Sigma embedding to (scale, shift); zero-initialized output so the
    block starts as an unmodulated pre-LN transformer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, shift = torch.chunk(self.net(t_emb), 2, dim=-1)
        return scale, shift


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int,
                                   flip_sin_to_cos: bool = True,
                                   downscale_freq_shift: float = 0.0,
                                   scale: float = 1.0,
                                   max_period: int = 10000) -> torch.Tensor:
    """diffusers' ``get_timestep_embedding`` (pure torch).

    The reference conditions its AdaLN modulation on
    ``Timesteps(num_channels=256, flip_sin_to_cos=True,
    downscale_freq_shift=0)`` followed by ``TimestepEmbedding``.  Feeding the
    raw scalar (``1000 * sigma``) into a ``Linear`` instead leaves activations
    of several hundred and makes the modulation path explode: the first Adam
    steps already drive the (zero-initialised) scale/shift far enough to blow
    the loss up.
    """
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)
    if dim % 2 == 1:
        emb = fn.pad(emb, (0, 1, 0, 0))
    return emb


class _TimestepEmbedding(nn.Module):
    """diffusers' ``TimestepEmbedding``: Linear -> SiLU -> Linear."""

    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class _Timesteps(nn.Module):
    """diffusers' ``Timesteps`` with the reference's settings."""

    def __init__(self, num_channels: int = 256, flip_sin_to_cos: bool = True,
                 downscale_freq_shift: float = 0.0) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return _sinusoidal_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift)


class GarfPoseNet(nn.Module):
    """SE(3) flow-matching denoiser with anchor injection."""

    def __init__(self, feat_dim: int = 128, hidden: int = 256,
                 layers: int = 6, heads: int = 8,
                 pose_freq: int = 10) -> None:
        super().__init__()
        self.encoder = BlockedPointEncoder(feat_dim)
        self.feat_dim = feat_dim
        pose_dim = 7 * (1 + 2 * pose_freq)
        self.pose_mlp = nn.Sequential(nn.Linear(pose_dim, hidden),
                                      nn.SiLU(inplace=True))
        # reference conditioning: Timesteps(256) -> TimestepEmbedding -> SiLU
        self.time_proj = _Timesteps(num_channels=256, flip_sin_to_cos=True,
                                    downscale_freq_shift=0.0)
        self.time_embed = _TimestepEmbedding(256, hidden)
        self.time_act = nn.SiLU()
        self.ref_embed = nn.Embedding(2, hidden)
        self.in_proj = nn.Linear(feat_dim + hidden, hidden)
        self.blocks = nn.ModuleList([
            _AdaLNAttnBlock(hidden, heads) for _ in range(layers)])
        self.ada_layers = nn.ModuleList(
            [_TimeModulation(hidden) for _ in range(layers)])
        self.mlp_t = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(inplace=True),
                                   nn.Linear(256, 3))
        self.mlp_r = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(inplace=True),
                                   nn.Linear(256, 3))
        self.pose_freq = pose_freq

    def forward(self, pcs: torch.Tensor, pose: torch.Tensor,
                timestep: torch.Tensor, anchor: torch.Tensor,
                valids: torch.Tensor,
                code: torch.Tensor | None = None) -> torch.Tensor:
        """pose [B, P, 7] (quat wxyz + trans) at ``timestep`` = 1000 * sigma.

        ``code`` may carry precomputed packed fragment codes [V, D] (the
        sampler reuses them across integration steps). Returns the
        predicted velocity [B, P, 6] = [trans(3), rot(3)].
        """
        b, p, _, _ = pcs.shape
        valid = valids > 0.5
        packed = code if code is not None else self.encoder(pcs[valid])
        code = torch.zeros(b, p, self.feat_dim, device=pcs.device)
        code[valid] = packed
        pose_feat = self.pose_mlp(fourier_embed(pose, self.pose_freq))
        ref_feat = self.ref_embed(anchor.long())
        node = self.in_proj(torch.cat([code, pose_feat + ref_feat], dim=-1))
        mask = valids < 0.5
        # per-scene conditioning (the reference embeds the timestep, not the
        # raw sigma: a Linear on 1000*sigma leaves ~4e2 activations)
        t_emb = self.time_act(self.time_embed(self.time_proj(timestep[:, 0])))
        for block, ada in zip(self.blocks, self.ada_layers):
            scale, shift = ada(t_emb)
            node = block(node, mask, scale, shift)
        return torch.cat([self.mlp_t(node), self.mlp_r(node)], dim=-1)

    @staticmethod
    def _uniform_quat(b: int, p: int, rng: torch.Generator) -> torch.Tensor:
        u = torch.rand(b, p, 3, device=rng.device, generator=rng)
        q = torch.stack([
            (1 - u[..., 0]).sqrt() * torch.sin(2 * math.pi * u[..., 1]),
            (1 - u[..., 0]).sqrt() * torch.cos(2 * math.pi * u[..., 1]),
            u[..., 0].sqrt() * torch.sin(2 * math.pi * u[..., 2]),
            u[..., 0].sqrt() * torch.cos(2 * math.pi * u[..., 2]),
        ], dim=-1)
        return q

    def add_noise(self, quat_gt: torch.Tensor, trans_gt: torch.Tensor,
                  anchor: torch.Tensor, rng: torch.Generator
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one sigma per scene from {1/1000, ..., 1} and corrupt
        the poses; returns x_sigma, v_target, timestep (1000 * sigma).

        The anchored fragments keep their GT pose as the input and carry
        a zero velocity target, as in the reference.
        """
        b, p = quat_gt.shape[:2]
        device = quat_gt.device
        sigma = torch.randint(1, 1001, (b, 1, 1), device=device,
                              generator=rng) / 1000.0

        noise_t = torch.randn(b, p, 3, device=device, generator=rng)
        x_t = (1 - sigma) * trans_gt + sigma * noise_t
        v_t = noise_t - trans_gt

        r0 = quat_to_rotmat(quat_gt)
        r1 = quat_to_rotmat(self._uniform_quat(b, p, rng))
        v_r = axis_angle_of(r1)                   # reference target: log R1
        r_t = exp_map(sigma * v_r) @ r0           # left multiplication
        q_t = matrix_to_quat(r_t)
        x_t = torch.where((anchor > 0.5).unsqueeze(-1), trans_gt, x_t)
        q_t = torch.where((anchor > 0.5).unsqueeze(-1), quat_gt, q_t)
        pose_t = torch.cat([q_t, x_t], dim=-1)
        v_gt = torch.cat([v_t, v_r], dim=-1)
        v_gt = v_gt * (1.0 - anchor.unsqueeze(-1).float())
        timestep = (1000.0 * sigma).reshape(b, 1).expand(b, p)
        return pose_t, v_gt, timestep

    def loss_step(self, pcs: torch.Tensor, quat_gt: torch.Tensor,
                  trans_gt: torch.Tensor, valids: torch.Tensor,
                  anchor: torch.Tensor | None = None) -> dict:
        """Sample sigma and supervise the velocity prediction (MSE) over
        all valid fragments."""
        if anchor is None:
            anchor = torch.zeros_like(valids)
            anchor[:, 0] = 1.0  # first fragment is the anchor by default
        rng = torch.Generator(device=pcs.device)
        rng.manual_seed(int(torch.randint(0, 2 ** 31, (1,),
                                          device=pcs.device).item()))
        pose_t, v_gt, timestep = self.add_noise(quat_gt, trans_gt,
                                                anchor, rng)
        v_pred = self(pcs, pose_t, timestep, anchor, valids)
        loss = ((v_pred - v_gt).pow(2).sum(-1) * valids).sum() \
            / valids.sum().clamp(min=1.0)
        return {"total": loss, "flow_mse": loss}

    @torch.no_grad()
    def predict(self, pcs: torch.Tensor, valids: torch.Tensor,
                anchor_quat: torch.Tensor | None = None,
                anchor_trans: torch.Tensor | None = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate sigma from 1 to 0 in 20 Euler steps.

        The anchor fragment (index 0) is pinned to the provided GT pose
        (used at evaluation) or to the identity pose; it is re-pinned
        after every step, as in the reference.
        """
        b, p = pcs.shape[:2]
        device = pcs.device
        code = self.encoder(pcs[valids > 0.5])
        rng = torch.Generator(device=device).manual_seed(0)
        trans = torch.randn(b, p, 3, device=device, generator=rng)
        quat = self._uniform_quat(b, p, rng)
        anchor_q = (anchor_quat if anchor_quat is not None
                    else torch.zeros(b, p, 4, device=device))
        anchor_t = (anchor_trans if anchor_trans is not None
                    else torch.zeros(b, p, 3, device=device))
        if anchor_quat is None:
            anchor_q[:, 0, 0] = 1.0
        anchor = torch.zeros(b, p, device=device)
        anchor[:, 0] = 1.0
        a_mask = anchor.unsqueeze(-1)

        d_sigma = -1.0 / INTEGRATION_STEPS
        for step in range(INTEGRATION_STEPS):
            sigma = 1.0 - step / INTEGRATION_STEPS
            timestep = torch.full((b, p), 1000.0 * sigma, device=device)
            pose = torch.cat([quat, trans], dim=-1)
            v = self(pcs, pose, timestep, anchor, valids, code=code)
            trans = trans + d_sigma * v[..., :3]
            quat = matrix_to_quat(exp_map(d_sigma * v[..., 3:])
                                  @ quat_to_rotmat(quat))
            # re-anchor to the GT pose
            quat = quat * (1 - a_mask) + anchor_q * a_mask
            quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            trans = trans * (1 - a_mask) + anchor_t * a_mask
        return quat, trans
