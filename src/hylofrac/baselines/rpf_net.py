"""RPF baseline: rectified point flow over fragment point clouds.

The flow formulation follows Sun et al., NeurIPS 2025 and its reference
code (GradientSpaces/Rectified-Point-Flow): the target x0 is the assembled
point cloud and the source x1 is Gaussian noise, the path is the straight
line x_t = (1 - t) x0 + t x1 with the constant velocity target x1 - x0,
training timesteps are drawn from the reference's u-shaped density
``asinh(u sinh a) / a`` (a = 4) clamped to [0.01, 1], and inference
integrates t from 1 to 0 with the reference config's
``inference_sampling_steps`` = 50 Euler steps, ``x_t <- x_t - dt * v``.  The
anchor fragment carries the assembled points as its input after every step
(the reference's ``_reset_anchor``) and a zero velocity target, the same
convention the GARF baseline of this repository uses for its anchored
fragment.  Fragment poses are recovered from the transported points by the
reference's SVD Procrustes fit (its ``solve_procrustes``: Kabsch with the
determinant correction, ``target ~ source @ R^T + t``).

House adaptation.  The reference conditions a dense point DiT on the whole
scene point set; the HyloFrac protocol feeds 1,000 points per fragment for
up to ``MAX_PARTS`` = 100 fragments (docs/benchmark.md, section 3), so the
per-point trunk is the shared blocked point-transformer encoder
(``blocked_point_encoder.py``, the same backbone the GARF baseline uses)
and global mixing runs over fragment tokens.  The velocity of a point is
produced by a point-wise head conditioned on its own path and observation
coordinates, the code and context of its fragment, and the timestep.  The
protocol supplies fragment point clouds without normals, so the reference's
normal conditioning is dropped.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as fn

from hylofrac.baselines.blocked_point_encoder import BlockedPointEncoder
from hylofrac.baselines.common import (MAX_PARTS, matrix_to_quat,
                                       rotate_point_clouds)

INTEGRATION_STEPS = 50
# reference _sample_timesteps: u-shaped density over [0, 1], clamped away
# from 0 to keep the velocity target finite
TIMESTEP_EPS = 0.01
TIMESTEP_A = 4.0


def solve_procrustes(source: torch.Tensor, target: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched Kabsch fit of ``target ~ source @ R^T + t``.

    ``source`` / ``target`` are [B, N, 3]; returns R [B, 3, 3] and t [B, 3].
    The cross-covariance is ``source_c^T @ target_c`` and the rotation is
    ``V U^T``, with the last row of ``V^T`` negated when det is negative --
    the reference solves one fragment at a time in exactly this order
    (``procrustes.py``), so the batched form here is the same estimator.
    """
    source_c = source - source.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    h = source_c.transpose(1, 2) @ target_c
    u, _s, vt = torch.linalg.svd(h)
    r = vt.transpose(1, 2) @ u.transpose(1, 2)
    flip = torch.det(r) < 0
    if bool(flip.any()):
        vt = vt.clone()
        vt[flip, -1, :] *= -1.0
        r = vt.transpose(1, 2) @ u.transpose(1, 2)
    t = target.mean(dim=1) - (source.mean(dim=1).unsqueeze(1)
                              @ r.transpose(1, 2)).squeeze(1)
    return r, t


def u_shaped_timesteps(batch: int, device: torch.device,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """Reference timestep sampling (``timestep_sampling: u_shaped``)."""
    u = torch.rand(batch, device=device, generator=generator) * 2.0 - 1.0
    u = torch.asinh(u * math.sinh(TIMESTEP_A)) / TIMESTEP_A
    return ((u + 1.0) / 2.0).clamp(TIMESTEP_EPS, 1.0)


def fourier_embed(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    """sin/cos features of the last dimension at ``2 ** i`` frequencies."""
    freq = 2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)
    arg = x.unsqueeze(-1) * freq
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1).flatten(-2)


class _TimeModulation(nn.Module):
    """Timestep embedding to (scale, shift); zero-initialized so the block
    starts unmodulated (reference DiT adaLN-Zero)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale, shift = torch.chunk(self.net(t_emb), 2, dim=-1)
        return scale, shift


class _FragmentBlock(nn.Module):
    """Pre-LN fragment-token block: attention modulated by the timestep
    embedding through (1 + scale) x + shift, then a GEGLU feed-forward
    network (the reference DiT block, applied to fragment tokens)."""

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
        h = self.norm1(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x + self.attn(h, h, h, key_padding_mask=mask,
                          need_weights=False)[0]
        h = self.norm2(x)
        return x + self.down(fn.gelu(self.gate(h)) * self.up(h))


class RpfPoseNet(nn.Module):
    """Rectified point flow with anchor injection."""

    def __init__(self, feat_dim: int = 128, hidden: int = 256,
                 layers: int = 4, heads: int = 8, point_dim: int = 128,
                 t_freq: int = 8) -> None:
        super().__init__()
        self.encoder = BlockedPointEncoder(feat_dim)
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.t_freq = t_freq
        self.part_embed = nn.Embedding(MAX_PARTS, hidden)
        self.anchor_embed = nn.Embedding(2, hidden)
        self.time_mlp = nn.Sequential(nn.Linear(2 * t_freq, hidden),
                                      nn.SiLU(inplace=True),
                                      nn.Linear(hidden, hidden))
        # fragment token: encoder code + current position in the path
        self.token_in = nn.Linear(feat_dim + 6 * t_freq, hidden)
        self.blocks = nn.ModuleList([
            _FragmentBlock(hidden, heads) for _ in range(layers)])
        self.ada_layers = nn.ModuleList(
            [_TimeModulation(hidden) for _ in range(layers)])
        # point head: path point, observation point, fragment context
        point_in = 3 + 3 + 6 * t_freq + 6 * t_freq + hidden
        self.point_mlp = nn.Sequential(
            nn.Linear(point_in, point_dim), nn.SiLU(inplace=True),
            nn.Linear(point_dim, point_dim), nn.SiLU(inplace=True),
            nn.Linear(point_dim, 3))

    def _fragment_context(self, pcs: torch.Tensor, x_t: torch.Tensor,
                          timestep: torch.Tensor, anchor: torch.Tensor,
                          valids: torch.Tensor, code: torch.Tensor,
                          ) -> torch.Tensor:
        """[B, P, hidden] context from the fragment codes and the current
        path positions, mixed over all fragments of the scene."""
        b, p = valids.shape
        part = torch.arange(p, device=pcs.device).clamp(max=MAX_PARTS - 1)
        t_emb = self.time_mlp(fourier_embed(timestep[:, :1], self.t_freq))
        node = self.token_in(torch.cat(
            [code, fourier_embed(x_t.mean(dim=2), self.t_freq)], dim=-1))
        node = node + self.part_embed(part)[None] \
            + self.anchor_embed(anchor.long().clamp(0, 1)) + t_emb[:, None]
        mask = valids < 0.5
        for block, ada in zip(self.blocks, self.ada_layers):
            scale, shift = ada(t_emb)
            node = block(node, mask, scale, shift)
        return node

    def forward(self, pcs: torch.Tensor, x_t: torch.Tensor,
                timestep: torch.Tensor, anchor: torch.Tensor,
                valids: torch.Tensor,
                code: torch.Tensor | None = None) -> torch.Tensor:
        """Velocity field at the path points ``x_t`` [B, P, N, 3].

        ``pcs`` [B, P, N, 3] are the observation clouds, ``timestep`` [B, 1]
        the flow time, ``anchor`` [B, P] the anchor flag.  ``code`` may carry
        precomputed fragment codes [V, feat_dim] (the sampler reuses them
        across integration steps).
        """
        b, p = valids.shape
        valid = valids > 0.5
        packed = code if code is not None else self.encoder(pcs[valid])
        code = torch.zeros(b, p, self.feat_dim, device=pcs.device)
        code[valid] = packed
        context = self._fragment_context(pcs, x_t, timestep, anchor, valids,
                                         code)
        feat = torch.cat([
            x_t, pcs,
            fourier_embed(x_t, self.t_freq),
            fourier_embed(pcs, self.t_freq),
            context[:, :, None, :].expand(-1, -1, x_t.shape[2], -1),
        ], dim=-1)
        return self.point_mlp(feat)

    def loss_step(self, pcs: torch.Tensor, quat_gt: torch.Tensor,
                  trans_gt: torch.Tensor, valids: torch.Tensor,
                  anchor: torch.Tensor | None = None) -> dict:
        """Supervise the rectified-flow velocity at a sampled timestep.

        The anchor fragment keeps its assembled points as the input and
        carries a zero velocity target, as in the reference.
        """
        if anchor is None:
            anchor = torch.zeros_like(valids)
            anchor[:, 0] = 1.0  # first fragment is the anchor by default
        device = pcs.device
        x0 = rotate_point_clouds(pcs, quat_gt, trans_gt)
        rng = torch.Generator(device=device)
        rng.manual_seed(int(torch.randint(0, 2 ** 31, (1,),
                                          device=device).item()))
        t = u_shaped_timesteps(pcs.shape[0], device, rng).view(-1, 1, 1, 1)
        x1 = torch.randn_like(x0)
        x_t = (1.0 - t) * x0 + t * x1
        v_gt = x1 - x0
        a_mask = (anchor > 0.5)[..., None, None]
        x_t = torch.where(a_mask, x0, x_t)
        v_gt = v_gt.masked_fill(a_mask, 0.0)

        v_pred = self(pcs, x_t, t.reshape(-1, 1), anchor, valids)
        point_valid = (valids > 0.5)[..., None].expand_as(v_gt[..., 0])
        loss = ((v_pred - v_gt).pow(2).sum(-1) * point_valid).sum() \
            / point_valid.sum().clamp(min=1.0)
        return {"total": loss, "flow_mse": loss}

    @torch.no_grad()
    def predict(self, pcs: torch.Tensor, valids: torch.Tensor,
                anchor_quat: torch.Tensor | None = None,
                anchor_trans: torch.Tensor | None = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate t from 1 to 0 and fit one rigid pose per fragment.

        The anchor fragment (index 0) is pinned to the assembled points of
        the provided GT pose (or to the identity pose) after every step, as
        the reference resets it.  Poses are then recovered by the batched
        Procrustes fit of the observation cloud to the transported cloud.
        """
        b, p, n, _ = pcs.shape
        device = pcs.device
        valid = valids > 0.5
        code = self.encoder(pcs[valid])
        rng = torch.Generator(device=device).manual_seed(0)
        x_t = torch.randn(b, p, n, 3, device=device, generator=rng)

        anchor = torch.zeros(b, p, device=device)
        anchor[:, 0] = 1.0
        a_mask = (anchor > 0.5)[..., None, None]
        # slot 0 carries the anchor pose given by the caller (the unified
        # inference path passes a single [1, 4] / [1, 3] pose, garf_net.py
        # reads it the same way); every other slot stays at the identity
        anchor_q = torch.zeros(b, p, 4, device=device)
        anchor_q[..., 0] = 1.0
        anchor_t = torch.zeros(b, p, 3, device=device)
        if anchor_quat is not None:
            anchor_q[:, 0] = anchor_quat.reshape(b, -1, 4)[:, 0]
        if anchor_trans is not None:
            anchor_t[:, 0] = anchor_trans.reshape(b, -1, 3)[:, 0]
        x0_anchor = rotate_point_clouds(pcs, anchor_q, anchor_t)
        x_t = torch.where(a_mask, x0_anchor, x_t)

        d_t = 1.0 / INTEGRATION_STEPS
        for step in range(INTEGRATION_STEPS):
            timestep = torch.full((b, 1), 1.0 - step * d_t, device=device)
            v = self(pcs, x_t, timestep, anchor, valids, code=code)
            x_t = x_t - d_t * v
            x_t = torch.where(a_mask, x0_anchor, x_t)

        flat_pcs = pcs.reshape(b * p, n, 3)
        flat_pred = x_t.reshape(b * p, n, 3)
        r, t = solve_procrustes(flat_pcs, flat_pred)
        return matrix_to_quat(r).reshape(b, p, 4), t.reshape(b, p, 3)
