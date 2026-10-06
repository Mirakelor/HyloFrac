"""Dense blocked point-transformer encoder (PTV3-style) for GARF.

Per-fragment point clouds are reordered along a voxel-lexicographic
(space-filling) order and refined by non-overlapping window attention:
every stage pads the sequence to whole windows of ``block_size`` tokens,
applies ``depth`` pre-LN window transformer blocks (self-attention +
FFN), and discards the padding; between stages the sequence is stride-2
downsampled so that later windows cover twice the spatial extent. Stage
channels follow the reference PointTransformerV3 configuration
(32/64/128, depths 2/2/6) and the coordinates are re-embedded at every
stage through a linear ``xyz`` embedding added to the tokens. A
projection stage lifts the tokens to 256 channels; mean pooling over the
last-stage tokens and a linear head yield the per-fragment code.

The windows are computed densely (plain batched attention over padded
windows with a key mask), which keeps the backbone free of sparse-kernel
dependencies while preserving the blocked receptive field of the
reference design.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as fn

VOXELS_PER_DIAGONAL = 32.0
# one axis spans at most VOXELS_PER_DIAGONAL + margin voxels
_VOXEL_BASE = 64


def _voxel_order(points: torch.Tensor) -> torch.Tensor:
    """Per-item space-filling order indices of [B, N, 3] point clouds.

    Coordinates are voxelized with one cell size per item (bounding-box
    diagonal / VOXELS_PER_DIAGONAL), packed into a single integer per
    point and sorted, so consecutive tokens are spatially close.
    """
    lo = points.min(dim=1, keepdim=True).values
    diag = (points.max(dim=1, keepdim=True).values - lo) \
        .norm(dim=-1, keepdim=True).clamp(min=1e-6)
    voxel = ((points - lo) / (diag / VOXELS_PER_DIAGONAL)).floor().long()
    ix, iy, iz = voxel.unbind(-1)
    # z has the highest place value: points sort by z, then y, then x
    key = (iz * _VOXEL_BASE + iy) * _VOXEL_BASE + ix
    return key.argsort(dim=-1, stable=True)


class _WindowBlock(nn.Module):
    """Pre-LN window transformer block (self-attention + FFN)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4),
                                 nn.SiLU(inplace=True),
                                 nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """x [B, W, C]; key_mask [B, W] True for padding tokens."""
        xn = self.norm1(x)
        h = self.attn(xn, xn, xn, key_padding_mask=key_mask,
                      need_weights=False)[0]
        x = x + h
        return x + self.ffn(self.norm2(x))


class _WindowStage(nn.Module):
    """Stage: xyz embedding, projection from the previous stage, blocks."""

    def __init__(self, in_dim: int | None, dim: int, depth: int,
                 block_size: int, heads: int) -> None:
        super().__init__()
        self.embed_xyz = nn.Linear(3, dim)
        self.in_proj = None if in_dim is None else nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList(
            [_WindowBlock(dim, heads) for _ in range(depth)])
        self.block_size = block_size

    def forward(self, x: torch.Tensor | None, pts: torch.Tensor
                ) -> torch.Tensor:
        """x [B, N, C] (None for the first stage) or features; pts [B, N, 3].

        Returns the refined tokens of the N points (padding removed).
        """
        tokens = self.embed_xyz(pts) if x is None \
            else self.in_proj(x) + self.embed_xyz(pts)
        b, n, c = tokens.shape
        block = self.block_size
        n_blocks = math.ceil(n / block)
        pad = n_blocks * block - n
        if pad:
            tokens = fn.pad(tokens, (0, 0, 0, pad), mode="replicate")
        tokens = tokens.reshape(b * n_blocks, block, c)
        key_mask = torch.zeros(b, n_blocks, block, dtype=torch.bool,
                               device=tokens.device)
        if pad:
            key_mask[:, -1, block - pad:] = True
        key_mask = key_mask.reshape(b * n_blocks, block)
        for blk in self.blocks:
            tokens = blk(tokens, key_mask)
        return tokens.reshape(b, n_blocks * block, c)[:, :n]


class BlockedPointEncoder(nn.Module):
    """Three-stride blocked transformer encoder for fragment point clouds.

    Stage sizes: channels 32/64/128 with 2/2/6 window layers and stride-2
    downsampling, followed by a 256-channel projection stage; mean pooling
    and a linear head map the tokens to the per-fragment code.
    """

    def __init__(self, feat_dim: int = 128, block_size: int = 64,
                 heads: int = 4) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.stages = nn.ModuleList([
            _WindowStage(None, 32, 2, block_size, heads),
            _WindowStage(32, 64, 2, block_size, heads),
            _WindowStage(64, 128, 6, block_size, heads),
        ])
        self.wide = nn.Linear(128, 256)
        self.head = nn.Linear(256, feat_dim)

    def forward(self, pcs: torch.Tensor) -> torch.Tensor:
        """pcs [B, N, 3] -> per-fragment codes [B, feat_dim]."""
        b = pcs.shape[0]
        order = _voxel_order(pcs)
        b_idx = torch.arange(b, device=pcs.device).view(b, 1)
        pts = pcs[b_idx, order]
        tokens = None
        for i, stage in enumerate(self.stages):
            tokens = stage(tokens, pts)
            if i < len(self.stages) - 1:
                pts = pts[:, 0::2]
                tokens = tokens[:, 0::2]
        return self.head(fn.silu(self.wide(tokens)).mean(dim=1))
