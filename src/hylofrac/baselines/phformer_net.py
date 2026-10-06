"""PHFormer baseline: proxy-level hybrid transformer.

Fragments are encoded by a PointNet2-style set-abstraction encoder
(FPS + ball-query grouping with per-cloud normalized radii) whose
last-stage centers act as proxy tokens per fragment; the grouping and
set-abstraction semantics follow the reference implementation
(Cui et al., AAAI 2024, 521piglet/PHFormer): FPS starts at point 0, the
ball query runs over per-cloud max-norm normalized coordinates and keeps
the first ``nsample`` inliers in scan order (undersized balls repeat the
first inlier), and every group feature is the concatenation of the group
and its difference to the center features. A hybrid transformer
alternates intra-fragment (self) and inter-fragment (cross) attention;
the self layers add a sinusoidal position encoding of the proxy centers
in voxel coordinates (126 dims over 21 sin/cos frequencies per axis,
zero-padded to 128), the cross layers mask out within-fragment pairs.
An adjacency-aware hierarchical head then predicts pairwise adjacency
and relative poses, propagates messages over the weighted graph, and
regresses absolute poses through the reference pose regressor (shared
MLP plus separate rotation and translation heads).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from hylofrac.baselines.common import _ConvMLP, PoseRegressor


def farthest_point_sample(points: torch.Tensor, n_centers: int) -> torch.Tensor:
    """Farthest point sampling starting at point 0; returns [B, M]."""
    b, n, _ = points.shape
    centers = torch.zeros(b, n_centers, dtype=torch.long, device=points.device)
    dist = torch.full((b, n), float("inf"), device=points.device)
    for i in range(1, n_centers):
        idx = centers[:, i - 1].reshape(b, 1, 1)
        d = ((points - points.gather(1, idx.expand(b, n, 3))) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        centers[:, i] = dist.argmax(dim=-1)
    return centers


def ball_query(points: torch.Tensor, centers_idx: torch.Tensor, radius: float,
               nsample: int) -> torch.Tensor:
    """Neighborhood indices of the first ``nsample`` inliers in scan
    order inside a ball of ``radius`` around every center.

    points [B, N, 3], centers_idx [B, M] -> indices [B, M, nsample].
    Distances are computed on per-cloud max-norm normalized coordinates
    (the radius is relative to the extent of every cloud), as in the
    reference QueryAndGroup. Undersized balls repeat their first inlier,
    as the reference CUDA kernel does; the center itself is always an
    inlier (distance 0).
    """
    b, n, _ = points.shape
    m = centers_idx.shape[1]
    scale = points.norm(dim=-1).max(dim=-1).values
    norm_pts = points / scale.view(b, 1, 1)
    ctr = norm_pts.gather(1, centers_idx.unsqueeze(-1).expand(b, m, 3))
    d2 = ((norm_pts.unsqueeze(1) - ctr.unsqueeze(2)) ** 2).sum(-1)  # [B,M,N]
    inside = d2 < radius ** 2
    pos = torch.where(inside,
                      torch.arange(n, device=points.device).view(1, 1, n),
                      torch.full((), n, device=points.device))
    vals, idx = pos.topk(nsample, dim=-1, largest=False)
    first = inside.long().argmax(dim=-1, keepdim=True)
    return torch.where(vals >= n, first.expand_as(idx), idx)


class SetAbstraction(nn.Module):
    """FPS downsampling, ball-query grouping and an MLP with max pooling.

    The group features are the relative coordinates (plus the grouped
    features from the second stage on) concatenated with their difference
    to the center features, doubling the input channels as in the
    reference edge-conv set abstraction.
    """

    def __init__(self, npoint: int, radius: float, nsample: int,
                 channels: tuple[int, int, int]) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        in_dim = (channels[0] + 3) * 2
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, channels[1], 1, bias=False),
            nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], 1, bias=False),
            nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True))

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """xyz [B, N, 3], features [B, N, C] or None (first stage).

        Returns the sampled centers [B, M, 3] and pooled features
        [B, M, C_out].
        """
        b, n, _ = xyz.shape
        centers_idx = farthest_point_sample(xyz, self.npoint)
        centers = xyz.gather(1, centers_idx.unsqueeze(-1).expand(
            b, self.npoint, 3))
        group = ball_query(xyz, centers_idx, self.radius, self.nsample)
        b_idx = torch.arange(b, device=xyz.device).view(b, 1, 1)
        rel = xyz[b_idx, group] - centers.unsqueeze(2)  # [B, M, ns, 3]
        if features is None:
            nf = rel
            # center reference: the absolute center coordinates
            ori = centers.unsqueeze(2)
        else:
            nf = torch.cat([rel, features[b_idx, group]], dim=-1)
            ori = torch.cat([centers, features.gather(
                1, centers_idx.unsqueeze(-1).expand(b, self.npoint,
                                                    features.shape[-1]))],
                dim=-1).unsqueeze(2)
        edge = torch.cat([nf, nf - ori], dim=-1)       # [B, M, ns, 2(C+3)]
        pooled = self.mlp(edge.permute(0, 3, 1, 2)).max(dim=-1).values
        return centers, pooled.transpose(1, 2)


class PointNet2Encoder(nn.Module):
    """Three-stage set abstraction; the final centers are the proxies."""

    def __init__(self, feat_dim: int = 128, n_proxies: int = 32) -> None:
        super().__init__()
        self.sa1 = SetAbstraction(256, 0.2, 50, (0, 64, 128))
        self.sa2 = SetAbstraction(64, 0.3, 25, (128, 128, 256))
        self.sa3 = SetAbstraction(n_proxies, 0.4, 10, (256, 512, feat_dim))

    def forward(self, pcs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pcs [B, N, 3] -> proxy features [B, M, D], centers [B, M, 3]."""
        pts, feat = self.sa1(pcs, None)
        pts, feat = self.sa2(pts, feat)
        return self.sa3(pts, feat)


class VolumetricPositionEncoding(nn.Module):
    """Sinusoidal encoding of proxy-center coordinates in voxel units.

    Per axis (dim // 3) // 2 = 21 sin/cos frequency pairs for dim 128,
    whose frequencies decay geometrically over the grid (the reference
    takes every other frequency of the per-axis grid and divides by
    dim // 3, i.e. exp(-ln 10000 * m / 21)); the 126-dim code is
    zero-padded to ``dim`` and added to the tokens of the self layers.
    """

    def __init__(self, dim: int, voxel_size: float = 0.005,
                 origin: tuple[float, float, float] = (-0.5, -0.5, -0.5)
                 ) -> None:
        super().__init__()
        per_axis = dim // 3
        n_freq = per_axis // 2
        freq = torch.exp(-math.log(10000.0)
                         * torch.arange(n_freq, dtype=torch.float) / n_freq)
        self.register_buffer("freq", freq.view(1, 1, -1))
        self.register_buffer("origin", torch.tensor(origin).view(1, 1, 3))
        self.voxel_size = voxel_size
        self.pad_dim = dim - 6 * n_freq

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz [..., M, 3] -> encoding [..., M, dim] (zero-padded)."""
        vox = (xyz - self.origin) / self.voxel_size          # [..., M, 3]
        phase = vox.unsqueeze(-1) * self.freq                # [..., M, 3, F]
        code = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        code = code.reshape(*code.shape[:-2], -1)            # [..., M, 6F]
        if self.pad_dim:
            zeros = code.new_zeros(*code.shape[:-1], self.pad_dim)
            code = torch.cat([code, zeros], dim=-1)
        return code


class TransformerLayer(nn.Module):
    """Post-LN transformer block with FFN, applied over proxy tokens."""

    def __init__(self, dim: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True),
                                 nn.Linear(ffn_dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, x, x, attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=False)[0])
        return self.norm2(x + self.ffn(x))


class GlobalEncoder(nn.Module):
    """Per-fragment MLP whose valids-weighted mean is the scene code."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = _ConvMLP([dim, dim * 2, dim])

    def forward(self, node: torch.Tensor, valids: torch.Tensor
                ) -> torch.Tensor:
        enc = self.mlp(node)                                  # [B, P, C]
        wsum = (enc * valids.unsqueeze(-1)).sum(dim=1,
                                                keepdim=True)  # [B, 1, C]
        denom = valids.sum(dim=1).clamp(min=1.0)               # [B]
        return wsum / denom.unsqueeze(-1).unsqueeze(-1)


class RelativePoseEstimator(nn.Module):
    """Adjacency and relative-pose heads over ordered pairs, with message
    passing over the adjacency-weighted graph."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.edge_encoder = _ConvMLP([2 * dim, dim, dim])
        self.adj_predictor = _ConvMLP([dim, dim // 2, 1])
        self.rel_regressor = PoseRegressor(dim)
        self.merge = _ConvMLP([2 * dim, dim, dim])

    def forward(self, node: torch.Tensor, valids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor]:
        """node [B, P, C] -> (rel_rot, rel_trans, new_node, adj_scores)."""
        b, p, c = node.shape
        pair = torch.cat([node.unsqueeze(2).expand(b, p, p, c),
                          node.unsqueeze(1).expand(b, p, p, c)], dim=-1)
        edge = self.edge_encoder(pair)                     # [B, P, P, C]
        adj_logit = self.adj_predictor(edge).squeeze(-1)   # [B, P, P]
        adj_scores = torch.sigmoid(adj_logit)
        rel_rot, rel_trans = self.rel_regressor(edge)
        mask = (valids.unsqueeze(1) * valids.unsqueeze(2)).unsqueeze(-1)
        relation = adj_scores.unsqueeze(-1) * mask         # [B, P, P, 1]
        msg = (edge * relation).sum(dim=2) \
            / relation.sum(dim=2).clamp(min=1e-6)
        new_node = self.merge(torch.cat([node, msg], dim=-1))
        return rel_rot, rel_trans, new_node, adj_scores


class PhformerPoseNet(nn.Module):
    """Hybrid transformer with hierarchical adjacency-aware pose head."""

    def __init__(self, feat_dim: int = 128, n_proxies: int = 32,
                 heads: int = 4, layers: int = 4, ffn_dim: int = 256,
                 refine_steps: int = 1) -> None:
        super().__init__()
        self.encoder = PointNet2Encoder(feat_dim, n_proxies)
        self.n_proxies = n_proxies
        self.feat_dim = feat_dim
        self.refine_steps = refine_steps
        self.voxel_pe = VolumetricPositionEncoding(feat_dim)

        self.layers = nn.ModuleList([
            TransformerLayer(feat_dim, heads, ffn_dim) for _ in range(layers)
        ])
        self.layer_kind = ["self", "cross"] * (layers // 2)

        self.rel_estimators = nn.ModuleList(
            [RelativePoseEstimator(feat_dim) for _ in range(refine_steps)])
        self.g_encoders = nn.ModuleList(
            [GlobalEncoder(feat_dim) for _ in range(refine_steps)])
        self.pose_predictors = nn.ModuleList(
            [PoseRegressor(feat_dim + 7) for _ in range(refine_steps)])

    def forward(self, pcs: torch.Tensor, valids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """pcs [B, MAX_PARTS, N, 3], valids [B, MAX_PARTS].

        Valid fragments are packed, encoded and refined together, then
        scattered back. Returns (quat, trans) for every fragment slot.
        """
        b, p, _, _ = pcs.shape
        valid = valids > 0.5

        centers, feat = self.encoder(pcs[valid])  # [V, M, 3], [V, M, D]
        m = self.n_proxies
        tokens = feat
        pe = self.voxel_pe(centers)               # [V, M, D]

        for kind, layer in zip(self.layer_kind, self.layers):
            if kind == "self":
                # intra-fragment attention over the M proxies of every
                # fragment (packed fragments form the batch); each self
                # layer re-adds the position encoding
                tokens = layer(tokens + pe)
            else:
                # inter-fragment attention per sample over its valid
                # fragments; the packed layout equals the reference
                # padded layout with padded tokens masked out
                out = tokens.new_empty(tokens.shape)
                start = 0
                for s in range(b):
                    v_s = int(valid[s].sum().item())
                    if v_s == 0:
                        continue
                    seg = tokens[start:start + v_s].reshape(1, v_s * m,
                                                            self.feat_dim)
                    idx = torch.arange(v_s * m, device=pcs.device) // m
                    mask = idx[:, None] == idx[None, :]
                    out[start:start + v_s] = layer(seg, attn_mask=mask) \
                        .reshape(v_s, m, self.feat_dim)
                    start += v_s
                tokens = out

        node = tokens.new_zeros(b, p, self.feat_dim)
        node[valid] = tokens.mean(dim=1)

        quat = torch.zeros(b, p, 4, device=pcs.device)
        trans = torch.zeros(b, p, 3, device=pcs.device)
        quat[..., 0] = 1.0
        for step in range(self.refine_steps):
            rel_rot, rel_trans, node_new, adj_scores = \
                self.rel_estimators[step](node, valids)
            self.rel_rot = rel_rot
            self.rel_trans = rel_trans
            self.adj_scores = adj_scores
            global_code = self.g_encoders[step](node_new, valids)
            node = node_new + global_code
            inp = torch.cat([node, quat, trans], dim=-1)
            quat, trans = self.pose_predictors[step](inp)
        return quat, trans

    def aux_loss(self, adj_gt: torch.Tensor, rel_quat_gt: torch.Tensor,
                 rel_trans_gt: torch.Tensor, valids: torch.Tensor,
                 w_adj: float = 1.0, w_rel_rot: float = 0.2,
                 w_rel_trans: float = 1.0) -> torch.Tensor:
        """Adjacency MSE (on the predicted probabilities) and relative
        pose losses on the adjacent ordered pairs.

        Requires a preceding forward call, which stores the pairwise
        adjacency scores and relative pose predictions.
        """
        scores = getattr(self, "adj_scores", None)
        if scores is None or not hasattr(self, "rel_rot"):
            raise RuntimeError("aux_loss requires a forward pass first")
        b, p = adj_gt.shape[:2]
        valid_pair = (valids.unsqueeze(1) * valids.unsqueeze(2)) > 0.5
        eye = torch.eye(p, device=adj_gt.device).bool()
        valid_pair = valid_pair & ~eye.unsqueeze(0)

        loss_adj = (scores - adj_gt).pow(2) * valid_pair.float()
        loss_adj = loss_adj.sum() / valid_pair.float().sum().clamp(min=1.0)

        pred_q = self.rel_rot
        pred_q = pred_q / pred_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pair_mask = (adj_gt > 0.5) & valid_pair
        n_pair = pair_mask.sum().clamp(min=1.0)
        rot = 1.0 - (pred_q * rel_quat_gt).sum(-1).abs()
        loss_rel_rot = (rot * pair_mask).sum() / n_pair
        loss_rel_trans = ((self.rel_trans - rel_trans_gt).pow(2).sum(-1)
                          * pair_mask).sum() / n_pair
        return (w_adj * loss_adj + w_rel_rot * loss_rel_rot
                + w_rel_trans * loss_rel_trans)
