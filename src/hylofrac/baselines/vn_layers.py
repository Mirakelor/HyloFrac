"""Vector-neuron building blocks (Deng et al., NeurIPS 2021) and the
VN-DGCNN backbone of the DiffAssemble baseline, following the reference
implementation (Scarpellini et al., CVPR 2024, iit-pavis/diffassemble).

Inside the backbone features are stacks of 3-dim vectors with the layout
[B, C, 3, N] (C vector channels over N points), extended to
[B, C, 3, N, k] by the kNN graph. Every operation is equivariant under
SO(3):

- VNLinear: shared linear map over the channel axis;
- VNBatchNorm: batch norm on the vector norms only;
- VNLinearLeakyReLU: VNLinear, VNBatchNorm, and a reflection leaky relu
  whose reflection direction is a learned linear map of the input.

The encoder mirrors the reference VN-DGCNN (k = 20): three edge-conv
stages of 64 // 3 vector channels with dynamic kNN graphs (self included,
distances over the flattened vector channels), mean pooling over the
neighbors, a convolution over the concatenation of the three stages, and
the global point mean concatenated and pooled again. The final mean
pooling makes the second half of the 768-dim code duplicate the first,
as in the reference's mean-pooling path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-6


class VNLinear(nn.Module):
    """Shared linear map over the channel axis of [B, C_in, 3, N...]."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.map_to_feat = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNBatchNorm(nn.Module):
    """Batch norm applied to the vector norms only."""

    def __init__(self, c: int, dim: int = 5) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(c) if dim == 5 else nn.BatchNorm1d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=2) + EPS        # [B, C, N(, k)]
        norm_bn = self.bn(norm).unsqueeze(2)
        return x / norm.unsqueeze(2) * norm_bn


class VNLinearLeakyReLU(nn.Module):
    """Linear map, norm batch norm, and a reflection leaky relu.

    The reflection direction is a learned map of the *input* (shared
    across channels when ``share_nonlinearity``), as in the reference.
    """

    def __init__(self, c_in: int, c_out: int, dim: int = 5,
                 share_nonlinearity: bool = False,
                 negative_slope: float = 0.2) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.map_to_feat = nn.Linear(c_in, c_out, bias=False)
        self.batchnorm = VNBatchNorm(c_out, dim)
        self.map_to_dir = nn.Linear(c_in, 1 if share_nonlinearity else c_out,
                                    bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        p = self.batchnorm(p)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return (self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)))


def _knn_indices(feat: torch.Tensor, k: int) -> torch.Tensor:
    """kNN indices [B, N, k] under the squared Euclidean distance of the
    flattened vector channels; the point itself is included (the zero
    distance always ranks first), as in the reference."""
    inner = -2 * torch.matmul(feat.transpose(1, 2), feat)
    xx = (feat * feat).sum(1, keepdim=True)
    dist = -xx - inner - xx.transpose(1, 2)   # negative squared distances
    return dist.topk(k, dim=-1)[1]


def _edge_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    """kNN edge features [B, C, 3, N] -> [B, 2C, 3, N, k].

    The neighbor block comes first (differences to the center), then the
    center block; both keep the [C, 3] vector layout per neighbor.
    """
    b, c, _, n = x.shape
    flat = x.reshape(b, -1, n)
    idx = _knn_indices(flat, k)                    # [B, N, k]
    idx = (idx + torch.arange(b, device=x.device).view(b, 1, 1) * n) \
        .reshape(-1)
    # per-point rows keep the [C, 3] layout of the flattened channels
    all_pts = x.permute(0, 3, 1, 2).reshape(b * n, c * 3)  # [B*N, C*3]
    neigh = all_pts[idx].reshape(b, n, k, c, 3)
    center = all_pts.reshape(b, n, 1, c, 3).expand(b, n, k, c, 3)
    return torch.cat([neigh - center, center], dim=3).permute(0, 3, 4, 1, 2)


class VNDGCNNEncoder(nn.Module):
    """Reference VN-DGCNN over fragment point clouds.

    Channel sizes are the reference's vector-neuron counts (64 // 3 per
    stage); the code stacks 2 * 128 vector channels (the last convolution
    output and its global mean) over the 3 components, flattened to 768
    scalars per fragment.
    """

    def __init__(self, k: int = 20, feat_dim: int = 128) -> None:
        super().__init__()
        self.k = k
        c = 64 // 3
        self.conv1 = VNLinearLeakyReLU(2, c)
        self.conv2 = VNLinearLeakyReLU(c, c)
        self.conv3 = VNLinearLeakyReLU(2 * c, c)
        self.conv4 = VNLinearLeakyReLU(c, c)
        self.conv5 = VNLinearLeakyReLU(2 * c, c)
        self.conv6 = VNLinearLeakyReLU(3 * c, feat_dim, dim=4,
                                       share_nonlinearity=True)

    def forward(self, pcs: torch.Tensor) -> torch.Tensor:
        """pcs [B, P, N, 3] -> per-fragment codes [B, P, 768]."""
        b, p, n, _ = pcs.shape
        x = pcs.reshape(b * p, n, 3).transpose(1, 2).unsqueeze(1)
        x = _edge_feature(x, self.k)          # [B', 2, 3, N, k]
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.mean(dim=-1)                   # [B', 21, 3, N]
        x = _edge_feature(x1, self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.mean(dim=-1)
        x = _edge_feature(x2, self.k)
        x = self.conv5(x)
        x3 = x.mean(dim=-1)
        x = self.conv6(torch.cat([x1, x2, x3], dim=1))  # [B', 128, 3, N]
        x = torch.cat([x, x.mean(dim=-1, keepdim=True).expand_as(x)], dim=1)
        code = x.mean(dim=-1)                 # [B', 256, 3]
        return code.reshape(b, p, -1)
