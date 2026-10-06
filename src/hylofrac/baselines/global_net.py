"""Global baseline: per-part and whole-cloud PointNet codes with an MLP
pose head (CompoNet, Schor et al., ICCV 2019, and PAGENet, Li et al.,
AAAI 2020; Breaking Bad B-Global).

A shared PointNet encodes every fragment; a second PointNet encodes the
whole scene cloud (all fragment slots concatenated, padded slots
included, as in the reference). An MLP pose regressor maps the
concatenated global and per-fragment codes to each fragment pose.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hylofrac.baselines.common import PointNetEncoder, PoseRegressor


class GlobalPoseNet(nn.Module):
    def __init__(self, feat_dim: int = 128) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(feat_dim)
        self.global_encoder = PointNetEncoder(feat_dim)
        self.pose_predictor = PoseRegressor(feat_dim * 2)

    def forward(self, pcs: torch.Tensor, valids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """pcs [B, MAX_PARTS, N, 3], valids [B, MAX_PARTS].

        Returns (quat [B, P, 4] scalar-first, trans [B, P, 3]).
        """
        b, p, n, _ = pcs.shape
        valid = valids > 0.5
        codes = torch.zeros(b, p, self.encoder.feat_dim, device=pcs.device)
        codes[valid] = self.encoder(pcs[valid])
        global_code = self.global_encoder(pcs.reshape(b, p * n, 3))
        global_code = global_code.unsqueeze(1).expand(b, p, -1)
        return self.pose_predictor(torch.cat([global_code, codes], dim=-1))
