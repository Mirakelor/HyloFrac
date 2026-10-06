"""DGL baseline: dynamic graph network (Huang et al., NeurIPS 2020;
Breaking Bad DGL).

Fragment PointNet codes are the graph nodes. Three rounds with
per-round parameters alternate: message passing over the current relation
matrix (the full graph in the first round; relations predicted from the
previous round's poses afterwards), node updates without a residual
connection, and per-round pose regression whose outputs feed the next
round (pose feedback). All rounds are supervised during training and only
the last round is used at evaluation. Geometric-equivalence node merging
is disabled because fragments are geometrically unique.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hylofrac.baselines.common import (GeometricLoss, PointNetEncoder,
                                       PoseRegressor)


class _MLP(nn.Module):
    """Shared MLP with batch norm and relu on every layer."""

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i]))
            layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        y = self.net(x.reshape(-1, x.shape[-1]))
        return y.reshape(*lead, -1)


class _RelationNet(nn.Module):
    """Edge relation predictor over pose-feature pairs (sigmoid)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class _PoseEncoder(nn.Module):
    """Pose (quat + trans) to pose features for the relation predictor."""

    def __init__(self, pose_dim: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DglPoseNet(nn.Module):
    def __init__(self, feat_dim: int = 128, gnn_iter: int = 3) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(feat_dim)
        self.iterations = gnn_iter
        mid = 512
        self.edge_mlps = nn.ModuleList([
            _MLP([2 * feat_dim, mid, mid, feat_dim])
            for _ in range(gnn_iter)])
        self.node_mlps = nn.ModuleList([
            _MLP([2 * feat_dim, mid, mid, feat_dim])
            for _ in range(gnn_iter)])
        self.relation_net = _RelationNet(feat_dim)
        self.pose_extractor = _PoseEncoder()
        self.pose_predictors = nn.ModuleList([
            PoseRegressor(feat_dim + 7) for _ in range(gnn_iter)])
        self._loss_fn = GeometricLoss()

    def _rounds(self, pcs: torch.Tensor, valids: torch.Tensor
                ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run all rounds; returns (quat, trans) per round."""
        b, p = pcs.shape[:2]
        device = pcs.device
        valid = valids > 0.5
        node = torch.zeros(b, p, self.encoder.feat_dim, device=device)
        node[valid] = self.encoder(pcs[valid])
        valid_matrix = (valids.unsqueeze(1) * valids.unsqueeze(2)).float()

        pred_pose = torch.zeros(b, p, 7, device=device)
        pred_pose[..., 0] = 1.0  # identity initial pose
        rounds = []
        for i in range(self.iterations):
            if i == 0:
                relation = valid_matrix
            else:
                # relations re-estimated from the previous round's poses
                pose_feats = self.pose_extractor(pred_pose)
                pair = torch.cat([
                    pose_feats.unsqueeze(2).expand(b, p, p, -1),
                    pose_feats.unsqueeze(1).expand(b, p, p, -1)], dim=-1)
                relation = self.relation_net(pair) * valid_matrix

            # directed pairs (i -> j): edge features over the node pair
            pair = torch.cat([
                node.unsqueeze(2).expand(b, p, p, -1),
                node.unsqueeze(1).expand(b, p, p, -1)], dim=-1)
            edge = self.edge_mlps[i](pair)          # [B, P, P, F]
            msg = (edge * relation.unsqueeze(-1)).sum(dim=2) \
                / relation.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            # node update without residual
            node = self.node_mlps[i](
                torch.cat([msg, node], dim=-1))

            rot, trans = self.pose_predictors[i](
                torch.cat([node, pred_pose], dim=-1))
            pred_pose = torch.cat([rot, trans], dim=-1)
            rounds.append((rot, trans))
        return rounds

    def forward(self, pcs: torch.Tensor, valids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Last-round poses (the reference returns only the last round at
        evaluation)."""
        rounds = self._rounds(pcs, valids)
        self.rounds = rounds
        return rounds[-1]

    def loss_step(self, pcs: torch.Tensor, quat_gt: torch.Tensor,
                  trans_gt: torch.Tensor, valids: torch.Tensor) -> dict:
        """Sum of the geometric pose loss over all rounds."""
        rounds = self._rounds(pcs, valids)
        total = sum(self._loss_fn(pcs, q, t, quat_gt, trans_gt, valids)
                    ["total"] for q, t in rounds)
        return {"total": total}
