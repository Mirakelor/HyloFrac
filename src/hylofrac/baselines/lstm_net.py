"""LSTM baseline: PointNet codes through a GRU sequence-to-sequence
network (PQ-Net, Wu et al., CVPR 2020; Breaking Bad B-LSTM; the
reference baseline is named LSTM while its cells are GRUs).

A shared PointNet encodes the fragments; a two-layer bidirectional GRU
encoder (hidden 256 per direction) reads the code sequence in fragment
order with the padded slots excluded, and a two-layer GRU decoder
reconstructs the code sequence: training mixes the ground-truth codes
into the decoder input with probability 0.5 (teacher forcing), evaluation
is autoregressive. The decoder output code of every fragment goes
through the shared MLP pose head; a per-layer 16-dim noise vector
initializes the decoder hidden state (fixed at evaluation for
reproducibility).
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from hylofrac.baselines.common import PointNetEncoder, PoseRegressor


class LstmPoseNet(nn.Module):
    def __init__(self, feat_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(feat_dim)
        self.gru_enc = nn.GRU(feat_dim, hidden_dim, num_layers=2,
                              bidirectional=True, dropout=0.2)
        self.gru_dec = nn.GRU(feat_dim, hidden_dim * 2 + 16, num_layers=2,
                              dropout=0.2)
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, 256), nn.LeakyReLU(inplace=True),
            nn.Linear(256, feat_dim),
        )
        self.pose_predictor = PoseRegressor(feat_dim)

    def forward(self, pcs: torch.Tensor, valids: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """pcs [B, MAX_PARTS, N, 3], valids [B, MAX_PARTS].

        Returns (quat [B, P, 4], trans [B, P, 3]) of the decoded codes.
        """
        b, p, _, _ = pcs.shape
        device = pcs.device
        valid = valids > 0.5
        feats = torch.zeros(b, p, self.encoder.feat_dim, device=device)
        feats[valid] = self.encoder(pcs[valid])
        seq = feats.transpose(0, 1)  # [P, B, D]

        lengths = valid.sum(dim=1).cpu()
        packed = pack_padded_sequence(seq, lengths, enforce_sorted=False)
        _, h_enc = self.gru_enc(packed)             # [4, B, hidden]
        h_enc = h_enc.view(2, 2, b, -1)             # layers x directions
        hidden = torch.cat([h_enc[:, 0], h_enc[:, 1]], dim=-1)  # [2, B, 2H]
        if self.training:
            noise = torch.randn(2, b, 16, device=device)
        else:
            rng = torch.Generator(device=device).manual_seed(0)
            noise = torch.randn(2, b, 16, device=device, generator=rng)
        hidden = torch.cat([hidden, noise], dim=-1)  # [2, B, 2H + 16]

        decoder_input = torch.zeros(1, b, self.encoder.feat_dim,
                                    device=device)
        force = self.training and random.random() < 0.5
        codes = []
        for i in range(p):
            _, hidden = self.gru_dec(decoder_input, hidden)
            # the reference (B-LSTM) takes the pose code from the FIRST
            # decoder layer's hidden state (``hidden1, hidden2 =
            # torch.split(hidden, 1, 0)`` then ``linear1(hidden1)``)
            code = self.decoder_fc(hidden[0])
            codes.append(code)
            decoder_input = seq[i:i + 1] if force \
                else code.detach().unsqueeze(0)
        out = torch.stack(codes, dim=0).transpose(0, 1)  # [B, P, D]
        return self.pose_predictor(out)
