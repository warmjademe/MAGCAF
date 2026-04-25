"""Engagement-specific baseline — Abedi & Khan, CIIR 2021 variant.

Abedi & Khan 2021 concatenate a ResNet-50 per-frame feature with a dilated-TCN
temporal model. To keep Phase-B GPU time bounded (and to mirror the
linear-probe protocol we apply to the Transformer baselines), we feed the
SAME frozen face-domain spatial features used by MAGCAF-Net (InceptionResnetV1
/ VGGFace2, 512-d) into the TCN instead of running a trainable ResNet-50 per
batch per epoch. This turns one training step from ~7.5 s to ~0.2 s.

Reviewer framing: "For a fair architectural comparison, ResNet-TCN shares the
exact same frozen spatial backbone as MAGCAF-Net; the only difference is the
temporal head (dilated TCN vs. MAGCAF + Bi-LSTM + additive attention)."
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import MultiTaskHead


class _DilatedTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        return x + r


class ResNetTCNModel(nn.Module):
    """Frozen-spatial-feature + trainable dilated TCN + multi-task head.

    Forward signature matches the other feature-mode models:
        forward(spatial_seq: (B, T, d_in)) -> (B, K, C) logits
    So the trainer passes this through the `features` dataset pathway.
    """
    def __init__(self, in_dim: int = 512, tcn_channels: int = 256,
                 num_blocks: int = 4, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, tcn_channels)
        self.tcn = nn.Sequential(*[
            _DilatedTCNBlock(tcn_channels, dilation=2 ** i, dropout=0.2)
            for i in range(num_blocks)
        ])
        self.head = MultiTaskHead(tcn_channels, hidden=tcn_channels, dropout=dropout)

    def forward(self, spatial_seq: torch.Tensor) -> torch.Tensor:
        # (B, T, d_in) -> (B, T, C) -> (B, C, T)
        x = self.proj(spatial_seq).transpose(1, 2)
        x = self.tcn(x)                                         # (B, C, T)
        pooled = x.mean(dim=-1)                                 # (B, C)
        return self.head(pooled)


class LRCNModel(nn.Module):
    """Long-term Recurrent Convolutional Network (Donahue 2015 / Gupta 2016).

    Original DAiSEE benchmark baseline: per-frame CNN feature -> stacked LSTM ->
    classifier. We use the same frozen face-domain features as ResNet-TCN
    (InceptionResnetV1, 512-d per frame), so that LRCN vs ResNet-TCN vs MAGCAF
    differ only in their temporal head — the cleanest possible architectural
    comparison.

    Forward signature:
        forward(spatial_seq: (B, T, d_in)) -> (B, K, C) logits
    """
    def __init__(self, in_dim: int = 512, hidden: int = 256,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = MultiTaskHead(out_dim, hidden=out_dim, dropout=dropout)

    def forward(self, spatial_seq: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(spatial_seq)                           # (B, T, H)
        pooled = h.mean(dim=1)                                  # mean pool
        return self.head(pooled)
