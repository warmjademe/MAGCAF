"""ViBED-Net (Gothwal et al., 2025) — Video-Based Engagement Detection Network.

Original ViBED-Net is a single-task engagement model with a dual-stream design:
  - Face stream:  per-frame face crop  -> EfficientNetV2-S -> 1280-d per frame
  - Scene stream: per-frame full frame -> EfficientNetV2-S -> 1280-d per frame
followed by per-stream LSTM, late-fusion concat, and an engagement classifier.

Under our unified 4-task multi-task protocol (T=16, 20 epochs, frozen
backbones, 3 seeds), we adapt ViBED-Net as follows:
  1. EfficientNetV2-S (frozen, ImageNet-21k -> ImageNet-1k) is pre-applied
     once per clip to produce face-stream and scene-stream feature sequences
     of shape (T=16, 1280); these are cached to disk and consumed at training
     time, mirroring the InceptionResnetV1 / VideoMAE / TimeSformer
     feature-cache pattern used by all other baselines.
  2. The engagement-only single-task head is replaced by our shared
     MultiTaskHead (4 tasks, 4 classes) so that ViBED-Net is evaluated on the
     same four affective dimensions as MAGCAF and the other baselines.
  3. The aggressive minority-class augmentations of the original paper
     (salt-and-pepper, elastic, etc.) are NOT applied — the unified protocol
     uses the standard preprocessing pipeline shared across all five models.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._common import MultiTaskHead


class ViBEDNetModel(nn.Module):
    """Dual-stream EfficientNetV2 + LSTM faithful to ViBED-Net (Gothwal 2025).

    Forward signature (cached-feature mode):
        forward(face_seq:  (B, T, d_face),
                scene_seq: (B, T, d_scene)) -> (B, K, C) logits
    """

    def __init__(
        self,
        in_dim_face: int = 1280,
        in_dim_scene: int = 1280,
        hidden: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.face_lstm = nn.LSTM(
            in_dim_face, hidden, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.scene_lstm = nn.LSTM(
            in_dim_scene, hidden, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = MultiTaskHead(2 * hidden, hidden=2 * hidden, dropout=dropout)

    def forward(
        self,
        face_seq: torch.Tensor,
        scene_seq: torch.Tensor,
    ) -> torch.Tensor:
        f_h, _ = self.face_lstm(face_seq)
        s_h, _ = self.scene_lstm(scene_seq)
        f_pool = f_h.mean(dim=1)
        s_pool = s_h.mean(dim=1)
        feat = torch.cat([f_pool, s_pool], dim=-1)
        return self.head(feat)
