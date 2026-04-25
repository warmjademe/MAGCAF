"""Transformer-era video baselines.

Default training path uses **pre-extracted frozen features** (see
`data/extract_transformer_features.py`). The cached-feature wrappers train a
tiny adapter + MultiTaskHead on top of the (768,) vector per clip,
reducing one training epoch for Phase-B from ~25 min to <1 min per model.

Fallback "end-to-end frozen" classes are kept for edge cases (running without
precomputed features) but are not the primary training path.

Models exposed (consumed by models/build.py):
    TimeSformerModel    <- CachedTimeSformerModel   (primary; feature input)
    VideoMAEModel       <- CachedVideoMAEModel      (primary)
    TimeSformerModelE2E                             (fallback; raw-frame input)
    VideoMAEModelE2E                                (fallback)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import MultiTaskHead, uniform_temporal_sample


# ---------------------------------------------------------------------------
# Primary path: cached feature -> adapter + head
# ---------------------------------------------------------------------------
class _CachedAdapterHead(nn.Module):
    def __init__(self, in_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = MultiTaskHead(in_dim, hidden=in_dim, dropout=dropout)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(self.adapter(feat))


class CachedTimeSformerModel(_CachedAdapterHead):
    """Trained on pre-extracted TimeSformer [CLS] embeddings (768,)."""
    pass


class CachedVideoMAEModel(_CachedAdapterHead):
    """Trained on pre-extracted VideoMAE mean-pooled features (768,)."""
    pass


# Public aliases used by build.py
TimeSformerModel = CachedTimeSformerModel
VideoMAEModel = CachedVideoMAEModel


# ---------------------------------------------------------------------------
# Fallback end-to-end frozen path (kept for completeness / debugging)
# ---------------------------------------------------------------------------
def _resize(frames: torch.Tensor, size: int) -> torch.Tensor:
    B, T, C, H, W = frames.shape
    if H == size and W == size:
        return frames
    flat = frames.reshape(B * T, C, H, W)
    flat = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False)
    return flat.view(B, T, C, size, size)


def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad = False
    module.eval()
    return module


class TimeSformerModelE2E(nn.Module):
    def __init__(self, dropout: float = 0.3, target_T: int = 8, img_size: int = 224):
        super().__init__()
        from transformers import TimesformerModel
        self.target_T = target_T; self.img_size = img_size
        self.backbone = _freeze(TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"))
        self.adapter = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU(), nn.Dropout(dropout))
        self.head = MultiTaskHead(768, hidden=768, dropout=dropout)

    def train(self, mode: bool = True):
        super().train(mode); self.backbone.eval(); return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        frames = uniform_temporal_sample(frames, self.target_T)
        frames = _resize(frames, self.img_size)
        with torch.no_grad():
            cls = self.backbone(pixel_values=frames).last_hidden_state[:, 0]
        return self.head(self.adapter(cls))


class VideoMAEModelE2E(nn.Module):
    def __init__(self, dropout: float = 0.3, target_T: int = 16, img_size: int = 224):
        super().__init__()
        from transformers import VideoMAEModel as _VMAE
        self.target_T = target_T; self.img_size = img_size
        self.backbone = _freeze(_VMAE.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"))
        self.adapter = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 768), nn.GELU(), nn.Dropout(dropout))
        self.head = MultiTaskHead(768, hidden=768, dropout=dropout)

    def train(self, mode: bool = True):
        super().train(mode); self.backbone.eval(); return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        frames = uniform_temporal_sample(frames, self.target_T)
        frames = _resize(frames, self.img_size)
        with torch.no_grad():
            pooled = self.backbone(pixel_values=frames).last_hidden_state.mean(dim=1)
        return self.head(self.adapter(pooled))
