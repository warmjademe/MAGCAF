"""Factory for every loss in the P3 imbalance ablation."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal import FocalLoss
from .class_balanced import ClassBalancedFocalLoss, effective_num_weights
from .ldam import LDAMLoss


def build_loss(name: str, class_counts: Sequence[int]) -> nn.Module:
    """
    Supported names (matches P3 row of REVISION_PLAN.md):
        ce              -> plain cross-entropy (S0, baseline)
        weighted_ce     -> class-weighted CE with sqrt frequency (S1)
        focal           -> Focal Loss gamma=2 (S2)
        cb_focal        -> Class-Balanced Focal Loss beta=0.9999 gamma=2 (S3) [default]
        ldam            -> LDAM without re-weight (S4, pair with DRW scheduler)
    """
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        counts = torch.tensor(class_counts, dtype=torch.float)
        inv_sqrt = 1.0 / counts.clamp_min(1.0).sqrt()
        w = (inv_sqrt / inv_sqrt.mean())
        return nn.CrossEntropyLoss(weight=w)
    if name == "focal":
        return FocalLoss(gamma=2.0)
    if name == "cb_focal":
        return ClassBalancedFocalLoss(class_counts, beta=0.9999, gamma=2.0)
    if name == "ldam":
        w = effective_num_weights(class_counts, beta=0.9999)
        return LDAMLoss(class_counts, weight=w)
    raise ValueError(f"Unknown loss: {name}")
