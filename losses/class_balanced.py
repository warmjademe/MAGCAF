"""Class-Balanced Focal Loss (Cui et al., CVPR 2019).

Per-class weight = (1 - beta) / (1 - beta^n_c), normalised to sum to num_classes.
This is our default loss for MAGCAF-Net -- directly answers R1 #3 & R2 #5.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .focal import FocalLoss


def effective_num_weights(class_counts: Sequence[int], beta: float = 0.9999) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.double)
    effective_num = 1.0 - torch.pow(beta, counts)
    w = (1.0 - beta) / effective_num.clamp_min(1e-12)
    # Normalise so average is 1.0 -> keeps scale comparable to plain CE.
    w = w / w.mean()
    return w.float()


class ClassBalancedFocalLoss(FocalLoss):
    def __init__(self, class_counts: Sequence[int], beta: float = 0.9999,
                 gamma: float = 2.0, reduction: str = "mean"):
        alpha = effective_num_weights(class_counts, beta)
        super().__init__(gamma=gamma, alpha=alpha, reduction=reduction)
        self.beta = beta
        self.class_counts = list(class_counts)
