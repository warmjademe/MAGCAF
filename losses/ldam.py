"""LDAM-DRW (Cao et al., NeurIPS 2019).

Label-Distribution-Aware Margin: enforces bigger margin for minority classes.
DRW (deferred re-weighting) is applied externally via training schedule.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    def __init__(self, class_counts: Sequence[int], max_m: float = 0.5,
                 scale: float = 30.0, weight: torch.Tensor | None = None):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(np.asarray(class_counts, dtype=np.float64)))
        m = m * (max_m / np.max(m))
        self.register_buffer("m_list", torch.tensor(m, dtype=torch.float))
        self.scale = scale
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        idx = torch.zeros_like(logits, dtype=torch.bool)
        idx.scatter_(1, target.unsqueeze(1), True)
        batch_m = self.m_list[target].unsqueeze(1)
        logits_m = logits - batch_m.expand_as(logits) * idx.float()
        return F.cross_entropy(self.scale * logits_m, target, weight=self.weight)
