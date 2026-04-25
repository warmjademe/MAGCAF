"""Focal Loss (Lin et al., ICCV 2017). Multi-class form via softmax + NLL
on the predicted-correct-class probability."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        focal = (1.0 - pt).pow(self.gamma) * (-log_pt)
        if self.alpha is not None:
            focal = self.alpha[target] * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal
