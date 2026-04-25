"""Shared utilities for baseline models:

    MultiTaskHead        -- (B, D) -> (B, K, C) four 4-way heads.
    TemporalPoolAdapter  -- (B, T, 3, H, W) -> (B, 3, T', H, W) for 3D/Transformer
                            models that expect clip tensors (T'=16 or 8).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_TASKS = 4
NUM_CLASSES = 4


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int | None = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or in_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, NUM_CLASSES),
            ) for _ in range(NUM_TASKS)
        ])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, in_dim) -> (B, K, C)."""
        return torch.stack([h(feat) for h in self.heads], dim=1)


def uniform_temporal_sample(frames: torch.Tensor, target_T: int) -> torch.Tensor:
    """frames: (B, T, 3, H, W) -> (B, target_T, 3, H, W) via uniform indices."""
    B, T, C, H, W = frames.shape
    if T == target_T:
        return frames
    idx = torch.linspace(0, T - 1, target_T).round().long().to(frames.device)
    return frames.index_select(1, idx)


# ---------------------------------------------------------------------------
# MAGCAF: task-correlation Omega head and Kendall-Gal uncertainty weighting
# ---------------------------------------------------------------------------
class OmegaHead(nn.Module):
    """Task-correlation head: couples K per-task logits through a learnable
    matrix Omega regularised toward an empirical-correlation prior."""
    def __init__(self, num_tasks: int, num_classes: int,
                 omega_prior: torch.Tensor | None = None,
                 tau: float = 1.0, reg_lambda: float = 0.01):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.tau = tau
        self.reg_lambda = reg_lambda
        if omega_prior is None:
            omega_prior = torch.eye(num_tasks)
        self.register_buffer("omega_prior", omega_prior.clone())
        self.omega = nn.Parameter(omega_prior.clone())

    def forward(self, task_logits: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjc->bic", self.omega, task_logits) / self.tau

    def regularizer_loss(self) -> torch.Tensor:
        return self.reg_lambda * torch.linalg.norm(
            self.omega - self.omega_prior, ord="fro"
        ) ** 2


class UncertaintyWeighter(nn.Module):
    """Kendall--Gal homoscedastic uncertainty weighting for K task losses."""
    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_sigma2 = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, task_losses: torch.Tensor) -> torch.Tensor:
        ls = self.log_sigma2.clamp(min=-10.0, max=10.0)
        return (torch.exp(-ls) * task_losses + 0.5 * ls).sum()
