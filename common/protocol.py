"""Unified experimental protocol for all models. Single source of truth for splits,
seeds, augmentations, and hyper-parameters so every reviewer-visible number is
produced under identical conditions.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------
AFFECTIVE_TASKS: tuple[str, ...] = ("boredom", "engagement", "confusion", "frustration")
NUM_CLASSES_PER_TASK: int = 4  # 0=Very Low, 1=Low, 2=High, 3=Very High
NUM_TASKS: int = len(AFFECTIVE_TASKS)

# Frozen 5-seed protocol
SEEDS: tuple[int, ...] = (42, 123, 2024, 7, 1337)


@dataclass(frozen=True)
class ProtocolConfig:
    # Data paths (filled at runtime via env vars or CLI)
    dataset_root: str = os.environ.get(
        "DAISEE_ROOT", "/home/qyb/datasets/DAiSEE/DAiSEE"
    )
    face_cache_dir: str = os.environ.get(
        "DAISEE_FACE_CACHE", "/home/qyb/datasets/DAiSEE/face_cache"
    )
    flow_cache_dir: str = os.environ.get(
        "DAISEE_FLOW_CACHE", "/home/qyb/datasets/DAiSEE/flow_cache"
    )

    # Clip shape (Plan C: uniform 16-frame sampling across the 10s clip)
    fps: int = 10
    clip_frames: int = 16
    face_size: int = 224
    flow_stack: int = 5  # stack 5 consecutive flows → 10 channels

    # Optim (Plan C + Lever 3: epochs 20, patience 3 since DAiSEE converges ~10-15 ep)
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    early_stop_patience: int = 3  # eval steps (= 6 no-improve epochs)
    eval_every_epochs: int = 2

    # Reporting
    seeds: tuple[int, ...] = SEEDS


CFG = ProtocolConfig()


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python / NumPy / Torch (CPU + CUDA) for a reproducible run.

    We do *not* set torch.backends.cudnn.deterministic=True because that
    cripples conv throughput on the 4090 and the variance it introduces is
    already absorbed by our 5-seed mean±std reporting.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seed mixed with base seed to avoid same RNG per worker."""
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------
def resolve_split_dir(root: str, split: str) -> str:
    assert split in ("Train", "Validation", "Test")
    return os.path.join(root, "DataSet", split)


def resolve_labels_csv(root: str, split: str) -> str:
    fname = {
        "Train": "TrainLabels.csv",
        "Validation": "ValidationLabels.csv",
        "Test": "TestLabels.csv",
    }[split]
    return os.path.join(root, "Labels", fname)
