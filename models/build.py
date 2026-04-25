"""Factory mapping model name (CLI) -> instantiated nn.Module.

Models exposed:
    magcaf       -- MAGCAF (ours): 4-source heterogeneous fusion with
                    task-conditioned gated cross-attention, Omega
                    task-correlation head, and Kendall-Gal uncertainty
                    weighting (alias: ours).
    timesformer  -- TimeSformer (Bertasius 2021), frozen + adapter.
    videomae     -- VideoMAE   (Tong   2022), frozen + adapter.
    resnet_tcn   -- ResNet-TCN (Abedi & Khan 2021).
    lrcn         -- LRCN       (Donahue 2015 / DAiSEE benchmark, Gupta 2016).

All models consume 16 uniformly-sampled frames per clip.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from .magcaf_v2 import MAGCAFv2Net, MAGCAFv2Config


def build_model(name: str, **kwargs: Any) -> nn.Module:
    name = name.lower()
    mcfg = kwargs.get("model_cfg", {})

    if name in ("magcaf", "magcaf_v2", "ours"):
        return MAGCAFv2Net(
            cfg=MAGCAFv2Config(**mcfg),
            omega_prior=kwargs.get("omega_prior"),
            class_counts_per_task=kwargs.get("class_counts_per_task"),
            ablate=kwargs.get("ablate"),
        )

    if name == "timesformer":
        from .baseline_transformer import TimeSformerModel
        return TimeSformerModel(**mcfg)
    if name == "videomae":
        from .baseline_transformer import VideoMAEModel
        return VideoMAEModel(**mcfg)
    if name == "resnet_tcn":
        from .baseline_engagement import ResNetTCNModel
        return ResNetTCNModel(**mcfg)
    if name == "lrcn":
        from .baseline_engagement import LRCNModel
        return LRCNModel(**mcfg)

    raise ValueError(f"Unknown model name: {name}")
