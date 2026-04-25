"""Lightweight logging: one metrics.json + one confusion_matrices.json per run,
plus a human-readable stdout/log tail. Everything the aggregation script needs
is in the two JSONs so downstream stays stateless.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict


def setup_logger(run_dir: str, name: str = "train") -> logging.Logger:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


def dump_json(path: str, obj: Dict[str, Any]) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
