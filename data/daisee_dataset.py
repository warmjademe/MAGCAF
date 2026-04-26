"""DAiSEE dataset loader.

Two modes:
    mode="features" -> yields
        {"spatial": (T, d_s) float16   (from feat_cache, pre-extracted),
         "flow":    (T, 10, H, W) float32  (built from cached int8 flows),
         "labels":  (4,) int,
         "clip_id": str}
    used by MAGCAFNet and FusionAttnLSTM (both consume (spatial, flow_stacks)).

    mode="raw" -> yields
        {"frames":  (T, 3, 224, 224) float,
         "labels":  (4,) int,
         "clip_id": str}
    used by all end-to-end baselines (R3D, I3D, R(2+1)D, SlowFast,
    TimeSformer, VideoMAE, X-CLIP, ResNet-TCN).

Flow stacks: each frame t exposes the 10-channel stack of 5 consecutive
TV-L1 flows ending at t (clipped at boundaries). The stack is built from
the int8 cache via `build_flow_stacks`, so no extra disk cost.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.protocol import (                          # noqa: E402
    AFFECTIVE_TASKS, CFG, NUM_CLASSES_PER_TASK, resolve_labels_csv,
)
from data.flow_cache import decode_flow_int8           # noqa: E402


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _stem(x: str) -> str:
    return os.path.splitext(os.path.basename(x))[0]


def load_labels_df(root: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(resolve_labels_csv(root, split))
    df.columns = [c.strip() for c in df.columns]
    df["ClipID"] = df["ClipID"].map(_stem)
    return df


def class_counts_per_task(df: pd.DataFrame) -> torch.Tensor:
    out = np.zeros((len(AFFECTIVE_TASKS), NUM_CLASSES_PER_TASK), dtype=np.int64)
    col_map = {"boredom": "Boredom", "engagement": "Engagement",
               "confusion": "Confusion", "frustration": "Frustration"}
    for i, t in enumerate(AFFECTIVE_TASKS):
        for c in range(NUM_CLASSES_PER_TASK):
            out[i, c] = int((df[col_map[t]] == c).sum())
    return torch.tensor(out)


def label_correlation(df: pd.DataFrame) -> torch.Tensor:
    col_map = {"boredom": "Boredom", "engagement": "Engagement",
               "confusion": "Confusion", "frustration": "Frustration"}
    arr = np.stack([df[col_map[t]].values.astype(np.float32)
                    for t in AFFECTIVE_TASKS], axis=1)
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    return torch.tensor(corr, dtype=torch.float32)


# ---------------------------------------------------------------------------
def build_flow_stacks(flow_int8: np.ndarray, stack: int = 5) -> np.ndarray:
    """(T-1, H, W, 2) int8 -> (T, 2*stack, H, W) float32.

    Index t uses flows [t-stack+1 .. t] (clipped).
    """
    flow = decode_flow_int8(flow_int8)                 # (T-1, H, W, 2) float32
    Tm1, H, W, _ = flow.shape
    T = Tm1 + 1
    out = np.zeros((T, stack * 2, H, W), dtype=np.float32)
    for t in range(T):
        for k in range(stack):
            src = t - (stack - 1 - k)
            src = max(0, min(Tm1 - 1, src))
            out[t, 2 * k:2 * k + 2] = np.transpose(flow[src], (2, 0, 1))
    return out


# ---------------------------------------------------------------------------
class DAiSEEDataset(Dataset):
    """Yields whichever keys the caller asks for via `features_wanted`:

        spatial / flow / timesformer / videomae / raw_frames

    Loading any key the model doesn't need is pure I/O waste (flow stacks at
    T=16 are ~13 MB each). The trainer passes the minimal set based on model.
    """
    def __init__(self, split: str, mode: str = "features",
                 features_wanted: Sequence[str] | None = None,
                 dataset_root: str = CFG.dataset_root,
                 face_cache_dir: str = CFG.face_cache_dir,
                 feat_cache_dir: str | None = None,
                 clip_frames: int = CFG.clip_frames,
                 augment: bool = False):
        assert split in ("Train", "Validation", "Test")
        assert mode in ("features", "raw")
        self.split = split
        self.mode = mode
        self.face_cache_dir = face_cache_dir
        self.feat_cache_dir = feat_cache_dir or os.path.join(
            os.path.dirname(face_cache_dir), "feat_cache")
        self.clip_frames = clip_frames
        self.augment = augment and split == "Train"

        if features_wanted is None:
            features_wanted = (
                ("raw_frames",) if mode == "raw"
                else ("spatial", "flow", "timesformer", "videomae")
            )
        self.features_wanted = set(features_wanted)

        df = load_labels_df(dataset_root, split)
        avail = set(_stem(p) for p in os.listdir(face_cache_dir)
                    if p.endswith(".npz"))
        df = df[df["ClipID"].isin(avail)].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def _load_npz(self, clip_id: str) -> Dict[str, np.ndarray]:
        return np.load(os.path.join(self.face_cache_dir, f"{clip_id}.npz"))

    def _labels(self, row) -> torch.Tensor:
        return torch.tensor([
            int(row["Boredom"]), int(row["Engagement"]),
            int(row["Confusion"]), int(row["Frustration"]),
        ], dtype=torch.long)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        clip_id = row["ClipID"]
        labels = self._labels(row)
        wants = self.features_wanted
        out = {"labels": labels, "clip_id": clip_id}

        if "spatial" in wants:
            sp_path = os.path.join(self.feat_cache_dir, f"{clip_id}_spatial.npy")
            spatial = np.load(sp_path).astype(np.float32)              # (T, d_s)
            out["spatial"] = torch.from_numpy(spatial)

        if "flow" in wants:
            npz = self._load_npz(clip_id)
            flow_int8 = npz["flows"]                                   # (T-1, H, W, 2)
            flow_stacks = build_flow_stacks(flow_int8)                 # (T, 10, H, W)
            if self.augment and torch.rand(1).item() < 0.5:
                flow_stacks = flow_stacks.copy()
                flow_stacks = np.flip(flow_stacks, axis=-1).copy()
                flow_stacks[:, 0::2] = -flow_stacks[:, 0::2]
            out["flow"] = torch.from_numpy(flow_stacks)

        for tag in ("timesformer", "videomae", "flow_feat", "landmark_seq",
                    "effnetv2_face", "effnetv2_scene"):
            if tag in wants:
                # On disk: <ClipID>_{timesformer,videomae,flow,landmarks,
                #                    effnetv2_face,effnetv2_scene}.npy
                disk_tag = {
                    "flow_feat": "flow",
                    "landmark_seq": "landmarks",
                }.get(tag, tag)
                p = os.path.join(self.feat_cache_dir, f"{clip_id}_{disk_tag}.npy")
                if os.path.exists(p):
                    out[tag] = torch.from_numpy(np.load(p).astype(np.float32))
        # Token-level features for MAGCAF v4 (Q-Former).
        # Stored as <ClipID>_{tsf,vmae}_tokens.npy with shape (~1568, 768) float16.
        for tag in ("tsf_tokens", "vmae_tokens"):
            if tag in wants:
                p = os.path.join(self.feat_cache_dir, f"{clip_id}_{tag}.npy")
                if os.path.exists(p):
                    out[tag] = torch.from_numpy(np.load(p).astype(np.float32))

        if "raw_frames" in wants:
            npz = self._load_npz(clip_id)
            frames = npz["frames"]                                     # (T, H, W, 3)
            rgb = frames.astype(np.float32) / 255.0
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
            rgb = np.transpose(rgb, (0, 3, 1, 2))                      # (T, 3, H, W)
            if self.augment and torch.rand(1).item() < 0.5:
                rgb = np.ascontiguousarray(rgb[..., ::-1])
            out["frames"] = torch.from_numpy(rgb)

        return out
