"""Pre-extract frozen Transformer features (TimeSformer + VideoMAE) per clip.

Since the HF backbones are frozen during training (our "linear-probe"
fine-tuning protocol), each clip's features are identical across all epochs.
Caching them on disk turns 16 forward passes over the heavy ViT into a single
pass per clip, then reduces training to adapter+head arithmetic only.

Outputs (per clip in `<face_cache_dir>/../feat_cache/`):
    <ClipID>_timesformer.npy   float16 (768,)   CLS embedding, target_T=8
    <ClipID>_videomae.npy      float16 (768,)   mean-pooled, target_T=16

Usage:
    python -m data.extract_transformer_features \
        --face-cache /home/qyb/datasets/DAiSEE/face_cache
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.protocol import CFG             # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def uniform_sample_np(frames: np.ndarray, target_T: int) -> np.ndarray:
    T = frames.shape[0]
    if T == target_T:
        return frames
    idx = np.round(np.linspace(0, T - 1, target_T)).astype(np.int64)
    return frames[idx]


def prepare_batch(frames_u8: np.ndarray, size: int, target_T: int,
                  device: str) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (1, target_T, 3, size, size) float32 on device."""
    frames = uniform_sample_np(frames_u8, target_T)
    x = torch.from_numpy(frames).to(device).float().div_(255.0)   # (T,H,W,3)
    x = x.permute(0, 3, 1, 2).contiguous()                        # (T,3,H,W)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x.unsqueeze(0)   # (1, T, 3, size, size)


@torch.no_grad()
def extract_timesformer(frames_u8: np.ndarray, model, device: str) -> np.ndarray:
    x = prepare_batch(frames_u8, size=224, target_T=8, device=device)
    out = model(pixel_values=x).last_hidden_state
    return out[:, 0].squeeze(0).float().cpu().numpy().astype(np.float16)


@torch.no_grad()
def extract_videomae(frames_u8: np.ndarray, model, device: str) -> np.ndarray:
    x = prepare_batch(frames_u8, size=224, target_T=16, device=device)
    out = model(pixel_values=x).last_hidden_state
    pooled = out.mean(dim=1)
    return pooled.squeeze(0).float().cpu().numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-cache", default=CFG.face_cache_dir)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.face_cache), "feat_cache")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    clips = sorted(p[:-4] for p in os.listdir(args.face_cache) if p.endswith(".npz"))
    print(f"[feat-trans] {len(clips)} clips")

    # Load once
    print("[feat-trans] loading TimeSformer...")
    from transformers import TimesformerModel, VideoMAEModel
    ts = TimesformerModel.from_pretrained(
        "facebook/timesformer-base-finetuned-k400").to(args.device).eval()
    for p in ts.parameters(): p.requires_grad = False

    print("[feat-trans] loading VideoMAE...")
    vm = VideoMAEModel.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics").to(args.device).eval()
    for p in vm.parameters(): p.requires_grad = False

    for clip_id in tqdm(clips):
        ts_path = os.path.join(out_dir, f"{clip_id}_timesformer.npy")
        vm_path = os.path.join(out_dir, f"{clip_id}_videomae.npy")
        if os.path.exists(ts_path) and os.path.exists(vm_path):
            continue
        npz = np.load(os.path.join(args.face_cache, f"{clip_id}.npz"))
        frames = npz["frames"]
        if not os.path.exists(ts_path):
            feat = extract_timesformer(frames, ts, args.device)
            np.save(ts_path, feat)
        if not os.path.exists(vm_path):
            feat = extract_videomae(frames, vm, args.device)
            np.save(vm_path, feat)


if __name__ == "__main__":
    main()
