"""Pre-extract FROZEN face-domain spatial features once per clip.

Using facenet-pytorch's InceptionResnetV1 (VGGFace2 pretrain) as the spatial
branch -- this is the publicly-reproducible stand-in for the paper's
"ResNet-50 + VGGFace2" claim (VGGFace2 weights are otherwise not pip-installable).
Output per clip: 512-d float16 vector per frame.

Cache path: <face_cache_dir>/../feat_cache/<ClipID>_spatial.npy

Temporal features are NOT precomputed -- MAGCAFNet and FusionAttnLSTM contain a
trainable lightweight 3D-CNN that consumes the raw TV-L1 flow stacks on the fly.

Usage:
    python -m data.extract_features --face-cache /home/qyb/datasets/DAiSEE/face_cache
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.protocol import CFG   # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SpatialBackbone(nn.Module):
    """Frozen InceptionResnetV1 (VGGFace2 pretrain) -> 512-d face embedding."""
    def __init__(self):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.backbone = InceptionResnetV1(pretrained="vggface2").eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.out_dim = 512

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # facenet-pytorch expects 160x160 faces
        x = F.interpolate(x, size=160, mode="bilinear", align_corners=False)
        return self.backbone(x)        # (B, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-cache", default=CFG.face_cache_dir)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.face_cache), "feat_cache")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    clips = sorted(p[:-4] for p in os.listdir(args.face_cache) if p.endswith(".npz"))
    print(f"[feat] {len(clips)} clips in {args.face_cache}")

    device = args.device
    backbone = SpatialBackbone().to(device).eval()
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    with torch.no_grad():
        for clip_id in tqdm(clips):
            sp_path = os.path.join(out_dir, f"{clip_id}_spatial.npy")
            if os.path.exists(sp_path):
                continue
            npz = np.load(os.path.join(args.face_cache, f"{clip_id}.npz"))
            frames = npz["frames"].astype(np.float32) / 255.0     # (T, H, W, 3)
            rgb = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2))).to(device)
            rgb = (rgb - mean) / std

            feats = []
            for i in range(0, rgb.shape[0], args.batch):
                emb = backbone(rgb[i:i + args.batch])
                feats.append(emb.float().cpu())
            feats = torch.cat(feats, 0).numpy().astype(np.float16)
            np.save(sp_path, feats)


if __name__ == "__main__":
    main()
