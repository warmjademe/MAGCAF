"""Pre-extract frozen EfficientNetV2-S features for ViBED-Net (face + scene).

Two streams per clip:
  <ClipID>_effnetv2_face.npy   (T=16, 1280)  float16 — face crops from face_cache
  <ClipID>_effnetv2_scene.npy  (T=16, 1280)  float16 — full-frame from .avi

Mirrors data/extract_transformer_features.py for TimeSformer / VideoMAE.

We use torchvision's EfficientNetV2-S (ImageNet-1k pretraining) instead of
timm's HuggingFace-hosted variant because the NAS firewall blocks
huggingface.co. The 21M-param backbone is identical in capacity to the
timm `tf_efficientnetv2_s` used by ViBED-Net's original paper; only the
21k -> 1k fine-tune differs.

Usage:
    python -m data.extract_vibednet_features \
        --weights /home/qyb/models/effnetv2s_torchvision.pth
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
from common.protocol import CFG  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_first_T_frames_avi(avi_path: str, T: int = 16, size: int = 224) -> np.ndarray | None:
    """Decode the leading T frames from .avi at 224x224, return (T,H,W,3) uint8."""
    import cv2
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        return None
    out = []
    while len(out) < T:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != size or frame.shape[1] != size:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        out.append(frame)
    cap.release()
    if not out:
        return None
    while len(out) < T:
        out.append(out[-1])
    return np.stack(out, axis=0).astype(np.uint8)


def build_avi_index(avi_root: str) -> dict[str, str]:
    m: dict[str, str] = {}
    for split in ("Train", "Validation", "Test"):
        sp = os.path.join(avi_root, split)
        if not os.path.isdir(sp):
            continue
        for user in os.listdir(sp):
            up = os.path.join(sp, user)
            if not os.path.isdir(up):
                continue
            for cid in os.listdir(up):
                cdir = os.path.join(up, cid)
                if not os.path.isdir(cdir):
                    continue
                for f in os.listdir(cdir):
                    if f.lower().endswith((".avi", ".mp4")):
                        m[cid] = os.path.join(cdir, f)
                        break
    return m


def prepare_batch(frames_u8: np.ndarray, size: int, device: str) -> torch.Tensor:
    x = torch.from_numpy(frames_u8).to(device).float().div_(255.0)
    x = x.permute(0, 3, 1, 2).contiguous()
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x


class _EffnetV2Trunk(nn.Module):
    """torchvision EfficientNetV2-S without classifier — outputs 1280-d feature."""

    def __init__(self, weights_path: str):
        super().__init__()
        from torchvision.models import efficientnet_v2_s
        m = efficientnet_v2_s(weights=None)
        sd = torch.load(weights_path, map_location="cpu")
        m.load_state_dict(sd, strict=True)
        self.features = m.features
        self.avgpool = m.avgpool
        # m.classifier = (Dropout, Linear(1280, 1000)); we drop both.
        self.feat_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


@torch.no_grad()
def forward_effnet(model, frames_u8: np.ndarray, device: str) -> np.ndarray:
    x = prepare_batch(frames_u8, size=224, device=device)
    feat = model(x)
    return feat.float().cpu().numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-cache", default=CFG.face_cache_dir)
    ap.add_argument("--avi-root",
                    default="/home/qyb/datasets/DAiSEE/DAiSEE/DataSet")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--weights",
        default="/home/qyb/models/effnetv2s_torchvision.pth",
        help="Local torchvision EfficientNetV2-S checkpoint .pth",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process the first N clips (smoke test).")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.face_cache), "feat_cache")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    clips = sorted(p[:-4] for p in os.listdir(args.face_cache) if p.endswith(".npz"))
    if args.limit > 0:
        clips = clips[: args.limit]
    print(f"[vibed-feat] {len(clips)} clips total")

    print(f"[vibed-feat] loading EfficientNetV2-S from {args.weights} ...")
    m = _EffnetV2Trunk(args.weights).to(args.device).eval()
    for p in m.parameters():
        p.requires_grad = False

    print(f"[vibed-feat] indexing .avi files under {args.avi_root} ...")
    avi_map = build_avi_index(args.avi_root)
    print(f"[vibed-feat] indexed {len(avi_map)} video files (.avi/.mp4)")

    n_face = n_scene = n_skip = n_no_avi = 0
    for clip_id in tqdm(clips):
        face_path = os.path.join(out_dir, f"{clip_id}_effnetv2_face.npy")
        scene_path = os.path.join(out_dir, f"{clip_id}_effnetv2_scene.npy")
        if os.path.exists(face_path) and os.path.exists(scene_path):
            n_skip += 1
            continue

        if not os.path.exists(face_path):
            npz = np.load(os.path.join(args.face_cache, f"{clip_id}.npz"))
            faces = npz["frames"]
            feat = forward_effnet(m, faces, args.device)
            np.save(face_path, feat)
            n_face += 1

        if not os.path.exists(scene_path):
            avi = avi_map.get(clip_id)
            if avi is None:
                n_no_avi += 1
                continue
            frames = load_first_T_frames_avi(avi, T=CFG.clip_frames, size=224)
            if frames is None:
                n_no_avi += 1
                continue
            feat = forward_effnet(m, frames, args.device)
            np.save(scene_path, feat)
            n_scene += 1

    print(f"[vibed-feat] done. face={n_face} scene={n_scene} "
          f"skipped(both-cached)={n_skip} no_avi={n_no_avi}")


if __name__ == "__main__":
    main()
