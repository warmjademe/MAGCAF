"""Single-pass GPU preprocessor: video.avi -> aligned face frames + RAFT flow.

Output per clip -> `<face_cache>/<ClipID>.npz`:
    frames   : uint8 (T, 224, 224, 3)   RGB aligned faces
    flows    : int8  (T-1, 224, 224, 2) RAFT (or TV-L1) optical flows
    hit_mask : bool  (T,)               per-frame face detection hit
    miss_rate: float

Plan A choice: RAFT-small on GPU replaces the paper's TV-L1 for both
throughput (~500 frame-pairs/s on RTX 4090 vs. ~2 f/s on CPU) and accuracy
(CVPR'20 best paper, still SOTA-competitive). See data/flow_cache.py for the
ablation-friendly `compute_tvl1_flows` legacy path.

Two-stage legacy flags still exist:
    --stage faces        : MTCNN-only, save _frames_raw.npy + _hit.npy stubs
    --stage flows        : read stubs, run RAFT, pack final .npz
    --stage all          : face + flow back-to-back (DEFAULT, fastest)

    --flow-backend raft  : default
    --flow-backend tvl1  : legacy CPU path (ablation)

Usage:
    python -m data.preprocess_clip --split Validation            # all stages, RAFT
    python -m data.preprocess_clip --split Train --flow-backend raft
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.protocol import CFG, resolve_split_dir           # noqa: E402
from data.flow_cache import compute_raft_flows, compute_tvl1_flows  # noqa: E402


STAGE1_SUFFIX = "_frames_raw.npy"
STAGE1_HIT_SUFFIX = "_hit.npy"


def _sample_frames(video_path: str, fps: int, target_T: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        return frames
    if len(frames) >= target_T:
        return frames[:target_T]
    return frames + [frames[-1]] * (target_T - len(frames))


def _walk_videos(split_root: str) -> List[str]:
    out = []
    for root, _, files in os.walk(split_root):
        for f in files:
            if f.lower().endswith((".avi", ".mp4", ".mov")):
                out.append(os.path.join(root, f))
    out.sort()
    return out


# ---------------------------------------------------------------------------
def process_single_clip(aligner, vp: str, face_cache: str,
                        flow_backend: str, device: str) -> None:
    clip_id = Path(vp).stem
    out_path = os.path.join(face_cache, f"{clip_id}.npz")
    if os.path.exists(out_path):
        return

    frames_bgr = _sample_frames(vp, CFG.fps, CFG.clip_frames)
    if not frames_bgr:
        print(f"[skip] empty {vp}", file=sys.stderr)
        return
    aligned_rgb, hit = aligner.align_batch(frames_bgr)
    if flow_backend == "raft":
        flows = compute_raft_flows(aligned_rgb, device=device)
    else:
        import cv2
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in aligned_rgb]
        flows = compute_tvl1_flows(gray)

    np.savez_compressed(
        out_path,
        frames=aligned_rgb,
        flows=flows,
        hit_mask=hit,
        miss_rate=float(1.0 - hit.mean()),
        clip_id=clip_id,
    )


def stage_all(dataset_root: str, split: str, face_cache: str,
              flow_backend: str, device: str):
    from data.face_pipeline import FaceAligner
    Path(face_cache).mkdir(parents=True, exist_ok=True)
    aligner = FaceAligner(device=device)

    split_root = resolve_split_dir(dataset_root, split)
    videos = _walk_videos(split_root)
    print(f"[{split}] {len(videos)} videos under {split_root}")

    # Also pick up clips stuck in the old 2-stage format (have stubs, no .npz)
    for vp in tqdm(videos, desc=f"{split}/{flow_backend}"):
        clip_id = Path(vp).stem
        out_path = os.path.join(face_cache, f"{clip_id}.npz")
        if os.path.exists(out_path):
            continue
        stubs_rgb = os.path.join(face_cache, f"{clip_id}{STAGE1_SUFFIX}")
        stubs_hit = os.path.join(face_cache, f"{clip_id}{STAGE1_HIT_SUFFIX}")
        try:
            if os.path.exists(stubs_rgb) and os.path.exists(stubs_hit):
                # Recover from legacy 2-stage intermediate
                aligned_rgb = np.load(stubs_rgb)
                hit = np.load(stubs_hit)
                if flow_backend == "raft":
                    flows = compute_raft_flows(aligned_rgb, device=device)
                else:
                    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in aligned_rgb]
                    flows = compute_tvl1_flows(gray)
                np.savez_compressed(
                    out_path,
                    frames=aligned_rgb,
                    flows=flows,
                    hit_mask=hit,
                    miss_rate=float(1.0 - hit.mean()),
                    clip_id=clip_id,
                )
                os.remove(stubs_rgb)
                os.remove(stubs_hit)
            else:
                process_single_clip(aligner, vp, face_cache, flow_backend, device)
        except Exception as e:
            print(f"[err] {vp}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["Train", "Validation", "Test", "all"],
                    default="all")
    ap.add_argument("--dataset-root", default=CFG.dataset_root)
    ap.add_argument("--face-cache", default=CFG.face_cache_dir)
    ap.add_argument("--flow-backend", choices=["raft", "tvl1"], default="raft")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    splits = ["Train", "Validation", "Test"] if args.split == "all" else [args.split]
    t0 = time.time()
    for s in splits:
        stage_all(args.dataset_root, s, args.face_cache,
                  args.flow_backend, args.device)
    print(f"[preprocess] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
