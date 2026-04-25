"""Per-clip facial-landmark features via MediaPipe FaceMesh.

Adds a 4th heterogeneous source to MAGCAF: structured per-frame facial
geometry. Unlike the three RGB-encoder sources (IRv1 / VideoMAE / TimeSformer),
landmarks live in a 3D coordinate space (478 keypoints x (x, y, z)) — a
genuinely orthogonal representation of the same face, capturing geometry
rather than appearance.

We run MediaPipe FaceMesh (refine_landmarks=True, 478 points incl. iris) on
each cached face crop, propagate the previous frame's landmarks if any frame
fails detection, and store the per-clip tensor.

Output: <feat_cache>/<ClipID>_landmarks.npy   shape (16, 478, 3) float16

The model loads this tensor at training time and applies its own temporal
encoder + projection inside the fusion layer (so the cache stays raw).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.protocol import CFG                            # noqa: E402


def main():
    import mediapipe as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-cache", default=CFG.face_cache_dir)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-detect-conf", type=float, default=0.3)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.face_cache), "feat_cache")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    clips = sorted(p[:-4] for p in os.listdir(args.face_cache)
                   if p.endswith(".npz"))
    print(f"[lm] {len(clips)} clips, out_dir={out_dir}")

    fmesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=args.min_detect_conf,
    )

    n_full_miss = 0
    for clip_id in tqdm(clips):
        out_path = os.path.join(out_dir, f"{clip_id}_landmarks.npy")
        if os.path.exists(out_path):
            continue
        npz = np.load(os.path.join(args.face_cache, f"{clip_id}.npz"))
        frames = npz["frames"]                                  # (T,H,W,3) uint8
        T = frames.shape[0]
        out = np.zeros((T, 478, 3), dtype=np.float32)
        last_valid = None
        miss = 0
        for t in range(T):
            r = fmesh.process(frames[t])
            if r.multi_face_landmarks:
                lms = r.multi_face_landmarks[0].landmark
                arr = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                               dtype=np.float32)
                out[t] = arr
                last_valid = arr
            else:
                miss += 1
                if last_valid is not None:
                    out[t] = last_valid
        if last_valid is None:
            # Whole clip failed; back-fill from any future success was already
            # tried via the loop but nothing came; leave zeros and mark.
            n_full_miss += 1
        np.save(out_path, out.astype(np.float16))

    fmesh.close()
    print(f"[lm] done. clips with no detection at all: {n_full_miss}")


if __name__ == "__main__":
    main()
