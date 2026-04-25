"""MTCNN face detection + 5-point alignment + 224×224 crop, with explicit
handling of missed frames (reviewer #2 comment 7: reproducibility).

Fallback policy when MTCNN misses a frame:
    1. Copy the nearest preceding successful frame's aligned crop.
    2. If no preceding success, copy the nearest following.
    3. If the entire clip fails, fall back to a center-crop of the raw frame.

Per-video miss rate is recorded in the preprocessed .npz for the dataset
object to expose as a dataset-level statistic.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


# Five canonical points (VGGFace2 alignment template) for 224x224 output
_REF_PTS_224 = np.array([
    [70.7450, 108.2370],   # left eye
    [153.2550, 108.2370],  # right eye
    [112.0000, 143.6210],  # nose
    [82.7610,  177.4580],  # left mouth
    [141.2390, 177.4580],  # right mouth
], dtype=np.float32)


class FaceAligner:
    """Wraps facenet-pytorch MTCNN. Keeps it on a single device for throughput."""

    def __init__(self, device: str = "cuda"):
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(
            image_size=224, margin=0, keep_all=False,
            select_largest=True, post_process=False, device=device,
        )
        self.device = device

    # ------------------------------------------------------------------
    def _align_with_landmarks(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Similarity warp using 5 landmarks to canonical 224×224."""
        M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), _REF_PTS_224,
                                           method=cv2.LMEDS)
        if M is None:
            h, w = img.shape[:2]
            s = 224 / min(h, w)
            resized = cv2.resize(img, (int(w * s), int(h * s)))
            y0 = (resized.shape[0] - 224) // 2
            x0 = (resized.shape[1] - 224) // 2
            return resized[y0:y0 + 224, x0:x0 + 224]
        return cv2.warpAffine(img, M, (224, 224), borderValue=(0, 0, 0))

    # ------------------------------------------------------------------
    def align_batch(self, frames_bgr: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (aligned_rgb: T,H,W,3 uint8, hit_mask: T bool).

        Missing frames are filled via nearest-neighbour policy.
        """
        T = len(frames_bgr)
        aligned = np.zeros((T, 224, 224, 3), dtype=np.uint8)
        hit = np.zeros(T, dtype=bool)

        # Run MTCNN on PIL RGB
        pil_batch = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        try:
            boxes, probs, lms = self.mtcnn.detect(pil_batch, landmarks=True)
        except Exception:
            boxes, probs, lms = [None] * T, [None] * T, [None] * T

        for i, (f, bbs, lm) in enumerate(zip(frames_bgr, boxes, lms)):
            if bbs is None or lm is None or len(bbs) == 0:
                continue
            idx = int(np.argmax([p if p is not None else 0.0 for p in probs[i]]))
            warped = self._align_with_landmarks(f, lm[idx])
            aligned[i] = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            hit[i] = True

        # NN fallback forward, then backward
        if hit.any():
            for i in range(T):
                if not hit[i]:
                    # try previous
                    j = i - 1
                    while j >= 0 and not hit[j]:
                        j -= 1
                    if j >= 0:
                        aligned[i] = aligned[j]
                        continue
                    # try next
                    j = i + 1
                    while j < T and not hit[j]:
                        j += 1
                    if j < T:
                        aligned[i] = aligned[j]
        else:
            # worst case: whole clip failed -- center-crop everything
            for i, f in enumerate(frames_bgr):
                h, w = f.shape[:2]
                s = 224 / min(h, w)
                resized = cv2.resize(f, (int(w * s), int(h * s)))
                y0 = (resized.shape[0] - 224) // 2
                x0 = (resized.shape[1] - 224) // 2
                aligned[i] = cv2.cvtColor(
                    resized[y0:y0 + 224, x0:x0 + 224], cv2.COLOR_BGR2RGB,
                )

        return aligned, hit
