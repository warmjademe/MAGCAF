"""Uniform-16 variant of the face cache for the frame-sampling check.

Decodes each listed clip at CFG.fps, samples CFG.clip_frames indices uniformly
across the full clip (np.linspace), and runs the same MTCNN alignment as the
main preprocessing. RAFT flow is skipped (no compared model consumes it).
Output tree is independent of the main cache.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.face_pipeline import FaceAligner
from common.protocol import CFG

UROOT = os.environ.get("SAMPLING_CHECK_ROOT", "/path/to/uni16")
OUTDIR = os.path.join(UROOT, "face_cache")
os.makedirs(OUTDIR, exist_ok=True)
clip_ids = [l.strip() for l in open(os.path.join(UROOT, "shadow_root", "clips.txt")) if l.strip()]

vmap = {}
for root, _, files in os.walk(os.path.join(CFG.dataset_root, "DataSet")):
    for f in files:
        if f.lower().endswith((".avi", ".mp4", ".mov")):
            vmap[os.path.splitext(f)[0]] = os.path.join(root, f)

aligner = FaceAligner(device="cuda")
done = fail = 0
for cid in clip_ids:
    out = os.path.join(OUTDIR, f"{cid}.npz")
    if os.path.exists(out):
        continue
    vp = vmap.get(cid)
    if vp is None:
        fail += 1
        continue
    cap = cv2.VideoCapture(vp)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / CFG.fps)))
    frames, idx = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(fr)
        idx += 1
    cap.release()
    if not frames:
        fail += 1
        continue
    sel = np.linspace(0, len(frames) - 1, CFG.clip_frames).round().astype(int)
    aligned, hit = aligner.align_batch([frames[j] for j in sel])
    np.savez_compressed(out, frames=aligned, flows=np.zeros((1,), np.int8),
                        hit_mask=hit, miss_rate=float(1.0 - hit.mean()), clip_id=cid)
    done += 1
print(f"uniform-16 preprocess: done={done} fail={fail}")
