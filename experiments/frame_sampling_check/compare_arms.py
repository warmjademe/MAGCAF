"""Aggregate the frame-sampling check (paper Table 2): per-model metrics for
both arms plus McNemar's paired two-sided test pooled over the four tasks."""
import json
import os

import numpy as np
from scipy.stats import binomtest

TASKS = ["boredom", "engagement", "confusion", "frustration"]
BASE = os.environ.get("RUNS_DIR", "runs/pilot") + "/"

def load(stem):
    m = json.load(open(BASE + stem + "__s42/metrics.json"))
    per = {t["task"]: t for t in m["test"]["per_task"]}
    d = np.load(BASE + stem + "__s42/predictions.npz")
    preds = {t: (d[f"prob_{t}"].argmax(1), d[f"true_{t}"]) for t in TASKS}
    return (np.mean([per[t]["accuracy"] for t in TASKS]) * 100,
            np.mean([per[t]["macro_f1"] for t in TASKS]),
            np.mean([per[t]["auc_ovr"] for t in TASKS]), preds)

for model in ["magcaf", "timesformer", "videomae"]:
    avgA, f1A, aucA, pA = load(f"sub33_first16_{model}")
    avgB, f1B, aucB, pB = load(f"uni16_{model}")
    b = c = 0
    for t in TASKS:
        prA, y = pA[t]
        prB, _ = pB[t]
        okA, okB = prA == y, prB == y
        b += int((okA & ~okB).sum())
        c += int((~okA & okB).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    print(f"{model:12s} f16 {avgA:.2f}/{f1A:.3f}/{aucA:.3f}  "
          f"u16 {avgB:.2f}/{f1B:.3f}/{aucB:.3f}  McNemar p={p:.4f}")
