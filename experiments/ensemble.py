"""Average softmax probabilities of multiple runs → ensemble metrics.

Usage:
    python experiments/ensemble.py \
        --runs runs/pilot/v4_ce_d512__s42 \
               runs/pilot/magcafv2_ce__s42 \
               runs/pilot/magcafv2_noom__s42
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.metrics import compute_all_metrics                # noqa: E402
from common.protocol import AFFECTIVE_TASKS                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    probs_per_task = {t: [] for t in AFFECTIVE_TASKS}
    trues = None
    for rd in args.runs:
        p = os.path.join(rd, "predictions.npz")
        if not os.path.exists(p):
            print(f"[warn] missing {p}"); continue
        npz = np.load(p)
        for t in AFFECTIVE_TASKS:
            probs_per_task[t].append(npz[f"prob_{t}"])
        if trues is None:
            trues = {t: npz[f"true_{t}"] for t in AFFECTIVE_TASKS}

    mean_probs = {t: np.mean(np.stack(probs_per_task[t], 0), axis=0)
                  for t in AFFECTIVE_TASKS}
    metrics = compute_all_metrics(trues, mean_probs)

    print(f"{'task':<14}  {'acc':>7}  {'macro-f1':>9}  {'auc':>7}")
    for m in metrics.per_task:
        print(f"{m.task:<14}  {m.accuracy:>7.4f}  {m.macro_f1:>9.4f}  {m.auc_ovr:>7.4f}")
    print("-" * 44)
    print(f"{'MEAN':<14}  {metrics.mean_accuracy:>7.4f}  "
          f"{metrics.mean_macro_f1:>9.4f}  {metrics.mean_auc:>7.4f}")


if __name__ == "__main__":
    main()
