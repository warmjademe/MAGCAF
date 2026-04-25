"""Aggregate per-task metrics across seeds and produce LaTeX-ready tables for
the revision.

Outputs to stdout three blocks:
    [RQ1] Table 2 rows with per-task accuracy for each modern baseline
          (3-seed mean, per-task Top-1).
    [RQ2] Component-ablation table: Full vs. no_magcaf vs. no_omega vs. no_uw
          × 3 seeds, per-task Top-1, with Macro-F1 and AUC.
    [RQ3] Per-task source-attention gate matrix alpha_{k,j} from the single
          gate-dump run.

Usage:
    python experiments/aggregate_tables.py --runs /home/qyb/TongBu/discover_artifical_intelligence/runs/pilot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

TASKS = ["boredom", "engagement", "confusion", "frustration"]
TASK_SHORT = ["Bor", "Eng", "Con", "Fru"]


def load_preds(run_dir: str):
    p = os.path.join(run_dir, "predictions.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)


def per_task_acc(run_dir: str):
    z = load_preds(run_dir)
    if z is None:
        return None
    out = {}
    for t in TASKS:
        pr = z[f"prob_{t}"].argmax(1)
        tr = z[f"true_{t}"]
        out[t] = float((pr == tr).mean())
    return out


def load_metrics(run_dir: str):
    m = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(m):
        return None
    return json.load(open(m))


def aggregate_seeds(runs_dir: str, base: str, seeds):
    """For each task + overall Macro-F1 + AUC, return mean±std across seeds."""
    per_task = {t: [] for t in TASKS}
    accs, f1s, aucs = [], [], []
    for s in seeds:
        rd = os.path.join(runs_dir, f"{base}__s{s}")
        met = load_metrics(rd)
        pta = per_task_acc(rd)
        if met is None or pta is None:
            continue
        for t in TASKS:
            per_task[t].append(pta[t])
        accs.append(met["test"]["mean_accuracy"])
        f1s.append(met["test"]["mean_macro_f1"])
        aucs.append(met["test"]["mean_auc"])
    n = len(accs)
    if n == 0:
        return None
    return {
        "n_seeds": n,
        "per_task_mean": {t: 100 * np.mean(per_task[t]) for t in TASKS},
        "per_task_std":  {t: 100 * np.std(per_task[t])  for t in TASKS},
        "avg_acc_mean": 100 * np.mean(accs),
        "avg_acc_std":  100 * np.std(accs),
        "f1_mean": np.mean(f1s),
        "f1_std":  np.std(f1s),
        "auc_mean": np.mean(aucs),
        "auc_std":  np.std(aucs),
    }


def fmt_pp(m, s, digits=2):
    return f"{m:.{digits}f}$\\pm${s:.{digits}f}"


def render_rq1_row(name, stats):
    if stats is None:
        return f"% MISSING: {name}"
    pt = stats["per_task_mean"]; ps = stats["per_task_std"]
    cells = [fmt_pp(pt[t], ps[t], 2) for t in TASKS]
    cells += [fmt_pp(stats["avg_acc_mean"], stats["avg_acc_std"], 2)]
    cells += [fmt_pp(stats["f1_mean"], stats["f1_std"], 3)]
    cells += [fmt_pp(stats["auc_mean"], stats["auc_std"], 3)]
    return f"\\quad {name} & " + " & ".join(cells) + " \\\\"


def render_rq2_row(label, stats):
    if stats is None:
        return f"% MISSING: {label}"
    pt = stats["per_task_mean"]; ps = stats["per_task_std"]
    cells = [fmt_pp(pt[t], ps[t], 2) for t in TASKS]
    cells += [fmt_pp(stats["avg_acc_mean"], stats["avg_acc_std"], 2)]
    cells += [fmt_pp(stats["f1_mean"], stats["f1_std"], 3)]
    return f"  {label} & " + " & ".join(cells) + " \\\\"


def render_rq3_gates(runs_dir):
    p = os.path.join(runs_dir, "rq3_gates__s42", "gate_attn.npy")
    if not os.path.exists(p):
        return "% MISSING: rq3_gates__s42/gate_attn.npy"
    attn = np.load(p)    # (N, K, n_src)
    mean_attn = attn.mean(0)   # (K, n_src)  -- per-task mean over test set
    sources = ["Face (IRv1)", "VideoMAE", "TimeSformer"]
    rows = [f"  \\textbf{{{TASK_SHORT[k]}}} & " +
            " & ".join(f"{mean_attn[k, j]:.3f}" for j in range(mean_attn.shape[1])) +
            " \\\\"
            for k in range(len(TASK_SHORT))]
    dom = np.argmax(mean_attn, axis=1)   # per task dominant source idx
    return {
        "rows": "\n".join(rows),
        "dominant_per_task": {TASK_SHORT[k]: sources[dom[k]]
                              for k in range(len(TASK_SHORT))},
        "mean_attn": mean_attn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/pilot")
    args = ap.parse_args()

    seeds_baseline = [42, 123, 2024]
    seeds_rq2 = [42, 123, 2024]

    print("% ====== [RQ1] MODERN BASELINES per-task (3-seed mean+/-std) ======")
    for name, base in [
        ("TimeSformer~\\cite{bertasius2021space}", "timesformer__ce"),
        ("VideoMAE~\\cite{tong2022videomae}",      "videomae__ce"),
        ("ResNet-TCN~\\cite{abedi2021improving}",  "resnet_tcn__ce"),
    ]:
        st = aggregate_seeds(args.runs, base, seeds_baseline)
        print(render_rq1_row(name, st))

    print("\n% ====== [RQ2] COMPONENT ABLATION (3-seed mean+/-std) ======")
    for label, base in [
        ("MAGCAF-Net (full)",              "rq2_full"),
        ("\\quad $-$ M1 (gated $\\to$ concat fusion)", "rq2_no_magcaf"),
        ("\\quad $-$ M2 ($\\Omega$ head)",              "rq2_no_omega"),
        ("\\quad $-$ M3 (uncertainty weighting)",       "rq2_no_uw"),
    ]:
        st = aggregate_seeds(args.runs, base, seeds_rq2)
        print(render_rq2_row(label, st))

    print("\n% ====== [RQ3] Per-task source-attention gates (single seed 42) ======")
    r3 = render_rq3_gates(args.runs)
    if isinstance(r3, dict):
        print(r3["rows"])
        print("% dominant source per task:")
        for k, v in r3["dominant_per_task"].items():
            print(f"%   {k}: {v}")
    else:
        print(r3)


if __name__ == "__main__":
    main()
