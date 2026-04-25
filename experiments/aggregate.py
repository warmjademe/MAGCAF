"""Aggregate all runs/<group>/*/metrics.json into mean±std LaTeX tables.

Usage:
    python experiments/aggregate.py --runs runs --group phaseB --out tables/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.protocol import AFFECTIVE_TASKS   # noqa: E402


def load_run(run_dir: str) -> Dict | None:
    mpath = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(mpath):
        return None
    with open(mpath, "r") as f:
        return json.load(f)


def collect(runs_dir: str, group: str) -> Dict[Tuple[str, str], List[Dict]]:
    """Key = (model, loss); value = list of run dicts across seeds."""
    out: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    gdir = os.path.join(runs_dir, group)
    if not os.path.isdir(gdir):
        return out
    for name in sorted(os.listdir(gdir)):
        rd = os.path.join(gdir, name)
        if not os.path.isdir(rd):
            continue
        # name = <model>__<loss>__s<seed>
        parts = name.split("__")
        if len(parts) < 3:
            continue
        model = parts[0]
        loss = parts[1]
        seed = parts[2].lstrip("s")
        m = load_run(rd)
        if m is None:
            continue
        m["_seed"] = int(seed)
        out[(model, loss)].append(m)
    return out


def fmt(mean: float, std: float, scale: float = 100.0, digits: int = 1) -> str:
    return f"{mean * scale:.{digits}f}$\\pm${std * scale:.{digits}f}"


def build_main_table(bucket) -> str:
    """Produce the revised Table 2 (Phase B): per-task Top-1 Acc + Macro-F1 + AUC."""
    rows = []
    rows.append(
        r"\begin{tabular}{lccccc|ccc}"
        "\n"
        r"\toprule"
        "\n"
        r"\multirow{2}{*}{\textbf{Model}} & "
        r"\multicolumn{4}{c}{\textbf{Top-1 Accuracy (\%)}} & "
        r"\textbf{Avg Acc} & "
        r"\textbf{Macro-F1} & \textbf{AUC} & \textbf{Kappa} \\"
        "\n"
        r" & Bor & Eng & Con & Fru & (\%) & & & \\"
        "\n"
        r"\midrule"
    )
    for (model, loss), runs in sorted(bucket.items()):
        accs_per_task = [[] for _ in AFFECTIVE_TASKS]
        accs_mean = []
        f1s = []
        aucs = []
        kappas = []
        for m in runs:
            pt = m["test"]["per_task"]
            for i, t in enumerate(pt):
                accs_per_task[i].append(t["accuracy"])
                kappas.append(t["cohens_kappa"])
            accs_mean.append(m["test"]["mean_accuracy"])
            f1s.append(m["test"]["mean_macro_f1"])
            aucs.append(m["test"]["mean_auc"])

        cells = [
            fmt(np.mean(pt), np.std(pt)) for pt in accs_per_task
        ]
        cells.append(fmt(np.mean(accs_mean), np.std(accs_mean)))
        cells.append(fmt(np.mean(f1s), np.std(f1s), scale=1, digits=3))
        cells.append(fmt(np.mean(aucs), np.std(aucs), scale=1, digits=3))
        cells.append(fmt(np.mean(kappas), np.std(kappas), scale=1, digits=3))

        label = f"{model} ({loss})" if loss != "cb_focal" else f"{model}"
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", default="tables")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    bucket = collect(args.runs, args.group)
    if not bucket:
        print(f"[aggregate] no runs found in {args.runs}/{args.group}")
        return

    tex = build_main_table(bucket)
    out_path = os.path.join(args.out, f"{args.group}.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[aggregate] wrote {out_path}")
    print(tex)


if __name__ == "__main__":
    main()
