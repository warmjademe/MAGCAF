"""Per-task + averaged metrics, plus paired-significance tests for reviewer 2/3.

We deliberately compute everything from raw predictions so the CSV output is
self-sufficient for the LaTeX aggregation step (no Torch objects persisted).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .protocol import AFFECTIVE_TASKS, NUM_CLASSES_PER_TASK


@dataclass
class TaskMetrics:
    task: str
    accuracy: float
    macro_f1: float
    auc_ovr: float
    cohens_kappa: float
    per_class_f1: List[float]         # length = NUM_CLASSES_PER_TASK
    per_class_support: List[int]
    confusion: List[List[int]]        # NUM_CLASSES × NUM_CLASSES


@dataclass
class ModelMetrics:
    per_task: List[TaskMetrics]
    mean_accuracy: float
    mean_macro_f1: float
    mean_auc: float

    def to_dict(self) -> Dict:
        return {
            "per_task": [asdict(t) for t in self.per_task],
            "mean_accuracy": self.mean_accuracy,
            "mean_macro_f1": self.mean_macro_f1,
            "mean_auc": self.mean_auc,
        }


def _task_metrics(task_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> TaskMetrics:
    """
    y_true: (N,) int labels in [0, C-1]
    y_prob: (N, C) probabilities
    """
    y_pred = y_prob.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        # OvR macro AUC; requires all classes present in y_true for stability.
        # We pass labels=[0..C-1] and drop silently if any class absent in y_true.
        present = np.unique(y_true)
        if len(present) > 1 and y_prob.shape[1] >= len(present):
            auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=list(range(NUM_CLASSES_PER_TASK)),
            )
        else:
            auc = float("nan")
    except ValueError:
        auc = float("nan")

    kappa = cohen_kappa_score(y_true, y_pred)
    _, _, f1s, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES_PER_TASK)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES_PER_TASK)))

    return TaskMetrics(
        task=task_name,
        accuracy=float(acc),
        macro_f1=float(macro_f1),
        auc_ovr=float(auc),
        cohens_kappa=float(kappa),
        per_class_f1=[float(x) for x in f1s],
        per_class_support=[int(x) for x in support],
        confusion=cm.tolist(),
    )


def compute_all_metrics(
    y_trues: Dict[str, np.ndarray],
    y_probs: Dict[str, np.ndarray],
) -> ModelMetrics:
    per_task = [_task_metrics(t, y_trues[t], y_probs[t]) for t in AFFECTIVE_TASKS]
    mean_acc = float(np.mean([m.accuracy for m in per_task]))
    mean_f1 = float(np.mean([m.macro_f1 for m in per_task]))
    # ignore NaN AUCs in the mean
    aucs = [m.auc_ovr for m in per_task if not np.isnan(m.auc_ovr)]
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    return ModelMetrics(per_task, mean_acc, mean_f1, mean_auc)


# ---------------------------------------------------------------------------
# McNemar test for paired classifier comparison (reviewer #2 comment 6)
# ---------------------------------------------------------------------------
def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Exact McNemar test. Returns statistic + two-tailed p-value.

    pred_a, pred_b, y_true: 1-D int arrays, same length.
    """
    from scipy.stats import binomtest

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "statistic": 0.0}
    # Exact binomial test: under H0, b ~ Binomial(n, 0.5)
    res = binomtest(b, n, 0.5, alternative="two-sided")
    return {"b": b, "c": c, "p_value": float(res.pvalue), "statistic": float(b - c)}
