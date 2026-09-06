"""Classification metrics, with an internal consistency guard.

The guard exists for a specific reason. AgriTL-ViT (ESWA 2025) reports Tomato
Early_blight as precision 0.977, recall 0.958, F1 0.857 - an F1 below both of its
inputs, which is impossible, since F1 is the harmonic mean and therefore always lies
between them. Rather than only pointing that out in the write-up, we make our own
tables incapable of carrying the same error.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

_TOL = 1e-6


class MetricConsistencyError(AssertionError):
    """Raised when per-class metrics violate an algebraic invariant."""


def _check_consistency(precision, recall, f1, labels) -> None:
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        if p == 0 and r == 0:
            continue  # F1 is defined as 0 here; the bound below is vacuous
        lo, hi = min(p, r), max(p, r)
        if not (lo - _TOL <= f <= hi + _TOL):
            raise MetricConsistencyError(
                f"class {labels[i]!r}: F1={f:.4f} lies outside [{lo:.4f}, {hi:.4f}] "
                f"(precision={p:.4f}, recall={r:.4f}). F1 is the harmonic mean of "
                f"precision and recall and must lie between them."
            )
        harmonic = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        if abs(harmonic - f) > 1e-4:
            raise MetricConsistencyError(
                f"class {labels[i]!r}: F1={f:.4f} but 2PR/(P+R)={harmonic:.4f}"
            )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """Overall + per-class metrics. Support columns are derived from ``y_true``.

    Support is computed here rather than copied between tables. AgriTL-ViT's Rice table
    reuses the Tomato support column verbatim across datasets of different sizes; a
    derived column cannot do that.
    """
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    _check_consistency(precision, recall, f1, class_names)

    total_support = int(support.sum())
    if total_support != len(y_true):
        raise MetricConsistencyError(
            f"support sums to {total_support} but there are {len(y_true)} samples"
        )

    # Average over the fixed label set rather than letting sklearn infer it. On the
    # cross-domain split some classes have zero support (PlantDoc has no Target_Spot
    # folder), and an inferred label set would include such a class only when the model
    # happened to predict it - making macro-F1 depend on the model's guesses rather than
    # on the data. macro_f1_present is the honest figure to quote there: it averages
    # only over classes the evaluation set actually contains.
    present = support > 0
    macro_f1 = float(f1.mean())
    macro_f1_present = float(f1[present].mean()) if present.any() else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
        "classes_present": int(present.sum()),
        "classes_total": len(class_names),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "n_samples": int(len(y_true)),
        "per_class": {
            name: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for name, p, r, f, s in zip(class_names, precision, recall, f1, support)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


class AverageMeter:
    """Running mean of a scalar, weighted by batch size."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0
