"""Turn predicted probabilities into decisions, and find the cheapest threshold.

A truck is predicted "pos" (1) when its probability >= threshold.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from src.metrics import COST_FN, COST_FP

# Fine steps near 0 (0.001, 0.002, ... 0.049), then 0.05, 0.06, ... 0.99.
# Low thresholds get finer steps because with a cost ratio of 50:1 the best one may be small.
DEFAULT_THRESHOLDS = np.unique(np.round(np.concatenate([
    np.arange(0.001, 0.05, 0.001),
    np.arange(0.05, 1.00, 0.01),
]), 3))


def predict_at(proba, threshold: float) -> np.ndarray:
    """0/1 decisions: 1 when probability >= threshold."""
    return (np.asarray(proba) >= threshold).astype(int)


def cost_at_thresholds(y_true, proba, thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """For every threshold: counts, precision, recall and official cost."""
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        pred = predict_at(proba, t)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        rows.append({
            "threshold": t,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "cost": COST_FP * fp + COST_FN * fn,
        })
    return pd.DataFrame(rows).set_index("threshold")


def best_threshold(table: pd.DataFrame) -> float:
    """Threshold with the lowest cost (if several tie, the smallest one)."""
    return float(table["cost"].idxmin())


def out_of_fold_proba(model, X, y, cv) -> np.ndarray:
    """Cross-validated probability of "pos" for every row (each row predicted
    by a model trained without it). The whole pipeline is refitted in each fold."""
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
