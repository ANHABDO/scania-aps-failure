"""Helpers for error analysis and feature importance."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def outcome_labels(y_true, y_pred) -> np.ndarray:
    """Name each prediction: TP (caught), FN (missed), FP (false alarm), TN (correct no)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.select(
        [(y_true == 1) & (y_pred == 1), (y_true == 1) & (y_pred == 0),
         (y_true == 0) & (y_pred == 1)],
        ["TP", "FN", "FP"], default="TN",
    )


def group_permutation_importance(model, X: pd.DataFrame, y, groups: dict,
                                 n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """Shuffle a WHOLE GROUP of columns together and measure the drop in average precision.

    Shuffling the group's rows together destroys the group's information all at once,
    so correlated columns inside the group cannot "cover" for each other.
    """
    rng = np.random.default_rng(seed)
    baseline = average_precision_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            order = rng.permutation(len(X))
            X_shuffled[cols] = X[cols].to_numpy()[order]     # same row order for all columns
            score = average_precision_score(y, model.predict_proba(X_shuffled)[:, 1])
            drops.append(baseline - score)
        rows.append({"group": name, "n_columns": len(cols),
                     "importance_mean": np.mean(drops), "importance_std": np.std(drops)})
    return pd.DataFrame(rows).set_index("group").sort_values("importance_mean", ascending=False)
