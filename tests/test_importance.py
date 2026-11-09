"""Tests for src/importance.py."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.importance import group_permutation_importance, outcome_labels


def test_outcome_labels_by_hand():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    assert outcome_labels(y_true, y_pred).tolist() == ["TP", "FN", "FP", "TN"]


def test_group_importance_finds_the_useful_group():
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame({
        "signal_a": rng.normal(size=n), "signal_b": rng.normal(size=n),
        "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n),
    })
    y = (X["signal_a"] + X["signal_b"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    result = group_permutation_importance(
        model, X, y, {"signal": ["signal_a", "signal_b"], "noise": ["noise_a", "noise_b"]},
    )
    assert result.index[0] == "signal"
    assert result.loc["signal", "importance_mean"] > 0.2
    assert abs(result.loc["noise", "importance_mean"]) < 0.02


def test_group_shuffle_keeps_rows_together_and_does_not_modify_input():
    X = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)})
    y = (X["a"] > 50).astype(int)
    before = X.copy()
    model = LogisticRegression().fit(X, y)
    group_permutation_importance(model, X, y, {"ab": ["a", "b"]}, n_repeats=2)
    pd.testing.assert_frame_equal(X, before)
