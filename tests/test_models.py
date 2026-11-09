"""Tests for src/models.py: the full pipeline trains and predicts on raw-looking data."""

import numpy as np
import pandas as pd

from src.models import make_logreg_pipeline
from src.preprocessing import AC_CODE_BASE


def raw_like_data(n=200, seed=0):
    """Small fake data with the quirks of the real data: missing values, ac_000 code."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "ac_000": rng.integers(0, 500, n).astype(float),
        "ad_000": rng.integers(0, 500, n).astype(float),
        "cf_000": rng.integers(0, 500, n).astype(float),
        "co_000": rng.integers(0, 500, n).astype(float),
        "x_000": rng.integers(0, 1000, n).astype(float),
    })
    y = pd.Series((X["x_000"] > 800).astype(int), name="target")   # learnable rule
    X.loc[::7, "x_000"] = np.nan
    X.loc[::5, "ac_000"] = AC_CODE_BASE
    return X, y


def test_pipeline_fits_and_predicts_zeros_and_ones():
    X, y = raw_like_data()
    model = make_logreg_pipeline().fit(X, y)
    pred = model.predict(X)
    assert len(pred) == len(X)
    assert set(np.unique(pred)) <= {0, 1}


def test_predicted_probabilities_are_between_0_and_1():
    X, y = raw_like_data()
    proba = make_logreg_pipeline(class_weight="balanced").fit(X, y).predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_learns_a_simple_rule():
    X, y = raw_like_data()
    model = make_logreg_pipeline().fit(X, y)
    X_new, y_new = raw_like_data(seed=1)
    assert (model.predict(X_new) == y_new).mean() > 0.8


# ---------------------------------------------------------------- gradient boosting
from src.models import make_hgb_pipeline  # noqa: E402


def test_hgb_handles_missing_values_without_imputation():
    X, y = raw_like_data()
    assert X.isna().any().any()                     # the data really has missing values
    proba = make_hgb_pipeline().fit(X, y).predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_hgb_learns_a_simple_rule():
    X, y = raw_like_data(n=400)
    model = make_hgb_pipeline().fit(X, y)
    X_new, y_new = raw_like_data(n=400, seed=1)
    assert (model.predict(X_new) == y_new).mean() > 0.8


def test_hgb_is_repeatable_with_fixed_seed():
    X, y = raw_like_data(n=400)
    a = make_hgb_pipeline().fit(X, y).predict_proba(X)[:, 1]
    b = make_hgb_pipeline().fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(a, b)
