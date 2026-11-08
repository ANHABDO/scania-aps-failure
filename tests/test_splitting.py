"""Tests for src/splitting.py on a small fake dataset (1,000 rows, 50 positives)."""

import numpy as np
import pandas as pd

from src.splitting import split_features_target, train_val_split


def fake_data(n=1000, n_pos=50):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0, 100, size=(n, 3)), columns=["a", "b", "c"])
    df.insert(0, "class", ["pos"] * n_pos + ["neg"] * (n - n_pos))
    return df


def test_target_is_encoded_pos_1_neg_0():
    X, y = split_features_target(fake_data())
    assert "class" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 50


def test_split_sizes_and_stratification():
    X, y = split_features_target(fake_data())
    X_tr, X_val, y_tr, y_val = train_val_split(X, y)
    assert len(X_val) == 200 and len(X_tr) == 800
    assert y_val.sum() == 10 and y_tr.sum() == 40      # 5% positives in both parts


def test_train_and_validation_do_not_overlap():
    X, y = split_features_target(fake_data())
    X_tr, X_val, _, _ = train_val_split(X, y)
    assert set(X_tr.index).isdisjoint(X_val.index)
    assert len(X_tr) + len(X_val) == len(X)


def test_same_seed_gives_same_split():
    X, y = split_features_target(fake_data())
    first = train_val_split(X, y)[1].index
    second = train_val_split(X, y)[1].index
    assert first.equals(second)


def test_different_seed_gives_different_split():
    X, y = split_features_target(fake_data())
    a = train_val_split(X, y, seed=1)[1].index
    b = train_val_split(X, y, seed=2)[1].index
    assert not a.equals(b)
