"""Tests for src/preprocessing.py on tiny hand-made data."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import AC_CODE_BASE, CleanRawValues, make_preprocessor


def tiny_train():
    return pd.DataFrame({
        "ac_000": [10.0, AC_CODE_BASE + 4, 30.0, np.nan],   # row 1 holds the code
        "ad_000": [1.0, 2.0, 8.58e9, 4.0],                    # row 2 is "corrupted"
        "cf_000": [1.0, 2.0, 3.0, 4.0],
        "co_000": [1.0, 2.0, 3.0, 4.0],
        "cd_000": [5.0, 5.0, 5.0, 5.0],                       # constant
        "x_000":  [0.0, 100.0, np.nan, 300.0],
    })


# ---------------------------------------------------------------- step 1: cleaning
def test_ac_code_becomes_flag_plus_missing():
    out = CleanRawValues().fit_transform(tiny_train())
    assert out["ac_000_is_code"].tolist() == [0, 1, 0, 0]
    assert np.isnan(out.loc[1, "ac_000"])
    assert out.loc[0, "ac_000"] == 10            # normal values untouched


def test_corrupted_values_become_missing():
    out = CleanRawValues().fit_transform(tiny_train())
    assert np.isnan(out.loc[2, "ad_000"])
    assert out.loc[0, "ad_000"] == 1


def test_constant_column_learned_on_train_is_dropped_everywhere():
    clean = CleanRawValues().fit(tiny_train())
    assert clean.constant_columns_ == ["cd_000"]
    val = tiny_train()
    val["cd_000"] = [1.0, 2.0, 3.0, 4.0]          # not constant in validation...
    assert "cd_000" not in clean.transform(val).columns   # ...but still dropped: train decides


def test_input_data_is_not_modified():
    data = tiny_train()
    before = data.copy()
    CleanRawValues().fit_transform(data)
    pd.testing.assert_frame_equal(data, before)


# ---------------------------------------------------------------- full pipeline
def test_pipeline_output_has_no_missing_values():
    out = make_preprocessor().fit_transform(tiny_train())
    assert out.isna().sum().sum() == 0


def test_pipeline_adds_missing_indicators():
    out = make_preprocessor().fit_transform(tiny_train())
    assert "missingindicator_ac_000" in out.columns
    assert "missingindicator_x_000" in out.columns


def test_train_columns_are_standardized():
    out = make_preprocessor().fit_transform(tiny_train())
    assert np.allclose(out.mean(), 0, atol=1e-9)


def test_missing_values_are_filled_with_train_median():
    pre = make_preprocessor().fit(tiny_train())
    impute = pre.named_steps["impute"]
    medians = dict(zip(impute.feature_names_in_, impute.statistics_))
    # x_000 train values: 0, 100, 300 -> log1p -> median = log1p(100)
    assert medians["x_000"] == pytest.approx(np.log1p(100))


def test_transforming_validation_does_not_relearn():
    pre = make_preprocessor().fit(tiny_train())
    scale_mean_before = pre.named_steps["scale"].mean_.copy()
    val = tiny_train() * 1000                     # very different data
    pre.transform(val)
    assert np.array_equal(pre.named_steps["scale"].mean_, scale_mean_before)
