"""Tests for src/features.py (usage-share features)."""

import numpy as np
import pandas as pd
import pytest

from src.features import AddUsageShares


def histogram_data():
    # one histogram "ag" with 3 bins (enough for the idea), plus another column
    return pd.DataFrame({
        "ag_000": [1.0, 100.0, 0.0, np.nan],
        "ag_001": [3.0, 300.0, 0.0, 5.0],
        "ag_002": [6.0, 600.0, 0.0, 5.0],
        "aa_000": [7.0, 8.0, 9.0, 10.0],
    })


def test_same_usage_pattern_gives_same_shares():
    out = AddUsageShares().fit_transform(histogram_data())
    # truck 0 (1, 3, 6) and truck 1 (100, 300, 600): different size, same pattern
    assert out.loc[0, "ag_000_share"] == pytest.approx(0.1)
    assert out.loc[1, "ag_000_share"] == pytest.approx(0.1)
    assert out.loc[0, ["ag_000_share", "ag_001_share", "ag_002_share"]].sum() == pytest.approx(1)


def test_zero_total_or_missing_bin_gives_missing_shares():
    out = AddUsageShares().fit_transform(histogram_data())
    assert out.loc[2, ["ag_000_share", "ag_001_share"]].isna().all()     # total = 0
    assert out.loc[3, ["ag_000_share", "ag_001_share"]].isna().all()     # one bin missing


def test_original_columns_are_kept_and_input_not_modified():
    data = histogram_data()
    before = data.copy()
    out = AddUsageShares().fit_transform(data)
    pd.testing.assert_frame_equal(out[before.columns], before)
    pd.testing.assert_frame_equal(data, before)
    assert list(out.columns[-4:]) == ["ag_000_share", "ag_001_share", "ag_002_share", "hist_total"]


def test_histogram_total_column():
    out = AddUsageShares().fit_transform(histogram_data())
    assert out.loc[1, "hist_total"] == 1000
