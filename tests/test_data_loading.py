"""Tests for src/data_loading.py.

Most tests use SMALL FAKE files written to a temporary folder (tmp_path),
so they run in a second and don't need the real data.
The last test uses the real training file and is skipped if it's not there.

Run from the project root:
    python -m pytest -v
"""

import pandas as pd
import pytest

from src import data_loading
from src.data_loading import find_header_line, load_raw

LICENSE = "This file is part of APS Failure.\n\nCopyright (c) Scania\n\n" + "-" * 72 + "\n\n"
FEATURES = [f"f{i:03d}_000" for i in range(170)]          # 170 features + class = 171 columns


def write_fake_csv(path, rows, header=None):
    """Write license text + header + rows, like the official files."""
    header = header or ["class"] + FEATURES
    lines = [",".join(header)] + [",".join(r) for r in rows]
    path.write_text(LICENSE + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def good_row(label="neg", first_value="1"):
    return [label, first_value] + ["2"] * (len(FEATURES) - 1)


@pytest.fixture
def fake_train(tmp_path, monkeypatch):
    """Point load_raw('train') at a fake file. Returns a function to write it."""
    path = tmp_path / "train.csv"
    monkeypatch.setitem(data_loading.FILES, "train", path)

    def _write(rows, header=None):
        return write_fake_csv(path, rows, header)

    return _write


# ---------------------------------------------------------------- happy path
def test_find_header_line_skips_license(tmp_path):
    path = write_fake_csv(tmp_path / "x.csv", [good_row()])
    line_number = find_header_line(path)
    assert path.read_text().splitlines()[line_number].startswith("class,")


def test_na_is_read_as_missing(fake_train):
    fake_train([good_row(first_value="na"), good_row()])
    df = load_raw("train")
    assert pd.isna(df.loc[0, "f000_000"])
    assert df.loc[1, "f000_000"] == 1


def test_features_are_numeric_and_shape_is_correct(fake_train):
    fake_train([good_row("neg"), good_row("pos")])
    df = load_raw("train")
    assert df.shape == (2, 171)
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURES)


# ---------------------------------------------------------------- validation catches problems
def test_other_missing_markers_are_not_silently_accepted(fake_train):
    # Only "na" means missing. "NA" or "null" must make validation fail, not hide.
    fake_train([good_row(first_value="NA"), good_row()])
    with pytest.raises(ValueError, match="not numeric"):
        load_raw("train")


def test_unexpected_class_label_raises(fake_train):
    fake_train([good_row(label="maybe")])
    with pytest.raises(ValueError, match="unexpected class labels"):
        load_raw("train")


def test_wrong_number_of_columns_raises(fake_train):
    short_header = ["class"] + FEATURES[:-1]                  # 170 columns instead of 171
    fake_train([good_row()[:-1]], header=short_header)
    with pytest.raises(ValueError, match="expected 171 columns"):
        load_raw("train")


def test_missing_header_raises(tmp_path):
    path = tmp_path / "no_header.csv"
    path.write_text(LICENSE + "neg,1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No header line"):
        find_header_line(path)


def test_unknown_split_raises():
    with pytest.raises(ValueError, match="split must be one of"):
        load_raw("validation")


# ---------------------------------------------------------------- real data (skipped if absent)
@pytest.mark.skipif(not data_loading.FILES["train"].exists(), reason="real training file not downloaded")
def test_real_training_set_matches_documentation():
    df = load_raw("train")
    assert df.shape == (60_000, 171)
    assert df["class"].value_counts().to_dict() == {"neg": 59_000, "pos": 1_000}
