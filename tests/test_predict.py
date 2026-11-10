"""Tests for saving/loading the model (src/model_io.py) and predicting (src/predict.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.model_io import META_FILE, load_model, save_model
from src.models import make_hgb_pipeline
from src.predict import find_feature_header, main, predict_dataframe, read_input

FEATURES = ["ac_000", "ad_000", "cf_000", "co_000", "x_000"]


def fake_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.integers(0, 1000, size=(n, len(FEATURES))).astype(float), columns=FEATURES)
    y = (X["x_000"] > 700).astype(int)
    X.loc[::9, "ad_000"] = np.nan
    return X, y


@pytest.fixture
def saved_model(tmp_path):
    X, y = fake_data()
    model = make_hgb_pipeline().fit(X, y)
    save_model(model, tmp_path, threshold=0.3, feature_names=FEATURES,
               extra_info={"model_name": "test_model"})
    return model, tmp_path


def test_saved_and_loaded_model_give_identical_predictions(saved_model):
    model, model_dir = saved_model
    loaded, _ = load_model(model_dir)
    X, _ = fake_data(seed=1)
    assert np.array_equal(model.predict_proba(X), loaded.predict_proba(X))


def test_metadata_records_threshold_features_and_versions(saved_model):
    _, model_dir = saved_model
    meta = json.loads((model_dir / META_FILE).read_text())
    assert meta["threshold"] == 0.3
    assert meta["feature_names"] == FEATURES
    assert "scikit-learn" in meta["versions"]
    assert meta["model_name"] == "test_model"


def test_loading_from_empty_folder_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train it first"):
        load_model(tmp_path)


def test_flags_follow_the_threshold(saved_model):
    _, model_dir = saved_model
    model, meta = load_model(model_dir)
    X, _ = fake_data(seed=2)
    out = predict_dataframe(model, meta, X)
    assert len(out) == len(X)
    assert out["probability"].between(0, 1).all()
    assert (out["flag_for_aps_check"] == (out["probability"] >= 0.3)).all()


def test_missing_column_raises(saved_model):
    _, model_dir = saved_model
    model, meta = load_model(model_dir)
    X, _ = fake_data()
    with pytest.raises(ValueError, match="missing 1 required columns"):
        predict_dataframe(model, meta, X.drop(columns="x_000"))


def test_extra_columns_and_different_order_are_handled(saved_model):
    _, model_dir = saved_model
    model, meta = load_model(model_dir)
    X, _ = fake_data(seed=3)
    shuffled = X[FEATURES[::-1]].assign(truck_id=range(len(X)))     # reversed order + extra column
    a = predict_dataframe(model, meta, X)
    b = predict_dataframe(model, meta, shuffled)
    pd.testing.assert_frame_equal(a, b)


def test_command_line_end_to_end(saved_model, tmp_path):
    _, model_dir = saved_model
    X, _ = fake_data(n=20, seed=4)
    input_csv = tmp_path / "in.csv"
    X.to_csv(input_csv, index=False, na_rep="na")
    output_csv = tmp_path / "out" / "pred.csv"
    main(["--input", str(input_csv), "--output", str(output_csv), "--model-dir", str(model_dir)])
    result = pd.read_csv(output_csv)
    assert list(result.columns) == ["row", "probability", "flag_for_aps_check"]
    assert len(result) == 20


def test_header_is_found_after_license_text(tmp_path):
    path = tmp_path / "official_like.csv"
    path.write_text("License text\n\n----\n\nclass,aa_000,ab_000\nneg,1,na\n", encoding="utf-8")
    assert find_feature_header(path) == 4                     # 0-based line of the header


def test_plain_csv_header_is_line_zero(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("aa_000,ab_000\n1,na\n", encoding="utf-8")
    assert find_feature_header(path) == 0
    assert read_input(path)["ab_000"].isna().all()           # "na" read as missing
