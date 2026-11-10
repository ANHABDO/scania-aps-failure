"""Predict APS-failure risk for new trucks with the saved model.

Input: a CSV file with the 170 feature columns (names as in the Scania data),
       missing values written as "na". Extra columns are ignored.
Output: a CSV with, for every input row:
       row                  position in the input file (0 = first truck)
       probability          the model's probability of an APS failure (0 to 1)
       flag_for_aps_check   1 if probability >= threshold, else 0

Run from the project root:
    python -m src.predict --input data/samples/sample_trucks.csv --output predictions.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data_loading import PROJECT_ROOT
from src.model_io import load_model

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def find_feature_header(path: Path, first_feature: str = "aa_000") -> int:
    """Line number of the header: the first line that contains a feature name.

    Works for plain CSV files (header on line 0) and for the official Scania files
    (license text first, header later). Returns 0 if no such line is found.
    """
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if first_feature in line.split(","):
                return line_number
    return 0


def read_input(path: Path) -> pd.DataFrame:
    """Read a CSV where only the text 'na' means missing (same rule as training)."""
    return pd.read_csv(path, skiprows=find_feature_header(path),
                       na_values=["na"], keep_default_na=False)


def prepare_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Check the input and return the features in the training column order."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing {len(missing)} required columns, e.g. {missing[:5]}")
    X = df[feature_names]                                   # ignores extra columns, fixes order
    non_numeric = [c for c in feature_names if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"these columns contain non-numeric values: {non_numeric[:5]}")
    return X


def predict_dataframe(model, meta: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Probabilities and 0/1 decisions for every row of df."""
    X = prepare_features(df, meta["feature_names"])
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "row": range(len(X)),
        "probability": proba.round(4),
        "flag_for_aps_check": (proba >= meta["threshold"]).astype(int),
    })


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict APS failures with the saved model.")
    parser.add_argument("--input", required=True, type=Path, help="CSV file with the 170 features")
    parser.add_argument("--output", required=True, type=Path, help="where to write predictions")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, type=Path,
                        help="folder with model.joblib and model_meta.json")
    args = parser.parse_args(argv)

    model, meta = load_model(args.model_dir)
    predictions = predict_dataframe(model, meta, read_input(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    n_flag = int(predictions["flag_for_aps_check"].sum())
    print(f"model: {meta.get('model_name', '?')} (threshold {meta['threshold']}, "
          f"trained {meta['trained_at_utc']})")
    print(f"{len(predictions)} trucks -> {n_flag} flagged for an APS check")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
