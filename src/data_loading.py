"""Load the raw APS Failure at Scania Trucks CSV files.

The official CSV files start with a block of license text, then the real
column header ("class,aa_000,..."), then the data. Missing values are
written as the text "na".
"""

from pathlib import Path

import pandas as pd

# Project root = the folder that contains src/, data/, notebooks/ ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

FILES = {
    "train": RAW_DIR / "aps_failure_training_set.csv",
    "test": RAW_DIR / "aps_failure_test_set.csv",
}

TARGET_COL = "class"
EXPECTED_N_COLUMNS = 171          # from the official documentation
EXPECTED_CLASSES = {"neg", "pos"}


def find_header_line(path: Path) -> int:
    """Return the 0-based line number where the real header starts.

    We search for it instead of hard-coding a number, so the code still
    works if the license block ever has a different length.
    """
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if line.startswith(TARGET_COL + ","):
                return line_number
    raise ValueError(f"No header line starting with '{TARGET_COL},' in {path}")


def load_raw(split: str = "train") -> pd.DataFrame:
    """Load one split ("train" or "test") with 'na' read as missing (NaN)."""
    if split not in FILES:
        raise ValueError(f"split must be one of {list(FILES)}, got {split!r}")

    path = FILES[split]
    header_line = find_header_line(path)

    df = pd.read_csv(
        path,
        skiprows=header_line,   # skip the license text above the header
        na_values=["na"],       # the dataset's official missing-value marker
        keep_default_na=False,  # ONLY "na" means missing; anything else odd stays visible
    )
    validate_raw(df, split)
    return df


def validate_raw(df: pd.DataFrame, split: str) -> None:
    """Basic format checks. Raises an error if the file is not what we expect."""
    if df.shape[1] != EXPECTED_N_COLUMNS:
        raise ValueError(f"{split}: expected {EXPECTED_N_COLUMNS} columns, got {df.shape[1]}")

    if df.columns[0] != TARGET_COL:
        raise ValueError(f"{split}: first column should be '{TARGET_COL}', got {df.columns[0]!r}")

    found_classes = set(df[TARGET_COL].unique())
    if not found_classes <= EXPECTED_CLASSES:
        raise ValueError(f"{split}: unexpected class labels {found_classes - EXPECTED_CLASSES}")

    feature_cols = df.columns.drop(TARGET_COL)
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"{split}: these feature columns are not numeric (unexpected text inside?): "
            f"{non_numeric[:10]}"
        )
