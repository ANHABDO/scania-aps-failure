"""First look at the APS TRAINING data: shape, types, classes, missingness.

Run from the project root:
    python -m src.inspect_data

Only the training set is explored. The test set is loaded once just to
confirm its format, and no statistics about it are printed.
"""

import pandas as pd

from src.data_loading import TARGET_COL, load_raw


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    train = load_raw("train")

    section("1. Shape")
    print(f"rows: {train.shape[0]:,}   columns: {train.shape[1]}")

    section("2. Column types")
    print(train.dtypes.value_counts().to_string())

    section("3. Target class balance")
    counts = train[TARGET_COL].value_counts()
    print(counts.to_string())
    print(f"positive rate: {counts.get('pos', 0) / len(train):.4%}")

    section("4. Missing values")
    missing_frac = train.drop(columns=TARGET_COL).isna().mean()
    total_cells = train.drop(columns=TARGET_COL).size
    print(f"missing cells overall: {missing_frac.mean():.2%} of {total_cells:,}")
    print(f"columns with no missing values: {(missing_frac == 0).sum()}")
    print(f"columns with > 50% missing:   {(missing_frac > 0.5).sum()}")
    print("\n10 columns with the most missing values:")
    print(missing_frac.sort_values(ascending=False).head(10).map("{:.1%}".format).to_string())

    rows_missing = train.drop(columns=TARGET_COL).isna().mean(axis=1)
    print("\nmissing fraction per row, by class (mean):")
    print(rows_missing.groupby(train[TARGET_COL]).mean().map("{:.1%}".format).to_string())

    section("5. Duplicates and constant columns")
    print(f"fully duplicated rows: {train.duplicated().sum()}")
    n_unique = train.drop(columns=TARGET_COL).nunique(dropna=True)
    print(f"constant columns (1 distinct value): {list(n_unique[n_unique <= 1].index)}")

    section("6. Histogram column groups (prefix before '_')")
    prefixes = pd.Series([c.split("_")[0] for c in train.columns.drop(TARGET_COL)])
    group_sizes = prefixes.value_counts()
    print(group_sizes[group_sizes > 1].to_string())

    section("7. Value ranges (5 example columns)")
    print(train.iloc[:, 1:6].describe().T.to_string())

    section("8. Test set format check (no statistics shown)")
    test = load_raw("test")
    same_columns = list(test.columns) == list(train.columns)
    print(f"test loaded and validated; same columns as train: {same_columns}")


if __name__ == "__main__":
    main()
