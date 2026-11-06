"""Session 3: feature distributions, scale, and histogram consistency (TRAINING set only).

Questions:
  Q1. Are any values negative? (counters and histogram bins should be >= 0)
  Q2. How skewed are the features, and how different are their scales?
  Q3. Does a log transform make the skewed features better behaved?
  Q4. Histogram groups: do the 7 histograms add up to the same total per truck?

Run from the project root:
    python -m src.explore_distributions

Describes only. Changes and saves nothing.
"""

import numpy as np
import pandas as pd

from src.data_loading import TARGET_COL, load_raw

AC_CODE_BASE = 0x7F000000   # coded state found in ac_000 in Session 2
HISTOGRAM_GROUPS = ["ag", "ay", "az", "ba", "cn", "cs", "ee"]


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def remove_ac_code(X: pd.DataFrame) -> pd.DataFrame:
    """Treat the ac_000 coded values as missing, so they don't distort statistics."""
    X = X.copy()
    X.loc[X["ac_000"] >= AC_CODE_BASE, "ac_000"] = np.nan
    return X


def check_negatives(X: pd.DataFrame) -> None:
    section("Q1. Negative values")
    n_neg = (X < 0).sum()
    print(f"columns with at least one negative value: {(n_neg > 0).sum()}")
    if (n_neg > 0).any():
        print(n_neg[n_neg > 0].to_string())


def check_scale_and_skew(X: pd.DataFrame) -> None:
    section("Q2. Zeros, scale and skewness")

    zero_share = (X == 0).sum() / X.notna().sum()      # among NON-missing values
    print("share of zeros per column (among non-missing values):")
    print(f"  columns with > 50% zeros: {(zero_share > 0.5).sum()}")
    print(f"  columns with > 90% zeros: {(zero_share > 0.9).sum()}")

    col_max = X.max()
    print("\nscale: maximum value per column")
    print(f"  smallest column maximum: {col_max.min():,.0f}  ({col_max.idxmin()})")
    print(f"  largest column maximum:  {col_max.max():,.0f}  ({col_max.idxmax()})")
    print(f"  ratio largest / smallest: {col_max.max() / max(col_max.min(), 1):,.0f}x")

    skew = X.skew()
    print("\nskewness (0 = symmetric, > 1 = long right tail):")
    print(f"  |skew| > 1:   {(skew.abs() > 1).sum()} of {len(skew)} columns")
    print(f"  |skew| > 10:  {(skew.abs() > 10).sum()} columns")
    print(f"  median skew:  {skew.median():.1f}")


def check_log_transform(X: pd.DataFrame) -> None:
    section("Q3. Effect of log(1 + x) on the 10 most skewed columns")
    skew = X.skew()
    top = skew.abs().sort_values(ascending=False).head(10).index
    # log1p(x) = log(1 + x): defined at 0, which matters because many values are 0
    after = np.log1p(X[top].clip(lower=0)).skew()
    table = pd.DataFrame({
        "skew_before": skew[top],
        "skew_after_log1p": after,
        "median": X[top].median(),
        "max": X[top].max(),
    })
    print(table.to_string(float_format=lambda v: f"{v:,.1f}"))
    all_after = np.log1p(X.clip(lower=0)).skew()
    print(f"\nall columns: |skew| > 1 before = {(skew.abs() > 1).sum()}, "
          f"after log1p = {(all_after.abs() > 1).sum()}")


def check_histograms(X: pd.DataFrame) -> None:
    section("Q4. Histogram groups: total per truck")
    sums = {}
    for g in HISTOGRAM_GROUPS:
        cols = [c for c in X.columns if c.startswith(g + "_")]
        complete = X[cols].notna().all(axis=1)
        sums[g] = X[cols].sum(axis=1).where(complete)    # NaN if any bin is missing
        print(f"  {g}: {len(cols)} bins, complete rows: {complete.sum():,}")
    sums = pd.DataFrame(sums).dropna()
    print(f"\nrows where all 7 histograms are complete: {len(sums):,}")

    print("\nmedian total per histogram:")
    print(sums.median().map("{:,.0f}".format).to_string())

    equal = pd.DataFrame(
        {a: {b: (sums[a] == sums[b]).mean() for b in sums} for a in sums}
    )
    print("\nshare of trucks where two histograms have EXACTLY the same total:")
    print(equal.to_string(float_format=lambda v: f"{v:5.0%}"))

    print("\nrank correlation of totals (1 = they rise and fall together):")
    print(sums.corr(method="spearman").to_string(float_format=lambda v: f"{v:5.2f}"))


def main() -> None:
    train = load_raw("train")
    X = remove_ac_code(train.drop(columns=TARGET_COL))
    check_negatives(X)
    check_scale_and_skew(X)
    check_log_transform(X)
    check_histograms(X)


if __name__ == "__main__":
    main()
