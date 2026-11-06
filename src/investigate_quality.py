"""Session 2: test two data-quality hypotheses on the TRAINING set only.

H1. Very large values (near 2.13e9) are error/sentinel codes, not real measurements.
H2. Missing values are not random: some columns go missing together,
    and positive trucks have more missing values.

Run from the project root:
    python -m src.investigate_quality

This script only DESCRIBES the data. It does not change or save anything.
"""

import numpy as np
import pandas as pd

from src.data_loading import TARGET_COL, load_raw

INT32_MAX = 2**31 - 1             # 2,147,483,647 = largest signed 32-bit integer
HUGE_THRESHOLD = 2_000_000_000    # "close to the 32-bit limit"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------- H1
def investigate_huge_values(X: pd.DataFrame, y: pd.Series) -> None:
    section("H1. Values close to the 32-bit integer limit")

    is_huge = X >= HUGE_THRESHOLD                 # True/False for every cell (NaN -> False)
    rows_with_huge = is_huge.any(axis=1)
    print(f"threshold: >= {HUGE_THRESHOLD:,}   (int32 max = {INT32_MAX:,})")
    print(f"cells above threshold: {int(is_huge.to_numpy().sum()):,}")
    print(f"rows with at least one: {int(rows_with_huge.sum()):,} of {len(X):,}")
    if not rows_with_huge.any():
        return

    # A) Which columns? And is there a GAP between normal values and huge ones?
    #    A real measurement usually grows smoothly; an error code sits far away.
    per_col = is_huge.sum()
    per_col = per_col[per_col > 0].sort_values(ascending=False)
    summary = pd.DataFrame({
        "n_huge": per_col,
        "largest_normal_value": [X.loc[~is_huge[c], c].max() for c in per_col.index],
        "p99_normal": [X.loc[~is_huge[c], c].quantile(0.99) for c in per_col.index],
    })
    print("\nA) columns containing huge values:")
    print(summary.to_string(float_format=lambda v: f"{v:,.0f}"))

    # B) Are they many different numbers, or the same few numbers repeated?
    #    Repeated identical values point to a code.
    huge_values = pd.Series(X.to_numpy()[is_huge.to_numpy()])
    print(f"\nB) distinct huge values: {huge_values.nunique()}")
    print("   most common (decimal | hexadecimal | count):")
    for value, count in huge_values.value_counts().head(10).items():
        v = int(value)
        print(f"   {v:>15,} | {v:#010x} | {count}")

    # D) Decode the pattern. The top values all look like 0x7F000000 + a small number.
    #    Hypothesis: a fixed "flag" was added on top of a real small counter value.
    #    If true, the leftover part should look like the column's NORMAL values.
    base = 0x7F000000                                   # 2,130,706,432
    rows = []
    for col in per_col.index:
        huge = X.loc[is_huge[col], col].astype("int64")
        normal = X.loc[~is_huge[col], col].dropna().astype("int64")
        has_prefix = (huge >= base) & (huge < base + 2**24)  # first byte is 0x7F
        leftover = huge[has_prefix] - base
        rows.append({
            "column": col,
            "n_huge": len(huge),
            "with_0x7F_prefix": f"{has_prefix.mean():.0%}",
            "leftover_min": leftover.min() if len(leftover) else None,
            "leftover_median": leftover.median() if len(leftover) else None,
            "leftover_max": leftover.max() if len(leftover) else None,
            "leftover_even": f"{(leftover % 2 == 0).mean():.0%}" if len(leftover) else "-",
            "normal_median": normal.median(),
            "normal_even": f"{(normal % 2 == 0).mean():.0%}",
        })
    print("\nD) huge value = 0x7F000000 + leftover?  (compare leftover with normal values)")
    print(pd.DataFrame(rows).set_index("column").to_string())

    n_per_row = is_huge.sum(axis=1)
    multi = is_huge[n_per_row > 1]
    print(f"\n   rows with more than one huge value: {len(multi)}")
    for idx, row in multi.iterrows():
        print(f"   row {idx}: {list(row[row].index)}  class={y[idx]}")

    # C) Do these rows belong to one class more than expected?
    print("\nC) class of rows containing a huge value:")
    print(y[rows_with_huge].value_counts().to_string())
    print(f"   positive rate in these rows: {(y[rows_with_huge] == 'pos').mean():.2%}"
          f"   (whole training set: {(y == 'pos').mean():.2%})")


# ---------------------------------------------------------------- H2
def investigate_missingness(X: pd.DataFrame, y: pd.Series) -> None:
    section("H2. Missing-value patterns")
    mask = X.isna()                               # True where the value is missing

    # A) Columns that are missing in EXACTLY the same rows.
    #    We turn each column's True/False pattern into bytes and group equal patterns.
    groups: dict[bytes, list[str]] = {}
    for col in mask.columns:
        if mask[col].any():
            key = np.packbits(mask[col].to_numpy()).tobytes()
            groups.setdefault(key, []).append(col)
    shared = [cols for cols in groups.values() if len(cols) > 1]
    shared.sort(key=lambda cols: -mask[cols[0]].mean())

    print(f"\nA) groups of columns missing in IDENTICAL rows: {len(shared)}")
    print(f"   columns in such groups: {sum(len(c) for c in shared)} of {mask.any().sum()} columns with missing values")
    for cols in shared:
        names = ", ".join(cols[:8]) + (" ..." if len(cols) > 8 else "")
        print(f"   {mask[cols[0]].mean():6.1%} missing | {len(cols):2d} cols | {names}")

    # A2) "Identical" is very strict: one extra missing cell breaks it.
    #     So for the most-missing columns we also measure OVERLAP:
    #     of the rows where at least one of two columns is missing,
    #     what share has BOTH missing? (100% = always missing together)
    heavy = mask.columns[mask.mean() > 0.5]
    if len(heavy) > 1:
        m = mask[heavy].to_numpy()
        both = m.T.astype(int) @ m.astype(int)            # rows where col i AND col j missing
        count = m.sum(axis=0)
        either = count[:, None] + count[None, :] - both     # rows where i OR j missing
        overlap = pd.DataFrame(both / either, index=heavy, columns=heavy)
        print("\nA2) overlap of missing rows between columns with > 50% missing:")
        print(overlap.to_string(float_format=lambda v: f"{v:5.0%}"))

    # B) Per column: missing rate for neg vs pos trucks.
    rate = mask.groupby(y.to_numpy()).mean().T    # rows = columns, cols = neg / pos
    rate["pos_minus_neg"] = rate["pos"] - rate["neg"]
    top = rate.reindex(rate["pos_minus_neg"].abs().sort_values(ascending=False).index).head(10)
    print("\nB) 10 columns where missing rate differs most between classes:")
    print(top.to_string(float_format=lambda v: f"{v:6.1%}"))

    # C) Rows grouped by how many values they miss. Is the positive rate different?
    n_missing = mask.sum(axis=1)
    bucket = pd.cut(n_missing, bins=[-1, 0, 10, 20, 50, 100, X.shape[1]],
                    labels=["0", "1-10", "11-20", "21-50", "51-100", f"101-{X.shape[1]}"]
                    ).rename("n_missing")
    table = pd.crosstab(bucket, y)
    table["rows"] = table["neg"] + table["pos"]
    table["pos_rate"] = (table["pos"] / table["rows"]).map("{:.2%}".format)
    print("\nC) rows grouped by number of missing values:")
    print(table.to_string())


def main() -> None:
    train = load_raw("train")
    X = train.drop(columns=TARGET_COL)
    y = train[TARGET_COL]
    investigate_huge_values(X, y)
    investigate_missingness(X, y)


if __name__ == "__main__":
    main()
