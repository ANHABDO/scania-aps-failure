"""Session 8: error analysis of the best model (hgb_balanced).

Part 1 - Mistakes: cross-validated predictions on TRAIN (800 failures), so validation stays clean.
Part 2 - Feature importance: permutation importance on VALIDATION (measurement only:
         we will NOT choose features from it, which would be tuning on validation).

Run from the project root (takes several minutes):
    python -m src.error_analysis
"""

import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from src.data_loading import PROJECT_ROOT, load_raw
from src.importance import group_permutation_importance, outcome_labels
from src.models import make_hgb_pipeline
from src.preprocessing import AC_CODE_BASE
from src.splitting import RANDOM_SEED, split_features_target, train_val_split
from src.threshold import best_threshold, cost_at_thresholds, out_of_fold_proba, predict_at

REPORTS_DIR = PROJECT_ROOT / "reports"
HISTOGRAM_GROUPS = ["ag", "ay", "az", "ba", "cn", "cs", "ee"]
pd.set_option("display.width", 140)


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def describe_groups(X: pd.DataFrame, outcome: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """One row per outcome group with simple descriptive numbers."""
    ag_cols = [c for c in X.columns if c.startswith("ag_")]
    info = pd.DataFrame({
        "outcome": outcome,
        "probability": proba,
        "missing_share": X.isna().mean(axis=1),
        "has_ac_code": X["ac_000"] >= AC_CODE_BASE,
        "histogram_total": X[ag_cols].sum(axis=1, min_count=len(ag_cols)),
        "aa_000": X["aa_000"],
    })
    table = info.groupby("outcome").agg(
        trucks=("probability", "size"),
        median_probability=("probability", "median"),
        mean_missing_share=("missing_share", "mean"),
        share_with_ac_code=("has_ac_code", "mean"),
        median_histogram_total=("histogram_total", "median"),
        median_aa_000=("aa_000", "median"),
    )
    return table.reindex(["TP", "FN", "FP", "TN"])


def main() -> None:
    start = time.perf_counter()
    X, y = split_features_target(load_raw("train"))
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    # ================================================================ PART 1: mistakes
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    print("cross-validated predictions on train...", flush=True)
    oof = out_of_fold_proba(make_hgb_pipeline(), X_train, y_train, cv)
    t = best_threshold(cost_at_thresholds(y_train, oof))
    outcome = outcome_labels(y_train, predict_at(oof, t))

    section(f"1. The four outcome groups (CV on train, threshold {t})")
    groups = describe_groups(X_train, outcome, oof)
    print(groups.to_string(formatters={
        "trucks": "{:,}".format, "median_probability": "{:.3f}".format,
        "mean_missing_share": "{:.1%}".format, "share_with_ac_code": "{:.1%}".format,
        "median_histogram_total": "{:,.0f}".format, "median_aa_000": "{:,.0f}".format,
    }))

    section("2. Missed failures: near misses or confidently wrong?")
    fn_proba = oof[outcome == "FN"]
    bins = [0, 0.01, 0.05, t / 2, t]
    labels = ["< 0.01 (confidently wrong)", "0.01 - 0.05", f"0.05 - {t / 2:.2f}",
              f"{t / 2:.2f} - {t} (near miss)"]
    print(pd.cut(fn_proba, bins=bins, labels=labels, include_lowest=True)
          .value_counts().reindex(labels).to_string())

    section("3. False alarms: how confident was the model?")
    fp_proba = oof[outcome == "FP"]
    fp_bins = [t, 0.3, 0.5, 0.8, 1.0]
    fp_labels = [f"{t} - 0.3", "0.3 - 0.5", "0.5 - 0.8", "0.8 - 1.0 (very confident)"]
    print(pd.cut(fp_proba, bins=fp_bins, labels=fp_labels, include_lowest=True)
          .value_counts().reindex(fp_labels).to_string())

    # ================================================================ PART 2: importance
    print("\nfitting on all of train for importance...", flush=True)
    model = make_hgb_pipeline().fit(X_train, y_train)

    section("4. Permutation importance, single columns (validation, drop in average precision)")
    print("   shuffling each of 170 columns 5 times... (the slow part)", flush=True)
    result = permutation_importance(model, X_val, y_val, scoring="average_precision",
                                    n_repeats=5, random_state=RANDOM_SEED)
    imp = pd.DataFrame({"importance_mean": result.importances_mean,
                        "importance_std": result.importances_std},
                       index=X_val.columns).sort_values("importance_mean", ascending=False)
    print(imp.head(15).to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\n   columns with importance <= 0 (shuffling them does not hurt): "
          f"{(imp['importance_mean'] <= 0).sum()} of {len(imp)}")

    section("5. Top 5 columns: median value in each outcome group (train, raw values)")
    top5 = list(imp.index[:5])
    medians = X_train[top5].groupby(outcome).median().reindex(["TP", "FN", "FP", "TN"])
    print(medians.to_string(float_format=lambda v: f"{v:,.0f}"))

    section("6. Histogram groups: shuffle all 10 bins TOGETHER vs one at a time")
    hist_groups = {g: [c for c in X_val.columns if c.startswith(g + "_")]
                   for g in HISTOGRAM_GROUPS}
    together = group_permutation_importance(model, X_val, y_val, hist_groups)
    together["sum_of_single_columns"] = [imp.loc[cols, "importance_mean"].sum()
                                         for cols in hist_groups.values()]
    print(together.to_string(float_format=lambda v: f"{v:.4f}"))

    REPORTS_DIR.mkdir(exist_ok=True)
    groups.to_csv(REPORTS_DIR / "error_groups.csv")
    imp.to_csv(REPORTS_DIR / "permutation_importance.csv")
    together.to_csv(REPORTS_DIR / "group_importance.csv")
    print(f"\nsaved: reports/error_groups.csv, permutation_importance.csv, group_importance.csv"
          f"\ntotal time: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
