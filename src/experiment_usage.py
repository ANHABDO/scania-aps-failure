"""Option B experiment: does adding usage-share features lower the cost?

Fair comparison with the Session 7 model:
  - same 48,000 train rows, same 5 folds (same seed), same class weighting
  - each variant gets its own cost-optimal threshold from its out-of-fold predictions
  - compared FOLD BY FOLD: a real improvement should appear in most folds
The validation set and the (already used) test set are NOT used for this decision.

Run from the project root (about 10 minutes):
    python -m src.experiment_usage
"""

import time

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.data_loading import PROJECT_ROOT, load_raw
from src.metrics import total_cost
from src.models import make_hgb_pipeline
from src.splitting import RANDOM_SEED, split_features_target, train_val_split
from src.threshold import best_threshold, cost_at_thresholds, out_of_fold_proba, predict_at

REPORTS_DIR = PROJECT_ROOT / "reports"

VARIANTS = {
    "hgb_balanced (current)": make_hgb_pipeline(usage_shares=False),
    "hgb_balanced + usage shares": make_hgb_pipeline(usage_shares=True),
}


def main() -> None:
    X, y = split_features_target(load_raw("train"))
    X_train, _, y_train, _ = train_val_split(X, y)            # validation not used
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    folds = [idx for _, idx in cv.split(X_train, y_train)]

    summary, per_fold = [], {}
    for name, model in VARIANTS.items():
        print(f"{name}: 5-fold cross-validation... ", end="", flush=True)
        start = time.perf_counter()
        oof = out_of_fold_proba(model, X_train, y_train, cv)
        table = cost_at_thresholds(y_train, oof)
        t = best_threshold(table)
        per_fold[name] = [total_cost(y_train.iloc[idx], predict_at(oof[idx], t)) for idx in folds]
        summary.append({
            "variant": name,
            "threshold": t,
            "cv_cost": table.loc[t, "cost"],
            "missed": table.loc[t, "fn"],
            "false_alarms": table.loc[t, "fp"],
            "recall": table.loc[t, "recall"],
            "avg_precision": average_precision_score(y_train, oof),
        })
        print(f"done in {time.perf_counter() - start:.0f} s")

    summary = pd.DataFrame(summary).set_index("variant")
    folds_table = pd.DataFrame(per_fold, index=[f"fold {i + 1}" for i in range(5)])
    current, new = folds_table.columns
    folds_table["difference"] = folds_table[new] - folds_table[current]

    print("\nA) Cross-validation summary (800 failures)")
    print(summary.to_string(formatters={
        "cv_cost": "{:,}".format, "recall": "{:.1%}".format, "avg_precision": "{:.3f}".format}))

    print("\nB) Fold by fold (cost; negative difference = usage shares cheaper)")
    print(folds_table.to_string(formatters={c: "{:+,}".format if c == "difference" else "{:,}".format
                                            for c in folds_table.columns}))
    better = int((folds_table["difference"] < 0).sum())
    print(f"\n   usage shares cheaper in {better} of 5 folds; "
          f"total difference {folds_table['difference'].sum():+,}")

    REPORTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(REPORTS_DIR / "experiment_usage_summary.csv")
    folds_table.to_csv(REPORTS_DIR / "experiment_usage_folds.csv")
    print("\nsaved: reports/experiment_usage_summary.csv, reports/experiment_usage_folds.csv")


if __name__ == "__main__":
    main()
