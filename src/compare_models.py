"""Session 7: compare logistic regression and gradient boosting FAIRLY.

Same treatment for every model:
  - same 5-fold stratified cross-validation on the same train rows
  - class_weight="balanced"
  - each model gets its OWN threshold, chosen on its cross-validated predictions
Main comparison = cross-validation (800 failures). Then ONE check on validation.

Run from the project root (takes a few minutes):
    python -m src.compare_models
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.data_loading import PROJECT_ROOT, load_raw
from src.metrics import evaluate, total_cost
from src.models import make_hgb_pipeline, make_logreg_pipeline
from src.splitting import RANDOM_SEED, split_features_target, train_val_split
from src.threshold import best_threshold, cost_at_thresholds, out_of_fold_proba, predict_at

REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS = {
    "logreg_balanced": make_logreg_pipeline(class_weight="balanced"),
    "hgb_balanced": make_hgb_pipeline(class_weight="balanced"),
}


def main() -> None:
    X, y = split_features_target(load_raw("train"))
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    rows = []
    for name, model in MODELS.items():
        print(f"{name}: cross-validation... ", end="", flush=True)
        start = time.perf_counter()

        # ---- 1. Cross-validation on train: choose the threshold, measure the cost
        oof = out_of_fold_proba(model, X_train, y_train, cv)
        table = cost_at_thresholds(y_train, oof)
        t = best_threshold(table)
        fold_costs = [                                   # same threshold, cost per fold
            total_cost(y_train.iloc[idx], predict_at(oof[idx], t))
            for _, idx in cv.split(X_train, y_train)
        ]

        # ---- 2. Train on all of train, check ONCE on validation
        print("final fit... ", end="", flush=True)
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        val = evaluate(y_val, predict_at(val_proba, t))
        seconds = time.perf_counter() - start
        print(f"done in {seconds:.0f} s")

        rows.append({
            "model": name,
            "threshold": t,
            "cv_cost": table.loc[t, "cost"],
            "cv_fold_cost_min": min(fold_costs),
            "cv_fold_cost_max": max(fold_costs),
            "cv_recall": table.loc[t, "recall"],
            "cv_avg_precision": average_precision_score(y_train, oof),
            "val_cost": val["cost"],
            "val_fn": val["fn"],
            "val_fp": val["fp"],
            "val_recall": val["recall"],
            "val_precision": val["precision"],
            "val_avg_precision": average_precision_score(y_val, val_proba),
            "seconds": round(seconds),
        })

    results = pd.DataFrame(rows).set_index("model")
    REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / "model_comparison.csv")

    pct = "{:.1%}".format
    num = "{:,.0f}".format
    print("\nA) Cross-validation on train (800 failures) - the main comparison")
    print(results[["threshold", "cv_cost", "cv_fold_cost_min", "cv_fold_cost_max",
                   "cv_recall", "cv_avg_precision"]].to_string(formatters={
        "cv_cost": num, "cv_fold_cost_min": num, "cv_fold_cost_max": num,
        "cv_recall": pct, "cv_avg_precision": "{:.3f}".format}))

    print("\nB) Validation (200 failures) - one confirmation, each model at its CV threshold")
    print(results[["val_cost", "val_fn", "val_fp", "val_recall", "val_precision",
                   "val_avg_precision"]].to_string(formatters={
        "val_cost": num, "val_recall": pct, "val_precision": pct,
        "val_avg_precision": "{:.3f}".format}))
    print("\nsaved: reports/model_comparison.csv")


if __name__ == "__main__":
    main()
