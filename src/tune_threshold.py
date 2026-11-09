"""Session 6: choose the decision threshold of logreg_balanced, honestly.

1. 5-fold cross-validation on the TRAIN part -> "out-of-fold" probabilities
   (every train truck is predicted by a model that never saw it).
2. Choose the threshold with the lowest cost on those probabilities.
3. Train on all of train, apply that threshold ONCE to validation.

Run from the project root (takes about 2 minutes):
    python -m src.tune_threshold
"""

import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data_loading import PROJECT_ROOT, load_raw
from src.metrics import evaluate
from src.models import make_logreg_pipeline
from src.splitting import RANDOM_SEED, split_features_target, train_val_split
from src.threshold import best_threshold, cost_at_thresholds, predict_at

REPORTS_DIR = PROJECT_ROOT / "reports"
SHOW = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def show(table: pd.DataFrame, thresholds) -> None:
    print(table.loc[sorted(set(thresholds))].to_string(formatters={
        "precision": "{:.1%}".format,
        "recall": "{:.1%}".format,
        "cost": "{:,}".format,
    }))


def main() -> None:
    X, y = split_features_target(load_raw("train"))
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    # ---- 1. Cross-validated probabilities on TRAIN only
    print("1. 5-fold cross-validation on train (48,000 rows)... ", end="", flush=True)
    start = time.perf_counter()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_proba = cross_val_predict(
        make_logreg_pipeline(class_weight="balanced"),
        X_train, y_train, cv=cv, method="predict_proba",
    )[:, 1]                                   # column 1 = probability of "pos"
    print(f"done in {time.perf_counter() - start:.0f} s")

    # ---- 2. Choose the threshold on those probabilities
    cv_table = cost_at_thresholds(y_train, oof_proba)
    chosen = best_threshold(cv_table)
    print(f"\n2. Cost on cross-validated train predictions "
          f"({int(y_train.sum())} failures; costs are about 4x validation size)")
    show(cv_table, SHOW + [chosen])
    print(f"\n   chosen threshold: {chosen}")
    if chosen in (cv_table.index.min(), cv_table.index.max()):
        print("   WARNING: the chosen threshold is at the edge of the tested range.")

    # ---- 3. Train on all of train, apply the chosen threshold ONCE on validation
    model = make_logreg_pipeline(class_weight="balanced").fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    results = pd.DataFrame([
        {"model": "logreg_balanced @ 0.50", **evaluate(y_val, predict_at(val_proba, 0.5))},
        {"model": f"logreg_balanced @ {chosen:g} (chosen by CV)",
         **evaluate(y_val, predict_at(val_proba, chosen))},
    ]).set_index("model")
    print("\n3. Validation (12,000 rows, 200 failures) - the honest check")
    print(results.to_string(formatters={
        "accuracy": "{:.2%}".format, "precision": "{:.2%}".format,
        "recall": "{:.2%}".format, "cost": "{:,}".format,
    }))

    # ---- For learning only: what if we had tuned on validation directly?
    val_table = cost_at_thresholds(y_val, val_proba)
    peek = best_threshold(val_table)
    print(f"\n4. FOR LEARNING ONLY (not used): tuning directly on validation would pick "
          f"{peek} with cost {val_table.loc[peek, 'cost']:,}")

    REPORTS_DIR.mkdir(exist_ok=True)
    cv_table.to_csv(REPORTS_DIR / "threshold_cv.csv")
    results.to_csv(REPORTS_DIR / "threshold_validation.csv")
    print("\nsaved: reports/threshold_cv.csv, reports/threshold_validation.csv")


if __name__ == "__main__":
    main()
