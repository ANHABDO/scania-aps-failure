"""Session 5, Step B: train logistic regression and compare it with the baselines.

Run from the project root (run src.run_baselines first):
    python -m src.train_logreg

Saves results to reports/logreg_results.csv
"""

import time

import pandas as pd

from src.data_loading import PROJECT_ROOT, load_raw
from src.metrics import evaluate
from src.models import make_logreg_pipeline
from src.splitting import split_features_target, train_val_split

REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    X, y = split_features_target(load_raw("train"))
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    variants = {
        "logreg_plain": make_logreg_pipeline(class_weight=None),
        "logreg_balanced": make_logreg_pipeline(class_weight="balanced"),
    }

    rows = []
    for name, model in variants.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)            # cleaning + model, learned on TRAIN only
        y_pred = model.predict(X_val)          # threshold 0.5 (scikit-learn default)
        seconds = time.perf_counter() - start
        rows.append({"model": name, **evaluate(y_val, y_pred), "fit_seconds": round(seconds, 1)})
        print(f"trained {name} in {seconds:.1f} s")

    results = pd.DataFrame(rows).set_index("model")

    REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / "logreg_results.csv")

    baseline_file = REPORTS_DIR / "baseline_results.csv"
    if baseline_file.exists():
        baselines = pd.read_csv(baseline_file, index_col="model")
        results = pd.concat([baselines, results])

    print("\nValidation results (cost: lower is better)")
    print(results.drop(columns="fit_seconds", errors="ignore").to_string(formatters={
        "accuracy": "{:.2%}".format,
        "precision": "{:.2%}".format,
        "recall": "{:.2%}".format,
        "cost": "{:,.0f}".format,
    }))
    print("\nsaved: reports/logreg_results.csv")


if __name__ == "__main__":
    main()
