"""Session 4: split the training data and score "dumb" baseline models on validation.

Run from the project root:
    python -m src.run_baselines

Saves the results to reports/baseline_results.csv
"""

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.data_loading import PROJECT_ROOT, load_raw
from src.metrics import evaluate
from src.splitting import RANDOM_SEED, split_features_target, train_val_split

REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    X, y = split_features_target(load_raw("train"))
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    print("Split (training file only; test set untouched)")
    for name, part in [("train", y_train), ("validation", y_val)]:
        print(f"  {name:<10}: {len(part):>6,} rows | {part.sum():>4} positives | "
              f"positive rate {part.mean():.2%}")

    # Baselines ignore the features completely. They set the minimum bar.
    baselines = {
        "always_neg": DummyClassifier(strategy="constant", constant=0),
        "always_pos": DummyClassifier(strategy="constant", constant=1),
        "random_guess": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
    }

    rows = []
    for name, model in baselines.items():
        model.fit(X_train, y_train)          # "learns" only the class counts
        y_pred = model.predict(X_val)        # predictions for the validation rows
        rows.append({"model": name, **evaluate(y_val, y_pred)})

    results = pd.DataFrame(rows).set_index("model")
    print("\nBaselines on the validation set")
    print(results.to_string(formatters={
        "accuracy": "{:.2%}".format,
        "precision": "{:.2%}".format,
        "recall": "{:.2%}".format,
        "cost": "{:,}".format,
    }))

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "baseline_results.csv"
    results.to_csv(out)
    print(f"\nsaved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
