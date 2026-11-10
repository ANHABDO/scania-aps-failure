"""Session 9: train the FINAL model on all 60,000 training-file rows and save it.

All decisions were frozen in Phase 4:
    model      = hgb_balanced (cleaning + HistGradientBoosting, class_weight="balanced")
    threshold  = 0.14 (chosen by 5-fold cross-validation, Session 7)
The official test set is NOT used here. It stays locked for the final evaluation.

Run from the project root:
    python -m src.train_final
"""

import time

import pandas as pd

from src.data_loading import PROJECT_ROOT, load_raw
from src.model_io import META_FILE, MODEL_FILE, save_model
from src.models import make_hgb_pipeline
from src.splitting import RANDOM_SEED, split_features_target

MODEL_DIR = PROJECT_ROOT / "models"
FINAL_THRESHOLD = 0.14          # frozen: CV-chosen threshold of hgb_balanced (Session 7)
SAMPLE_FILE = PROJECT_ROOT / "data" / "samples" / "sample_trucks.csv"


def main() -> None:
    X, y = split_features_target(load_raw("train"))

    print(f"training hgb_balanced on {len(X):,} rows ({int(y.sum())} failures)...", flush=True)
    start = time.perf_counter()
    model = make_hgb_pipeline(class_weight="balanced").fit(X, y)
    seconds = time.perf_counter() - start
    print(f"done in {seconds:.0f} s")

    cv_info = {}
    comparison = PROJECT_ROOT / "reports" / "model_comparison.csv"
    if comparison.exists():
        row = pd.read_csv(comparison, index_col="model").loc["hgb_balanced"]
        cv_info = {"cv_cost_on_48000_rows": float(row["cv_cost"]),
                   "cv_recall": float(row["cv_recall"]),
                   "cv_avg_precision": float(row["cv_avg_precision"])}

    save_model(model, MODEL_DIR, FINAL_THRESHOLD, list(X.columns), extra_info={
        "model_name": "hgb_balanced",
        "training_rows": len(X),
        "training_failures": int(y.sum()),
        "random_seed": RANDOM_SEED,
        "training_seconds": round(seconds),
        "cv_results_session7": cv_info,
        "note": "Trained on the full training file. Official test set not used.",
    })
    print(f"saved: models/{MODEL_FILE} and models/{META_FILE}")

    # A small example input for trying the prediction script.
    # These rows come from the TRAINING file (never the locked test set), so the
    # model has seen them: use this file to test the tool, not to judge quality.
    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample = pd.concat([X[y == 1].head(5), X[y == 0].head(15)])
    sample.to_csv(SAMPLE_FILE, index=False, na_rep="na")
    print(f"example input: data/samples/{SAMPLE_FILE.name} (20 training trucks, 5 of them failures)")


if __name__ == "__main__":
    main()
