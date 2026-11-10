"""Save and load the final model together with its "label" (metadata).

Files written to the model folder:
    model.joblib      the whole fitted Pipeline (cleaning + gradient boosting)
    model_meta.json   threshold, feature names, training info, library versions

SECURITY: a .joblib file can run code when it is loaded.
Only load model files you created yourself or fully trust.
"""

import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

MODEL_FILE = "model.joblib"
META_FILE = "model_meta.json"


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def save_model(model, model_dir: Path, threshold: float, feature_names: list,
               extra_info: dict | None = None) -> None:
    """Write the model and its metadata to model_dir."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / MODEL_FILE)

    meta = {
        "threshold": threshold,
        "feature_names": list(feature_names),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": library_versions(),
        **(extra_info or {}),
    }
    (model_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model(model_dir: Path):
    """Return (model, meta). Warns if scikit-learn's version differs from training."""
    model_dir = Path(model_dir)
    if not (model_dir / MODEL_FILE).exists():
        raise FileNotFoundError(
            f"No model in {model_dir}. Train it first: python -m src.train_final"
        )
    meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
    trained_with = meta["versions"]["scikit-learn"]
    if trained_with != sklearn.__version__:
        warnings.warn(
            f"Model trained with scikit-learn {trained_with}, "
            f"but {sklearn.__version__} is installed. Predictions may differ or fail."
        )
    model = joblib.load(model_dir / MODEL_FILE)
    return model, meta
