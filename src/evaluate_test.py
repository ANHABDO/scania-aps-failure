"""Session 10, Step A: the FINAL exam - evaluate the frozen model ONCE on the official test set.

Rules enforced by this script:
  1. Everything must be committed first (clean Git state), so the commit recorded in the
     results proves the decisions were frozen BEFORE the test score was seen.
  2. It refuses to overwrite an existing results file: one official run.
     (To check reproducibility later, write to another file with --output.)

Run from the project root (after: git add . ; git commit ...):
    python -m src.evaluate_test
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import average_precision_score

from src.data_loading import PROJECT_ROOT, load_raw
from src.metrics import evaluate
from src.model_io import load_model
from src.splitting import split_features_target
from src.threshold import predict_at

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "test_results.json"
MODEL_DIR = PROJECT_ROOT / "models"

# Reference points (cost on the SAME official test set and cost rule)
IDA_2016_BEST_COST = 9_920          # winner of the IDA 2016 Industrial Challenge (UCI documentation)
VALIDATION_COST = 6_670             # our hgb_balanced on validation, Session 7 (12,000 trucks)
VALIDATION_ROWS = 12_000


def git_state() -> tuple[str, bool]:
    """Return (current commit hash, True if there are no uncommitted changes)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True, cwd=PROJECT_ROOT).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                 text=True, check=True, cwd=PROJECT_ROOT).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Git is needed to record the frozen state: {error}")
    return commit, changes == ""


def check_ready(output: Path) -> str:
    """Stop before touching the test set if the rules are not met. Returns the commit."""
    if output.exists():
        raise SystemExit(
            f"{output.name} already exists: the official test evaluation was already run.\n"
            f"To check reproducibility, use another file: --output reports/test_results_check.json"
        )
    commit, clean = git_state()
    if not clean:
        raise SystemExit("Uncommitted changes found. Commit everything first (git add . ; git commit).")
    return commit


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Final, one-time evaluation on the test set.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    commit = check_ready(args.output)                 # rules first, data second

    model, meta = load_model(MODEL_DIR)
    X_test, y_test = split_features_target(load_raw("test"))
    proba = model.predict_proba(X_test)[:, 1]
    result = evaluate(y_test, predict_at(proba, meta["threshold"]))
    result["avg_precision"] = average_precision_score(y_test, proba)

    n = len(y_test)
    per_1000 = result["cost"] / n * 1000
    val_per_1000 = VALIDATION_COST / VALIDATION_ROWS * 1000

    print("FINAL TEST EVALUATION (official test set, used once)")
    print(f"  commit:     {commit[:7]}   model trained: {meta['trained_at_utc']}   "
          f"threshold: {meta['threshold']}")
    print(f"  trucks:     {n:,}   real APS failures: {int(y_test.sum())}")
    print(f"  caught:     {result['tp']}   missed: {result['fn']}   "
          f"false alarms: {result['fp']}")
    print(f"  recall:     {result['recall']:.1%}   precision: {result['precision']:.1%}   "
          f"accuracy: {result['accuracy']:.2%}   avg precision: {result['avg_precision']:.3f}")
    print(f"  COST:       {result['cost']:,}")
    print("\n  Reference points")
    print(f"  cost per 1,000 trucks: test {per_1000:,.0f} | validation {val_per_1000:,.0f}")
    print(f"  IDA 2016 challenge winner on this test set: {IDA_2016_BEST_COST:,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "model_trained_at_utc": meta["trained_at_utc"],
        "threshold": meta["threshold"],
        "test_rows": n,
        "test_failures": int(y_test.sum()),
        **{k: (float(v) if k in ("accuracy", "precision", "recall", "avg_precision") else v)
           for k, v in result.items()},
        "cost_per_1000_trucks": round(per_1000, 1),
        "ida_2016_best_cost": IDA_2016_BEST_COST,
    }, indent=2), encoding="utf-8")
    print(f"\nsaved: {args.output.relative_to(PROJECT_ROOT) if args.output.is_absolute() else args.output}")


if __name__ == "__main__":
    main()
