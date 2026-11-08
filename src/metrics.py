"""Evaluation metrics, including the dataset's official misclassification cost.

Labels: 1 = pos (APS component failure), 0 = neg.
Official cost (from the dataset documentation):
    false positive (unnecessary workshop check) = 10
    false negative (missed faulty truck)        = 500
"""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

COST_FP = 10
COST_FN = 500


def confusion_counts(y_true, y_pred) -> dict:
    """Count the four outcomes: tn, fp, fn, tp."""
    # labels=[0, 1] forces a full 2x2 table, even if a model only ever predicts one class
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def total_cost(y_true, y_pred, cost_fp: int = COST_FP, cost_fn: int = COST_FN) -> int:
    """Official cost = cost_fp * false positives + cost_fn * false negatives."""
    c = confusion_counts(y_true, y_pred)
    return cost_fp * c["fp"] + cost_fn * c["fn"]


def evaluate(y_true, y_pred) -> dict:
    """All the numbers we report for one set of predictions."""
    return {
        **confusion_counts(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        # zero_division=0: if the model never predicts pos, precision is reported as 0
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "cost": total_cost(y_true, y_pred),
    }
