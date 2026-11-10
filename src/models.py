"""Model pipelines: preprocessing + a classifier, chained into ONE object.

Because preprocessing and model live in the same Pipeline, calling
    model.fit(X_train, y_train)
fits every cleaning step AND the classifier on training data only, and
    model.predict(X_val)
applies exactly the same cleaning to validation data before predicting.
"""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import AddUsageShares
from src.preprocessing import CleanRawValues, make_preprocessor
from src.splitting import RANDOM_SEED


def make_logreg_pipeline(class_weight=None) -> Pipeline:
    """Preprocessing + logistic regression.

    class_weight=None       : every truck counts the same.
    class_weight="balanced" : each class counts in inverse proportion to its size,
                              so the 800 positives weigh as much as the 47,200 negatives.
    """
    return Pipeline([
        ("pre", make_preprocessor()),
        ("model", LogisticRegression(class_weight=class_weight, max_iter=2000)),
    ])


def make_hgb_pipeline(class_weight="balanced", usage_shares: bool = False) -> Pipeline:
    """Cleaning + histogram gradient boosting (a team of small decision trees).

    Only the cleaning step is needed:
      - no imputation: the trees handle missing values themselves
      - no log / scaling: trees only use the ORDER of values
    random_state fixes the internal random choices (early stopping uses a
    random 10% of the training rows), so results are repeatable.

    usage_shares=True adds the option-B features (histogram bins as shares of
    their total). Default False = the final model of Project 1, unchanged.
    """
    steps = [("clean", CleanRawValues())]
    if usage_shares:
        steps.append(("usage", AddUsageShares()))
    steps.append(("model", HistGradientBoostingClassifier(
        class_weight=class_weight, random_state=RANDOM_SEED,
    )))
    return Pipeline(steps)
