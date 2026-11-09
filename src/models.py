"""Model pipelines: preprocessing + a classifier, chained into ONE object.

Because preprocessing and model live in the same Pipeline, calling
    model.fit(X_train, y_train)
fits every cleaning step AND the classifier on training data only, and
    model.predict(X_val)
applies exactly the same cleaning to validation data before predicting.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import make_preprocessor


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
