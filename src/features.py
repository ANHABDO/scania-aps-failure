"""Feature engineering experiment (option B): usage-normalized histogram features.

Idea (from the Session 8 error analysis): the model mostly tracks HOW MUCH a truck was
used. Shares describe HOW it was used, independent of how much:
    share of a bin = bin value / total of its histogram
Example: 1,000 of 10,000 hours and 100 of 1,000 hours are both a 10% share.

The original columns are KEPT; new columns are ADDED, so the model can ignore them.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

HISTOGRAM_GROUPS = ["ag", "ay", "az", "ba", "cn", "cs", "ee"]


class AddUsageShares(BaseEstimator, TransformerMixin):
    """Add <bin>_share for every histogram bin, plus hist_total (mean of the 7 totals)."""

    def fit(self, X: pd.DataFrame, y=None):
        self.groups_ = {g: [c for c in X.columns if c.startswith(g + "_")]
                        for g in HISTOGRAM_GROUPS}
        self.groups_ = {g: cols for g, cols in self.groups_.items() if cols}
        new = [f"{c}_share" for cols in self.groups_.values() for c in cols] + ["hist_total"]
        self.feature_names_out_ = list(X.columns) + new
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new_columns = {}
        totals = []
        for cols in self.groups_.values():
            # min_count=len(cols): the total is missing if ANY bin is missing
            total = X[cols].sum(axis=1, min_count=len(cols))
            totals.append(total)
            safe_total = total.replace(0, np.nan)          # avoid dividing by zero
            for c in cols:
                new_columns[f"{c}_share"] = X[c] / safe_total
        new_columns["hist_total"] = pd.concat(totals, axis=1).mean(axis=1)
        return pd.concat([X, pd.DataFrame(new_columns, index=X.index)], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
