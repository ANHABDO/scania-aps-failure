"""Preprocessing pipeline: turns raw Scania features into clean numbers for a model.

Golden rule: every step LEARNS only from the data given to fit() (the training set),
then applies exactly the same rule to validation and test data. This prevents leakage.

Steps (decisions from Phase 2, see PROJECT_STATUS.md):
  1. clean  : drop constant columns, ac_000 code -> flag + missing, corrupted values -> missing
  2. log    : log(1 + x) to squeeze extreme values
  3. impute : fill missing values with the TRAIN median + add a 0/1 "was missing" column
  4. scale  : standardize every column (mean 0, standard deviation 1 on TRAIN)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

AC_CODE_BASE = 0x7F000000     # 2,130,706,432: coded state found in ac_000 (Session 2)

# Session 2-3: one row (45977) has three nearly identical values of about 8.58e9
# in columns whose normal maximum is below 1e6 -> treated as corrupted.
# Not applied to dq_000: its large values are a genuine heavy tail.
SUSPECT_LIMITS = {"ad_000": 2e9, "cf_000": 2e9, "co_000": 2e9}


class CleanRawValues(BaseEstimator, TransformerMixin):
    """Step 1: apply our Phase 2 data-quality decisions.

    fit()       learns which columns are constant in the TRAINING data.
    transform() applies the cleaning rules to any data (train, validation, test).
    """

    def fit(self, X: pd.DataFrame, y=None):
        n_unique = X.nunique(dropna=True)
        self.constant_columns_ = list(n_unique[n_unique <= 1].index)
        kept = [c for c in X.columns if c not in self.constant_columns_]
        self.feature_names_out_ = kept + ["ac_000_is_code"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(columns=self.constant_columns_).copy()   # copy: never modify the caller's data

        is_code = X["ac_000"] >= AC_CODE_BASE
        X["ac_000_is_code"] = is_code.astype(float)         # keep the signal as a flag...
        X.loc[is_code, "ac_000"] = np.nan                    # ...and drop the fake number

        for col, limit in SUSPECT_LIMITS.items():
            if col in X.columns:
                X.loc[X[col] >= limit, col] = np.nan

        return X[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


def make_preprocessor() -> Pipeline:
    """Build the 4-step preprocessing pipeline (not fitted yet)."""
    return Pipeline([
        ("clean", CleanRawValues()),
        # log1p on the 0/1 flag gives 0/0.69; after scaling this makes no difference
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]).set_output(transform="pandas")    # keep column names, so we can inspect the result
