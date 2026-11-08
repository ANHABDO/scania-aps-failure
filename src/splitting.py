"""Separate features from the target, and split the TRAINING data into train / validation.

The official test set is NOT used here. It stays locked until the final evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loading import TARGET_COL

RANDOM_SEED = 42         # any fixed number works; fixing it makes the split repeatable
VALIDATION_SIZE = 0.2    # 20% of the training rows become the validation set


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (the 170 feature columns) and y (1 = pos, 0 = neg)."""
    X = df.drop(columns=TARGET_COL)
    y = (df[TARGET_COL] == "pos").astype(int).rename("target")
    return X, y


def train_val_split(X: pd.DataFrame, y: pd.Series,
                    val_size: float = VALIDATION_SIZE, seed: int = RANDOM_SEED):
    """Stratified split: train and validation keep the same positive rate.

    Returns X_train, X_val, y_train, y_val.
    """
    return train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)
