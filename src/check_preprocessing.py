"""Session 5, Step A: fit the preprocessing pipeline on TRAIN and inspect what it produces.

Run from the project root:
    python -m src.check_preprocessing
"""

from src.data_loading import load_raw
from src.preprocessing import make_preprocessor
from src.splitting import split_features_target, train_val_split


def main() -> None:
    X, y = split_features_target(load_raw("train"))
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    pre = make_preprocessor()
    Xt_train = pre.fit_transform(X_train)   # LEARN on train, then transform train
    Xt_val = pre.transform(X_val)           # only APPLY what was learned

    clean = pre.named_steps["clean"]
    impute = pre.named_steps["impute"]
    n_indicators = sum(c.startswith("missingindicator_") for c in Xt_train.columns)

    print("1. Shape")
    print(f"   train: {X_train.shape} -> {Xt_train.shape}")
    print(f"   val:   {X_val.shape} -> {Xt_val.shape}")

    print("\n2. What the cleaning step learned / did")
    print(f"   constant columns dropped (learned on train): {clean.constant_columns_}")
    print(f"   ac_000 coded values flagged: train {int((X_train['ac_000'] >= 0x7F000000).sum()):,}, "
          f"val {int((X_val['ac_000'] >= 0x7F000000).sum()):,}")
    for col in ["ad_000", "cf_000", "co_000"]:
        n_tr = int((X_train[col] >= 2e9).sum())
        n_val = int((X_val[col] >= 2e9).sum())
        print(f"   {col} corrupted values set to missing: train {n_tr}, val {n_val}")

    print("\n3. Missing values")
    print(f"   before: train {int(X_train.isna().sum().sum()):,} missing cells")
    print(f"   after:  train {int(Xt_train.isna().sum().sum())}, val {int(Xt_val.isna().sum().sum())}")
    print(f"   'was missing' indicator columns added: {n_indicators}")

    print("\n4. Medians used for filling (learned on train, log scale) - first 5")
    medians = dict(zip(impute.feature_names_in_, impute.statistics_))
    for col in list(medians)[:5]:
        print(f"   {col}: {medians[col]:.3f}")

    print("\n5. Scaling check (mean should be 0 and std 1 on TRAIN; only close on VAL)")
    for name, data in [("train", Xt_train), ("val", Xt_val)]:
        means, stds = data.mean(), data.std()
        print(f"   {name:<5}: column means from {means.min():+.3f} to {means.max():+.3f} | "
              f"stds from {stds.min():.3f} to {stds.max():.3f}")


if __name__ == "__main__":
    main()
