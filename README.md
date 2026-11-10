# Predicting APS Failures in Scania Trucks

A reproducible machine-learning project on real, messy industrial data: predicting whether a heavy truck's failure comes from a specific component of its **Air Pressure System (APS)**, with a cost-aware evaluation.

> **Final result on the untouched official test set (16,000 trucks, 375 APS failures):**
> 361 failures caught (**recall 96.3%**), 14 missed, 506 false alarms, **cost 12,060**
> (official cost rule; the IDA 2016 challenge winner scored 9,920 on the same test set).

---

## 1. The problem

The data comes from heavy Scania trucks in everyday use. Every row is a truck that had a failure:

- **positive (`pos`)**: the failure is in **one specific component of the APS**;
- **negative (`neg`)**: the failure is in **another component, unrelated to the APS**.

So the task is **not** "will this truck break down?" but "is this failure caused by that APS component?".

Errors do not cost the same (official cost rule of the dataset):

| Mistake | Meaning | Cost |
|---|---|---|
| False positive | unnecessary check by a mechanic | 10 |
| False negative | faulty truck missed, may break down | 500 |

**Total cost = 10 × FP + 500 × FN** (lower is better). Accuracy is misleading here: a model that always says "no APS failure" is 98.3% accurate and catches nothing.

## 2. Data

- **Source:** [APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks), UCI Machine Learning Repository, DOI [10.24432/C51S51](https://doi.org/10.24432/C51S51). Creator: Scania CV AB.
- **License:** listed as CC BY 4.0 on the UCI page; the files themselves contain a GPL-3.0 notice. The data is **not** included in this repository.
- **Size:** training file 60,000 rows (1,000 positive, 1.67%); test file 16,000 rows (375 positive, 2.34%).
- **Features:** 170 anonymized columns (single counters + 7 histograms of 10 bins), plus `class`.

### Data-quality findings (training data only)

| Finding | Handling |
|---|---|
| Files start with license text; real header comes later | Header line located automatically |
| Missing values written as `na` (8.33% of cells) | Only `na` is read as missing; anything else unexpected fails validation |
| Missingness is informative (e.g. `br_000` missing for 83% of negatives but 9% of positives) and comes in blocks (24 groups cover 136 columns) | Kept as information: missing indicators (logistic regression) or native handling (gradient boosting) |
| `ac_000`: 9,468 values = `0x7F000000` + a small number, far above the normal range | Coded value → flag column + set to missing |
| One record with ≈ 8.58 × 10⁹ in three columns at once (`ad_000`, `cf_000`, `co_000`) | Treated as corrupted → set to missing |
| `cd_000` is constant | Dropped (learned on training data) |
| Extreme skew (median skewness 18.4) and scales from 2 to 8.58 × 10⁹ | `log1p` + standardization for logistic regression only; not needed for trees |
| The 7 histograms have identical totals for 97–100% of trucks | Used as a consistency check |

## 3. Method

1. **Split:** the training file is split into train (48,000) and validation (12,000), stratified, seed 42. The official test set is locked until the very end.
2. **Pipelines (scikit-learn):** all preprocessing is inside a `Pipeline`, so everything is learned from training data only (no leakage).
3. **Baselines:** always "no", always "yes", random guessing.
4. **Models:** logistic regression (with `log1p`, median imputation + missing indicators, scaling) and histogram gradient boosting (cleaning step only). Both with `class_weight="balanced"`.
5. **Fair comparison:** same 5-fold stratified cross-validation; **each model gets its own cost-optimal threshold**, chosen on out-of-fold predictions (never on validation).
6. **Error analysis:** missed failures and false alarms studied on cross-validated predictions; permutation importance (single columns and whole histograms).
7. **Final model:** gradient boosting retrained on all 60,000 training rows, threshold 0.14, saved with its metadata. Evaluated **once** on the test set, from a committed and clean Git state.

## 4. Results

### Model selection (5-fold cross-validation on train, 800 failures)

| Model | Threshold | CV cost | Recall | Average precision |
|---|---|---|---|---|
| Logistic regression (balanced) | 0.39 | 42,880 | 94.0% | 0.779 |
| **Gradient boosting (balanced)** | **0.14** | **35,150** | **94.8%** | **0.828** |

### Validation (12,000 trucks, 200 failures)

| Model | Missed | False alarms | Recall | Cost |
|---|---|---|---|---|
| Always "no failure" | 200 | 0 | 0% | 100,000 |
| Random guess | 194 | 176 | 3.0% | 98,760 |
| Logistic regression (balanced, threshold 0.39) | 13 | 529 | 93.5% | 11,790 |
| **Gradient boosting (balanced, threshold 0.14)** | **6** | **367** | **97.0%** | **6,670** |

### Final test (16,000 trucks, 375 failures) — evaluated once

| Missed | False alarms | Recall | Precision | Average precision | **Cost** |
|---|---|---|---|---|---|
| 14 | 506 | 96.3% | 41.6% | 0.880 | **12,060** |

**Reading the numbers honestly**

- Cost per 1,000 trucks: validation **556**, cross-validation **732**, test **754**. Cross-validation (more data, 5 rotations) predicted the test result well; the single validation split was optimistic.
- The test set has more failures (2.34% vs 1.67%), which raises cost per 1,000 trucks. Per failure, the test miss rate (3.7%) was lower than in cross-validation (5.2%).
- Reference: IDA 2016 Industrial Challenge on the same test set — 1st 9,920, 2nd 10,900, 3rd 11,480. This project used default model settings and no feature engineering.

### What the model relies on (hypotheses)

- One column, `ck_000`, dominates permutation importance (average precision drops by 0.21 when shuffled), then `bj_000`, histogram `ag`, `aa_000`.
- Missed failures look like **lightly used** trucks and false alarms like **heavily used** ones (e.g. median histogram total: caught failures 34.3M, false alarms 20.8M, missed failures 7.0M, correct negatives 2.0M). The model seems to rely strongly on usage-like magnitude.
- 30 of 42 cross-validated misses had a probability below 0.05: they are not near misses, so a lower threshold would not catch them.

## 5. Limitations

- **Anonymized features:** we can say which columns matter, not what they physically measure.
- **Negative ≠ healthy:** negatives are trucks with *other* failures; some confident false alarms may share symptoms with APS failures.
- **Informative missingness:** the model partly uses *how data was recorded*. This helps on this benchmark but may not transfer to a system that records data differently.
- **Single dataset, fixed period:** no evidence about other truck models, other years, or data drift. This is a benchmark result, **not a production-ready system**.
- **Cost rule is fixed** at 10 / 500; a different business cost would need a different threshold.
- **Default hyperparameters** for gradient boosting; no tuning was done.

## 6. Reproduce

Requirements: Python 3.14 (tested on Windows with Python 3.14.6, CPU only, 8 GB RAM); exact library versions in `requirements.txt`. Timings below are from a ThinkPad T480.

```powershell
git clone https://github.com/ANHABDO/scania-aps-failure.git
cd scania-aps-failure
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the dataset from the UCI page and put these two files in `data/raw/`:
`aps_failure_training_set.csv`, `aps_failure_test_set.csv`.

```powershell
python -m pytest                     # 61 tests (about 1 minute)
python -m src.run_baselines          # dummy baselines            -> reports/baseline_results.csv
python -m src.compare_models         # CV comparison (~6 min)     -> reports/model_comparison.csv
python -m src.error_analysis         # error analysis (~6 min)    -> reports/error_groups.csv, ...
python -m src.train_final            # final model (~1.5 min)     -> models/
python -m src.predict --input data/samples/sample_trucks.csv --output reports/sample_predictions.csv
python -m src.evaluate_test --output reports/test_results_check.json   # re-check the test result
```

`reports/test_results.json` holds the official one-time result; the script refuses to overwrite it.

### Predict for new trucks

```powershell
python -m src.predict --input my_trucks.csv --output predictions.csv
```
Input: a CSV with the 170 feature columns (missing values as `na`; the official file format with license text also works). Output: `row`, `probability`, `flag_for_aps_check` (1 if probability ≥ 0.14).

> Only load model files you created yourself: loading a `.joblib` file can run code.

## 7. Project structure

```
src/
  data_loading.py        load raw files, validate format
  splitting.py           features/target, stratified train/validation split
  preprocessing.py       cleaning step + logistic-regression preprocessing pipeline
  models.py              logistic regression and gradient boosting pipelines
  metrics.py             confusion counts, official cost, evaluation
  threshold.py           threshold search, cross-validated probabilities
  importance.py          outcome labels, group permutation importance
  model_io.py            save/load model + metadata
  train_final.py         train and save the final model
  predict.py             command-line predictions
  evaluate_test.py       one-time final test evaluation
  features.py            usage-share features (experiment, section 8)
  (exploration scripts)  inspect_data, investigate_quality, explore_distributions,
                         check_preprocessing, run_baselines, train_logreg,
                         tune_threshold, compare_models, error_analysis,
                         experiment_usage
tests/                   61 pytest tests
reports/                 saved results (CSV / JSON)
docs/                    GLOSSARY.md, INTERVIEW_NOTES.md
PROJECT_STATUS.md        session-by-session log of decisions and results
```

## 8. Experiment after the final evaluation: usage-share features

Motivated by the error analysis, I added 70 features (each histogram bin divided by its histogram's total) plus the histogram total, keeping all original columns, and compared with the final model on the same 5 cross-validation folds (`python -m src.experiment_usage`, about 12 minutes).

| Variant | Threshold | CV cost | Missed | False alarms | Average precision |
|---|---|---|---|---|---|
| Final model | 0.14 | 35,150 | 42 | 1,415 | 0.828 |
| + usage shares | 0.09 | 34,320 | 37 | 1,582 | 0.842 |

Fold by fold the shares were cheaper in **3 of 5 folds** (differences −150, −700, −290, +200, +110). The total gain (−830, about 2.4%) is the size of one or two trucks per fold, so it is **small and not robust**. The final model was **not changed**: the evidence is weak, and the test set has already been used, so a changed model could not get an unbiased test score. This is reported as an exploratory cross-validation result only.

## 9. Future work

- Usage-share features: confirm or reject the small gain with repeated cross-validation (several random fold splits).
- Hyperparameter tuning of gradient boosting (with cross-validation).
- Grouped missing indicators (one per missingness block instead of 168).
- Cost-sensitive analysis for other cost ratios.
