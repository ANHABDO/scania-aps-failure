# Project Status — APS Failure at Scania Trucks

## Current phase
**Project 1 — COMPLETE (2026-11-10).** All five phases done; final test evaluated once; reproduced exactly from a fresh clone.
Optional follow-up: usage-normalized features experiment (option B). Next project: Project 2 — AI Job Market Intelligence & Skill Gap Analyzer.

## Final result (official test set, evaluated once at commit `68fe1ea`)
| Trucks | Failures | Caught | Missed | False alarms | Recall | Precision | Avg precision | **Cost** |
|---|---|---|---|---|---|---|---|---|
| 16,000 | 375 | 361 | 14 | 506 | 96.3% | 41.6% | 0.880 | **12,060** |

Cost per 1,000 trucks: validation 556 · cross-validation 732 · **test 754**. IDA 2016 challenge on the same test set: 9,920 / 10,900 / 11,480 (top three).

## Master brief — definition of done (Project 1)
| Criterion | Status |
|---|---|
| Explain the prediction problem and dataset limitations | ✅ README §1, §5; interview notes |
| Repeatable data loading and validation | ✅ `data_loading.py` + tests |
| Preprocessing and modeling without leakage | ✅ Pipelines; CV; test locked until the end |
| Baseline + at least two models compared fairly | ✅ dummies, logistic regression, gradient boosting (same CV, own thresholds) |
| Imbalance-aware metrics with error analysis | ✅ cost, recall, precision, average precision; Session 8 |
| Training command saves real metrics and a trained model | ✅ `train_final.py` (+ `reports/`) |
| Prediction path loads the saved model | ✅ `predict.py` |
| Key behaviour covered by tests | ✅ 57 tests |
| Clean environment installs and runs the workflow | ✅ fresh clone, Session 10 |
| README with method, results, source, limitations, reproduction | ✅ `README.md` |
| User can explain the main decisions without Claude | ⏳ self-check: answer `docs/INTERVIEW_NOTES.md` questions without reading them |

## Phase overview
| Phase | Content | Status |
|---|---|---|
| 0 | Setup: folders, venv, requirements, Git | done |
| 1 | AI4I warm-up (optional) | skipped |
| 2 | Data understanding + first tests | done (Sessions 1–3) |
| 3 | Train/validation split, cost metric, baselines, preprocessing pipeline, logistic regression | done (Sessions 4–5) |
| 4 | Threshold (S6), tree model + fair comparison (S7), error analysis (S8) | done |
| 5 | Final training + saving, prediction script (S9); final test evaluation, README, fresh-install check (S10) | done |

## Key numbers to beat (validation set, 12,000 rows, 200 positives)
| Model | Cost | Recall | Accuracy |
|---|---|---|---|
| always_neg | 100,000 | 0% | 98.33% |
| always_pos | 118,000 | 100% | 1.67% |
| random_guess | 98,760 | 3% | 96.92% |
| logreg_plain (threshold 0.5) | 30,800 | 69.5% | 99.24% |
| logreg_balanced (threshold 0.5) | 11,410 | 93.0% | 96.21% |
| logreg_balanced (threshold 0.39, chosen by 5-fold CV) | 11,790 | 93.5% | 95.48% |
| **hgb_balanced (threshold 0.14, chosen by 5-fold CV) — current best** | **6,670** | **97.0%** | 96.89% |

Main comparison is CV on train (800 failures): logreg 42,880 vs hgb 35,150 (about −18%).

Noise level: 1 truck = 500 cost on validation → differences below about 1,000 are not meaningful.

## Open decisions for Phase 3 (preprocessing)
- `ac_000` coded values (`0x7F000000` + leftover): default = flag column + set value to missing (option b). Option (a) "flag + keep leftover" to test later as an experiment.
- Missing-value indicators: needed; consider one indicator per identical-missingness group instead of 169.
- Row 45977 (`ad_000`, `cf_000`, `co_000` ≈ 8.58 × 10⁹, nearly identical): **updated recommendation — treat these three values as missing** (strong evidence of corruption). Keep the row.
- Logistic regression: `log1p` transform then standardize. Tree models: no transform needed.
- Do not drop zero-heavy columns by zero rate alone (rare non-zero values may be informative).
- Experiment for later: histogram bins as share of histogram total.
- Any feature selection based on the target must happen after the train/validation split.

---

## Session 10 — 2026-11-10

**Objective:** one-time final test evaluation, README, fresh-install reproduction.

### Files created / changed
- `src/evaluate_test.py` (new) — refuses to run with uncommitted changes or if `reports/test_results.json` exists; records the Git commit; evaluates the saved model once on the official test set.
- `tests/test_evaluate_test.py` (new, 3 tests).
- `README.md` (new) — problem, data and license, data-quality findings, method, results (CV / validation / test), limitations, reproduction, structure, future work.
- `reports/test_results.json` (official result), `docs/INTERVIEW_NOTES.md`, `PROJECT_STATUS.md`.

### Commands actually run (by the user)
- `python -m pytest -v` → **57 passed**.
- Commit `68fe1ea`, then `python -m src.evaluate_test` → official result above.
- Fresh clone `scania-check` (commit `9341805`): new venv, `pip install -r requirements.txt` OK, `pytest` **57 passed**, `train_final` (98 s), `evaluate_test --output reports/test_results_check.json` → **identical result** (cost 12,060; 361 / 14 / 506).

### Interpretation
- Test cost per 1,000 trucks (754) matches the cross-validation estimate (732); validation (556) was optimistic.
- Test set has a higher failure rate (2.34% vs 1.67%); per failure the miss rate was lower on test (3.7%) than in CV (5.2%).
- About 22% above the IDA 2016 winner, with default hyperparameters and no feature engineering.

### Understood
- Why the test set is used once, from a frozen and committed state.
- Why CV is a better estimate than one validation split (confirmed by the test).
- What a fresh-install check proves.
- (Corrected) Gradient boosting is itself a tree ensemble; the gap to the winners is not "trees vs boosting" but tuning and feature engineering (their exact methods unknown).

### Next recommended task
1. Self-check: answer the questions in `docs/INTERVIEW_NOTES.md` aloud without reading the answers.
2. Optional: publish the repository on GitHub (without `data/` and `models/`).
3. Optional: option B experiment (usage-normalized features, evaluated with CV — the test set must not be used again for decisions).
4. Then start Project 2.

---

## Session 9 — 2026-11-10

**Objective:** train the final model, save it with metadata, and build a prediction command.

### Files created / changed
- `src/model_io.py` (new) — `save_model()` (joblib + `model_meta.json`), `load_model()` (clear error if missing, warning if scikit-learn version differs).
- `src/train_final.py` (new) — fits `hgb_balanced` on all 60,000 training-file rows, `FINAL_THRESHOLD = 0.14`, saves to `models/`, writes `data/samples/sample_trucks.csv` (20 training trucks, 5 failures — for testing the tool only).
- `src/predict.py` (new) — CLI (`--input`, `--output`, `--model-dir`); finds the header line (plain CSV or official file with license text); validates columns; outputs `row`, `probability`, `flag_for_aps_check`.
- `tests/test_predict.py` (new, 9 tests).
- `docs/GLOSSARY.md`, `docs/INTERVIEW_NOTES.md` — updated.

### Commands actually run (by the user)
- `python -m pytest -v` → **54 passed**.
- `python -m src.train_final` → trained in 78 s on the T480; saved `models/model.joblib` and `models/model_meta.json`.
- `python -m src.predict --help` → help text shown.
- `python -m src.predict --input data/samples/sample_trucks.csv --output reports/sample_predictions.csv` → 20 trucks, 5 flagged.

### Notes
- The 20 sample trucks come from the training file, so the model has seen them: the sample run checks that the tool works, not how good the model is.
- `models/` and `data/` are not committed (`.gitignore`); the model is re-created exactly with `python -m src.train_final`.
- The official test set is still untouched.

### Understood
- Saving/loading a whole Pipeline; metadata as the model's label; version pinning; joblib security caution.
- argparse and a command-line interface.
- (Corrected) Correct predictions on training rows prove nothing about quality, because the model has already seen them.

### Next recommended task
Session 10: README (method, real results, source/license, limitations, reproduction steps), fresh-install check (new virtual environment from `requirements.txt`, run the tests), and the single final evaluation on the official test set.

---

## Session 8 — 2026-11-09

**Objective:** error analysis of `hgb_balanced` — which trucks it gets wrong, and which features it relies on.

### Files created / changed
- `src/importance.py` (new) — `outcome_labels()`, `group_permutation_importance()`.
- `src/error_analysis.py` (new) — outcome groups and probability bins on CV predictions (train); permutation importance on validation (measurement only); top-5 feature medians by group; histogram group shuffling. Saves `reports/error_groups.csv`, `reports/permutation_importance.csv`, `reports/group_importance.csv`.
- `tests/test_importance.py` (new, 3 tests).
- `docs/GLOSSARY.md`, `docs/INTERVIEW_NOTES.md` — updated.

### Commands actually run (by the user)
- `python -m pytest -v` → **45 passed**.
- `python -m src.error_analysis` → ran successfully (368 s on the T480). CV threshold reproduced: 0.14.

### Verified results
**Outcome groups (CV on train, threshold 0.14)**
| Group | Trucks | Median prob. | Mean missing | With `ac_000` code | Median histogram total | Median `aa_000` |
|---|---|---|---|---|---|---|
| TP | 758 | 0.976 | 17.3% | 1.1% | 34,291,130 | 591,949 |
| FN | 42 | 0.021 | 12.9% | 11.9% | 7,043,936 | 121,536 |
| FP | 1,415 | 0.405 | 18.7% | 3.0% | 20,776,325 | 338,784 |
| TN | 45,785 | 0.008 | 7.8% | 16.5% | 2,025,425 | 29,722 |

- Missed failures: 8 below 0.01, 22 at 0.01–0.05, 3 at 0.05–0.07, 9 near misses (0.07–0.14).
- False alarms: 514 at 0.14–0.3, 324 at 0.3–0.5, 318 at 0.5–0.8, 259 above 0.8.
- Permutation importance (validation, drop in average precision): `ck_000` 0.210, `bj_000` 0.072, `ag_002` 0.065, `aa_000` 0.059, `ay_006` 0.042. 57 of 170 columns ≤ 0.
- Top-5 medians: failures (TP) have much larger `ck_000`, `bj_000`, `aa_000` than TN; FN sit in between; `ag_002` is non-zero mainly for TP; `ay_006` is highest for FP (non-monotonic pattern).
- Histogram groups shuffled together: `ag` 0.119 (sum of single bins 0.104), `ay` 0.100 (0.076), `ee` 0.015 (0.009); for `cs`, `cn`, `az` the single-bin sum is larger than the group value.

### Interpretation (hypotheses, not proven)
- The model seems to rely heavily on the magnitude of usage-like counters (`ck_000`, `bj_000`, `aa_000`, histogram totals). Missed failures look like lightly used trucks; false alarms look like heavily used trucks.
- Most misses are confidently wrong → threshold changes cannot fix them; the data may lack clues for these trucks. Consistent with the flat threshold valley.
- Very confident false alarms may be trucks failing in another component (the negative class is "other failures", not "healthy").
- Single-column importance is unreliable with correlated columns, in both directions (covering effect, and unrealistic shuffled rows breaking the equal histogram totals).
- Anonymized names prevent any physical interpretation.

### Decisions
- No feature removal based on validation importance (would be tuning on validation).
- Documented possible experiment (optional, must be evaluated with CV): usage-normalized features (histogram bins as share of total; counters relative to `aa_000`), motivated by the usage pattern above.
- **User decision:** do Phase 5 first (option A); try the usage-normalized experiment (option B) after Phase 5 if time allows.

### Next recommended task
Phase 5, Session 9: final training script that fits the chosen model (`hgb_balanced`, threshold 0.14) on all 60,000 training rows and saves it with its metrics; prediction script that loads it; tests for saving/loading and prediction.

---

## Session 7 — 2026-11-09

**Objective:** add a tree-based model and compare it fairly with logistic regression.

### Files created / changed
- `src/models.py` — added `make_hgb_pipeline()`: `CleanRawValues` + `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)` (no imputation, log or scaling).
- `src/threshold.py` — added `out_of_fold_proba()` (shared CV helper).
- `src/compare_models.py` (new) — same 5-fold CV for each model, own CV threshold, per-fold costs, average precision, one validation check; saves `reports/model_comparison.csv`.
- `tests/test_models.py` — 3 boosting tests (missing values left in, learns a rule, repeatable).
- `docs/GLOSSARY.md`, `docs/INTERVIEW_NOTES.md` — updated (files moved from root to `docs/`).

### Commands actually run (by the user)
- `git log` → Session 6 commit `2be7a57`.
- `python -m pytest -v` → **42 passed**.
- `python -m src.compare_models` → ran successfully (logreg 85 s, boosting 275 s on the T480).

### Verified results
**A) CV on train (800 failures)**
| Model | Threshold | CV cost | Fold range | Recall | Avg precision |
|---|---|---|---|---|---|
| logreg_balanced | 0.39 | 42,880 | 7,140 – 9,430 | 94.0% | 0.779 |
| hgb_balanced | 0.14 | **35,150** | 5,670 – 8,470 | 94.8% | **0.828** |

**B) Validation (200 failures)**
| Model | FN | FP | Recall | Precision | Avg precision | Cost |
|---|---|---|---|---|---|---|
| logreg_balanced | 13 | 529 | 93.5% | 26.1% | 0.772 | 11,790 |
| hgb_balanced | **6** | **367** | 97.0% | 34.6% | 0.849 | **6,670** |

### Interpretation
- Boosting is better on every measure; on validation it reduces misses and false alarms at the same time.
- Validation gap (5,120) is well above noise (about 1,000).
- Honest size of improvement: about 18% (CV, 800 failures). Validation's 43% is likely partly favourable luck. Per-fold ranges overlap.
- Different thresholds (0.14 vs 0.39) confirm that each model needs its own threshold.
- Trade-off: boosting is about 3× slower to train.
- Why boosting wins: native missing-value handling (missingness is informative) and non-linear rules / interactions. Not because it skips log and scaling (those keep the order of values and remove no information).

### Decisions
- **`hgb_balanced` with threshold 0.14 is the current best model.**
- Master brief criterion "a baseline and at least two suitable models compared fairly" is met.

### Understood
- Gradient boosting (sequential trees correcting errors), early stopping, native missing handling.
- Why each model needs its own threshold; average precision as a threshold-free ranking score.
- Predicted correctly that boosting would win; reason refined (missing handling + non-linearity, not "no preprocessing").

### Next recommended task
Session 8: error analysis on the current best model — study the missed failures and the false alarms (cross-validated on train), and which features drive the predictions (permutation importance).

---

## Session 6 — 2026-11-09

**Objective:** choose the decision threshold of `logreg_balanced` for the cost, without tuning on validation.

### Files created / changed
- `src/threshold.py` (new) — `predict_at()`, `cost_at_thresholds()` (144 thresholds: 0.001–0.049 by 0.001, then 0.05–0.99 by 0.01), `best_threshold()`.
- `src/tune_threshold.py` (new) — 5-fold stratified CV out-of-fold probabilities on train → choose threshold → apply once on validation; saves `reports/threshold_cv.csv`, `reports/threshold_validation.csv`.
- `tests/test_threshold.py` (new, 5 tests).
- `docs/GLOSSARY.md`, `docs/INTERVIEW_NOTES.md` — updated.

### Commands actually run (by the user)
- `python -m pytest -v` → **39 passed**.
- `python -m src.tune_threshold` → ran successfully (CV 58 s on the T480).

### Verified results
**CV on train (800 failures)**
| Threshold | FP | FN | Precision | Recall | Cost |
|---|---|---|---|---|---|
| 0.10 | 4,035 | 29 | 16.0% | 96.4% | 54,850 |
| 0.30 | 2,296 | 43 | 24.8% | 94.6% | 44,460 |
| **0.39 (chosen)** | 1,888 | 48 | 28.5% | 94.0% | **42,880** |
| 0.50 | 1,525 | 59 | 32.7% | 92.6% | 44,750 |
| 0.90 | 540 | 125 | 55.6% | 84.4% | 67,900 |

**Validation (200 failures)**
| Threshold | TN | FP | FN | TP | Recall | Cost |
|---|---|---|---|---|---|---|
| 0.50 | 11,359 | 441 | 14 | 186 | 93.0% | 11,410 |
| 0.39 (CV) | 11,271 | 529 | 13 | 187 | 93.5% | 11,790 |
| (peeking, not used) 0.26 | — | — | — | — | — | 10,830 |

### Interpretation
- Flat optimum between about 0.3 and 0.5 (CV cost within about 4%).
- Validation difference 0.39 vs 0.5 (+380) = 1 fewer miss, 88 more alarms → noise (about 1 truck).
- Best threshold is near 0.5, not near 0.02, because `class_weight="balanced"` already shifted the probabilities (same job as lowering the threshold).
- Tuning on validation would pick 0.26 → shows how validation-tuning overfits to 200 specific failures.

### Decisions
- Keep threshold **0.39** (procedure fixed before seeing validation). Do not switch to 0.5 because validation preferred it — that would be tuning on validation.
- Threshold is not the main lever for this model; next gains must come from a better model.
- For model comparison, prefer CV cost (800 failures) over validation alone (200 failures, noisier).

### Understood
- `predict_proba` and choosing our own threshold.
- k-fold cross-validation and out-of-fold predictions; why tuning on validation is optimistic.
- Predicted correctly that the best threshold is below 0.5.

### Next recommended task
Session 7: a tree-based model (`HistGradientBoostingClassifier`), compared fairly with `logreg_balanced` using the same 5-fold CV procedure (cost at each model's own CV-chosen threshold), then confirmed on validation.

---

## Session 5 — 2026-11-08 / 2026-11-09

**Objective:** build the preprocessing pipeline (Phase 2 decisions, fitted on train only) and train the first real model.

### Files created / changed
- `src/preprocessing.py` (new) — `CleanRawValues` custom transformer (learns constant columns; `ac_000` code → `ac_000_is_code` flag + missing; `ad_000`/`cf_000`/`co_000` ≥ 2e9 → missing) + `make_preprocessor()`: clean → `log1p` → median imputation with missing indicators → `StandardScaler`.
- `src/check_preprocessing.py` (new) — inspects what the pipeline learned and produced.
- `src/models.py` (new) — `make_logreg_pipeline(class_weight)`: preprocessing + `LogisticRegression(max_iter=2000)`.
- `src/train_logreg.py` (new) — trains plain and balanced logistic regression, compares with baselines, saves `reports/logreg_results.csv`.
- `tests/test_preprocessing.py` (new, 9 tests), `tests/test_models.py` (new, 3 tests).
- `docs/GLOSSARY.md` — preprocessing and logistic-regression terms added.
- `docs/INTERVIEW_NOTES.md` (new) — interview questions and answers with real numbers.

### Commands actually run (by the user)
- `python -m pytest -v` → **34 passed**.
- `python -m src.check_preprocessing` → ran successfully.
- `python -m src.train_logreg` → ran successfully (plain 8.9 s, balanced 18.0 s on the T480). No warnings shown.

### Verified results
**Preprocessing (fit on train)**
- 170 → 338 columns: 170 − `cd_000` + 1 flag + 168 missing indicators (only columns with missing values in train).
- `ac_000` codes flagged: train 7,622 + val 1,846 = 9,468 (matches Session 2).
- Corrupted values: 1 each in `ad_000`, `cf_000`, `co_000`, all in train (row 45977).
- Missing cells: 679,122 in train → 0 in train and val.
- Train medians match earlier sessions after undoing the log: `aa_000` ≈ 30,800, `ac_000` ≈ 74, `ad_000` ≈ 126.
- Scaling: train means exactly 0 / std 1; validation means −0.024 to +0.023, std 0.835 to 1.218 → scaler learned on train only.

**Logistic regression (validation, threshold 0.5)**
| Model | TN | FP | FN | TP | Precision | Recall | Cost |
|---|---|---|---|---|---|---|---|
| logreg_plain | 11,770 | 30 | 61 | 139 | 82.25% | 69.50% | 30,800 |
| logreg_balanced | 11,359 | 441 | 14 | 186 | 29.67% | 93.00% | 11,410 |

- Balanced: about 8.7× cheaper than the best baseline (98,760), but lower accuracy (96.21%) than always_neg (98.33%) → accuracy trap confirmed on a real model.
- Balanced cost breakdown: misses 14 × 500 = 7,000; false alarms 441 × 10 = 4,410.

### Decisions
- `class_weight="balanced"` adopted for logistic regression.
- 168 individual missing indicators kept for now; grouping them (about 57) remains a later experiment.

### Caution for Phase 4
- If we choose the threshold by looking at validation results, the validation cost at that threshold becomes optimistic. Use cross-validation on the training part to choose it, then confirm on validation.

### Understood
- `fit` / `transform` / `fit_transform` and why validation only gets `transform`.
- Validation means "close to but not exactly 0" as evidence of no leakage.
- Median vs mean vs 0 for imputation; StandardScaler vs MinMax vs Robust.
- Weights, bias, sigmoid; `class_weight` as a fix for imbalance.
- Predicted correctly that balanced weights raise recall.

### Next recommended task
Phase 4 start: choose the decision threshold for `logreg_balanced` using cross-validated probabilities on the training part (cost-based), then confirm on validation.

---

## Session 4 — 2026-11-08

**Objective:** build the evaluation framework (train/validation split, official cost metric) and score dummy baselines.

### Files created / changed
- `src/splitting.py` (new) — `split_features_target()` (pos → 1, neg → 0), `train_val_split()` (stratified, 20% validation, `RANDOM_SEED = 42`).
- `src/metrics.py` (new) — `confusion_counts()`, `total_cost()` (10·FP + 500·FN), `evaluate()` (counts, accuracy, precision, recall, cost).
- `src/run_baselines.py` (new) — split + 3 `DummyClassifier` baselines, saves `reports/baseline_results.csv`.
- `tests/test_metrics.py` (new, 8 tests) and `tests/test_splitting.py` (new, 5 tests).
- `docs/GLOSSARY.md` (new) — technical terms with project examples; updated every session.
- `reports/baseline_results.csv` (generated).
- `requirements.txt` — regenerated with `pip freeze` (adds scikit-learn 1.9.1).

### Commands actually run (by the user)
- `python -m pip install scikit-learn` → scikit-learn 1.9.1.
- `python -m pytest -v` → **22 passed**.
- `python -m src.run_baselines` → ran successfully; results below.
- `git log --oneline` → Session 3 commit confirmed: `62e6477`.

### Verified results (validation set)
- Split: train 48,000 rows / 800 positives; validation 12,000 rows / 200 positives; both 1.67%.
- Baselines: see "Key numbers to beat" at the top.
- Highest accuracy (always_neg, 98.33%) ≠ lowest cost (random_guess, 98,760). Recall 100% (always_pos) gives the worst cost.
- The random baseline gave exactly the same numbers on a fake file with the same class counts: neither the stratified split's class pattern nor the dummy predictions depend on feature values.

### Understood
- Train / validation / test roles; data leakage.
- Stratified split and why it matters with 1.67% positives; random seed for reproducibility.
- Confusion matrix (TP, FP, FN, TN), accuracy, precision, recall, cost.
- `fit` / `predict`.
- Break-even: one caught failure (saves 500) is worth up to 50 false alarms (10 each) → flagging pays off whenever precision is above about 2%.

### Next recommended task
Session 5: preprocessing pipeline fitted on train only (drop `cd_000`; `ac_000` code → flag + missing; row 45977 values → missing; missing indicators; median imputation; `log1p`; standardization) + first logistic regression, scored with the cost metric on validation.

---

## Session 3 — 2026-11-06

**Objective:** finish Phase 2 (negatives, skewness/scale, log transform, histogram consistency) and add first automated tests.

### Files created / changed
- `src/explore_distributions.py` (new) — Q1–Q4 below; treats `ac_000` coded values as missing before statistics.
- `tests/test_data_loading.py` (new) — 8 tests on fake files (`tmp_path`, `monkeypatch`, `pytest.raises`) + 1 test on the real training file (skipped if absent). *Was first saved in `src/`; to be moved to `tests/`.*
- `requirements.txt` — regenerated with `pip freeze` (now includes pytest 9.2.0 and all dependencies).

### Commands actually run (by the user)
- `python -m pip install pytest` → pytest 9.2.0 installed.
- `python -m pytest -v` → **9 passed** (including the real-data test: 60,000 × 171, neg 59,000 / pos 1,000).
- `python -m src.explore_distributions` → ran successfully.
- `git log --oneline` → Session 2 commit confirmed: `ac37c62`.

### Verified results (training set)
| Check | Result |
|---|---|
| Negative values | none in any column |
| Zero-heavy columns | 56 columns > 50% zeros; 33 columns > 90% zeros |
| Scale | column maximum from 2 (`ch_000`) to 8,584,297,742 (`ad_000`) |
| Skewness | 163 of 170 columns \|skew\| > 1; 115 > 10; median skew 18.4 |
| After `log1p` | \|skew\| > 1 drops from 163 to 85 columns; e.g. `ad_000` 212.5 → −0.4, `cf_000` 212.5 → 3.3 |
| Still skewed after log | mostly zero-heavy columns (e.g. `ag_000` median 0, skew 20.8 after log) |
| `ad_000`, `cf_000`, `co_000` maxima | all ≈ 8.58 × 10⁹ — the same row 45977 |
| Histogram totals | 7 groups have the same median total (2,137,520); totals exactly equal for 97–100% of trucks; Spearman correlation 1.00 |

### Understood
- Logistic regression: inputs can be any number; only the output (via sigmoid) is between 0 and 1. Huge inputs dominate `z = w·x + b`, which is why we log-transform and standardize.
- Tree models are insensitive to feature scale.
- `log1p` is used because `log(0)` is undefined.
- A log transform cannot fix "mostly zeros" skew.
- What a test is; fake data in `tmp_path`; tests that check errors are raised.

### Corrected misunderstanding
- Initially thought logistic regression needs inputs between 0 and 1 → corrected (see above).

### Next recommended task
Phase 3 start: stratified train/validation split of the training set (fixed seed), the cost metric (10·FP + 500·FN) as a tested function, and a dummy baseline.

---

## Session 2 — 2026-11-06

**Objective:** test the Session 1 data-quality hypotheses (huge values, missingness) on the training set.

### Files created / changed
- `src/investigate_quality.py` (new) — H1 huge-value checks A–D, H2 missingness checks A, A2, B, C. Describes only; changes nothing.
- `requirements.txt` — to be regenerated with `pip freeze` (numpy now imported directly).
- `PROJECT_STATUS.md` — this section.

### Commands actually run (by the user)
- `python -m src.investigate_quality` → ran successfully on the real training data (twice; second run added check D).

### Verified results (training set)
**H1 — huge values (≥ 2,000,000,000)**
| Finding | Result |
|---|---|
| Cells / rows | 9,513 cells in 9,511 rows |
| `ac_000` | 9,468 huge values; largest normal value 265,766 (≈ 8,000× gap) |
| Pattern | 100% of `ac_000` huge values = `0x7F000000` (2,130,706,432) + leftover; leftover 0–364, median 0, normal `ac_000` median 74 |
| Most common | `0x7F000000` exactly: 6,188 times |
| Class | rows with a huge value: 0.33% positive (vs 1.67% overall) → informative |
| `dq_000` | 42 huge values, no `0x7F` prefix; normal values reach 1.99 × 10⁹ → genuine heavy tail |
| Row 45977 | only row with huge `ad_000`, `cf_000`, `co_000` (same missing group), class neg, no prefix → suspected corrupted record |
| Evenness | every column has 100% even normal values (dataset-wide property) → evenness is not evidence for the code |

**H2 — missingness**
| Finding | Result |
|---|---|
| Identical-missingness groups | 24 groups covering 136 of 169 columns with missing values |
| `ab_000` + `cr_000` | identical (77.2%); `bo_000` has the same rate but only 59% overlap → same rate ≠ same rows |
| `bm_000`…`br_000` | nested "staircase", 80–99% overlap |
| Per-class (strongest) | `br_000` missing 83.3% neg vs 9.0% pos; `ak_000` 6.6% neg vs 53.2% pos |
| Rows by n_missing | 0 → 12.86% pos; 1–10 → 0.42%; 11–20 → 4.24%; 21–50 → 2.71%; 51–100 → 10.44%; 101–170 → 0.69% |

### Hypothesis status
- Huge values are codes → **confirmed for `ac_000`** (systematic `0x7F` prefix, gap, informative). Meaning of leftover → **unproven**. Not true for `dq_000`.
- Same missing rate ⇒ missing together → **partly wrong** (`ab_000`/`cr_000` yes, `bo_000` no).
- Missingness is informative → **confirmed descriptively, in both directions per column**; "positives miss more" was too simple.

### Limitation to document in README
Missingness patterns are strongly linked to the class. A model may learn how the data was recorded rather than truck behaviour; this helps on the benchmark but may not transfer to a deployment where data is recorded differently.

### Understood
- Gap, repetition and class checks to tell codes from real extreme values.
- Reading numbers in hexadecimal to spot coded values.
- Same missing rate does not mean same missing rows.
- Deleting rows needs a documented reason; deleting the 9,511 coded rows would remove 16% of the data and a real signal.

### Still unclear / to review
- Answering "why" with evidence when interpreting results (practise next session).
- Cost metric (FP = 10, FN = 500) — Phase 4.

### Next recommended task
Finish Phase 2: feature distributions and scale (skewness, need for log transform), histogram-bin consistency check, and first `pytest` tests for `data_loading.py`.

---

## Session 1 — 2026-11-05

**Objective:** understand the problem and dataset, set up the project, load the raw data correctly, first inspection.

### Completed
- Problem framing agreed: predict whether a truck's failure belongs to the positive class (failure of one specific APS component) vs. the negative class (failures of components not related to the APS). Not "predict any truck failure".
- Downloaded official files from UCI (dataset id 421) into `data/raw/`: training set, test set, description `.txt`.
- Moved project out of OneDrive to `C:\Users\nasrn\projects\scania-aps-failure` (avoid sync/lock problems with `.venv` and data).
- Inspected raw CSV text manually: files start with a license block; the real header (`class,aa_000,...`) comes after it; missing values written as `na`.
- Created virtual environment and installed `requirements.txt`.
- Ran the first inspection script on the training set.

### Files created
- `src/__init__.py`
- `src/data_loading.py` — finds the header line, loads with `na` as missing (only `na`), validates format (171 columns, `class` first, labels only `neg`/`pos`, all features numeric).
- `src/inspect_data.py` — training-set report; test set only format-checked.
- `requirements.txt` — `pandas==3.0.5`
- `.gitignore` — excludes `.venv/`, caches, `data/`, `models/`
- `PROJECT_STATUS.md`

### Commands actually run (by the user, Python 3.14.6, Windows)
- `python -m src.inspect_data` → ran successfully on the real data. Results below.
- `git log --oneline` → `ff91bc1 Add data loading, validation and first inspection script` (Git initialised, first commit confirmed on branch `main`).

### Verified results (training set)
| Check | Result |
|---|---|
| Shape | 60,000 rows × 171 columns (matches documentation) |
| Classes | neg 59,000 / pos 1,000 → positive rate 1.67% (matches documentation) |
| Types | 169 float64, 1 int64 (`aa_000`, the only column with no missing values), 1 str (`class`) |
| Missing cells overall | 8.33% |
| Columns > 50% missing | 8 (top: `br_000` 82.1%, `bq_000` 81.2%, `bp_000` 79.6%) |
| Mean missing per row | neg 8.2%, pos 17.1% |
| Duplicate rows | 0 |
| Constant columns | `cd_000` |
| Histogram groups | 7 groups × 10 bins (`ag`, `ay`, `az`, `ba`, `cn`, `cs`, `ee`) = 70 columns |
| Test set | loads and validates; same columns as train (no statistics inspected) |

### Decisions
- Test set is **locked**: no statistics, no tuning decisions based on it, until the final evaluation.
- Only the string `na` counts as missing (`keep_default_na=False`) so unexpected values stay visible.
- `cd_000` will be dropped during preprocessing (constant in training data).

### Hypotheses raised (tested in Session 2 — see above)
- Missingness is informative (positives have about 2× more missing values) → may add "was missing" indicator features.
- Columns with identical missing rates (e.g. `ab_000`, `bo_000`, `cr_000` at 77.2%) may be missing together.
- Very large values like `2130706438` in `ac_000` (near the 32-bit integer limit) may be sentinel/error codes rather than real measurements.

### Understood
- Why to look at raw text before `pd.read_csv`.
- Why to search for the header instead of hard-coding `skiprows`.
- Why lowercase `na` needs `na_values` (not in pandas' default list).
- Why the test set must stay untouched.
- Why integer columns become float when they contain `NaN`.

### Still unclear / to review
- Cost metric (FP = 10, FN = 500) and why accuracy is misleading — to be practised in Phase 4.

### Next recommended task
Investigate the two data-quality hypotheses on the training set:
1. Suspicious large values: how many rows have values near 2.13 × 10⁹, in which columns, and in which class?
2. Missingness patterns: do the high-missing columns go missing in the same rows?
