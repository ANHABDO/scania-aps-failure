# Glossary

Short definitions of the technical terms used in this project, each with an example from the Scania data.
Grouped by the session where the term first appeared.

---

## Data and files (Sessions 1–2)

| Term | Simple meaning | In our project |
|---|---|---|
| **Raw data** | The original files, exactly as downloaded. Never edited. | `data/raw/aps_failure_training_set.csv` |
| **CSV** | A text file where each line is a row and commas separate the columns. | Both Scania files are CSV, not Excel. |
| **Header** | The line that gives the column names. | `class,aa_000,ab_000,...` comes after the license text. |
| **Missing value / NaN** | A cell with no value. pandas shows it as `NaN` ("Not a Number"). | Written as `na` in the files; 8.33% of cells. |
| **Data validation** | Automatic checks that the data has the shape and content we expect. | `validate_raw()`: 171 columns, labels only `neg`/`pos`, numeric features. |
| **Sentinel / coded value** | A special number that means "something special happened" rather than a real measurement. | `ac_000` values starting with `0x7F000000`. |
| **Hexadecimal** | Writing numbers in base 16 (digits 0–9 and a–f). Computers' internal codes often look "round" in it. | 2,130,706,432 = `0x7F000000`. |
| **Informative missingness** | When *whether* a value is missing tells you something about the answer. | `br_000` missing in 83% of neg trucks but only 9% of pos trucks. |
| **Histogram feature** | A group of columns, each counting how often a truck was in one range of a condition. | 7 groups × 10 bins, e.g. `ag_000`…`ag_009`. |

## Distributions (Session 3)

| Term | Simple meaning | In our project |
|---|---|---|
| **Distribution** | How the values of a column are spread out. | Most columns: many small values, a few huge ones. |
| **Skewness** | How lopsided a distribution is. 0 = balanced; > 1 = long right tail. | Median skew of our columns: 18.4. |
| **Log transform (`log1p`)** | Replace x with log(1 + x). Squeezes huge values closer together, keeps the order. | Columns with \|skew\| > 1 drop from 163 to 85. |
| **Scale** | The typical size of a column's values. | From 2 (`ch_000`) to 8.58 billion (`ad_000`). |
| **Test (pytest)** | Code that checks other code automatically and reports pass/fail. | `tests/test_data_loading.py` |

## Models (preview, Session 3)

| Term | Simple meaning | In our project |
|---|---|---|
| **Logistic regression** | A model that adds up `weight × value` for every feature, then turns the sum into a probability between 0 and 1. | Our first real model (Session 5). Needs log + scaling. |
| **Sigmoid** | The function that turns any number into a value between 0 and 1. | Last step of logistic regression. |
| **Decision tree** | A model that predicts with a flowchart of yes/no questions like "is `ac_000` > 500?". | Phase 4. Doesn't care about scale. |

## Evaluation (Session 4)

| Term | Simple meaning | In our project |
|---|---|---|
| **Training set** | Rows the model learns from. | 48,000 rows after our split. |
| **Validation set** | Rows kept aside to compare models and choices: the "practice exam". | 12,000 rows from the training file. |
| **Test set** | Rows used once, at the very end: the "final exam". | The official 16,000-row test file. Locked. |
| **Data leakage** | When information from validation/test data sneaks into training, making scores look better than they really are. | Why we never look at the test set. |
| **Stratified split** | A split that keeps the same class proportions in each part. | Both parts keep 1.67% positives. |
| **Random seed** | A fixed number that makes "random" operations give the same result every run. | `RANDOM_SEED = 42`. |
| **Class imbalance** | One class is much rarer than the other. | 1,000 pos vs 59,000 neg. |
| **Confusion matrix** | A 2×2 table counting each kind of right and wrong prediction. | tn, fp, fn, tp in `metrics.py`. |
| **True positive (TP)** | Predicted failure, and it really is a failure. | APS failure caught. |
| **False positive (FP)** | Predicted failure, but it isn't one: a false alarm. | Unnecessary workshop check, cost 10. |
| **False negative (FN)** | Predicted no failure, but it is one: a missed failure. | Faulty truck missed, cost 500. |
| **True negative (TN)** | Predicted no failure, and correctly so. | — |
| **Accuracy** | Share of all predictions that are correct. Misleading when classes are imbalanced. | "Always neg" scores about 98% while catching nothing. |
| **Precision** | Of the trucks we flagged, how many really failed? TP / (TP + FP) | High precision = few false alarms. |
| **Recall** | Of the trucks that really failed, how many did we catch? TP / (TP + FN) | High recall = few missed failures. |
| **Cost metric** | The dataset's official score: 10 × FP + 500 × FN. Lower is better. | Our main metric. |
| **Baseline** | A deliberately simple model that sets the minimum bar. | "Always neg", "always pos", random guessing. |
| **fit / predict** | scikit-learn's two main steps: `fit` = learn from training data, `predict` = give answers for new rows. | Every model we use. |

## Preprocessing (Session 5, Step A)

| Term | Simple meaning | In our project |
|---|---|---|
| **Preprocessing** | Preparing raw data so a model can use it: fixing, filling, rescaling. | The 4 steps in `preprocessing.py`. |
| **Pipeline** | A chain of steps that runs in order as one object, like a kitchen preparation line. | `make_preprocessor()`; `make_logreg_pipeline()`. |
| **Transformer** | A pipeline step that changes the data (it has `fit` and `transform`). | `CleanRawValues`, `SimpleImputer`, `StandardScaler`. |
| **transform** | Apply what was learned, without learning anything new. | Used on validation (and later test) data. |
| **fit_transform** | Learn, then apply, in one call. Only on training data. | `pre.fit_transform(X_train)`. |
| **Class (Python)** | A blueprint that keeps data and functions together. | `class CleanRawValues(BaseEstimator, TransformerMixin)`. |
| **Custom transformer** | Our own pipeline step, built on scikit-learn's base classes so it fits in a Pipeline. | `CleanRawValues` applies our Phase 2 decisions. |
| **Imputation** | Filling missing values with a reasonable guess. | Median of each column, learned on train. |
| **Median** | The middle value when values are sorted. Not pulled by extreme values (unlike the mean). | `aa_000` train median ≈ 30,800. |
| **Missing indicator** | A 0/1 column saying "this value was missing". Keeps the information that something was missing. | 168 columns like `missingindicator_br_000`. |
| **Standardization** | Rescaling a column to mean 0 and standard deviation 1: z = (x − mean) / std. | `StandardScaler` (step 4). |
| **Standard deviation (std)** | How spread out the values are around the mean. | After scaling: 1 on train. |
| **MinMaxScaler / RobustScaler** | Other scalers: squeeze into 0–1 / use median and middle 50%. | Not chosen; see Session 5 notes. |

## Logistic regression (Session 5, Step B)

| Term | Simple meaning | In our project |
|---|---|---|
| **Weight (coefficient)** | How much a feature pushes the prediction up or down. Learned in training. | 338 weights, one per column. |
| **Bias (intercept)** | The starting value of z before any feature is added. | The `b` in z = w·x + b. |
| **Training** | Searching for the weights that make predictions match the true labels. | `model.fit(X_train, y_train)`. |
| **Probability (predict_proba)** | The model's estimated chance that a truck is positive, between 0 and 1. | Output of the sigmoid. |
| **Decision threshold** | The probability above which we say "failure". Default 0.5. | To be tuned in Phase 4. |
| **class_weight="balanced"** | Makes mistakes on the rare class count more in training, so both classes carry equal total weight. | Each positive counts ≈ 59× a negative. Cost 30,800 → 11,410. |
| **Convergence / max_iter** | Training is a step-by-step search; it "converges" when it stops improving. max_iter = maximum number of steps. | `max_iter=2000`, no warning. |
| **Oversampling / SMOTE** | Other ways to handle imbalance: repeat positives / create synthetic positives. | Alternatives for Phase 4. |

## Threshold and cross-validation (Session 6)

| Term | Simple meaning | In our project |
|---|---|---|
| **predict_proba** | Gives probabilities instead of yes/no. Two columns: [P(neg), P(pos)]; we keep column 1. | `model.predict_proba(X_val)[:, 1]` |
| **Threshold tuning** | Choosing the probability cut-off that best fits our goal (here: lowest cost). | 144 thresholds tested; 0.39 chosen. |
| **Cross-validation (k-fold)** | Cut the data into k parts; train on k−1, test on the remaining one; rotate k times. Several "practice exams" instead of one. | 5-fold on the 48,000 train rows. |
| **StratifiedKFold** | k-fold where every part keeps the same class proportions. | Each fold has 1.67% positives. |
| **Out-of-fold predictions** | Predictions for each row made by a model that never saw that row during training. | Used to choose the threshold. |
| **Overfitting to validation** | Tuning choices until they fit the validation set's specific rows; the score then looks better than it would on new data. | "Peeking" picks 0.26 (cost 10,830) instead of CV's 0.39. |
| **Noise** | Random variation that is not a real difference. | 1 truck = 500 cost; differences below about 1,000 on validation are not meaningful. |
| **Flat optimum** | A range of settings that all perform about equally well. | Thresholds 0.3–0.5: CV cost within about 4%. |

## Tree models and fair comparison (Session 7)

| Term | Simple meaning | In our project |
|---|---|---|
| **Gradient boosting** | Many small decision trees built one after another; each new tree corrects the mistakes of the previous ones (like golf shots). | `HistGradientBoostingClassifier`, CV cost 35,150 vs 42,880. |
| **Ensemble** | A model made of many smaller models combined. | Gradient boosting (sequential trees), Random Forest (independent trees that vote). |
| **Histogram-based ("Hist")** | Values are first sorted into up to 255 buckets, which makes training fast and light on memory. | Why it runs on the T480. |
| **Native missing-value handling** | The model deals with missing values itself; at each split it learns which side missing values should go. | No imputation in the boosting pipeline. |
| **Early stopping** | Stop adding trees when a held-out part of the training data stops improving, to avoid memorizing. | Automatic (on 10% of train), fixed by `random_state=42`. |
| **Overfitting** | Memorizing the training data, including its noise, instead of learning general rules. Good on train, worse on new data. | What early stopping and cross-validation protect against. |
| **Linear vs non-linear model** | Linear: each feature pushes by a fixed amount, added up. Non-linear: can learn curves, thresholds and combinations. | Logistic regression is linear; trees are non-linear. |
| **Interaction** | When the effect of one feature depends on another ("only when X and Y together"). | Trees can learn them; logistic regression cannot by itself. |
| **Average precision (AP)** | A 0–1 score of how well a model ranks real failures at the top of the list, regardless of the threshold. Random ≈ positive rate (0.017). | CV AP: 0.828 (boosting) vs 0.779 (logreg). |
| **Fold-to-fold variation** | How much a score changes between the CV folds; shows how big the noise is. | Boosting fold costs 5,670–8,470. |
| **Fair comparison** | Same data, same CV, same class weighting, each model with its own tuned threshold. | `compare_models.py`. |

## Error analysis (Session 8)

| Term | Simple meaning | In our project |
|---|---|---|
| **Error analysis** | Studying the actual cases a model gets wrong, not just its score. Like a coach watching the goals conceded. | 42 misses and 1,415 false alarms (CV on train). |
| **Outcome groups (TP / FN / FP / TN)** | Every prediction labelled as caught, missed, false alarm, or correct "no". | `outcome_labels()`. |
| **Near miss** | A missed case with probability just below the threshold; a lower threshold would catch it. | 9 of 42 misses. |
| **Confidently wrong** | A missed case with a very low probability; no threshold change can fix it. | 30 of 42 misses below 0.05. |
| **Permutation importance** | Shuffle one column's values between rows and measure how much the score drops. Big drop = important column. | `ck_000` drops average precision by 0.21. |
| **Group permutation importance** | Shuffle several related columns together, so they cannot "cover" for each other. | Histogram `ag`: 0.119 together vs 0.104 summed. |
| **Correlated features** | Columns that carry overlapping information. They make single-column importance unreliable (too low or too high). | The 10 bins of a histogram. |
| **Proxy** | A measurable value used to stand for something we cannot measure directly. | Histogram total and `aa_000` as possible proxies for how much a truck was used (hypothesis). |

## Packaging the model (Session 9)

| Term | Simple meaning | In our project |
|---|---|---|
| **Final model** | The chosen model, retrained on all allowed data once every decision is frozen. | `hgb_balanced` on all 60,000 training-file rows. |
| **Model artifact** | The saved file that contains a trained model. | `models/model.joblib`. |
| **Serialization (save / load)** | Turning a Python object into a file and back. | `joblib.dump()` / `joblib.load()`. |
| **joblib** | A library that saves and loads Python objects, efficient for large arrays. Installed with scikit-learn. | Used in `model_io.py`. |
| **Metadata** | "Data about data": the label on the jar. | `models/model_meta.json`: threshold, features, versions, CV results. |
| **JSON** | A simple text format for structured information, readable by humans and programs. | The metadata file. |
| **Inference / prediction** | Using a trained model on new data, without training again. | `python -m src.predict`. |
| **Command-line interface (CLI)** | A program you run from the terminal with options like `--input`. | `predict.py`. |
| **argparse** | Python's standard library for reading command-line options and producing `--help`. | `--input`, `--output`, `--model-dir`. |
| **Input validation** | Checking that incoming data has the expected shape before using it. | Missing columns or non-numeric values raise a clear error. |
| **Version pinning** | Recording exact library versions so results can be reproduced. | `requirements.txt` + versions on the model label. |
| **Untrusted pickle / joblib files** | Loading a `.joblib` file can run code, so only load files you created or trust. | Warning in `model_io.py`. |
