# Project Status — APS Failure at Scania Trucks

## Current phase
Phase 2 — Data understanding: **complete**. Next: Phase 3 — preprocessing and baseline.

## Open decisions for Phase 3 (preprocessing)
- `ac_000` coded values (`0x7F000000` + leftover): default = flag column + set value to missing (option b). Option (a) "flag + keep leftover" to test later as an experiment.
- Missing-value indicators: needed; consider one indicator per identical-missingness group instead of 169.
- Row 45977 (`ad_000`, `cf_000`, `co_000` ≈ 8.58 × 10⁹, nearly identical): **updated recommendation — treat these three values as missing** (strong evidence of corruption). Keep the row.
- Logistic regression: `log1p` transform then standardize. Tree models: no transform needed.
- Do not drop zero-heavy columns by zero rate alone (rare non-zero values may be informative).
- Experiment for later: histogram bins as share of histogram total.
- Any feature selection based on the target must happen after the train/validation split.

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
