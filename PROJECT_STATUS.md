# Project Status — APS Failure at Scania Trucks

## Current phase
Phase 2 — Data understanding (started). No preprocessing or modeling yet.

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

### Hypotheses (NOT yet verified)
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
