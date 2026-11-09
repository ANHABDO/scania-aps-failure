# Interview Notes

Short, clear answers to questions an interviewer could ask about this project.
Every number comes from a script actually run on the real data.
Updated every session.

---

## The project in 30 seconds

> "I built a predictive-maintenance classifier on real operational data from Scania trucks. The goal is to predict whether a truck's failure comes from one specific component of its air-pressure system. The data is messy: 8% missing values, coded values, a corrupted record, and only 1.67% positive cases. Missing a failure costs 50 times more than a false alarm, so I evaluate with that cost instead of accuracy. My current best model, a class-balanced logistic regression inside a leak-free scikit-learn pipeline with a threshold chosen by cross-validation, catches about 93% of failures on validation and cuts the cost from about 98,760 for the best dummy baseline to about 11,800."

---

## About the data

**Q: What is the prediction target exactly?**
Whether a truck's failure belongs to one specific component of the Air Pressure System (positive), or to another component unrelated to the APS (negative). It is *not* "will the truck break down".

**Q: Why this dataset instead of an easier one like AI4I?**
AI4I is synthetic and clean. Scania is real and messy, so it trains the skills real projects need: missing values, coded values, imbalance, and a business cost.

**Q: What data-quality problems did you find?**
- The files start with license text, so I locate the header line instead of hard-coding it.
- Missing values are written `na`, which pandas does not recognize by default.
- `ac_000` contains a coded value: 9,468 values equal to `0x7F000000` plus a small number, far above the normal range (largest normal value 265,766).
- One record has three nearly identical values of about 8.58 billion in three different columns: corrupted.
- One column (`cd_000`) is constant, so it carries no information.

**Q: How did you decide the huge values were codes and not real measurements?**
Three checks: a huge **gap** above the normal values, the **same few values repeated** (all starting with `0x7F` in hexadecimal), and a different **class mix** (0.33% positive instead of 1.67%).

**Q: Is the missing data random?**
No. Columns go missing in blocks (24 groups covering 136 columns), and missingness is linked to the class in both directions. For example, `br_000` is missing for 83% of negative trucks but only 9% of positive ones. So I keep "was missing" indicator columns.

**Q: Any limitation in that?**
Yes. A model may partly learn *how the data was recorded* rather than how the truck behaves. That works on this benchmark but might not transfer to a system that records data differently.

---

## About the method

**Q: Why not use accuracy?**
Only 1.67% of trucks are positive. A model that always says "no failure" gets 98.33% accuracy and catches nothing. My balanced model has *lower* accuracy (96.21%) than that dummy, yet it is about 9 times cheaper.

**Q: How do you evaluate?**
With the dataset's official cost, 10 × false alarms + 500 × missed failures (lower is better), plus recall and precision.

**Q: How do you avoid data leakage?**
- The official test set is locked until the final evaluation.
- I compare models on a validation set cut from the training data.
- All preprocessing is inside a scikit-learn Pipeline, so medians and scaling are learned on training data only. Proof: after scaling, train means are exactly 0, while validation means are only close to 0 (−0.024 to +0.023).

**Q: Why a stratified split?**
With so few positives, a random split could put too few failures in validation by bad luck. Stratifying keeps 1.67% positives in both parts (800 in train, 200 in validation).

**Q: How do you fill missing values, and why the median?**
With the training median of each column, plus a 0/1 indicator column. Zero would be confused with real zeros (very common here), and the mean is pulled by extreme values.

**Q: Why log-transform and scale?**
The features are extremely skewed (median skewness 18.4) and their maximums range from 2 to 8.58 billion. Logistic regression multiplies values by weights, so one huge column would dominate. `log1p` reduced strongly skewed columns from 163 to 85, then StandardScaler puts every column on the same scale. Tree models would not need this, because they only use the order of values.

**Q: How do you handle the class imbalance?**
`class_weight="balanced"`: errors on the rare positive class count about 59 times more during training. It raised recall from 69.5% to 93% and cut the cost from 30,800 to 11,410.

**Q: How did you choose the decision threshold?**
I tested 144 thresholds on 5-fold cross-validated predictions from the training data and kept the one with the lowest cost (0.39). Then I applied it once to validation. I did not tune on validation, because that would overfit to its 200 failures: tuning there would have picked 0.26, a very different value.

**Q: Did threshold tuning help?**
Honestly, not much for this model. The cost is almost flat between 0.3 and 0.5, and on validation 0.39 cost 11,790 versus 11,410 at 0.5. That difference is about one truck, so it is noise. The class weighting had already shifted the probabilities, which does the same job as lowering the threshold.

**Q: Why not oversampling or SMOTE?**
Class weights give a similar effect without creating artificial data. Those methods remain options to compare later.

---

## Engineering

**Q: How do you make it reproducible?**
A fixed random seed (42), pinned dependencies (`requirements.txt` from `pip freeze`), Git history, and scripts that save their results to `reports/`.

**Q: How do you test it?**
34 automated tests with pytest. They check that loading reads `na` correctly and fails loudly on unexpected values, that the cost is right on hand-checked examples, that the split is stratified and repeatable, and that the pipeline learns only from training data.

---

## Results so far (validation set: 12,000 trucks, 200 failures)

| Model | Cost | Recall | Precision | Accuracy |
|---|---|---|---|---|
| Always "no failure" | 100,000 | 0% | 0% | 98.33% |
| Random guess | 98,760 | 3.0% | 3.3% | 96.92% |
| Logistic regression | 30,800 | 69.5% | 82.3% | 99.24% |
| Logistic regression, balanced (threshold 0.5) | 11,410 | 93.0% | 29.7% | 96.21% |
| **Logistic regression, balanced (threshold 0.39, chosen by CV)** | **11,790** | **93.5%** | 26.1% | 95.48% |

These are validation results. With only 200 failures, differences below about 1,000 are noise. The final test-set score will be measured only once, at the end.
