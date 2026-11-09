# Interview Notes

Short, clear answers to questions an interviewer could ask about this project.
Every number comes from a script actually run on the real data.
Updated every session.

---

## The project in 30 seconds

> "I built a predictive-maintenance classifier on real operational data from Scania trucks. The goal is to predict whether a truck's failure comes from one specific component of its air-pressure system. The data is messy: 8% missing values, coded values, a corrupted record, and only 1.67% positive cases. Missing a failure costs 50 times more than a false alarm, so I evaluate with that cost instead of accuracy. I compared a logistic regression and a gradient-boosting model under identical cross-validation. Gradient boosting won: about 18% lower cross-validated cost, and on validation it catches 97% of failures, cutting the cost from about 98,760 for the best dummy baseline to 6,670."

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

**Q: Why did gradient boosting beat logistic regression?**
Two reasons. First, it handles missing values itself and learns at each split which way missing trucks should go, and missingness is a strong clue in this data. Second, it is non-linear: it can learn combinations like "this value is coded AND that one is missing", while logistic regression only adds a fixed push per feature. It also needs no log or scaling, because trees only use the order of values.

**Q: How did you make the comparison fair?**
Same train rows, same 5-fold cross-validation, same class weighting, and each model got its own cost-optimal threshold (0.39 for logistic regression, 0.14 for boosting), because their probabilities are on different scales. The main comparison uses cross-validation on 800 failures; validation is a single confirmation.

**Q: How big is the improvement, honestly?**
About 18% lower cost in cross-validation (35,150 vs 42,880). Validation shows 43% (6,670 vs 11,790), but with only 200 failures that is partly luck, so I quote the cross-validation figure. On validation, boosting made fewer misses (6 vs 13) and fewer false alarms (367 vs 529) at the same time. The trade-off: it trains about 3 times slower.

**Q: What is average precision and why report it?**
A 0–1 score of how well the model ranks real failures at the top of the list, independent of the threshold. Boosting: 0.828 vs 0.779 in cross-validation. It confirms the ranking itself is better, not just the threshold choice.

**Q: What did your error analysis show?**
On cross-validated predictions (800 failures), the model missed 42. Most of them (30) had a probability below 0.05, so they are confidently wrong, not near misses: lowering the threshold would not catch them. The missed failures look like lightly used trucks, while the false alarms look like heavily used trucks. For example, the median histogram total is about 34 million for caught failures, 21 million for false alarms, 7 million for missed failures and 2 million for correct negatives. My hypothesis is that the model relies a lot on how intensively a truck was used. Because the feature names are anonymized, I cannot confirm what these columns physically measure.

**Q: Which features matter most?**
Permutation importance on validation: one column, `ck_000`, dominates (average precision drops by 0.21 when it is shuffled), followed by `bj_000`, a bin of histogram `ag`, and `aa_000`. 57 of 170 columns do not help at all. At group level, histograms `ag` and `ay` matter most. I did not remove features based on this, because choosing features on validation would be tuning on it.

**Q: What are the limits of permutation importance?**
With correlated columns, such as the 10 bins of one histogram, shuffling one column can under- or over-state its importance: the other bins can cover for it, or the shuffle can create unrealistic rows. So I also shuffled each histogram as a whole group.

**Q: Why might some false alarms be very confident?**
259 false alarms have a probability above 0.8. In this dataset the negative class is not "healthy trucks" but "trucks with failures in other components", so some of these may show similar symptoms from a different failure. That is a hypothesis I cannot verify with this data.

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
| Logistic regression, balanced (threshold 0.39, chosen by CV) | 11,790 | 93.5% | 26.1% | 95.48% |
| **Gradient boosting, balanced (threshold 0.14, chosen by CV)** | **6,670** | **97.0%** | 34.6% | 96.89% |

These are validation results. With only 200 failures, differences below about 1,000 are noise. The final test-set score will be measured only once, at the end.
