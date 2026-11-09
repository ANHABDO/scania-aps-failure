"""Tests for src/threshold.py on a 4-truck example you can check by hand."""

import numpy as np

from src.threshold import best_threshold, cost_at_thresholds, predict_at

#          truck:  A     B     C     D
Y_TRUE =         [0,    0,    1,    1]
PROBA =          [0.1,  0.4,  0.35, 0.8]


def test_predict_at_uses_greater_or_equal():
    assert predict_at(PROBA, 0.4).tolist() == [0, 1, 0, 1]


def test_threshold_0_5_misses_truck_c():
    # predictions [0, 0, 0, 1] -> C is missed: 1 FN, 0 FP
    row = cost_at_thresholds(Y_TRUE, PROBA, [0.5]).loc[0.5]
    assert (row["fp"], row["fn"], row["cost"]) == (0, 1, 500)


def test_threshold_0_3_catches_c_with_one_false_alarm():
    # predictions [0, 1, 1, 1] -> B is a false alarm: 1 FP, 0 FN
    row = cost_at_thresholds(Y_TRUE, PROBA, [0.3]).loc[0.3]
    assert (row["fp"], row["fn"], row["cost"]) == (1, 0, 10)


def test_best_threshold_picks_lowest_cost():
    table = cost_at_thresholds(Y_TRUE, PROBA, [0.3, 0.5, 0.9])
    assert best_threshold(table) == 0.3


def test_lower_threshold_never_lowers_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = rng.random(500)
    recall = cost_at_thresholds(y, proba)["recall"]
    assert recall.is_monotonic_decreasing      # thresholds go up -> recall goes down
