"""Tests for src/metrics.py, using tiny examples we can check by hand."""

import pytest

from src.metrics import confusion_counts, evaluate, total_cost

#            truck: 1  2  3  4  5
Y_TRUE =          [0, 0, 1, 1, 0]
Y_PRED =          [0, 1, 1, 0, 0]
# truck 1: tn | truck 2: fp (false alarm) | truck 3: tp | truck 4: fn (missed) | truck 5: tn


def test_confusion_counts_by_hand():
    assert confusion_counts(Y_TRUE, Y_PRED) == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}


def test_cost_by_hand():
    assert total_cost(Y_TRUE, Y_PRED) == 10 * 1 + 500 * 1     # 510


def test_perfect_predictions_cost_nothing():
    assert total_cost(Y_TRUE, Y_TRUE) == 0


def test_always_neg_costs_500_per_positive():
    assert total_cost(Y_TRUE, [0] * 5) == 500 * 2


def test_always_pos_costs_10_per_negative():
    assert total_cost(Y_TRUE, [1] * 5) == 10 * 3


def test_constant_predictions_still_give_four_counts():
    counts = confusion_counts([0, 0, 0], [0, 0, 0])
    assert counts == {"tn": 3, "fp": 0, "fn": 0, "tp": 0}


def test_evaluate_scores_by_hand():
    r = evaluate(Y_TRUE, Y_PRED)
    assert r["accuracy"] == pytest.approx(3 / 5)     # 3 correct out of 5
    assert r["precision"] == pytest.approx(1 / 2)    # 1 real failure among 2 alarms
    assert r["recall"] == pytest.approx(1 / 2)       # 1 caught among 2 real failures


def test_never_predicting_pos_gives_zero_precision_not_error():
    r = evaluate(Y_TRUE, [0] * 5)
    assert r["precision"] == 0 and r["recall"] == 0
