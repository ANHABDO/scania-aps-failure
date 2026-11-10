"""Tests for the safety rules of src/evaluate_test.py (no test data is touched)."""

import pytest

from src import evaluate_test


def test_refuses_when_results_already_exist(tmp_path, monkeypatch):
    output = tmp_path / "test_results.json"
    output.write_text("{}")
    monkeypatch.setattr(evaluate_test, "git_state", lambda: ("abc1234", True))
    with pytest.raises(SystemExit, match="already exists"):
        evaluate_test.check_ready(output)


def test_refuses_with_uncommitted_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_test, "git_state", lambda: ("abc1234", False))
    with pytest.raises(SystemExit, match="Commit everything first"):
        evaluate_test.check_ready(tmp_path / "test_results.json")


def test_ready_when_clean_and_no_previous_results(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_test, "git_state", lambda: ("abc1234", True))
    assert evaluate_test.check_ready(tmp_path / "test_results.json") == "abc1234"
