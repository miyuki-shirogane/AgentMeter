"""Tests for JudgeResult schema and validation."""

import pytest

from agentmeter import JudgeResult


def test_judge_result_valid():
    result = JudgeResult(
        passed=True,
        score=0.92,
        reason="Agent 正确理解了用户意图。",
        violations=["a"],
        evidence=["b"],
    )
    assert result.passed is True
    assert result.score == 0.92
    assert result.reason
    assert result.violations == ["a"]
    assert result.evidence == ["b"]


def test_judge_result_defaults():
    result = JudgeResult(passed=False, score=0.4)
    assert result.reason == ""
    assert result.violations == []
    assert result.evidence == []


def test_judge_result_score_must_be_within_range():
    with pytest.raises(ValueError):
        JudgeResult(passed=True, score=1.5)
    with pytest.raises(ValueError):
        JudgeResult(passed=True, score=-0.1)


def test_judge_result_passed_is_required():
    with pytest.raises(ValueError):
        JudgeResult(score=0.5)


def test_judge_result_coerces_numeric_string():
    result = JudgeResult(passed=True, score="0.92")
    assert result.score == 0.92


def test_judge_result_ignores_extra_keys():
    result = JudgeResult.model_validate({"passed": True, "score": 0.9, "notes": "whatever"})
    assert result.score == 0.9
