"""Tests for Evaluator and deterministic output evaluators."""

import pytest

from agentmeter import (
    EvaluationResult,
    Evaluator,
    OutputContainsEvaluator,
    OutputEqualsEvaluator,
    OutputNotContainsEvaluator,
    OutputRegexEvaluator,
    Trace,
    Verdict,
)


def _trace(output: str) -> Trace:
    return Trace(input="input", final_output=output)


async def test_output_contains_passes():
    result = await OutputContainsEvaluator("北京").evaluate(_trace("你好，北京！"))
    assert isinstance(result, EvaluationResult)
    assert result.passed is True
    assert result.score == 1.0
    assert result.evaluator == "OutputContainsEvaluator"


async def test_output_contains_fails():
    result = await OutputContainsEvaluator("上海").evaluate(_trace("你好，北京！"))
    assert result.passed is False
    assert result.score == 0.0
    assert "does not contain" in result.reason


async def test_output_equals_passes():
    result = await OutputEqualsEvaluator("你好，北京！").evaluate(_trace("你好，北京！"))
    assert result.passed is True
    assert result.score == 1.0


async def test_output_equals_fails():
    result = await OutputEqualsEvaluator("你好，上海！").evaluate(_trace("你好，北京！"))
    assert result.passed is False
    assert result.score == 0.0


async def test_evaluator_handles_none_final_output():
    trace = Trace(input="input", final_output=None)
    result = await OutputContainsEvaluator("北京").evaluate(trace)
    assert result.passed is False


def test_evaluator_is_abstract():
    assert Evaluator.__abstractmethods__ == {"evaluate"}


def test_evaluation_result_defaults():
    result = EvaluationResult(evaluator="E", verdict=Verdict.PASS, score=1.0, reason="ok")
    assert result.metadata == {}


def test_evaluation_result_passed_derived_from_verdict():
    assert (
        EvaluationResult(evaluator="E", verdict=Verdict.PASS, score=1.0, reason="").passed is True
    )
    assert (
        EvaluationResult(evaluator="E", verdict=Verdict.FAIL, score=0.0, reason="").passed is False
    )
    assert (
        EvaluationResult(evaluator="E", verdict=Verdict.ERROR, score=0.0, reason="").passed is False
    )


def test_evaluation_result_score_must_be_within_range():
    with pytest.raises(ValueError):
        EvaluationResult(evaluator="E", verdict=Verdict.PASS, score=1.5, reason="")
    with pytest.raises(ValueError):
        EvaluationResult(evaluator="E", verdict=Verdict.PASS, score=-0.1, reason="")


async def test_output_not_contains_passes():
    result = await OutputNotContainsEvaluator("上海").evaluate(_trace("你好，北京！"))
    assert result.passed is True
    assert result.verdict == Verdict.PASS


async def test_output_not_contains_fails():
    result = await OutputNotContainsEvaluator("北京").evaluate(_trace("你好，北京！"))
    assert result.passed is False
    assert result.verdict == Verdict.FAIL


async def test_output_regex_passes():
    result = await OutputRegexEvaluator(r"^你好，\w+！$").evaluate(_trace("你好，北京！"))
    assert result.passed is True


async def test_output_regex_fails():
    result = await OutputRegexEvaluator(r"\d{4}").evaluate(_trace("你好，北京！"))
    assert result.passed is False
    assert result.score == 0.0


def test_output_regex_invalid_pattern_fails_fast():
    with pytest.raises(ValueError):
        OutputRegexEvaluator("(")
