"""Tests for composite evaluators (Not / AllOf / AnyOf)."""

import pytest

from agentmeter import (
    AllOfEvaluator,
    AnyOfEvaluator,
    EvaluationResult,
    Evaluator,
    NotEvaluator,
    OutputContainsEvaluator,
    StateEvaluator,
    StateSnapshotEvent,
    Trace,
    Verdict,
)


class _FixedEvaluator(Evaluator):
    """Returns a canned verdict/score, so composite logic is tested in isolation."""

    def __init__(self, verdict: Verdict, score: float = 1.0, name: str = "fixed") -> None:
        self._verdict = verdict
        self._score = score
        self._name = name

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        return EvaluationResult(
            evaluator=self._name,
            verdict=self._verdict,
            score=self._score,
            reason="fixed",
        )


def _trace() -> Trace:
    return Trace(input="in")


# --------------------------------------------------------------------------
# NotEvaluator
# --------------------------------------------------------------------------


async def test_not_inverts_pass_to_fail():
    result = await NotEvaluator(_FixedEvaluator(Verdict.PASS)).evaluate(_trace())
    assert result.verdict == Verdict.FAIL
    assert result.passed is False


async def test_not_inverts_fail_to_pass():
    result = await NotEvaluator(_FixedEvaluator(Verdict.FAIL, score=0.0)).evaluate(_trace())
    assert result.verdict == Verdict.PASS
    assert result.passed is True


async def test_not_propagates_error():
    result = await NotEvaluator(_FixedEvaluator(Verdict.ERROR, score=0.0)).evaluate(_trace())
    assert result.verdict == Verdict.ERROR


async def test_not_inverts_score():
    result = await NotEvaluator(_FixedEvaluator(Verdict.PASS, score=0.25)).evaluate(_trace())
    assert result.score == pytest.approx(0.75)


# --------------------------------------------------------------------------
# AllOfEvaluator
# --------------------------------------------------------------------------


async def test_all_of_passes_when_every_child_passes():
    evaluator = AllOfEvaluator(
        [_FixedEvaluator(Verdict.PASS), _FixedEvaluator(Verdict.PASS, score=0.5)]
    )
    result = await evaluator.evaluate(_trace())
    assert result.verdict == Verdict.PASS
    assert result.score == pytest.approx(0.5)  # weakest link


async def test_all_of_fails_when_one_child_fails():
    evaluator = AllOfEvaluator(
        [_FixedEvaluator(Verdict.PASS), _FixedEvaluator(Verdict.FAIL, score=0.0)]
    )
    result = await evaluator.evaluate(_trace())
    assert result.verdict == Verdict.FAIL


async def test_all_of_error_dominates_fail():
    evaluator = AllOfEvaluator(
        [_FixedEvaluator(Verdict.FAIL, score=0.0), _FixedEvaluator(Verdict.ERROR, score=0.0)]
    )
    result = await evaluator.evaluate(_trace())
    assert result.verdict == Verdict.ERROR


def test_all_of_requires_at_least_one_evaluator():
    with pytest.raises(ValueError, match="at least one"):
        AllOfEvaluator([])


# --------------------------------------------------------------------------
# AnyOfEvaluator
# --------------------------------------------------------------------------


async def test_any_of_passes_when_one_child_passes():
    evaluator = AnyOfEvaluator(
        [_FixedEvaluator(Verdict.FAIL, score=0.0), _FixedEvaluator(Verdict.PASS, score=0.9)]
    )
    result = await evaluator.evaluate(_trace())
    assert result.verdict == Verdict.PASS
    assert result.score == pytest.approx(0.9)  # best alternative


async def test_any_of_fails_when_no_child_passes():
    evaluator = AnyOfEvaluator(
        [_FixedEvaluator(Verdict.FAIL, score=0.0), _FixedEvaluator(Verdict.FAIL, score=0.3)]
    )
    result = await evaluator.evaluate(_trace())
    assert result.verdict == Verdict.FAIL


async def test_any_of_pass_wins_over_error():
    evaluator = AnyOfEvaluator(
        [_FixedEvaluator(Verdict.ERROR, score=0.0), _FixedEvaluator(Verdict.PASS, score=1.0)]
    )
    result = await evaluator.evaluate(_trace())
    assert result.verdict == Verdict.PASS


async def test_any_of_errors_when_nothing_passes_and_one_errors():
    evaluator = AnyOfEvaluator(
        [_FixedEvaluator(Verdict.FAIL, score=0.0), _FixedEvaluator(Verdict.ERROR, score=0.0)]
    )
    result = await evaluator.evaluate(_trace())
    assert result.verdict == Verdict.ERROR


def test_any_of_requires_at_least_one_evaluator():
    with pytest.raises(ValueError, match="at least one"):
        AnyOfEvaluator([])


# --------------------------------------------------------------------------
# Composition with real evaluators
# --------------------------------------------------------------------------


async def test_not_with_real_evaluator():
    trace = Trace(input="in", final_output="好的，我帮你处理")
    result = await NotEvaluator(OutputContainsEvaluator("抱歉")).evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_any_of_with_real_evaluators():
    """'refunded OR cancelled' — one state field may satisfy either way."""
    trace = Trace(input="in")
    trace.add_event(StateSnapshotEvent(state={"status": "cancelled"}))
    evaluator = AnyOfEvaluator(
        [
            StateEvaluator("status", "eq", "refunded"),
            StateEvaluator("status", "eq", "cancelled"),
        ]
    )
    assert (await evaluator.evaluate(trace)).verdict == Verdict.PASS


async def test_all_of_with_nested_not():
    trace = Trace(input="in", final_output="done")
    evaluator = AllOfEvaluator(
        [
            OutputContainsEvaluator("done"),
            NotEvaluator(OutputContainsEvaluator("error")),
        ]
    )
    assert (await evaluator.evaluate(trace)).verdict == Verdict.PASS
