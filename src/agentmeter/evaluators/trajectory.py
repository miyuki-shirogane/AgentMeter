"""Deterministic trajectory evaluators.

Trajectory evaluators reason about the *order and shape* of tool calls
recorded in the trace, not about their semantics. Semantic judgments (e.g.
"was this action actually necessary?") are left to a future LLM judge.
"""

from __future__ import annotations

from agentmeter.core.results import EvaluationResult
from agentmeter.core.trace import Trace
from agentmeter.core.verdict import Verdict
from agentmeter.evaluators.base import Evaluator
from agentmeter.evaluators.tool import ToolCalledEvaluator, ToolNotCalledEvaluator


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    """True when ``expected`` appears in ``actual`` preserving relative order."""
    it = iter(actual)
    return all(any(item == seen for seen in it) for item in expected)


class ToolOrderEvaluator(Evaluator):
    """Passes when tool calls appear in the expected relative order.

    Matching is by subsequence: other tool calls may happen in between, but
    the expected calls must keep their relative order. For example, expected
    ``["search", "calculate", "answer"]`` fails against the actual order
    ``["calculate", "search", "answer"]``.
    """

    def __init__(self, expected: list[str]) -> None:
        self._expected = list(expected)

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        actual = trace.tool_call_names()
        passed = _is_subsequence(self._expected, actual)
        reason = (
            f"tool order {self._expected!r} observed within {actual!r}"
            if passed
            else f"tool order {self._expected!r} not observed in {actual!r}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"expected_order": self._expected, "actual_order": actual},
        )


class MaximumToolCallsEvaluator(Evaluator):
    """Passes when the total number of tool calls does not exceed a bound."""

    def __init__(self, max_calls: int) -> None:
        self._max_calls = max_calls

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        actual = len(trace.tool_calls())
        passed = actual <= self._max_calls
        reason = (
            f"tool calls {actual} <= max {self._max_calls}"
            if passed
            else f"tool calls {actual} exceed max {self._max_calls}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"actual": actual, "max_calls": self._max_calls},
        )


class RequiredToolEvaluator(ToolCalledEvaluator):
    """Trajectory alias: the named tool must appear in the run."""


class ForbiddenToolEvaluator(ToolNotCalledEvaluator):
    """Trajectory alias: the named tool must never appear in the run."""
