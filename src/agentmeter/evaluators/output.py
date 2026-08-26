"""Deterministic output evaluators."""

from __future__ import annotations

import re

from agentmeter.core.results import EvaluationResult
from agentmeter.core.trace import Trace
from agentmeter.core.verdict import Verdict
from agentmeter.evaluators.base import Evaluator


def _output(trace: Trace) -> str:
    return trace.final_output or ""


class OutputEqualsEvaluator(Evaluator):
    """Passes when the agent's final output equals the expected text."""

    def __init__(self, expected: str) -> None:
        self._expected = expected

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        output = _output(trace)
        passed = output == self._expected
        reason = (
            f"final output equals {self._expected!r}"
            if passed
            else f"final output {output!r} != {self._expected!r}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"expected": self._expected},
        )


class OutputContainsEvaluator(Evaluator):
    """Passes when the agent's final output contains the expected text."""

    def __init__(self, expected: str) -> None:
        self._expected = expected

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        output = _output(trace)
        passed = self._expected in output
        reason = (
            f"final output contains {self._expected!r}"
            if passed
            else f"final output does not contain {self._expected!r}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"expected": self._expected},
        )


class OutputNotContainsEvaluator(Evaluator):
    """Passes when the agent's final output does NOT contain the expected text."""

    def __init__(self, expected: str) -> None:
        self._expected = expected

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        output = _output(trace)
        passed = self._expected not in output
        reason = (
            f"final output does not contain {self._expected!r}"
            if passed
            else f"final output unexpectedly contains {self._expected!r}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"expected": self._expected},
        )


class OutputRegexEvaluator(Evaluator):
    """Passes when the agent's final output matches the regex pattern.

    The pattern is compiled at construction time, so an invalid regex fails
    fast instead of surfacing mid-evaluation.
    """

    def __init__(self, pattern: str) -> None:
        try:
            self._pattern = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        output = _output(trace)
        passed = self._pattern.search(output) is not None
        reason = (
            f"final output matches {self._pattern.pattern!r}"
            if passed
            else f"final output does not match {self._pattern.pattern!r}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"pattern": self._pattern.pattern},
        )
