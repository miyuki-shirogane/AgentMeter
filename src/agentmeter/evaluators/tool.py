"""Deterministic tool-call and tool-argument evaluators."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from agentmeter.core.results import EvaluationResult
from agentmeter.core.trace import ToolCallEvent, Trace
from agentmeter.core.verdict import Verdict
from agentmeter.evaluators.base import Evaluator

_MISSING = object()


def _tool_calls_named(trace: Trace, name: str) -> list[ToolCallEvent]:
    return [call for call in trace.tool_calls() if call.name == name]


class ToolCalledEvaluator(Evaluator):
    """Passes when the named tool was called at least once."""

    def __init__(self, name: str) -> None:
        self._name = name

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        calls = _tool_calls_named(trace, self._name)
        passed = len(calls) > 0
        reason = (
            f"tool {self._name!r} was called {len(calls)} time(s)"
            if passed
            else f"tool {self._name!r} was never called"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"tool": self._name, "call_count": len(calls)},
        )


class ToolNotCalledEvaluator(Evaluator):
    """Passes when the named tool was never called."""

    def __init__(self, name: str) -> None:
        self._name = name

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        calls = _tool_calls_named(trace, self._name)
        passed = len(calls) == 0
        reason = (
            f"tool {self._name!r} was never called"
            if passed
            else f"forbidden tool {self._name!r} was called {len(calls)} time(s)"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"tool": self._name, "call_count": len(calls)},
        )


class ToolCallCountMode(StrEnum):
    EXACTLY = "exactly"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"


class ToolCallCountEvaluator(Evaluator):
    """Checks how many times a named tool was called."""

    def __init__(
        self,
        name: str,
        count: int,
        mode: ToolCallCountMode = ToolCallCountMode.EXACTLY,
    ) -> None:
        self._name = name
        self._count = count
        self._mode = mode

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        calls = _tool_calls_named(trace, self._name)
        actual = len(calls)
        expected = self._count
        if self._mode == ToolCallCountMode.EXACTLY:
            passed = actual == expected
            op = "exactly"
        elif self._mode == ToolCallCountMode.AT_LEAST:
            passed = actual >= expected
            op = "at least"
        else:
            passed = actual <= expected
            op = "at most"
        reason = (
            f"tool {self._name!r} called {actual} time(s) ({op} {expected})"
            if passed
            else f"tool {self._name!r} called {actual} time(s), expected {op} {expected}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={
                "tool": self._name,
                "call_count": actual,
                "expected": expected,
                "mode": self._mode.value,
            },
        )


def _get_path(data: dict[str, Any], path: str) -> Any:
    """Traverse a dotted path like ``options.language`` over a dict.

    Returns :data:`_MISSING` when any segment is absent. This deliberately
    stays small; a full JSONPath/JMESPath engine is out of scope.
    """
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


class ToolArgumentEvaluator(Evaluator):
    """Checks the arguments of a named tool call.

    With ``field=None`` the whole arguments dict must equal ``expected``
    (exact match). With ``field="query"`` or ``field="options.language"``
    only that (possibly nested) field is compared. The tool may be called
    multiple times: the evaluator passes when at least one call satisfies
    the expectation.
    """

    def __init__(self, name: str, expected: Any, field: str | None = None) -> None:
        self._name = name
        self._expected = expected
        self._field = field

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        calls = _tool_calls_named(trace, self._name)
        if not calls:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.FAIL,
                score=0.0,
                reason=f"tool {self._name!r} was never called",
                metadata={"tool": self._name, "field": self._field},
            )

        matched = False
        for call in calls:
            if self._field is None:
                if call.arguments == self._expected:
                    matched = True
                    break
            else:
                actual = _get_path(call.arguments, self._field)
                if actual is not _MISSING and actual == self._expected:
                    matched = True
                    break

        subject = "arguments" if self._field is None else f"field {self._field!r}"
        reason = (
            f"tool {self._name!r} {subject} match {self._expected!r}"
            if matched
            else f"tool {self._name!r} {subject} did not match {self._expected!r}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if matched else Verdict.FAIL,
            score=1.0 if matched else 0.0,
            reason=reason,
            metadata={"tool": self._name, "field": self._field, "expected": self._expected},
        )
