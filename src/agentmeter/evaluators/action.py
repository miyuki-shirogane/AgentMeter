"""Deterministic environment-action evaluators.

Environment interactions are recorded as :class:`ActionEvent`, which is
distinct from a :class:`ToolCallEvent` (agent → tool). These evaluators mirror
the tool family but read actions, so tests can assert that an agent performed
(or avoided) a given action, used the right arguments, and followed a required
order inside an environment.

This is how "forbidden action" / "required action" checks (e.g. cheat
detection) are expressed without putting any game-specific rule into core.
"""

from __future__ import annotations

from typing import Any

from agentmeter.core.results import EvaluationResult
from agentmeter.core.trace import ActionEvent, Trace
from agentmeter.core.verdict import Verdict
from agentmeter.environments.base import UNSET, resolve_path
from agentmeter.evaluators.base import Evaluator


def _actions_named(trace: Trace, name: str) -> list[ActionEvent]:
    return [action for action in trace.actions() if action.name == name]


class ActionCalledEvaluator(Evaluator):
    """Passes when the named environment action was taken at least once."""

    def __init__(self, name: str) -> None:
        self._name = name

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        actions = _actions_named(trace, self._name)
        passed = len(actions) > 0
        reason = (
            f"action {self._name!r} was taken {len(actions)} time(s)"
            if passed
            else f"action {self._name!r} was never taken"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"action": self._name, "action_count": len(actions)},
        )


class ActionNotCalledEvaluator(Evaluator):
    """Passes when the named environment action was never taken (forbidden action)."""

    def __init__(self, name: str) -> None:
        self._name = name

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        actions = _actions_named(trace, self._name)
        passed = len(actions) == 0
        reason = (
            f"action {self._name!r} was never taken"
            if passed
            else f"forbidden action {self._name!r} was taken {len(actions)} time(s)"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"action": self._name, "action_count": len(actions)},
        )


class ActionArgumentEvaluator(Evaluator):
    """Checks the arguments of a named environment action.

    With ``field=None`` the whole arguments dict must equal ``expected``;
    otherwise only the (possibly nested) ``field`` is compared. Passes when
    at least one invocation satisfies the expectation.
    """

    def __init__(self, name: str, expected: Any, field: str | None = None) -> None:
        self._name = name
        self._expected = expected
        self._field = field

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        actions = _actions_named(trace, self._name)
        if not actions:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.FAIL,
                score=0.0,
                reason=f"action {self._name!r} was never taken",
                metadata={"action": self._name, "field": self._field},
            )

        matched = False
        for action in actions:
            if self._field is None:
                if action.arguments == self._expected:
                    matched = True
                    break
            else:
                actual = resolve_path(action.arguments, self._field)
                if actual is not UNSET and actual == self._expected:
                    matched = True
                    break

        subject = "arguments" if self._field is None else f"field {self._field!r}"
        reason = (
            f"action {self._name!r} {subject} match {self._expected!r}"
            if matched
            else f"action {self._name!r} {subject} did not match {self._expected!r}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if matched else Verdict.FAIL,
            score=1.0 if matched else 0.0,
            reason=reason,
            metadata={"action": self._name, "field": self._field, "expected": self._expected},
        )


class ActionOrderEvaluator(Evaluator):
    """Passes when actions appear in the expected relative order."""

    def __init__(self, expected: list[str]) -> None:
        self._expected = list(expected)

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        actual = trace.action_names()
        passed = _is_subsequence(self._expected, actual)
        reason = (
            f"action order {self._expected!r} observed within {actual!r}"
            if passed
            else f"action order {self._expected!r} not observed in {actual!r}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={"expected_order": self._expected, "actual_order": actual},
        )


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    it = iter(actual)
    return all(any(item == seen for seen in it) for item in expected)
