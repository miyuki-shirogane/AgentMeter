"""Deterministic state and reward evaluators.

State is generic structured data (``{"boss": {"status": "dead"}, "reward":
100}``). A :class:`StateEvaluator` inspects the *final* state snapshot
recorded in a trace and checks a predicate over a nested path:

    StateEvaluator("boss.status", "eq", "dead")
    StateEvaluator("reward", "gte", 100)

Operators cover field equality, inequality, numeric ordering, nester paths,
and arbitrary custom predicates. Reward is optional and never forced into the
core model: :class:`RewardEvaluator` simply reads the last ``reward`` event if
the environment emitted one.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from agentmeter.core.results import EvaluationResult
from agentmeter.core.trace import Trace
from agentmeter.core.verdict import Verdict
from agentmeter.environments.base import UNSET, resolve_path
from agentmeter.evaluators.base import Evaluator

StateData = dict[str, Any]
StatePredicate = Callable[[StateData], bool]


class StateOperator(StrEnum):
    """Comparison operators supported by the state / reward evaluators."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EXISTS = "exists"


def compare_value(actual: Any, operator: StateOperator, expected: Any) -> bool:
    """Compare a resolved ``actual`` value against ``expected``.

    ``UNSET`` (field missing) only satisfies ``exists``; any other operator
    fails against a missing field rather than treating it as falsy.
    """
    if operator == StateOperator.EXISTS:
        return actual is not UNSET
    if actual is UNSET:
        return False
    if operator == StateOperator.EQ:
        return actual == expected
    if operator == StateOperator.NE:
        return actual != expected
    if operator == StateOperator.GT:
        return actual > expected
    if operator == StateOperator.GTE:
        return actual >= expected
    if operator == StateOperator.LT:
        return actual < expected
    if operator == StateOperator.LTE:
        return actual <= expected
    raise ValueError(f"unsupported operator: {operator!r}")


def make_state_predicate(
    path: str,
    operator: StateOperator | str = StateOperator.EQ,
    expected: Any = None,
    *,
    predicate: Callable[[Any], bool] | None = None,
) -> StatePredicate:
    """Build a ``(state_dict) -> bool`` predicate for ``path``.

    ``operator`` may be a :class:`StateOperator` or its string value. When
    ``predicate`` is given it receives the resolved value and its truth is
    used directly, allowing arbitrary custom logic.
    """
    op = StateOperator(operator)

    if predicate is not None:
        return lambda state: predicate(_resolve(state, path))

    return lambda state: compare_value(_resolve(state, path), op, expected)


def _resolve(state: StateData, path: str) -> Any:
    return resolve_path(state, path)


def _latest_state_snapshot(trace: Trace) -> StateData | None:
    snapshots = trace.state_snapshots()
    if not snapshots:
        return None
    return snapshots[-1].state


class StateEvaluator(Evaluator):
    """Passes when the final state snapshot satisfies a predicate.

    Args:
        path: dotted path into the state, e.g. ``"boss.status"`` or
            ``"$.reward"``.
        operator: a :class:`StateOperator` (or its string value).
        expected: the expected value for the operator (ignored by ``exists``).
        predicate: optional custom ``(value) -> bool``; overrides ``operator``.
        allow_missing: when ``True``, a missing state snapshot is reported as
            an ERROR instead of a FAIL. Defaults to ``False`` (FAIL).
    """

    def __init__(
        self,
        path: str,
        operator: StateOperator | str = StateOperator.EQ,
        expected: Any = None,
        *,
        predicate: Callable[[Any], bool] | None = None,
        allow_missing: bool = False,
    ) -> None:
        self._path = path
        self._operator = StateOperator(operator)
        self._expected = expected
        self._custom = predicate is not None
        self._predicate = make_state_predicate(path, self._operator, expected, predicate=predicate)
        self._allow_missing = allow_missing

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        state = _latest_state_snapshot(trace)
        if state is None:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.ERROR if self._allow_missing else Verdict.FAIL,
                score=0.0,
                reason="no state snapshot recorded in trace",
                metadata={"path": self._path},
            )

        passed = bool(self._predicate(state))
        reason = (
            f"state {self._path!r} matches {self._describe()}"
            if passed
            else f"state {self._path!r} does not match {self._describe()}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={
                "path": self._path,
                "operator": self._operator.value,
                "expected": self._expected,
            },
        )

    def _describe(self) -> str:
        if self._custom:
            return "custom predicate"
        if self._operator == StateOperator.EXISTS:
            return "exists"
        return f"{self._operator.value} {self._expected!r}"


class RewardEvaluator(Evaluator):
    """Passes when the last recorded reward satisfies an operator.

    Reward is optional: if the environment never emitted a reward event, this
    evaluator reports a FAIL (the agent did not earn the expected reward).
    """

    def __init__(
        self,
        operator: StateOperator | str = StateOperator.GTE,
        expected: float = 0.0,
        *,
        predicate: Callable[[float], bool] | None = None,
    ) -> None:
        self._operator = StateOperator(operator)
        self._expected = expected
        self._predicate = predicate

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        rewards = trace.rewards()
        if not rewards:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.FAIL,
                score=0.0,
                reason="no reward event recorded in trace",
                metadata={"operator": self._operator.value, "expected": self._expected},
            )

        latest = rewards[-1].value
        if self._predicate is not None:
            passed = bool(self._predicate(latest))
        else:
            passed = compare_value(latest, self._operator, self._expected)

        description = (
            "custom predicate"
            if self._predicate is not None
            else f"{self._operator.value} {self._expected}"
        )
        reason = (
            f"reward {latest} matches {description}"
            if passed
            else f"reward {latest} does not match {description}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={
                "reward": latest,
                "operator": self._operator.value,
                "expected": self._expected,
            },
        )
