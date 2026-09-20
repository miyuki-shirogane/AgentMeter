"""Evaluators over the *history* of environment state.

:class:`~agentmeter.evaluators.state.StateEvaluator` only inspects the final
snapshot, so a value that is briefly moved to an illegal value and then
corrected is invisible to it. These evaluators read the recorded timeline
instead:

- :class:`StateChangeEvaluator` checks each reported delta as it happens.
- :class:`StateTransitionEvaluator` checks the legal transitions of a label.
- :class:`StateMonotonicEvaluator` checks a numeric path never reverses.
- :class:`StateUnchangedEvaluator` checks a path is never touched at all.

All four rely on the snapshot timeline being trustworthy. ``StateSnapshotEvent``
and ``StateChangeEvent`` deep-copy their payload on construction, so recorded
history cannot be rewritten by a later in-place mutation of the environment's
state dict.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal

from agentmeter.core.results import EvaluationResult
from agentmeter.core.trace import StateChangeEvent, Trace
from agentmeter.core.verdict import Verdict
from agentmeter.environments.base import UNSET, resolve_path
from agentmeter.evaluators.base import Evaluator
from agentmeter.evaluators.state import StateOperator, compare_value

MonotonicDirection = Literal[
    "non_decreasing",
    "non_increasing",
    "strictly_increasing",
    "strictly_decreasing",
]

_MONOTONIC_DIRECTIONS: tuple[str, ...] = (
    "non_decreasing",
    "non_increasing",
    "strictly_increasing",
    "strictly_decreasing",
)


def _describe(predicate: bool, operator: StateOperator, expected: Any) -> str:
    return "custom predicate" if predicate else f"{operator.value} {expected!r}"


class StateChangeEvaluator(Evaluator):
    """Passes when every recorded change of ``path`` satisfies a predicate.

    Whereas :class:`~agentmeter.evaluators.state.StateEvaluator` answers "is
    the value right *now*?", this answers "was every step right?". A value that
    is briefly driven to an illegal level and then restored is caught here and
    missed there.

    Only deltas the environment actually reported are visible: a
    ``StateChangeEvent`` is emitted only when ``changes`` is non-empty, so a
    step that changed nothing produces no event. ``changes`` is also a
    *partial* delta — fields the environment does not report are invisible.

    Args:
        path: dotted path into the delta, e.g. ``"total"``.
        operator: a :class:`StateOperator` (or its string value).
        expected: the expected value for the operator.
        predicate: optional custom ``(value) -> bool``; overrides ``operator``.
        require_at_least_one: when ``True`` (default), a trace in which no
            change mentions ``path`` is reported as a FAIL — the assertion was
            never exercised, so it proved nothing.
    """

    def __init__(
        self,
        path: str,
        operator: StateOperator | str = StateOperator.EQ,
        expected: Any = None,
        *,
        predicate: Callable[[Any], bool] | None = None,
        require_at_least_one: bool = True,
    ) -> None:
        self._path = path
        self._operator = StateOperator(operator)
        self._expected = expected
        self._predicate = predicate
        self._require_at_least_one = require_at_least_one

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        inspected: list[tuple[Any, StateChangeEvent]] = []
        for event in trace.state_changes():
            value = resolve_path(event.changes, self._path)
            if value is not UNSET:
                inspected.append((value, event))

        if not inspected:
            if self._require_at_least_one:
                return EvaluationResult(
                    evaluator=type(self).__name__,
                    verdict=Verdict.FAIL,
                    score=0.0,
                    reason=f"no state change recorded for {self._path!r}; assertion not exercised",
                    metadata={"path": self._path},
                )
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.PASS,
                score=1.0,
                reason=f"no state change recorded for {self._path!r} (allowed)",
                metadata={"path": self._path, "inspected": 0},
            )

        violations: list[dict[str, Any]] = []
        for value, event in inspected:
            if self._predicate is not None:
                ok = bool(self._predicate(value))
            else:
                ok = compare_value(value, self._operator, self._expected)
            if not ok:
                violations.append({"value": value, "action_id": event.action_id})

        passed = not violations
        description = _describe(self._predicate is not None, self._operator, self._expected)
        reason = (
            f"all {len(inspected)} change(s) of {self._path!r} satisfy {description}"
            if passed
            else f"{len(violations)} of {len(inspected)} change(s) of "
            f"{self._path!r} violate {description}: {violations}"
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
                "inspected": len(inspected),
                "violations": violations,
            },
        )


class StateTransitionEvaluator(Evaluator):
    """Passes when every transition of ``path`` is in the allowed set.

    Reads the snapshot timeline (one snapshot per action, plus the initial
    reset), compares each pair of consecutive values, and requires every
    observed transition to be listed in ``allowed``. For example, a
    ``status`` field that may only go ``draft -> paid -> refunded`` is
    expressed as ``allowed={("draft", "paid"), ("paid", "refunded")}``; an
    observed ``("paid", "draft")`` fails.

    Args:
        path: dotted path to a discrete label, e.g. ``"status"``.
        allowed: the legal transitions as ``(from, to)`` pairs.
        allow_unchanged: when ``True`` (default), a snapshot whose value equals
            the previous one is not counted as a transition.
        on_missing: what to do when ``path`` is absent from a snapshot —
            ``"error"`` (default) reports ERROR because the history cannot be
            read completely, ``"skip"`` ignores that snapshot.
        require_at_least_one: when ``True`` (default), a trace with no
            transition at all is reported as a FAIL (nothing was exercised).
    """

    def __init__(
        self,
        path: str,
        allowed: Iterable[tuple[Any, Any]],
        *,
        allow_unchanged: bool = True,
        on_missing: Literal["error", "skip"] = "error",
        require_at_least_one: bool = True,
    ) -> None:
        if on_missing not in ("error", "skip"):
            raise ValueError(f"on_missing must be 'error' or 'skip', got {on_missing!r}")
        self._path = path
        self._allowed = [tuple(pair) for pair in allowed]
        self._allow_unchanged = allow_unchanged
        self._on_missing = on_missing
        self._require_at_least_one = require_at_least_one

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        snapshots = trace.state_snapshots()
        if not snapshots:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.ERROR,
                score=0.0,
                reason="no state snapshots recorded in trace",
                metadata={"path": self._path},
            )

        values: list[Any] = []
        for index, snapshot in enumerate(snapshots):
            value = resolve_path(snapshot.state, self._path)
            if value is UNSET:
                if self._on_missing == "error":
                    return EvaluationResult(
                        evaluator=type(self).__name__,
                        verdict=Verdict.ERROR,
                        score=0.0,
                        reason=f"state {self._path!r} missing in snapshot #{index}",
                        metadata={"path": self._path, "snapshot": index},
                    )
                continue
            values.append(value)

        transitions: list[tuple[Any, Any]] = []
        for previous, current in zip(values, values[1:], strict=False):
            if previous == current and self._allow_unchanged:
                continue
            transitions.append((previous, current))

        if not transitions and self._require_at_least_one:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.FAIL,
                score=0.0,
                reason=f"no transition of {self._path!r} observed; assertion not exercised",
                metadata={"path": self._path, "allowed": self._allowed},
            )

        violations = [pair for pair in transitions if pair not in self._allowed]
        passed = not violations
        reason = (
            f"all {len(transitions)} transition(s) of {self._path!r} are allowed"
            if passed
            else f"forbidden transition(s) of {self._path!r}: "
            f"{[f'{a} -> {b}' for a, b in violations]} (observed {transitions})"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={
                "path": self._path,
                "allowed": [list(pair) for pair in self._allowed],
                "observed": [list(pair) for pair in transitions],
                "violations": [list(pair) for pair in violations],
            },
        )


class StateMonotonicEvaluator(Evaluator):
    """Passes when a numeric state path never reverses direction.

    Reads the snapshot timeline and checks consecutive values. Pick the path by
    business meaning: a cart ``total`` is *not* monotonic (a coupon legitimately
    lowers it), while counters such as ``processed`` or ``retries`` are.

    Args:
        path: dotted path to a numeric value, e.g. ``"processed"``.
        direction: ``"non_decreasing"`` (default), ``"non_increasing"``,
            ``"strictly_increasing"`` or ``"strictly_decreasing"``.
        tolerance: values within this absolute tolerance count as equal, which
            absorbs float rounding (``total * 0.9`` is not exactly
            representable). Required for money-like values.

    A missing path, or one whose value is not a number, is reported as ERROR
    rather than silently passing or failing.
    """

    def __init__(
        self,
        path: str,
        direction: MonotonicDirection = "non_decreasing",
        *,
        tolerance: float = 0.0,
    ) -> None:
        if direction not in _MONOTONIC_DIRECTIONS:
            raise ValueError(f"direction must be one of {_MONOTONIC_DIRECTIONS}, got {direction!r}")
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self._path = path
        self._direction = direction
        self._tolerance = tolerance

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        snapshots = trace.state_snapshots()
        if not snapshots:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.ERROR,
                score=0.0,
                reason="no state snapshots recorded in trace",
                metadata={"path": self._path},
            )

        values: list[float] = []
        for index, snapshot in enumerate(snapshots):
            value = resolve_path(snapshot.state, self._path)
            if value is UNSET:
                return EvaluationResult(
                    evaluator=type(self).__name__,
                    verdict=Verdict.ERROR,
                    score=0.0,
                    reason=f"state {self._path!r} missing in snapshot #{index}",
                    metadata={"path": self._path, "snapshot": index},
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return EvaluationResult(
                    evaluator=type(self).__name__,
                    verdict=Verdict.ERROR,
                    score=0.0,
                    reason=f"state {self._path!r} is not numeric in snapshot "
                    f"#{index} (got {type(value).__name__})",
                    metadata={"path": self._path, "snapshot": index},
                )
            values.append(float(value))

        violations: list[dict[str, Any]] = []
        for index, (previous, current) in enumerate(zip(values, values[1:], strict=False), start=1):
            if self._direction == "non_decreasing":
                ok = current >= previous - self._tolerance
            elif self._direction == "non_increasing":
                ok = current <= previous + self._tolerance
            elif self._direction == "strictly_increasing":
                ok = current > previous + self._tolerance
            else:
                ok = current < previous - self._tolerance
            if not ok:
                violations.append({"snapshot": index, "from": previous, "to": current})

        passed = not violations
        reason = (
            f"{self._path!r} is {self._direction} across {len(values)} snapshot(s)"
            if passed
            else f"{self._path!r} violates {self._direction}: {violations}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={
                "path": self._path,
                "direction": self._direction,
                "tolerance": self._tolerance,
                "values": values,
                "violations": violations,
            },
        )


class StateUnchangedEvaluator(Evaluator):
    """Passes when a state path has the same value in every snapshot.

    Reads the snapshot timeline rather than state-change deltas, because
    "nothing changed" produces no delta event and is therefore invisible on
    that channel.

    This is stronger than forbidding an action: an agent can often reach a
    forbidden state through an *allowed* endpoint (e.g. lowering a price by
    applying a coupon instead of calling the admin endpoint), and asserting the
    state never changed does not depend on which endpoint was used.

    Args:
        path: dotted path, e.g. ``"owner"`` or ``"unit_price"``.
        allow_missing: when ``True``, a snapshot that lacks ``path`` is skipped;
            by default that is an ERROR, because the history cannot be checked
            completely.
    """

    def __init__(self, path: str, *, allow_missing: bool = False) -> None:
        self._path = path
        self._allow_missing = allow_missing

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        snapshots = trace.state_snapshots()
        if not snapshots:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.ERROR,
                score=0.0,
                reason="no state snapshots recorded in trace",
                metadata={"path": self._path},
            )

        observed: list[tuple[int, Any]] = []
        for index, snapshot in enumerate(snapshots):
            value = resolve_path(snapshot.state, self._path)
            if value is UNSET:
                if self._allow_missing:
                    continue
                return EvaluationResult(
                    evaluator=type(self).__name__,
                    verdict=Verdict.ERROR,
                    score=0.0,
                    reason=f"state {self._path!r} missing in snapshot #{index}",
                    metadata={"path": self._path, "snapshot": index},
                )
            observed.append((index, value))

        if len(observed) < 2:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.ERROR,
                score=0.0,
                reason=f"need at least 2 snapshots containing {self._path!r}, got {len(observed)}",
                metadata={"path": self._path, "observed": len(observed)},
            )

        first_index, first = observed[0]
        changed = [
            {"snapshot": index, "value": value} for index, value in observed[1:] if value != first
        ]

        passed = not changed
        reason = (
            f"{self._path!r} kept the value {first!r} across {len(observed)} snapshot(s)"
            if passed
            else f"{self._path!r} changed from {first!r} (snapshot #{first_index}): {changed}"
        )
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 if passed else 0.0,
            reason=reason,
            metadata={
                "path": self._path,
                "value": first,
                "observed": len(observed),
                "changes": changed,
            },
        )
