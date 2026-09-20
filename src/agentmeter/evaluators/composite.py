"""Composite evaluators.

These do not inspect the trace themselves — they combine the results of other
evaluators, so the same checks can be negated, AND-ed or OR-ed without adding
a bespoke class per combination.

Verdicts follow the framework-wide rule that ``ERROR`` dominates ``FAIL``
dominates ``PASS``:

- :class:`NotEvaluator` inverts PASS and FAIL; ERROR stays ERROR (if the inner
  check could not be determined, neither can its negation).
- :class:`AllOfEvaluator` passes only when *every* child passes.
- :class:`AnyOfEvaluator` passes when *at least one* child passes; a satisfied
  alternative settles the outcome, so it never downgrades to ERROR.

``score`` is informational and never drives the verdict: ``AllOf`` reports the
minimum child score, ``AnyOf`` the maximum, and ``Not`` reports ``1 - score``.
"""

from __future__ import annotations

from agentmeter.core.results import EvaluationResult
from agentmeter.core.trace import Trace
from agentmeter.core.verdict import Verdict
from agentmeter.evaluators.base import Evaluator


def _detail(results: list[EvaluationResult]) -> str:
    """A compact ``Name=VERDICT`` summary of child results, for ``reason``."""
    return ", ".join(f"{result.evaluator}={result.verdict.value}" for result in results)


def _summarize(results: list[EvaluationResult]) -> list[dict[str, object]]:
    """A compact, JSON-serializable summary of child results, for ``metadata``."""
    return [
        {"evaluator": result.evaluator, "verdict": result.verdict.value, "score": result.score}
        for result in results
    ]


class NotEvaluator(Evaluator):
    """Inverts the verdict of a single evaluator.

    PASS becomes FAIL and FAIL becomes PASS. An ERROR is propagated unchanged:
    when the inner check could not be determined, neither can its negation.

    Args:
        inner: the evaluator to negate.

    Example:
        ``NotEvaluator(OutputContainsEvaluator("抱歉"))`` passes when the agent
        does *not* apologize.
    """

    def __init__(self, inner: Evaluator) -> None:
        self._inner = inner

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        result = await self._inner.evaluate(trace)
        metadata = {"inner": result.evaluator, "inner_verdict": result.verdict.value}

        if result.verdict == Verdict.ERROR:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.ERROR,
                score=0.0,
                reason=f"not({result.evaluator}) is undetermined: {result.reason}",
                metadata=metadata,
            )

        passed = result.verdict == Verdict.FAIL
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=1.0 - result.score,
            reason=f"not({result.evaluator}) -> {result.verdict.value}",
            metadata=metadata,
        )


class AllOfEvaluator(Evaluator):
    """Passes only when every child evaluator passes.

    Args:
        evaluators: the child evaluators; must be non-empty.

    An ERROR in any child makes the composite ERROR (ERROR dominates FAIL,
    matching ``TestRunResult.verdict``).
    """

    def __init__(self, evaluators: list[Evaluator]) -> None:
        if not evaluators:
            raise ValueError("AllOfEvaluator requires at least one evaluator")
        self._evaluators = list(evaluators)

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        results = [await evaluator.evaluate(trace) for evaluator in self._evaluators]

        if any(result.verdict == Verdict.ERROR for result in results):
            verdict = Verdict.ERROR
        elif any(result.verdict == Verdict.FAIL for result in results):
            verdict = Verdict.FAIL
        else:
            verdict = Verdict.PASS

        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=verdict,
            score=min(result.score for result in results),
            reason=f"all of {len(results)} -> {verdict.value} [{_detail(results)}]",
            metadata={"children": _summarize(results)},
        )


class AnyOfEvaluator(Evaluator):
    """Passes when at least one child evaluator passes.

    Args:
        evaluators: the child evaluators; must be non-empty.

    A satisfied alternative settles the outcome as PASS. Only when no child
    passes does an ERROR in any child make the composite ERROR; otherwise it
    is FAIL.
    """

    def __init__(self, evaluators: list[Evaluator]) -> None:
        if not evaluators:
            raise ValueError("AnyOfEvaluator requires at least one evaluator")
        self._evaluators = list(evaluators)

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        results = [await evaluator.evaluate(trace) for evaluator in self._evaluators]

        if any(result.verdict == Verdict.PASS for result in results):
            verdict = Verdict.PASS
        elif any(result.verdict == Verdict.ERROR for result in results):
            verdict = Verdict.ERROR
        else:
            verdict = Verdict.FAIL

        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=verdict,
            score=max(result.score for result in results),
            reason=f"any of {len(results)} -> {verdict.value} [{_detail(results)}]",
            metadata={"children": _summarize(results)},
        )
