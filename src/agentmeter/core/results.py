"""Evaluation result models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentmeter.core.trace import Trace
from agentmeter.core.verdict import Score, Verdict


class EvaluationResult(BaseModel):
    """The outcome of a single evaluator applied to a trace.

    ``verdict`` is the source of truth (PASS / FAIL / ERROR); ``passed`` is
    a derived convenience property. ``score`` is always within [0.0, 1.0]
    and is independent from ``passed``.
    """

    evaluator: str
    verdict: Verdict
    score: Score
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS


class TestRunResult(BaseModel):
    """The outcome of a single agent run.

    This model represents exactly one execution. Statistical aggregation
    over many runs lives in :class:`AggregateResult`.
    """

    testcase_name: str
    trace: Trace | None = None
    results: list[EvaluationResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def verdict(self) -> Verdict:
        """Overall verdict: any ERROR -> ERROR, else any FAIL -> FAIL, else PASS."""
        if any(result.verdict == Verdict.ERROR for result in self.results):
            return Verdict.ERROR
        if any(result.verdict == Verdict.FAIL for result in self.results):
            return Verdict.FAIL
        return Verdict.PASS

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS

    @property
    def error(self) -> bool:
        return self.verdict == Verdict.ERROR

    @property
    def score(self) -> Score:
        """Average score across all evaluators (0.0 when none ran)."""
        if not self.results:
            return 0.0
        return sum(result.score for result in self.results) / len(self.results)


class AggregateResult(BaseModel):
    """Statistical summary of many runs of the same test case.

    ``pass_rate`` counts only PASS runs; ERROR runs are never counted as
    passes and are surfaced separately via ``error_rate``.
    """

    testcase_name: str
    total_runs: int = Field(ge=0)
    passed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    error_runs: int = Field(ge=0)
    pass_rate: Score
    error_rate: Score
    average_score: Score
    required_pass_rate: Score | None = None
    verdict: Verdict
    reason: str = ""

    @classmethod
    def from_runs(
        cls,
        testcase_name: str,
        results: list[TestRunResult],
        *,
        required_pass_rate: float | None = None,
    ) -> AggregateResult:
        """Aggregate a list of per-run results into summary statistics."""
        total_runs = len(results)
        passed_runs = sum(1 for result in results if result.verdict == Verdict.PASS)
        failed_runs = sum(1 for result in results if result.verdict == Verdict.FAIL)
        error_runs = sum(1 for result in results if result.verdict == Verdict.ERROR)

        pass_rate = passed_runs / total_runs if total_runs else 0.0
        error_rate = error_runs / total_runs if total_runs else 0.0
        average_score = sum(result.score for result in results) / total_runs if total_runs else 0.0

        if total_runs == 0:
            verdict = Verdict.ERROR
        elif error_runs == total_runs:
            verdict = Verdict.ERROR
        elif required_pass_rate is not None:
            verdict = Verdict.PASS if pass_rate >= required_pass_rate else Verdict.FAIL
        else:
            verdict = Verdict.PASS if failed_runs == 0 and error_runs == 0 else Verdict.FAIL

        required_txt = (
            f", required>={required_pass_rate:.2f}" if required_pass_rate is not None else ""
        )
        reason = (
            f"{passed_runs}/{total_runs} passed (pass_rate={pass_rate:.2f}{required_txt}), "
            f"{failed_runs} failed, {error_runs} errored (error_rate={error_rate:.2f}), "
            f"average_score={average_score:.3f}"
        )

        return cls(
            testcase_name=testcase_name,
            total_runs=total_runs,
            passed_runs=passed_runs,
            failed_runs=failed_runs,
            error_runs=error_runs,
            pass_rate=pass_rate,
            error_rate=error_rate,
            average_score=average_score,
            required_pass_rate=required_pass_rate,
            verdict=verdict,
            reason=reason,
        )

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS
