"""Tests for result semantics: PASS / FAIL / ERROR, score, aggregation."""

import pytest

from agentmeter import (
    AggregateResult,
    EvaluationResult,
    Trace,
    Verdict,
)
from agentmeter import (
    TestRunResult as AgentTestRunResult,
)


def _result(verdict: Verdict, score: float = 1.0) -> EvaluationResult:
    return EvaluationResult(evaluator="e", verdict=verdict, score=score, reason="")


def _run(*results: EvaluationResult) -> AgentTestRunResult:
    return AgentTestRunResult(testcase_name="t", trace=Trace(input="in"), results=list(results))


def test_evaluation_result_verdict():
    assert _result(Verdict.PASS).passed is True
    assert _result(Verdict.FAIL).passed is False
    assert _result(Verdict.ERROR).passed is False


def test_test_run_result_verdict_all_pass():
    assert _run(_result(Verdict.PASS)).verdict == Verdict.PASS
    assert _run(_result(Verdict.PASS)).passed is True


def test_test_run_result_verdict_any_fail():
    run = _run(_result(Verdict.PASS), _result(Verdict.FAIL))
    assert run.verdict == Verdict.FAIL
    assert run.passed is False


def test_test_run_result_verdict_any_error():
    run = _run(_result(Verdict.PASS), _result(Verdict.ERROR))
    assert run.verdict == Verdict.ERROR
    assert run.error is True


def test_error_not_silently_converted_to_fail():
    run = _run(_result(Verdict.ERROR), _result(Verdict.FAIL))
    assert run.verdict == Verdict.ERROR


def test_score_and_passed_are_independent():
    result = EvaluationResult(evaluator="e", verdict=Verdict.FAIL, score=0.72, reason="")
    assert result.passed is False
    assert result.score == 0.72


def test_score_validation():
    with pytest.raises(ValueError):
        EvaluationResult(evaluator="e", verdict=Verdict.PASS, score=1.1, reason="")
    with pytest.raises(ValueError):
        AggregateResult(
            testcase_name="t",
            total_runs=1,
            passed_runs=1,
            failed_runs=0,
            error_runs=0,
            pass_rate=1.5,
            error_rate=0.0,
            average_score=1.0,
            verdict=Verdict.PASS,
        )


def test_test_run_result_score_is_average():
    run = _run(
        _result(Verdict.PASS, score=1.0),
        _result(Verdict.FAIL, score=0.0),
    )
    assert run.score == 0.5


def test_aggregate_result_all_pass():
    agg = AggregateResult.from_runs("t", [_run(_result(Verdict.PASS)) for _ in range(10)])
    assert agg.total_runs == 10
    assert agg.passed_runs == 10
    assert agg.failed_runs == 0
    assert agg.error_runs == 0
    assert agg.pass_rate == 1.0
    assert agg.error_rate == 0.0
    assert agg.verdict == Verdict.PASS
    assert agg.passed is True


def test_aggregate_result_partial_failure():
    runs = [_run(_result(Verdict.PASS)) for _ in range(8)]
    runs += [_run(_result(Verdict.FAIL)), _run(_result(Verdict.ERROR))]
    agg = AggregateResult.from_runs("t", runs)

    assert agg.passed_runs == 8
    assert agg.failed_runs == 1
    assert agg.error_runs == 1
    assert agg.pass_rate == 0.8
    assert agg.error_rate == 0.1
    assert agg.verdict == Verdict.FAIL
    assert "8/10 passed" in agg.reason
    assert "1 errored" in agg.reason


def test_aggregate_error_never_counts_as_pass():
    runs = [_run(_result(Verdict.ERROR)) for _ in range(5)]
    agg = AggregateResult.from_runs("t", runs)
    assert agg.passed_runs == 0
    assert agg.error_runs == 5
    assert agg.pass_rate == 0.0
    assert agg.verdict == Verdict.ERROR


def test_aggregate_average_score():
    runs = [_run(_result(Verdict.PASS, score=1.0)) for _ in range(2)]
    runs += [_run(_result(Verdict.FAIL, score=0.0))]
    agg = AggregateResult.from_runs("t", runs)
    assert agg.average_score == pytest.approx(2 / 3)


def test_aggregate_threshold_pass_and_fail():
    passing = [_run(_result(Verdict.PASS)) for _ in range(18)]
    failing = [_run(_result(Verdict.FAIL)) for _ in range(2)]

    twenty = passing + failing
    assert AggregateResult.from_runs("t", twenty, required_pass_rate=0.9).passed is True

    seventeen = passing[:17] + failing
    assert AggregateResult.from_runs("t", seventeen, required_pass_rate=0.9).passed is False
