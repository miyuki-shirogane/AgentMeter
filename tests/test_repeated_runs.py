"""Tests for repeated runs, pass rate, threshold, and error handling."""

from agentmeter import (
    AggregateResult,
    OutputEqualsEvaluator,
    Runner,
    Verdict,
)
from agentmeter import (
    TestCase as AgentTestCase,
)


def _testcase(agent, evaluators=None) -> AgentTestCase:
    return AgentTestCase(
        name="repeated",
        input="hi",
        agent=agent,
        evaluators=evaluators or [OutputEqualsEvaluator("ok")],
    )


async def test_repeated_runs_all_pass():
    async def agent(message: str) -> str:
        return "ok"

    agg = await Runner().run_many(_testcase(agent), runs=10)

    assert isinstance(agg, AggregateResult)
    assert agg.total_runs == 10
    assert agg.passed_runs == 10
    assert agg.pass_rate == 1.0
    assert agg.verdict == Verdict.PASS


async def test_repeated_runs_partial_failure():
    counter = 0

    async def agent(message: str) -> str:
        nonlocal counter
        counter += 1
        return "bad" if counter == 3 else "ok"

    agg = await Runner().run_many(_testcase(agent), runs=10)

    assert agg.passed_runs == 9
    assert agg.failed_runs == 1
    assert agg.error_runs == 0
    assert agg.pass_rate == 0.9
    assert agg.verdict == Verdict.FAIL


async def test_repeated_runs_all_failure():
    async def agent(message: str) -> str:
        return "bad"

    agg = await Runner().run_many(_testcase(agent), runs=5)

    assert agg.passed_runs == 0
    assert agg.failed_runs == 5
    assert agg.pass_rate == 0.0
    assert agg.verdict == Verdict.FAIL


async def test_repeated_runs_agent_error_not_counted_as_pass():
    counter = 0

    async def agent(message: str) -> str:
        nonlocal counter
        counter += 1
        if counter == 4:
            raise RuntimeError("transient failure")
        return "ok"

    agg = await Runner().run_many(_testcase(agent), runs=10)

    assert agg.passed_runs == 9
    assert agg.error_runs == 1
    assert agg.failed_runs == 0
    assert agg.pass_rate == 0.9
    assert agg.error_rate == 0.1
    assert "1 errored" in agg.reason


async def test_repeated_runs_agent_error_not_hidden_by_threshold():
    async def agent(message: str) -> str:
        raise RuntimeError("always broken")

    agg = await Runner().run_many(_testcase(agent), runs=5)

    assert agg.passed_runs == 0
    assert agg.error_runs == 5
    assert agg.error_rate == 1.0
    assert agg.verdict == Verdict.ERROR
    assert "always broken" in agg.reason or agg.reason


async def test_repeated_runs_threshold():
    def flaky() -> AgentTestCase:
        counter = 0

        async def agent(message: str) -> str:
            nonlocal counter
            counter += 1
            return "bad" if counter in (4, 5) else "ok"

        return _testcase(agent)

    agg = await Runner().run_many(flaky(), runs=20, required_pass_rate=0.9)
    assert agg.pass_rate == 0.9
    assert agg.passed is True

    agg = await Runner().run_many(flaky(), runs=20, required_pass_rate=0.95)
    assert agg.pass_rate == 0.9
    assert agg.passed is False


async def test_repeated_runs_isolated_between_runs():
    seen = []

    async def agent(message: str) -> str:
        nonlocal seen
        seen.append(len(seen))
        return "ok"

    agg = await Runner().run_many(_testcase(agent), runs=3)

    assert seen == [0, 1, 2]
    assert agg.total_runs == 3
    assert agg.passed_runs == 3


async def test_repeated_runs_average_score():
    counter = 0

    async def agent(message: str) -> str:
        nonlocal counter
        counter += 1
        return "bad" if counter % 2 == 0 else "ok"

    agg = await Runner().run_many(_testcase(agent), runs=4)

    assert agg.average_score == 0.5
