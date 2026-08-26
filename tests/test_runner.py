"""Tests for Runner."""

from agentmeter import (
    OutputContainsEvaluator,
    OutputEqualsEvaluator,
    Runner,
    Verdict,
)
from agentmeter import (
    TestCase as AgentTestCase,
)
from agentmeter import (
    TestRunResult as AgentTestRunResult,
)


async def test_runner_returns_test_run_result():
    async def fake_agent(message: str) -> str:
        return "你好，北京！"

    testcase = AgentTestCase(
        name="hello_test",
        input="请向北京打招呼",
        agent=fake_agent,
        evaluators=[OutputContainsEvaluator("北京")],
    )

    result = await Runner().run(testcase)

    assert isinstance(result, AgentTestRunResult)
    assert result.testcase_name == "hello_test"
    assert result.passed is True
    assert result.score == 1.0
    assert len(result.results) == 1
    assert result.trace.final_output == "你好，北京！"


async def test_runner_marks_failure_when_any_evaluator_fails():
    async def fake_agent(message: str) -> str:
        return "你好，北京！"

    testcase = AgentTestCase(
        name="failing_test",
        input="hi",
        agent=fake_agent,
        evaluators=[
            OutputContainsEvaluator("北京"),
            OutputEqualsEvaluator("hello"),
        ],
    )

    result = await Runner().run(testcase)

    assert result.passed is False
    assert result.score == 0.5
    assert [r.passed for r in result.results] == [True, False]


async def test_runner_runs_async_agent():
    async def fake_agent(message: str) -> str:
        return "ok"

    testcase = AgentTestCase(
        name="async_agent",
        input="hi",
        agent=fake_agent,
        evaluators=[OutputEqualsEvaluator("ok")],
    )

    result = await Runner().run(testcase)

    assert result.passed is True


async def test_runner_agent_exception_becomes_error():
    async def broken_agent(message: str) -> str:
        raise RuntimeError("agent exploded")

    testcase = AgentTestCase(
        name="broken",
        input="hi",
        agent=broken_agent,
        evaluators=[OutputContainsEvaluator("x")],
    )

    result = await Runner().run(testcase)

    assert result.verdict == Verdict.ERROR
    assert result.passed is False
    assert result.trace is None
    assert result.results[0].evaluator == "agent"
    assert result.results[0].verdict == Verdict.ERROR
    assert "agent exploded" in result.results[0].reason


async def test_runner_evaluator_exception_becomes_error():
    async def fake_agent(message: str) -> str:
        return "ok"

    class ExplodingEvaluator(OutputContainsEvaluator):
        async def evaluate(self, trace):
            raise ValueError("bad evaluator")

    testcase = AgentTestCase(
        name="evaluator_broken",
        input="hi",
        agent=fake_agent,
        evaluators=[ExplodingEvaluator("ok")],
    )

    result = await Runner().run(testcase)

    assert result.verdict == Verdict.ERROR
    assert result.trace is not None
    assert result.results[0].verdict == Verdict.ERROR
    assert "bad evaluator" in result.results[0].reason


async def test_runner_mixed_verdicts():
    async def fake_agent(message: str) -> str:
        return "你好，北京！"

    class ExplodingEvaluator(OutputContainsEvaluator):
        async def evaluate(self, trace):
            raise ValueError("boom")

    testcase = AgentTestCase(
        name="mixed",
        input="hi",
        agent=fake_agent,
        evaluators=[
            OutputContainsEvaluator("北京"),
            ExplodingEvaluator("北京"),
        ],
    )

    result = await Runner().run(testcase)

    assert result.verdict == Verdict.ERROR
    assert result.results[0].passed is True
    assert result.results[1].verdict == Verdict.ERROR
