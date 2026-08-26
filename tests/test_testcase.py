"""Tests for TestCase."""

from agentmeter import (
    CallableAgentAdapter,
    OutputContainsEvaluator,
)
from agentmeter import (
    TestCase as AgentTestCase,
)


def test_testcase_wraps_callable_into_adapter():
    async def fake_agent(message: str) -> str:
        return message

    testcase = AgentTestCase(
        name="hello_test",
        input="hi",
        agent=fake_agent,
        evaluators=[OutputContainsEvaluator("hi")],
    )
    assert isinstance(testcase.agent, CallableAgentAdapter)
    assert testcase.name == "hello_test"
    assert testcase.input == "hi"


def test_testcase_accepts_existing_adapter():
    adapter = CallableAgentAdapter(lambda message: message)
    testcase = AgentTestCase(
        name="t",
        input="hi",
        agent=adapter,
        evaluators=[OutputContainsEvaluator("hi")],
    )
    assert testcase.agent is adapter


def test_testcase_metadata_defaults():
    testcase = AgentTestCase(
        name="t",
        input="hi",
        agent=lambda message: message,
        evaluators=[],
    )
    assert testcase.metadata == {}
