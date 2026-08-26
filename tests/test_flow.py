"""End-to-end test: the complete fake agent flow."""

from agentmeter import (
    AgentMessageEvent,
    OutputContainsEvaluator,
    Runner,
    UserMessageEvent,
)
from agentmeter import (
    TestCase as AgentTestCase,
)


async def test_full_fake_agent_flow():
    async def fake_agent(message: str) -> str:
        assert message == "请向北京打招呼"
        return "你好，北京！"

    testcase = AgentTestCase(
        name="hello_test",
        input="请向北京打招呼",
        agent=fake_agent,
        evaluators=[OutputContainsEvaluator("北京")],
    )

    result = await Runner().run(testcase)

    assert result.passed
    assert result.trace.input == "请向北京打招呼"
    assert result.trace.final_output == "你好，北京！"
    assert len(result.trace.events) == 2
    assert isinstance(result.trace.events[0], UserMessageEvent)
    assert isinstance(result.trace.events[1], AgentMessageEvent)
    assert result.trace.events[1].content == "你好，北京！"
    assert result.results[0].passed is True
