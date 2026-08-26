"""Tests for AgentAdapter and CallableAgentAdapter."""

from agentmeter import (
    AgentAdapter,
    AgentMessageEvent,
    CallableAgentAdapter,
    Trace,
    UserMessageEvent,
)


async def test_callable_adapter_runs_async_callable():
    async def fake_agent(message: str) -> str:
        return f"echo: {message}"

    adapter = CallableAgentAdapter(fake_agent)
    trace = await adapter.run("ping")

    assert isinstance(trace, Trace)
    assert trace.input == "ping"
    assert trace.final_output == "echo: ping"
    assert len(trace.events) == 2
    assert isinstance(trace.events[0], UserMessageEvent)
    assert trace.events[0].content == "ping"
    assert isinstance(trace.events[1], AgentMessageEvent)
    assert trace.events[1].content == "echo: ping"


async def test_callable_adapter_runs_sync_callable():
    adapter = CallableAgentAdapter(lambda message: f"sync:{message}")
    trace = await adapter.run("hello")
    assert trace.final_output == "sync:hello"


async def test_callable_adapter_is_an_agent_adapter():
    adapter = CallableAgentAdapter(lambda message: message)
    assert isinstance(adapter, AgentAdapter)


def test_agent_adapter_is_abstract():
    assert AgentAdapter.__abstractmethods__ == {"run"}
