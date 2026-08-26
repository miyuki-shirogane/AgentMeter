"""Tests for Trace and TraceEvent."""

import pytest

from agentmeter import (
    AgentMessageEvent,
    CustomEvent,
    ToolCallEvent,
    ToolResultEvent,
    Trace,
    UserMessageEvent,
)


def test_trace_event_types():
    events = [
        UserMessageEvent(content="hi"),
        AgentMessageEvent(content="hello"),
        ToolCallEvent(name="get_weather", arguments={"city": "beijing"}),
        ToolResultEvent(call_id="c1", name="get_weather", result={"temp": 25}),
        CustomEvent(name="thinking", data={"step": 1}),
    ]
    for event in events:
        assert event.type == type(event).model_fields["type"].default


def test_trace_discriminates_union():
    raw = {"type": "agent_message", "content": "world"}
    event = Trace.model_validate(
        {
            "input": "ping",
            "events": [raw],
            "final_output": "world",
        }
    ).events[0]
    assert isinstance(event, AgentMessageEvent)


def test_trace_records_events_and_final_output():
    trace = Trace(input="你好")
    trace.add_event(UserMessageEvent(content="你好"))
    trace.add_event(AgentMessageEvent(content="你好，北京！"))
    trace.final_output = "你好，北京！"

    assert trace.input == "你好"
    assert trace.final_output == "你好，北京！"
    assert len(trace.events) == 2
    assert isinstance(trace.events[0], UserMessageEvent)
    assert isinstance(trace.events[1], AgentMessageEvent)


def test_trace_id_is_unique():
    assert Trace(input="a").trace_id != Trace(input="a").trace_id


def test_trace_serializes_to_json():
    trace = Trace(input="hi", final_output="hello")
    trace.add_event(AgentMessageEvent(content="hello"))
    data = trace.model_dump_json()
    assert '"user_message"' in data or '"agent_message"' in data
    restored = Trace.model_validate_json(data)
    assert restored.input == "hi"
    assert restored.final_output == "hello"


def test_trace_metadata_defaults_to_empty_dict():
    assert Trace(input="x").metadata == {}


def test_tool_events_are_reserved_but_valid():
    call = ToolCallEvent(name="search", arguments={"q": "ai"})
    trace = Trace(input="query")
    trace.add_event(call)
    trace.add_event(ToolResultEvent(call_id=call.call_id, name="search", result="no results"))
    assert trace.events[0].type == "tool_call"
    assert trace.events[1].type == "tool_result"
    assert trace.events[1].result == "no results"
    assert trace.events[1].call_id == call.call_id


def test_trace_event_type_is_literal():
    with pytest.raises(ValueError):
        Trace.model_validate(
            {
                "input": "x",
                "events": [{"type": "unknown", "content": "boom"}],
            }
        )


def test_tool_call_ids_are_unique():
    assert ToolCallEvent(name="search").call_id != ToolCallEvent(name="search").call_id


def test_tool_result_requires_call_id():
    with pytest.raises(ValueError):
        ToolResultEvent(name="search", result="x")


def test_result_pairs_with_matching_call():
    trace = Trace(input="q")
    call = ToolCallEvent(name="search", arguments={"query": "北京"})
    trace.add_event(call)
    trace.add_event(ToolResultEvent(call_id=call.call_id, name="search", result={"hits": 3}))

    assert trace.result_for_call(call.call_id).result == {"hits": 3}


def test_result_does_not_pair_with_unrelated_call():
    trace = Trace(input="q")
    trace.add_event(ToolCallEvent(name="search"))
    trace.add_event(ToolResultEvent(call_id="other-id", name="search", result="x"))
    assert trace.result_for_call("other-id") is not None
    assert trace.result_for_call("missing-id") is None


def test_same_tool_multiple_calls_pair_independently():
    trace = Trace(input="q")
    first = ToolCallEvent(name="search", arguments={"query": "北京"})
    second = ToolCallEvent(name="search", arguments={"query": "上海"})
    trace.add_event(first)
    trace.add_event(second)
    trace.add_event(ToolResultEvent(call_id=first.call_id, name="search", result="beijing"))
    trace.add_event(ToolResultEvent(call_id=second.call_id, name="search", result="shanghai"))

    assert trace.result_for_call(first.call_id).result == "beijing"
    assert trace.result_for_call(second.call_id).result == "shanghai"


def test_interleaved_results_pair_correctly():
    trace = Trace(input="q")
    search = ToolCallEvent(name="search")
    calc = ToolCallEvent(name="calculate")
    trace.add_event(search)
    trace.add_event(calc)
    trace.add_event(ToolResultEvent(call_id=calc.call_id, name="calculate", result=42))
    trace.add_event(ToolResultEvent(call_id=search.call_id, name="search", result="docs"))

    assert trace.result_for_call(search.call_id).result == "docs"
    assert trace.result_for_call(calc.call_id).result == 42


def test_tool_call_query_helpers():
    trace = Trace(input="q")
    trace.add_event(ToolCallEvent(name="search"))
    trace.add_event(ToolCallEvent(name="search"))
    trace.add_event(ToolCallEvent(name="calculate"))

    assert trace.tool_call_names() == ["search", "search", "calculate"]
    assert len(trace.tool_calls()) == 3
    assert len(trace.tool_results()) == 0
