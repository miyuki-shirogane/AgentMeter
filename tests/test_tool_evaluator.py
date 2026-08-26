"""Tests for tool-call and tool-argument evaluators."""

from agentmeter import (
    ToolArgumentEvaluator,
    ToolCallCountEvaluator,
    ToolCallCountMode,
    ToolCalledEvaluator,
    ToolCallEvent,
    ToolNotCalledEvaluator,
    ToolResultEvent,
    Trace,
    Verdict,
)


def _trace_with_calls(calls) -> Trace:
    trace = Trace(input="in")
    for name, arguments in calls:
        call = ToolCallEvent(name=name, arguments=arguments)
        trace.add_event(call)
        trace.add_event(ToolResultEvent(call_id=call.call_id, name=name, result={}))
    return trace


async def test_tool_called_passes():
    trace = _trace_with_calls([("search", {"query": "Python"})])
    result = await ToolCalledEvaluator("search").evaluate(trace)
    assert result.passed is True
    assert result.verdict == Verdict.PASS


async def test_tool_called_fails_when_never_called():
    trace = _trace_with_calls([("calculate", {})])
    result = await ToolCalledEvaluator("search").evaluate(trace)
    assert result.passed is False
    assert result.verdict == Verdict.FAIL
    assert "never called" in result.reason


async def test_tool_not_called_passes():
    trace = _trace_with_calls([("search", {})])
    result = await ToolNotCalledEvaluator("delete_account").evaluate(trace)
    assert result.passed is True


async def test_tool_not_called_fails_when_called():
    trace = _trace_with_calls([("delete_account", {"id": 1})])
    result = await ToolNotCalledEvaluator("delete_account").evaluate(trace)
    assert result.passed is False
    assert result.verdict == Verdict.FAIL
    assert "delete_account" in result.reason


async def test_tool_call_count_exactly():
    trace = _trace_with_calls([("search", {}), ("search", {})])
    assert (await ToolCallCountEvaluator("search", 2).evaluate(trace)).passed is True
    assert (await ToolCallCountEvaluator("search", 1).evaluate(trace)).passed is False


async def test_tool_call_count_at_least():
    trace = _trace_with_calls([("search", {}), ("search", {})])
    eval_ = ToolCallCountEvaluator("search", 1, mode=ToolCallCountMode.AT_LEAST)
    assert (await eval_.evaluate(trace)).passed is True
    eval_ = ToolCallCountEvaluator("search", 3, mode=ToolCallCountMode.AT_LEAST)
    assert (await eval_.evaluate(trace)).passed is False


async def test_tool_call_count_at_most():
    trace = _trace_with_calls([("search", {}), ("search", {}), ("search", {})])
    eval_ = ToolCallCountEvaluator("search", 3, mode=ToolCallCountMode.AT_MOST)
    assert (await eval_.evaluate(trace)).passed is True
    eval_ = ToolCallCountEvaluator("search", 2, mode=ToolCallCountMode.AT_MOST)
    assert (await eval_.evaluate(trace)).passed is False


async def test_tool_call_count_ignores_other_tools():
    trace = _trace_with_calls([("search", {}), ("calculate", {})])
    result = await ToolCallCountEvaluator("search", 1).evaluate(trace)
    assert result.passed is True


async def test_tool_argument_exact_match():
    trace = _trace_with_calls([("search", {"query": "Python", "lang": "zh"})])
    result = await ToolArgumentEvaluator(
        "search", expected={"query": "Python", "lang": "zh"}
    ).evaluate(trace)
    assert result.passed is True


async def test_tool_argument_exact_mismatch():
    trace = _trace_with_calls([("search", {"query": "Java", "lang": "zh"})])
    result = await ToolArgumentEvaluator("search", expected={"query": "Python"}).evaluate(trace)
    assert result.passed is False
    assert result.verdict == Verdict.FAIL


async def test_tool_argument_field_match():
    trace = _trace_with_calls([("search", {"query": "Python", "lang": "zh"})])
    result = await ToolArgumentEvaluator("search", expected="Python", field="query").evaluate(trace)
    assert result.passed is True


async def test_tool_argument_field_mismatch():
    trace = _trace_with_calls([("search", {"query": "Java"})])
    result = await ToolArgumentEvaluator("search", expected="Python", field="query").evaluate(trace)
    assert result.passed is False


async def test_tool_argument_nested_field():
    trace = _trace_with_calls([("search", {"options": {"language": "zh"}})])
    result = await ToolArgumentEvaluator(
        "search", expected="zh", field="options.language"
    ).evaluate(trace)
    assert result.passed is True


async def test_tool_argument_nested_field_mismatch():
    trace = _trace_with_calls([("search", {"options": {"language": "en"}})])
    result = await ToolArgumentEvaluator(
        "search", expected="zh", field="options.language"
    ).evaluate(trace)
    assert result.passed is False


async def test_tool_argument_missing_field_fails():
    trace = _trace_with_calls([("search", {"query": "Python"})])
    result = await ToolArgumentEvaluator(
        "search", expected="zh", field="options.language"
    ).evaluate(trace)
    assert result.passed is False


async def test_tool_argument_requires_at_least_one_matching_call():
    trace = _trace_with_calls([("search", {"query": "Java"}), ("search", {"query": "Python"})])
    result = await ToolArgumentEvaluator("search", expected="Python", field="query").evaluate(trace)
    assert result.passed is True


async def test_tool_argument_when_tool_never_called():
    trace = _trace_with_calls([("other", {})])
    result = await ToolArgumentEvaluator("search", expected="Python", field="query").evaluate(trace)
    assert result.passed is False
    assert "never called" in result.reason
