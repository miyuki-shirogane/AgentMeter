"""Tests for trajectory evaluators (order, required, forbidden, limits)."""

from agentmeter import (
    ForbiddenToolEvaluator,
    MaximumToolCallsEvaluator,
    RequiredToolEvaluator,
    ToolCallEvent,
    ToolOrderEvaluator,
    Trace,
    Verdict,
)


def _trace(names: list[str]) -> Trace:
    trace = Trace(input="in")
    for name in names:
        trace.add_event(ToolCallEvent(name=name))
    return trace


async def test_tool_order_correct():
    trace = _trace(["search", "calculate", "answer"])
    result = await ToolOrderEvaluator(["search", "calculate", "answer"]).evaluate(trace)
    assert result.passed is True


async def test_tool_order_wrong_relative_order():
    trace = _trace(["calculate", "search", "answer"])
    result = await ToolOrderEvaluator(["search", "calculate", "answer"]).evaluate(trace)
    assert result.passed is False
    assert result.verdict == Verdict.FAIL


async def test_tool_order_allows_interleaving():
    trace = _trace(["search", "think", "calculate", "think", "answer"])
    result = await ToolOrderEvaluator(["search", "calculate", "answer"]).evaluate(trace)
    assert result.passed is True


async def test_tool_order_missing_action():
    trace = _trace(["search", "answer"])
    result = await ToolOrderEvaluator(["search", "calculate", "answer"]).evaluate(trace)
    assert result.passed is False


async def test_required_tool_present():
    trace = _trace(["search", "answer"])
    result = await RequiredToolEvaluator("search").evaluate(trace)
    assert result.passed is True


async def test_required_tool_missing():
    trace = _trace(["answer"])
    result = await RequiredToolEvaluator("search").evaluate(trace)
    assert result.passed is False


async def test_forbidden_tool_absent():
    trace = _trace(["search", "answer"])
    result = await ForbiddenToolEvaluator("delete_user").evaluate(trace)
    assert result.passed is True


async def test_forbidden_tool_present():
    trace = _trace(["search", "delete_user", "answer"])
    result = await ForbiddenToolEvaluator("delete_user").evaluate(trace)
    assert result.passed is False
    assert result.verdict == Verdict.FAIL


async def test_maximum_tool_calls_within_bound():
    trace = _trace(["a", "b", "c", "d", "e"])
    result = await MaximumToolCallsEvaluator(max_calls=5).evaluate(trace)
    assert result.passed is True


async def test_maximum_tool_calls_exceeded():
    trace = _trace(["a", "b", "c", "d", "e", "f"])
    result = await MaximumToolCallsEvaluator(max_calls=5).evaluate(trace)
    assert result.passed is False
    assert "6" in result.reason
