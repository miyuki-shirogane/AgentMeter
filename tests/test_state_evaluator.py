"""Tests for StateEvaluator, StateOperator, and the state predicate builder."""

import pytest

from agentmeter import (
    StateEvaluator,
    StateOperator,
    StateSnapshotEvent,
    Verdict,
    compare_value,
    make_state_predicate,
)
from agentmeter.core.trace import Trace


def _trace_with_state(state: dict) -> Trace:
    trace = Trace(input="in")
    trace.add_event(StateSnapshotEvent(state=state))
    return trace


# ---- operators -----------------------------------------------------------

def test_compare_value_all_operators():
    assert compare_value(5, StateOperator.EQ, 5) is True
    assert compare_value(5, StateOperator.NE, 6) is True
    assert compare_value(5, StateOperator.GT, 4) is True
    assert compare_value(5, StateOperator.GTE, 5) is True
    assert compare_value(5, StateOperator.LT, 6) is True
    assert compare_value(5, StateOperator.LTE, 5) is True


def test_compare_value_exists():
    from agentmeter.environments.base import UNSET

    assert compare_value(UNSET, StateOperator.EXISTS, None) is False
    assert compare_value(123, StateOperator.EXISTS, None) is True


def test_compare_value_missing_fails_for_value_operators():
    from agentmeter.environments.base import UNSET

    assert compare_value(UNSET, StateOperator.EQ, 0) is False
    assert compare_value(UNSET, StateOperator.NE, 5) is False  # never "not equal"
    assert compare_value(UNSET, StateOperator.GT, 0) is False


def test_compare_value_accepts_string_operator():
    assert compare_value(3, "gte", 3) is True


def test_unsupported_operator_raises():
    with pytest.raises(ValueError):
        compare_value(1, "banana", 1)


# ---- predicates over nested paths ---------------------------------------

def test_predicate_equals_and_not_equals():
    pred = make_state_predicate("boss.status", "eq", "dead")
    assert pred({"boss": {"status": "dead"}}) is True
    assert pred({"boss": {"status": "alive"}}) is False

    pred = make_state_predicate("boss.status", "ne", "dead")
    assert pred({"boss": {"status": "alive"}}) is True


def test_predicate_numeric_ordering():
    assert make_state_predicate("reward", "gte", 100)({"reward": 100}) is True
    assert make_state_predicate("reward", "gt", 100)({"reward": 100}) is False
    assert make_state_predicate("reward", "lt", 100)({"reward": 99}) is True


def test_predicate_dollar_prefix():
    pred = make_state_predicate("$.boss.status", "eq", "dead")
    assert pred({"boss": {"status": "dead"}}) is True


def test_predicate_custom_function():
    pred = make_state_predicate("boss.hp", predicate=lambda hp: hp <= 0)
    assert pred({"boss": {"hp": 0}}) is True
    assert pred({"boss": {"hp": 50}}) is False


# ---- StateEvaluator ------------------------------------------------------

async def test_state_evaluator_passes_on_matching_final_state():
    result = await StateEvaluator("boss.status", "eq", "dead").evaluate(
        _trace_with_state({"boss": {"status": "dead"}, "reward": 100})
    )
    assert result.passed is True
    assert result.verdict == Verdict.PASS
    assert result.score == 1.0
    assert result.metadata["path"] == "boss.status"


async def test_state_evaluator_fails_on_mismatch():
    result = await StateEvaluator("boss.status", "eq", "dead").evaluate(
        _trace_with_state({"boss": {"status": "alive"}})
    )
    assert result.passed is False
    assert result.verdict == Verdict.FAIL


async def test_state_evaluator_uses_last_snapshot():
    trace = Trace(input="in")
    trace.add_event(StateSnapshotEvent(state={"boss": {"status": "alive"}}))
    trace.add_event(StateSnapshotEvent(state={"boss": {"status": "dead"}}))
    result = await StateEvaluator("boss.status", "eq", "dead").evaluate(trace)
    assert result.passed is True


async def test_state_evaluator_missing_state_fails_by_default():
    result = await StateEvaluator("boss.status", "eq", "dead").evaluate(Trace(input="in"))
    assert result.verdict == Verdict.FAIL
    assert "no state snapshot" in result.reason


async def test_state_evaluator_missing_state_errors_when_allowed():
    result = await StateEvaluator("boss.status", "eq", "dead", allow_missing=True).evaluate(
        Trace(input="in")
    )
    assert result.verdict == Verdict.ERROR


async def test_state_evaluator_exists_operator():
    present = await StateEvaluator("boss.hp", "exists").evaluate(
        _trace_with_state({"boss": {"status": "alive", "hp": 100}})
    )
    assert present.passed is True

    absent = await StateEvaluator("boss.hp", "exists").evaluate(
        _trace_with_state({"boss": {"status": "dead"}})
    )
    assert absent.passed is False


async def test_state_evaluator_custom_predicate():
    result = await StateEvaluator("reward", predicate=lambda r: r >= 100).evaluate(
        _trace_with_state({"reward": 150})
    )
    assert result.passed is True


async def test_state_evaluator_lists_in_snapshot_are_stable():
    state = {"boss": {"status": "dead"}, "reward": 100}
    result = await StateEvaluator("$.reward", "gte", 100).evaluate(_trace_with_state(state))
    assert result.passed is True
