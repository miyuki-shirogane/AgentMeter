"""Tests for history-of-state evaluators and the snapshot isolation fix."""

import pytest

from agentmeter import (
    Action,
    ActionResult,
    Environment,
    EnvironmentAgentAdapter,
    State,
    StateChangeEvaluator,
    StateChangeEvent,
    StateMonotonicEvaluator,
    StateSnapshotEvent,
    StateTransitionEvaluator,
    StateUnchangedEvaluator,
    Trace,
    Verdict,
)
from agentmeter.environments.mock_order_api import OrderEnvironment


def _trace_with_states(states: list[dict], *, action_ids: list[str] | None = None) -> Trace:
    trace = Trace(input="in")
    for index, state in enumerate(states):
        action_id = action_ids[index] if action_ids else None
        trace.add_event(StateSnapshotEvent(state=state, action_id=action_id))
    return trace


def _trace_with_changes(changes: list[dict]) -> Trace:
    trace = Trace(input="in")
    for index, change in enumerate(changes):
        trace.add_event(StateChangeEvent(action_id=f"a{index}", changes=change))
    return trace


# --------------------------------------------------------------------------
# Snapshot isolation (step 0) — regression test
# --------------------------------------------------------------------------


class _InPlaceMutatingEnvironment(Environment):
    """A deliberately sloppy env that mutates the SAME nested structures in
    place and hands the same object back from ``get_state``. Without the
    defensive deep copy, every historical snapshot would show the final value.
    """

    def __init__(self) -> None:
        self._state: dict = {"items": [], "nested": {"x": 0}}

    async def reset(self) -> State:
        self._state = {"items": [], "nested": {"x": 0}}
        return State(data=self._state)

    async def execute_action(self, action: Action) -> ActionResult:
        self._state["items"].append(action.name)
        self._state["nested"]["x"] += 1
        return ActionResult(changes={"nested": {"x": self._state["nested"]["x"]}})

    async def get_state(self) -> State:
        return State(data=self._state)


async def test_snapshots_are_isolated_from_in_place_mutation():
    decisions = iter([Action(name="a1"), Action(name="a2"), "done"])

    async def decide(state: State, trace: Trace) -> Action | str:
        return next(decisions)

    trace = await EnvironmentAgentAdapter(_InPlaceMutatingEnvironment(), decide).run("go")

    assert [snapshot.state for snapshot in trace.state_snapshots()] == [
        {"items": [], "nested": {"x": 0}},
        {"items": ["a1"], "nested": {"x": 1}},
        {"items": ["a1", "a2"], "nested": {"x": 2}},
    ]
    assert [change.changes for change in trace.state_changes()] == [
        {"nested": {"x": 1}},
        {"nested": {"x": 2}},
    ]


async def test_snapshots_carry_the_action_id():
    decisions = iter([Action(name="a1"), Action(name="a2"), "done"])

    async def decide(state: State, trace: Trace) -> Action | str:
        return next(decisions)

    trace = await EnvironmentAgentAdapter(_InPlaceMutatingEnvironment(), decide).run("go")

    action_ids = [action.action_id for action in trace.actions()]
    assert [snapshot.action_id for snapshot in trace.state_snapshots()] == [
        None,
        *action_ids,
    ]


# --------------------------------------------------------------------------
# StateChangeEvaluator
# --------------------------------------------------------------------------


async def test_state_change_passes_when_every_change_satisfies():
    trace = _trace_with_changes([{"total": 299.0}, {"total": 348.0}])
    result = await StateChangeEvaluator("total", "lte", 5000).evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_state_change_fails_on_a_single_violating_delta():
    trace = _trace_with_changes([{"total": 299.0}, {"total": 9999.0}])
    result = await StateChangeEvaluator("total", "lte", 5000).evaluate(trace)
    assert result.verdict == Verdict.FAIL
    assert result.metadata["violations"][0]["action_id"] == "a1"


async def test_state_change_fails_when_assertion_was_never_exercised():
    trace = _trace_with_changes([{"status": "paid"}])
    result = await StateChangeEvaluator("total", "lte", 5000).evaluate(trace)
    assert result.verdict == Verdict.FAIL
    assert "not exercised" in result.reason


async def test_state_change_can_allow_no_matching_change():
    trace = _trace_with_changes([{"status": "paid"}])
    result = await StateChangeEvaluator("total", "lte", 5000, require_at_least_one=False).evaluate(
        trace
    )
    assert result.verdict == Verdict.PASS


async def test_state_change_supports_custom_predicate():
    trace = _trace_with_changes([{"total": 299.0}, {"total": 349.0}])
    result = await StateChangeEvaluator("total", predicate=lambda v: v % 100 != 0).evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_state_change_predicate_violation_fails():
    trace = _trace_with_changes([{"total": 299.0}, {"total": 400.0}])
    result = await StateChangeEvaluator("total", predicate=lambda v: v % 100 != 0).evaluate(trace)
    assert result.verdict == Verdict.FAIL


# --------------------------------------------------------------------------
# StateTransitionEvaluator
# --------------------------------------------------------------------------

ALLOWED_STATUS = [("draft", "paid"), ("paid", "refunded")]


async def test_transition_passes_on_allowed_path():
    trace = _trace_with_states([{"status": "draft"}, {"status": "paid"}, {"status": "refunded"}])
    result = await StateTransitionEvaluator("status", ALLOWED_STATUS).evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_transition_fails_on_forbidden_transition():
    trace = _trace_with_states([{"status": "draft"}, {"status": "paid"}, {"status": "draft"}])
    result = await StateTransitionEvaluator("status", ALLOWED_STATUS).evaluate(trace)
    assert result.verdict == Verdict.FAIL
    assert result.metadata["violations"] == [["paid", "draft"]]


async def test_transition_ignores_repeated_values_by_default():
    trace = _trace_with_states([{"status": "draft"}, {"status": "draft"}, {"status": "paid"}])
    result = await StateTransitionEvaluator("status", ALLOWED_STATUS).evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_transition_counts_repeats_when_unchanged_is_not_allowed():
    trace = _trace_with_states([{"status": "draft"}, {"status": "draft"}, {"status": "paid"}])
    result = await StateTransitionEvaluator(
        "status", ALLOWED_STATUS, allow_unchanged=False
    ).evaluate(trace)
    assert result.verdict == Verdict.FAIL


async def test_transition_errors_when_path_missing():
    trace = _trace_with_states([{"status": "draft"}, {"other": 1}])
    result = await StateTransitionEvaluator("status", ALLOWED_STATUS).evaluate(trace)
    assert result.verdict == Verdict.ERROR


async def test_transition_can_skip_missing_path():
    trace = _trace_with_states([{"status": "draft"}, {"other": 1}, {"status": "paid"}])
    result = await StateTransitionEvaluator("status", ALLOWED_STATUS, on_missing="skip").evaluate(
        trace
    )
    assert result.verdict == Verdict.PASS


async def test_transition_errors_without_snapshots():
    result = await StateTransitionEvaluator("status", ALLOWED_STATUS).evaluate(Trace(input="in"))
    assert result.verdict == Verdict.ERROR


async def test_transition_fails_when_nothing_was_exercised():
    trace = _trace_with_states([{"status": "draft"}])
    result = await StateTransitionEvaluator("status", ALLOWED_STATUS).evaluate(trace)
    assert result.verdict == Verdict.FAIL
    assert "not exercised" in result.reason


async def test_transition_rejects_invalid_on_missing():
    with pytest.raises(ValueError, match="on_missing"):
        StateTransitionEvaluator("status", ALLOWED_STATUS, on_missing="whatever")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# StateMonotonicEvaluator
# --------------------------------------------------------------------------


async def test_monotonic_non_decreasing_passes():
    trace = _trace_with_states([{"processed": 0}, {"processed": 100}, {"processed": 100}])
    result = await StateMonotonicEvaluator("processed").evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_monotonic_non_decreasing_fails_on_decrease():
    trace = _trace_with_states([{"processed": 100}, {"processed": 42}])
    result = await StateMonotonicEvaluator("processed").evaluate(trace)
    assert result.verdict == Verdict.FAIL


async def test_monotonic_strictly_increasing_rejects_equal_values():
    trace = _trace_with_states([{"processed": 1}, {"processed": 1}])
    result = await StateMonotonicEvaluator("processed", "strictly_increasing").evaluate(trace)
    assert result.verdict == Verdict.FAIL


async def test_monotonic_non_increasing_passes():
    trace = _trace_with_states([{"budget": 100}, {"budget": 40}])
    result = await StateMonotonicEvaluator("budget", "non_increasing").evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_monotonic_strictly_decreasing_passes():
    trace = _trace_with_states([{"budget": 100}, {"budget": 40}])
    result = await StateMonotonicEvaluator("budget", "strictly_decreasing").evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_monotonic_errors_without_snapshots():
    result = await StateMonotonicEvaluator("processed").evaluate(Trace(input="in"))
    assert result.verdict == Verdict.ERROR


async def test_monotonic_tolerance_absorbs_float_rounding():
    """Two values that differ only by float noise must not read as a decrease."""
    trace = _trace_with_states([{"total": 269.1}, {"total": 269.0999999999999}])
    lenient = await StateMonotonicEvaluator("total", "non_decreasing", tolerance=0.01).evaluate(
        trace
    )
    strict = await StateMonotonicEvaluator("total", "non_decreasing").evaluate(trace)
    assert lenient.verdict == Verdict.PASS
    assert strict.verdict == Verdict.FAIL


async def test_monotonic_errors_on_non_numeric_value():
    trace = _trace_with_states([{"processed": 1}, {"processed": "many"}])
    result = await StateMonotonicEvaluator("processed").evaluate(trace)
    assert result.verdict == Verdict.ERROR


async def test_monotonic_errors_on_missing_path():
    trace = _trace_with_states([{"processed": 1}, {"other": 2}])
    result = await StateMonotonicEvaluator("processed").evaluate(trace)
    assert result.verdict == Verdict.ERROR


async def test_monotonic_rejects_bad_arguments():
    with pytest.raises(ValueError, match="direction"):
        StateMonotonicEvaluator("x", "sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tolerance"):
        StateMonotonicEvaluator("x", tolerance=-1.0)


# --------------------------------------------------------------------------
# StateUnchangedEvaluator
# --------------------------------------------------------------------------


async def test_unchanged_passes_when_value_is_stable():
    trace = _trace_with_states(
        [{"owner": "alice", "total": 0}, {"owner": "alice", "total": 299}, {"owner": "alice"}]
    )
    result = await StateUnchangedEvaluator("owner").evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_unchanged_fails_when_value_moves():
    trace = _trace_with_states([{"owner": "alice"}, {"owner": "bob"}])
    result = await StateUnchangedEvaluator("owner").evaluate(trace)
    assert result.verdict == Verdict.FAIL
    assert result.metadata["changes"] == [{"snapshot": 1, "value": "bob"}]


async def test_unchanged_errors_when_path_is_missing():
    trace = _trace_with_states([{"owner": "alice"}, {"total": 1}])
    result = await StateUnchangedEvaluator("owner").evaluate(trace)
    assert result.verdict == Verdict.ERROR


async def test_unchanged_can_skip_missing_path():
    trace = _trace_with_states([{"owner": "alice"}, {"total": 1}, {"owner": "alice"}])
    result = await StateUnchangedEvaluator("owner", allow_missing=True).evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_unchanged_errors_with_too_few_observations():
    trace = _trace_with_states([{"owner": "alice"}])
    result = await StateUnchangedEvaluator("owner").evaluate(trace)
    assert result.verdict == Verdict.ERROR


async def test_unchanged_errors_without_snapshots():
    result = await StateUnchangedEvaluator("owner").evaluate(Trace(input="in"))
    assert result.verdict == Verdict.ERROR


# --------------------------------------------------------------------------
# End to end against the reference environment
# --------------------------------------------------------------------------


async def _run_order_flow() -> Trace:
    async def decide(state: State, trace: Trace) -> Action | str:
        if state.get("status") == "draft" and not state.get("items"):
            return Action(name="add_item", arguments={"sku": "iphone", "qty": 1})
        if state.get("status") == "draft":
            return Action(name="checkout", arguments={})
        if state.get("status") == "paid":
            return Action(name="request_refund", arguments={"reason": "unwanted"})
        return "refunded"

    return await EnvironmentAgentAdapter(OrderEnvironment(), decide).run("下单并退款")


async def test_order_flow_respects_the_status_state_machine():
    trace = await _run_order_flow()
    allowed = [("draft", "paid"), ("paid", "refunded")]
    result = await StateTransitionEvaluator("status", allowed).evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_order_flow_keeps_owner_immutable():
    trace = await _run_order_flow()
    result = await StateUnchangedEvaluator("owner").evaluate(trace)
    assert result.verdict == Verdict.PASS


async def test_order_flow_total_is_non_decreasing():
    trace = await _run_order_flow()
    assert (await StateMonotonicEvaluator("total").evaluate(trace)).verdict == Verdict.PASS


async def test_coupon_breaks_monotonic_total():
    """Monotonicity must be chosen by business meaning: a coupon legitimately
    lowers the total, so 'total only increases' is the wrong assertion here."""

    async def decide(state: State, trace: Trace) -> Action | str:
        if not state.get("items"):
            return Action(name="add_item", arguments={"sku": "iphone", "qty": 1})
        if state.get("status") == "draft":
            return Action(name="apply_coupon", arguments={"code": "SAVE10"})
        return "discounted"

    trace = await EnvironmentAgentAdapter(OrderEnvironment(), decide).run("用券买 iphone")
    result = await StateMonotonicEvaluator("total").evaluate(trace)
    assert result.verdict == Verdict.FAIL
    assert result.metadata["violations"][0]["to"] == pytest.approx(269.1)
