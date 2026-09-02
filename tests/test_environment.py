"""Tests for the environment interface, State model, environment adapter,
and the Mock Game (an environment used only for testing)."""


from agentmeter import (
    Action,
    ActionEvent,
    ActionResult,
    AgentAdapter,
    AgentMessageEvent,
    Environment,
    EnvironmentAgentAdapter,
    EnvironmentEvent,
    RewardEvaluator,
    RewardEvent,
    Runner,
    State,
    StateChangeEvent,
    StateSnapshotEvent,
    Trace,
    UserMessageEvent,
    Verdict,
)
from agentmeter import (
    TestCase as AgentTestCase,
)

# --------------------------------------------------------------------------
# State model + path resolution
# --------------------------------------------------------------------------

def test_state_resolves_nested_path():
    state = State(data={"boss": {"status": "dead"}, "reward": 100})
    assert state.get("boss.status") == "dead"
    assert state.get("$.boss.status") == "dead"
    assert state.get("reward") == 100


def test_state_missing_path_returns_default():
    state = State(data={"boss": {"status": "dead"}})
    assert state.get("boss.hp") is None
    assert state.get("boss.hp", default=-1) == -1


def test_state_getitem_and_as_dict():
    state = State(data={"reward": 100})
    assert state["reward"] == 100
    assert state.as_dict() == {"reward": 100}


def test_state_supports_list_indexing():
    state = State(data={"players": [{"name": "alice"}]})
    assert state.get("players.0.name") == "alice"


# --------------------------------------------------------------------------
# Environment interface contract + adapter driving loop
# --------------------------------------------------------------------------

def test_environment_is_abstract():
    assert Environment.__abstractmethods__ == {"reset", "execute_action", "get_state"}


class RecordingEnvironment(Environment):
    """A stateful env: each attack deals 50 damage; done once HP hits 0."""

    def __init__(self) -> None:
        self._hp = 100

    async def reset(self) -> State:
        self._hp = 100
        return State(data={"hp": 100, "reward": 0})

    async def execute_action(self, action: Action) -> ActionResult:
        self._hp = max(0, self._hp - 50)
        done = self._hp == 0
        return ActionResult(
            reward=100 if done else None,
            observations=[f"did {action.name}"],
            changes={"hp": self._hp},
            done=done,
        )

    async def get_state(self) -> State:
        return State(data={"hp": self._hp, "reward": 100 if self._hp == 0 else 0})


async def test_environment_adapter_records_full_trace():
    decisions = iter([Action(name="attack", arguments={"target": "boss"}), "boss defeated"])

    async def decide(state: State, trace: Trace) -> Action | str:
        return next(decisions)

    adapter = EnvironmentAgentAdapter(RecordingEnvironment(), decide)
    assert isinstance(adapter, AgentAdapter)
    trace = await adapter.run("defeat the boss")

    assert isinstance(trace.events[0], UserMessageEvent)
    assert isinstance(trace.events[1], StateSnapshotEvent)
    assert trace.events[1].state == {"hp": 100, "reward": 0}

    action_events = [e for e in trace.events if isinstance(e, ActionEvent)]
    assert len(action_events) == 1
    assert action_events[0].name == "attack"
    assert action_events[0].arguments == {"target": "boss"}

    assert any(isinstance(e, EnvironmentEvent) for e in trace.events)
    assert any(isinstance(e, StateChangeEvent) for e in trace.events)
    assert trace.final_state == {"hp": 50, "reward": 0}
    assert trace.actions()[0].name == "attack"
    assert trace.final_output == "boss defeated"


async def test_environment_adapter_stops_on_string():
    async def decide(state: State, trace: Trace) -> Action | str:
        return "boss defeated"

    adapter = EnvironmentAgentAdapter(RecordingEnvironment(), decide)
    trace = await adapter.run("defeat the boss")

    assert trace.final_output == "boss defeated"
    assert isinstance(trace.events[-1], AgentMessageEvent)
    assert trace.actions() == []


async def test_environment_adapter_loop_breaks_when_env_is_done():
    """When the environment terminates, the loop stops even if the agent keeps
    returning actions; the final output stays empty (no agent summary)."""
    calls = {"n": 0}

    async def decide(state: State, trace: Trace) -> Action | str:
        calls["n"] += 1
        return Action(name="attack", arguments={"target": "boss"})

    adapter = EnvironmentAgentAdapter(RecordingEnvironment(), decide)
    trace = await adapter.run("fight")

    # Two attacks drive hp 100 -> 50 -> 0; after the second one the env is
    # done and the loop breaks without calling decide again.
    assert calls["n"] == 2
    assert len(trace.actions()) == 2
    assert trace.final_state == {"hp": 0, "reward": 100}
    assert trace.final_output == ""


async def test_environment_adapter_rejects_invalid_decision():
    async def decide(state: State, trace: Trace):
        return 42  # type: ignore[return-value]

    adapter = EnvironmentAgentAdapter(RecordingEnvironment(), decide)
    testcase = AgentTestCase(name="bad", input="hi", agent=adapter, evaluators=[])
    result = await Runner().run(testcase)
    assert result.verdict == Verdict.ERROR


# --------------------------------------------------------------------------
# Reward evaluator
# --------------------------------------------------------------------------

def _trace_with_reward(value: float) -> Trace:
    trace = Trace(input="in")
    trace.add_event(RewardEvent(value=value))
    return trace


async def test_reward_evaluator_passes_threshold():
    result = await RewardEvaluator("gte", 100).evaluate(_trace_with_reward(150))
    assert result.passed is True
    assert result.verdict == Verdict.PASS


async def test_reward_evaluator_fails_below_threshold():
    result = await RewardEvaluator("gte", 100).evaluate(_trace_with_reward(50))
    assert result.passed is False
    assert result.verdict == Verdict.FAIL


async def test_reward_evaluator_fails_when_no_reward():
    result = await RewardEvaluator("gte", 100).evaluate(Trace(input="in"))
    assert result.passed is False
    assert "no reward" in result.reason
