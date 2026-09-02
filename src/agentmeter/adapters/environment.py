"""Adapter that drives an agent inside an environment.

A plain chat agent produces text. A *task* agent is different: it keeps
observing the environment state and issuing actions until it produces a final
answer. This adapter translates that interaction loop into a standardized
:class:`~agentmeter.core.trace.Trace`, recording every action, environment
event, state snapshot, state change and reward along the way so the existing
evaluators (tool, trajectory, state, reward, LLM judge) can inspect it.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from agentmeter.adapters.base import AgentAdapter
from agentmeter.core.trace import (
    ActionEvent,
    AgentMessageEvent,
    EnvironmentEvent,
    RewardEvent,
    StateChangeEvent,
    StateSnapshotEvent,
    Trace,
    UserMessageEvent,
)
from agentmeter.environments.base import Action, Environment, State

# The agent decides what to do next given the current state and its trace so
# far. It returns either a final answer string (stop) or an Action (keep going).
EnvironmentDecider = Callable[
    [State, Trace],
    Awaitable[Action | str] | Action | str,
]


class EnvironmentAgentAdapter(AgentAdapter):
    """Runs an agent against an :class:`Environment`, recording a full trace.

    Args:
        environment: the environment the agent interacts with.
        decide: ``(state, trace) -> Action | str``; returning a string ends the
            run and becomes the agent's final output.
        max_steps: safety bound on actions per run to avoid infinite loops.
    """

    def __init__(
        self,
        environment: Environment,
        decide: EnvironmentDecider,
        *,
        max_steps: int = 100,
    ) -> None:
        self._environment = environment
        self._decide = decide
        self._max_steps = max_steps

    async def run(self, message: str) -> Trace:
        trace = Trace(input=message)
        trace.add_event(UserMessageEvent(content=message))

        state = await self._environment.reset()
        trace.add_event(StateSnapshotEvent(state=state.as_dict()))

        for _ in range(self._max_steps):
            decision = self._decide(state, trace)
            if inspect.isawaitable(decision):
                decision = await decision

            if isinstance(decision, str):
                trace.final_output = decision
                trace.add_event(AgentMessageEvent(content=decision))
                return trace

            action = self._coerce_action(decision)
            trace.add_event(ActionEvent(name=action.name, arguments=action.arguments))
            action_id = trace.events[-1].action_id

            outcome = await self._environment.execute_action(action)

            for observation in outcome.observations:
                trace.add_event(
                    EnvironmentEvent(
                        source=type(self._environment).__name__,
                        data={"action": action.name, "observation": observation},
                    )
                )

            if outcome.changes:
                trace.add_event(
                    StateChangeEvent(action_id=action_id, changes=outcome.changes)
                )

            if outcome.reward is not None:
                trace.add_event(RewardEvent(value=outcome.reward))

            state = await self._environment.get_state()
            trace.add_event(StateSnapshotEvent(state=state.as_dict()))

            if outcome.done:
                break

        if trace.final_output is None:
            trace.final_output = ""
            trace.add_event(AgentMessageEvent(content=""))
        return trace

    @staticmethod
    def _coerce_action(decision: object) -> Action:
        if isinstance(decision, Action):
            return decision
        if isinstance(decision, dict) and "name" in decision:
            return Action(
                name=decision["name"],
                arguments=decision.get("arguments", {}),
            )
        raise TypeError(
            f"decider must return Action or str, got {type(decision).__name__}: {decision!r}"
        )
