"""Framework-agnostic agent execution trace.

A ``Trace`` is the "dashcam" of an agent run: it records every observable
event that happened while the agent executed, plus the final output.

The trace model intentionally knows nothing about any specific agent
framework (OpenAI SDK, LangGraph, ...). Framework adapters are responsible
for translating framework-specific internals into standardized trace events.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class UserMessageEvent(BaseModel):
    """A message sent from the user (or the harness) to the agent."""

    type: Literal["user_message"] = "user_message"
    content: str


class AgentMessageEvent(BaseModel):
    """A message produced by the agent."""

    type: Literal["agent_message"] = "agent_message"
    content: str


class ToolCallEvent(BaseModel):
    """The agent requested a tool invocation.

    ``call_id`` uniquely identifies this invocation; a matching
    :class:`ToolResultEvent` references the same id so that calls and
    results can be paired even when multiple calls interleave or the same
    tool is invoked repeatedly. It is framework agnostic and must never be
    inferred from event ordering.
    """

    type: Literal["tool_call"] = "tool_call"
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(BaseModel):
    """The result returned by a tool invocation.

    ``call_id`` must match the :class:`ToolCallEvent` it answers.
    """

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    result: Any = None


class CustomEvent(BaseModel):
    """An application-specific event recorded by the agent/adapter."""

    type: Literal["custom_event"] = "custom_event"
    name: str
    data: dict[str, Any] = Field(default_factory=dict)


class ActionEvent(BaseModel):
    """The agent requested an action inside an environment.

    ``action_id`` uniquely identifies this invocation; a correlated
    :class:`StateChangeEvent` references it. It is distinct from a
    :class:`ToolCallEvent` because an action is agent → environment, whereas a
    tool call is agent → tool.
    """

    type: Literal["action"] = "action"
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class EnvironmentEvent(BaseModel):
    """Feedback produced by the environment in response to an action."""

    type: Literal["environment_event"] = "environment_event"
    source: str
    data: dict[str, Any] = Field(default_factory=dict)


class StateSnapshotEvent(BaseModel):
    """A full snapshot of the environment's state at a point in time.

    ``action_id`` ties the snapshot to the :class:`ActionEvent` that produced
    it (``None`` for the initial, pre-action snapshot), so history evaluators
    can attribute a change to a specific action.

    ``state`` is deep-copied on construction: a snapshot must be immutable
    history, and an environment that mutates its state dict in place would
    otherwise rewrite every earlier snapshot.
    """

    type: Literal["state_snapshot"] = "state_snapshot"
    action_id: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state")
    @classmethod
    def _isolate_state(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Detach the snapshot from the caller's (mutable) state object."""
        return deepcopy(value)


class StateChangeEvent(BaseModel):
    """A (partial) delta describing how the state changed after an action.

    ``changes`` is deep-copied on construction for the same reason as
    :class:`StateSnapshotEvent`: recorded history must not alias a dict the
    environment keeps mutating.
    """

    type: Literal["state_change"] = "state_change"
    action_id: str | None = None
    changes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("changes")
    @classmethod
    def _isolate_changes(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Detach the delta from the caller's (mutable) dict."""
        return deepcopy(value)


class MetricEvent(BaseModel):
    """An optional named numeric metric emitted by the environment.

    Metrics are not required: environments that report none simply never
    record this event. The channel is deliberately generic — ``name`` carries
    the metric identity (``"reward"``, ``"quality"``, ``"cost"``, ...), so a
    single trace can carry several metrics instead of one anonymous scalar.
    Positive/negative real values are allowed.
    """

    type: Literal["metric"] = "metric"
    name: str
    value: float


TraceEvent = Annotated[
    UserMessageEvent
    | AgentMessageEvent
    | ToolCallEvent
    | ToolResultEvent
    | CustomEvent
    | ActionEvent
    | EnvironmentEvent
    | StateSnapshotEvent
    | StateChangeEvent
    | MetricEvent,
    Field(discriminator="type"),
]


class Trace(BaseModel):
    """A standardized, framework-agnostic record of a single agent run."""

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    input: str
    events: list[TraceEvent] = Field(default_factory=list)
    final_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_event(self, event: TraceEvent) -> None:
        """Append an event to the trace."""
        self.events.append(event)

    def tool_calls(self) -> list[ToolCallEvent]:
        """All tool call events, in execution order."""
        return [event for event in self.events if isinstance(event, ToolCallEvent)]

    def tool_results(self) -> list[ToolResultEvent]:
        """All tool result events, in execution order."""
        return [event for event in self.events if isinstance(event, ToolResultEvent)]

    def tool_call_names(self) -> list[str]:
        """Names of every tool call, in execution order."""
        return [event.name for event in self.tool_calls()]

    def result_for_call(self, call_id: str) -> ToolResultEvent | None:
        """The result answering ``call_id``, or ``None`` if there is none.

        The most recent matching result is returned, so interleaved or
        repeated calls of the same tool pair correctly.
        """
        for event in reversed(self.tool_results()):
            if event.call_id == call_id:
                return event
        return None

    def actions(self) -> list[ActionEvent]:
        """All agent actions recorded, in execution order."""
        return [event for event in self.events if isinstance(event, ActionEvent)]

    def action_names(self) -> list[str]:
        """Names of every action, in execution order."""
        return [event.name for event in self.actions()]

    def state_snapshots(self) -> list[StateSnapshotEvent]:
        """All state snapshots recorded, in chronological order."""
        return [event for event in self.events if isinstance(event, StateSnapshotEvent)]

    def state_changes(self) -> list[StateChangeEvent]:
        """All state-change deltas recorded, in execution order."""
        return [event for event in self.events if isinstance(event, StateChangeEvent)]

    def metrics(self) -> list[MetricEvent]:
        """All metric events recorded, in execution order."""
        return [event for event in self.events if isinstance(event, MetricEvent)]

    def metric(self, name: str) -> float | None:
        """The most recent value of the named metric, or ``None`` if absent.

        The most recent emission wins, so a metric reported repeatedly (or
        reported again after a later action) reflects its latest value.
        """
        for event in reversed(self.metrics()):
            if event.name == name:
                return event.value
        return None

    @property
    def final_state(self) -> dict[str, Any] | None:
        """The most recent state snapshot, or ``None`` if none was recorded."""
        snapshots = self.state_snapshots()
        return snapshots[-1].state if snapshots else None
