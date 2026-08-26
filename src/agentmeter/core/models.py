"""Core domain model: TestCase."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentmeter.adapters.base import AgentAdapter
from agentmeter.adapters.callable import CallableAgentAdapter
from agentmeter.evaluators.base import Evaluator


class TestCase(BaseModel):
    """A single agent test case.

    A test case binds an agent (via an ``AgentAdapter``) to a list of
    evaluators, along with the input that will be sent to the agent.

    For convenience, a plain Python callable may be passed as ``agent``; it
    is automatically wrapped in a :class:`CallableAgentAdapter`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    input: str
    agent: AgentAdapter
    evaluators: list[Evaluator]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _wrap_callable_agent(cls, data: Any) -> Any:
        if isinstance(data, dict):
            agent = data.get("agent")
            if agent is not None and not isinstance(agent, AgentAdapter):
                data = {**data, "agent": CallableAgentAdapter(agent)}
        return data
