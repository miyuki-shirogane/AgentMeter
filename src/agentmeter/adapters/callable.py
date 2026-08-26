"""Adapter for plain Python callables.

Phase 1 only needs to run ordinary ``async def`` (or ``def``) functions as
agents. The adapter records a ``user_message`` event, invokes the callable,
and records the resulting ``agent_message`` event.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from agentmeter.adapters.base import AgentAdapter
from agentmeter.core.trace import AgentMessageEvent, Trace, UserMessageEvent

CallableAgent = Callable[[str], Awaitable[str] | str]


class CallableAgentAdapter(AgentAdapter):
    """Adapts any ``message -> str`` callable into an agent."""

    def __init__(self, fn: CallableAgent) -> None:
        self._fn = fn

    async def run(self, message: str) -> Trace:
        events = [UserMessageEvent(content=message)]

        output = self._fn(message)
        if inspect.isawaitable(output):
            output = await output
        final_output = str(output)

        events.append(AgentMessageEvent(content=final_output))
        return Trace(
            input=message,
            events=events,
            final_output=final_output,
            metadata={"adapter": type(self).__name__},
        )
