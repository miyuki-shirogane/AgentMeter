"""Abstract agent adapter interface.

``AgentAdapter`` is the seam between AgentMeter and any concrete agent
framework. The core framework only ever talks to this interface; each
framework integration (OpenAI SDK, LangGraph, ...) implements it and
translates framework-specific execution details into a standardized
:class:`~agentmeter.core.trace.Trace`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentmeter.core.trace import Trace


class AgentAdapter(ABC):
    """Runs an agent and returns its standardized execution trace."""

    @abstractmethod
    async def run(self, message: str) -> Trace:
        """Execute the agent with ``message`` and return its trace."""
        raise NotImplementedError
