"""Abstract evaluator interface.

An evaluator inspects a :class:`~agentmeter.core.trace.Trace` and produces an
:class:`~agentmeter.core.results.EvaluationResult`. Evaluators are framework
agnostic: they never depend on any specific agent framework or LLM SDK.

Contract:
- ``result.verdict`` must be one of PASS / FAIL / ERROR.
- ``result.score`` must satisfy 0.0 <= score <= 1.0.
- ``result.passed`` (a derived property) is independent from ``score``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentmeter.core.results import EvaluationResult
from agentmeter.core.trace import Trace


class Evaluator(ABC):
    """Evaluates an agent trace and returns a verdict."""

    @abstractmethod
    async def evaluate(self, trace: Trace) -> EvaluationResult:
        """Evaluate ``trace`` and return the verdict."""
        raise NotImplementedError
