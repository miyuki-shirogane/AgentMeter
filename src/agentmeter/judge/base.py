"""Abstract judge provider interface.

``JudgeProvider`` is the seam between the LLM-judge evaluator and any
concrete LLM API. The evaluator only ever talks to this interface; each
provider implementation (OpenAI-compatible, Anthropic, Gemini, Ollama, ...)
translates an API call into a validated :class:`JudgeResult`.

Concrete providers must raise :class:`JudgeError` for any failure they can
observe (timeout, HTTP error, malformed body, schema violation) instead of
letting raw transport exceptions leak into evaluation logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentmeter.judge.result import JudgeResult


class JudgeError(Exception):
    """Raised by a judge provider when the judge call cannot complete."""


class JudgeProvider(ABC):
    """Asks an LLM to judge behavior and returns a validated result."""

    @abstractmethod
    async def judge(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        """Send the prompts to the LLM and return the validated verdict.

        Raises :class:`JudgeError` on timeout, API error, malformed JSON, or
        a response that fails schema validation.
        """
        raise NotImplementedError
