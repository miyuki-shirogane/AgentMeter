"""Structured result returned by a judge provider.

A judge must never answer with prose alone ("I think it's fine"). It returns
this structured, validated object so that evaluation semantics stay precise.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentmeter.core.verdict import Score


class JudgeResult(BaseModel):
    """A validated, structured verdict from an LLM judge.

    ``score`` is validated to the closed interval [0.0, 1.0]; ``passed`` is
    the judge's own pass decision and may differ from a caller-chosen
    threshold on ``score``.
    """

    passed: bool
    score: Score
    reason: str = ""
    violations: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
