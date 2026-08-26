"""Shared verdict and score semantics.

``Verdict`` is the single source of truth for evaluation outcome:

- PASS  : the agent executed normally and met the test condition.
- FAIL  : the agent executed normally but behaved differently than expected.
- ERROR : the agent (or evaluation infrastructure) raised, so the test could
          not complete normally. ERROR is never silently converted to FAIL.

``Score`` is a float constrained to the closed interval [0.0, 1.0], where
0.0 means "completely unsatisfied" and 1.0 means "completely satisfied".
``passed`` and ``score`` are independent: e.g. ``score=0.72, passed=False``
is valid (a future LLM judge may return continuous scores and its own
pass threshold).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


Score = Annotated[float, Field(ge=0.0, le=1.0)]
