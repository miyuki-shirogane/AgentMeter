"""Framework-agnostic environment interface.

AgentMeter is not limited to chat agents. It can also evaluate agents that
take actions inside an environment (a game, a browser, an API, a database, a
workflow, a simulation, ...) and then keep acting based on the resulting
state:

    Agent -> Action -> Environment -> State -> Agent keeps acting

This module defines the *general* contract an environment must satisfy. It
knows nothing about games, players, HP, parties, or any other domain concept.
Those live in concrete implementations (e.g. ``agentmeter.environments.mock_order_api``)
and must never be imported by the core framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

# Sentinel returned by path resolution when a segment is absent. Exposed so
# that evaluators can distinguish "field missing" from "field present with a
# falsy value" (e.g. ``0``, ``False``, ``""``).
UNSET = object()


def resolve_path(data: dict[str, Any], path: str) -> Any:
    """Resolve a (possibly nested) path over a dict-like structure.

    Supports plain dotted paths (``boss.status``), a leading ``$`` selector
    (``$.boss.status``), and numeric segments that index into a list
    (``players.0.name``). Returns :data:`UNSET` when any segment cannot be
    resolved, instead of raising.
    """
    if path.startswith("$."):
        path = path[2:]
    elif path.startswith("$"):
        path = path[1:]

    if path == "":
        return data

    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return UNSET
    return current


def set_by_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Set ``value`` at a dotted ``path``, creating intermediate dicts.

    Used by environments that expose a direct state-mutation action (the
    "cheat" action in the mock game) so such actions can be tested as a
    violation rather than silently ignored.
    """
    if path.startswith("$."):
        path = path[2:]
    elif path.startswith("$"):
        path = path[1:]

    parts = [part for part in path.split(".") if part != ""]
    if not parts:
        return

    current = data
    for part in parts[:-1]:
        if not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


class Action(BaseModel):
    """A single action an agent requests from the environment."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """The environment's response to an :class:`Action`.

    ``reward`` is optional: not every environment rewards actions. ``changes``
    is a (partial) delta describing what changed, ``observations`` are
    human-readable feedback for the agent, and ``done`` signals that the
    environment reached a terminal state.
    """

    reward: float | None = None
    observations: list[str] = Field(default_factory=list)
    changes: dict[str, Any] = Field(default_factory=dict)
    done: bool = False


class State(BaseModel):
    """Generic, structured environment state.

    ``data`` is an arbitrary JSON-like structure; ``get`` resolves a nested
    path so callers (and evaluators) never need to guess field names.
    """

    data: dict[str, Any] = Field(default_factory=dict)

    def get(self, path: str, default: Any = None) -> Any:
        """Resolve a nested ``path``, returning ``default`` when missing."""
        value = resolve_path(self.data, path)
        return default if value is UNSET else value

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def as_dict(self) -> dict[str, Any]:
        """Return the underlying state as a plain dict."""
        return self.data


class Environment(ABC):
    """The interface every environment must implement.

    The three methods are intentionally minimal: reset the world, execute an
    action, and read the current state. No games, players, HP, or other
    domain concepts are assumed. Concrete environments (games, browsers,
    APIs, databases, ...) implement these three methods and nothing more.
    """

    @abstractmethod
    async def reset(self) -> State:
        """Reset the environment to its initial state and return it."""
        raise NotImplementedError

    @abstractmethod
    async def execute_action(self, action: Action) -> ActionResult:
        """Apply ``action`` and return the resulting feedback."""
        raise NotImplementedError

    @abstractmethod
    async def get_state(self) -> State:
        """Return the environment's current state."""
        raise NotImplementedError
