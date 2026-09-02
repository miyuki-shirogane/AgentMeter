"""Environment package.

Exposes the framework-agnostic environment contract. Concrete environments
(like the Mock Game) live in submodules and are *not* re-exported here, so
that game-specific rules never leak into the framework's public surface.
"""

from agentmeter.environments.base import (
    Action,
    ActionResult,
    Environment,
    State,
    resolve_path,
    set_by_path,
)

__all__ = [
    "Action",
    "ActionResult",
    "Environment",
    "State",
    "resolve_path",
    "set_by_path",
]
