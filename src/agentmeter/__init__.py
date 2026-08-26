"""AgentMeter - a pytest-inspired evaluation framework for AI Agents."""

from agentmeter.adapters.base import AgentAdapter
from agentmeter.adapters.callable import CallableAgentAdapter
from agentmeter.core.models import TestCase
from agentmeter.core.results import AggregateResult, EvaluationResult, TestRunResult
from agentmeter.core.trace import (
    AgentMessageEvent,
    CustomEvent,
    ToolCallEvent,
    ToolResultEvent,
    Trace,
    TraceEvent,
    UserMessageEvent,
)
from agentmeter.core.verdict import Score, Verdict
from agentmeter.evaluators.base import Evaluator
from agentmeter.evaluators.output import (
    OutputContainsEvaluator,
    OutputEqualsEvaluator,
    OutputNotContainsEvaluator,
    OutputRegexEvaluator,
)
from agentmeter.evaluators.tool import (
    ToolArgumentEvaluator,
    ToolCallCountEvaluator,
    ToolCallCountMode,
    ToolCalledEvaluator,
    ToolNotCalledEvaluator,
)
from agentmeter.evaluators.trajectory import (
    ForbiddenToolEvaluator,
    MaximumToolCallsEvaluator,
    RequiredToolEvaluator,
    ToolOrderEvaluator,
)
from agentmeter.runner.runner import Runner

__all__ = [
    "AgentAdapter",
    "CallableAgentAdapter",
    "TestCase",
    "AggregateResult",
    "EvaluationResult",
    "TestRunResult",
    "AgentMessageEvent",
    "CustomEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "Trace",
    "TraceEvent",
    "UserMessageEvent",
    "Score",
    "Verdict",
    "Evaluator",
    "OutputContainsEvaluator",
    "OutputEqualsEvaluator",
    "OutputNotContainsEvaluator",
    "OutputRegexEvaluator",
    "ToolArgumentEvaluator",
    "ToolCallCountEvaluator",
    "ToolCallCountMode",
    "ToolCalledEvaluator",
    "ToolNotCalledEvaluator",
    "ForbiddenToolEvaluator",
    "MaximumToolCallsEvaluator",
    "RequiredToolEvaluator",
    "ToolOrderEvaluator",
    "Runner",
]

__version__ = "0.2.0"
