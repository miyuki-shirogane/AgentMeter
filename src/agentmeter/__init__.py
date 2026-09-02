"""AgentMeter - a pytest-inspired evaluation framework for AI Agents."""

from agentmeter.adapters.base import AgentAdapter
from agentmeter.adapters.callable import CallableAgentAdapter
from agentmeter.adapters.environment import EnvironmentAgentAdapter
from agentmeter.core.models import TestCase
from agentmeter.core.results import AggregateResult, EvaluationResult, TestRunResult
from agentmeter.core.trace import (
    ActionEvent,
    AgentMessageEvent,
    CustomEvent,
    EnvironmentEvent,
    RewardEvent,
    StateChangeEvent,
    StateSnapshotEvent,
    ToolCallEvent,
    ToolResultEvent,
    Trace,
    TraceEvent,
    UserMessageEvent,
)
from agentmeter.core.verdict import Score, Verdict
from agentmeter.environments.base import Action, ActionResult, Environment, State
from agentmeter.evaluators.action import (
    ActionArgumentEvaluator,
    ActionCalledEvaluator,
    ActionNotCalledEvaluator,
    ActionOrderEvaluator,
)
from agentmeter.evaluators.base import Evaluator
from agentmeter.evaluators.judge import LLMJudgeEvaluator
from agentmeter.evaluators.output import (
    OutputContainsEvaluator,
    OutputEqualsEvaluator,
    OutputNotContainsEvaluator,
    OutputRegexEvaluator,
)
from agentmeter.evaluators.state import (
    RewardEvaluator,
    StateEvaluator,
    StateOperator,
    compare_value,
    make_state_predicate,
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
from agentmeter.judge.base import JudgeError, JudgeProvider
from agentmeter.judge.openai import OpenAIJudgeProvider
from agentmeter.judge.result import JudgeResult
from agentmeter.runner.runner import Runner

__all__ = [
    "AgentAdapter",
    "CallableAgentAdapter",
    "EnvironmentAgentAdapter",
    "TestCase",
    "AggregateResult",
    "EvaluationResult",
    "TestRunResult",
    "ActionEvent",
    "AgentMessageEvent",
    "CustomEvent",
    "EnvironmentEvent",
    "RewardEvent",
    "StateChangeEvent",
    "StateSnapshotEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "Trace",
    "TraceEvent",
    "UserMessageEvent",
    "Action",
    "ActionResult",
    "Environment",
    "State",
    "ActionArgumentEvaluator",
    "ActionCalledEvaluator",
    "ActionNotCalledEvaluator",
    "ActionOrderEvaluator",
    "Score",
    "Verdict",
    "Evaluator",
    "LLMJudgeEvaluator",
    "OutputContainsEvaluator",
    "OutputEqualsEvaluator",
    "OutputNotContainsEvaluator",
    "OutputRegexEvaluator",
    "RewardEvaluator",
    "StateEvaluator",
    "StateOperator",
    "compare_value",
    "make_state_predicate",
    "ToolArgumentEvaluator",
    "ToolCallCountEvaluator",
    "ToolCallCountMode",
    "ToolCalledEvaluator",
    "ToolNotCalledEvaluator",
    "ForbiddenToolEvaluator",
    "MaximumToolCallsEvaluator",
    "RequiredToolEvaluator",
    "ToolOrderEvaluator",
    "JudgeError",
    "JudgeProvider",
    "JudgeResult",
    "OpenAIJudgeProvider",
    "Runner",
]

__version__ = "0.4.0"
