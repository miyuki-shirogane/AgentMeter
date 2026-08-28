"""LLM-as-a-Judge evaluator.

Used for semantic judgments that deterministic checks cannot express: did
the agent understand the user's intent, stay in character, avoid
hallucination, actually complete the task, etc. Deterministic checks
should always be preferred; reach for this evaluator only when the verdict
needs natural-language reasoning.
"""

from __future__ import annotations

from agentmeter.core.results import EvaluationResult
from agentmeter.core.trace import ToolCallEvent, ToolResultEvent, Trace
from agentmeter.core.verdict import Verdict
from agentmeter.evaluators.base import Evaluator
from agentmeter.judge.base import JudgeError, JudgeProvider
from agentmeter.judge.prompts import build_judge_prompts


def _summarize_trace(trace: Trace) -> str | None:
    lines: list[str] = []
    for event in trace.events:
        if isinstance(event, ToolCallEvent):
            lines.append(f"tool_call: {event.name}({event.arguments})")
        elif isinstance(event, ToolResultEvent):
            lines.append(f"tool_result: {event.name} -> {event.result}")
    return "\n".join(lines) or None


class LLMJudgeEvaluator(Evaluator):
    """Asks a :class:`JudgeProvider` to evaluate an agent's behavior.

    Args:
        provider: the judge provider to call.
        criteria: natural-language evaluation standard, e.g. "判断 Agent 是否
            理解用户希望查询北京天气这一意图".
        expected_behavior: optional description of the desired behavior.
        pass_threshold: optional score in [0, 1]. When set, the verdict is
            PASS iff ``score >= pass_threshold`` (ignoring the judge's own
            ``passed`` field).
    """

    def __init__(
        self,
        provider: JudgeProvider,
        criteria: str,
        *,
        expected_behavior: str | None = None,
        pass_threshold: float | None = None,
    ) -> None:
        if pass_threshold is not None and not 0.0 <= pass_threshold <= 1.0:
            raise ValueError(f"pass_threshold must be in [0, 1], got {pass_threshold}")
        self._provider = provider
        self._criteria = criteria
        self._expected_behavior = expected_behavior
        self._pass_threshold = pass_threshold

    async def evaluate(self, trace: Trace) -> EvaluationResult:
        system_prompt, user_prompt = build_judge_prompts(
            criteria=self._criteria,
            user_input=trace.input,
            agent_output=trace.final_output or "",
            expected_behavior=self._expected_behavior,
            trace_summary=_summarize_trace(trace),
        )
        try:
            judged = await self._provider.judge(system_prompt, user_prompt)
        except JudgeError as exc:
            return EvaluationResult(
                evaluator=type(self).__name__,
                verdict=Verdict.ERROR,
                score=0.0,
                reason=f"judge failed: {exc}",
                metadata={"criteria": self._criteria, "judge_error": str(exc)},
            )

        passed = judged.passed
        if self._pass_threshold is not None:
            passed = judged.score >= self._pass_threshold
        return EvaluationResult(
            evaluator=type(self).__name__,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=judged.score,
            reason=judged.reason,
            metadata={
                "criteria": self._criteria,
                "expected_behavior": self._expected_behavior,
                "pass_threshold": self._pass_threshold,
                "violations": judged.violations,
                "evidence": judged.evidence,
                "provider": type(self._provider).__name__,
            },
        )
