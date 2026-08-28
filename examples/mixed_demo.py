"""Demo: mixing base / tool / LLM-judge evaluators in one TestCase.

Scenario: an agent that queries weather for Beijing and Shanghai, then
compares them. The single TestCase combines:

- base (deterministic): output contains checks
- tool (deterministic): called / count / argument checks
- trajectory (deterministic): maximum tool calls
- LLM judge (semantic): did it really complete the multi-step task?

The LLM judge calls a real OpenAI-compatible provider (DeepSeek). Credentials
are read from a ``.env`` file at the repo root (``DEEPSEEK_API_KEY`` /
``DEEPSEEK_BASE_URL`` / ``DEEPSEEK_MODEL``), see ``_load_dotenv``.

Result semantics: any evaluator ERROR -> overall ERROR, else any FAIL ->
overall FAIL, else PASS. Each evaluator is evaluated independently.
"""

import asyncio
import os
from pathlib import Path

from agentmeter import (
    AgentAdapter,
    AgentMessageEvent,
    JudgeProvider,
    LLMJudgeEvaluator,
    MaximumToolCallsEvaluator,
    OpenAIJudgeProvider,
    OutputContainsEvaluator,
    Runner,
    TestCase,
    ToolArgumentEvaluator,
    ToolCallCountEvaluator,
    ToolCalledEvaluator,
    ToolCallEvent,
    ToolResultEvent,
    Trace,
    UserMessageEvent,
)


class WeatherCompareAgent(AgentAdapter):
    """Queries weather for Beijing and Shanghai, then compares them."""

    async def run(self, message: str) -> Trace:
        trace = Trace(input=message)
        trace.add_event(UserMessageEvent(content=message))

        temps = {"Beijing": 21, "Shanghai": 24}
        for city, temp in temps.items():
            call = ToolCallEvent(name="weather", arguments={"city": city})
            trace.add_event(call)
            trace.add_event(
                ToolResultEvent(call_id=call.call_id, name="weather", result={"temp": temp})
            )

        trace.final_output = "北京 21℃，上海 24℃，上海更热。"
        trace.add_event(AgentMessageEvent(content=trace.final_output))
        return trace


def _load_dotenv() -> None:
    """Load ``KEY=VALUE`` pairs from the repo-root ``.env`` into os.environ.

    Deliberately tiny and dependency-free so the demo needs no extra package.
    Values already present in the environment are kept (setdefault).
    """
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def build_judge_provider() -> JudgeProvider:
    """Real DeepSeek (OpenAI-compatible) judge, configured from ``.env``.

    Requires ``DEEPSEEK_API_KEY``; there is deliberately no fake fallback so
    the demo always exercises the real model. To point at a different
    provider, edit ``DEEPSEEK_BASE_URL`` / ``DEEPSEEK_MODEL`` in ``.env``.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key or "REPLACE_ME" in api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置。请在仓库根目录 .env 中填写 DEEPSEEK_API_KEY=sk-... "
            "（当前为占位符）"
        )
    return OpenAIJudgeProvider(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=api_key,
    )


JUDGE_CRITERIA = """
判断 Agent 是否完整完成了"比较北京和上海今天谁更热"的任务。

必须完成的关键步骤（缺任一步即 FAIL）：
1. 查询了北京气温
2. 查询了上海气温
3. 明确比较了两者
4. 给出了结论（哪边更热）

判定规则：
- 若 Agent 回复中的数字与工具返回结果矛盾，视为幻觉，FAIL。
- 只调用了一个城市的天气即下结论，视为未完成任务，FAIL。
"""


def build_testcase(agent: AgentAdapter) -> TestCase:
    return TestCase(
        name="weather_compare_mixed",
        input="比较一下北京和上海今天谁更热",
        agent=agent,
        evaluators=[
            # ---- tool（确定性）----
            ToolCalledEvaluator("weather"),
            ToolCallCountEvaluator("weather", 2),
            ToolArgumentEvaluator("weather", expected="Beijing", field="city"),
            ToolArgumentEvaluator("weather", expected="Shanghai", field="city"),
            # ---- trajectory（确定性）----
            MaximumToolCallsEvaluator(2),
            # ---- base output（确定性）----
            OutputContainsEvaluator("北京"),
            OutputContainsEvaluator("上海"),
            # ---- LLM judge（语义）----
            LLMJudgeEvaluator(
                provider=build_judge_provider(),
                criteria=JUDGE_CRITERIA,
                expected_behavior="应查询两城气温并给出明确比较结论",
                pass_threshold=0.8,
            ),
        ],
    )


def print_trace(trace: Trace) -> None:
    """Dump the full trace: input, every event (all fields), final output."""
    print("=== trace detail ===")
    print(f"trace_id:     {trace.trace_id}")
    print(f"input:        {trace.input!r}")
    print("events:")
    for i, event in enumerate(trace.events, start=1):
        print(f"  [{i}] {event.type}")
        for field, value in event.model_dump(exclude={"type"}).items():
            print(f"      {field}: {value!r}")
    print(f"final_output: {trace.final_output!r}")
    if trace.metadata:
        print(f"metadata:     {trace.metadata}")
    print()


async def main() -> None:
    _load_dotenv()
    result = await Runner().run(build_testcase(WeatherCompareAgent()))

    print_trace(result.trace)
    print("--- evaluations ---")
    for item in result.results:
        print(f"{item.evaluator:28s} {item.verdict.value:6s} score={item.score}  {item.reason}")
    print(f"overall: {result.verdict.value} (score={result.score:.3f})")
    assert result.passed


if __name__ == "__main__":
    asyncio.run(main())
