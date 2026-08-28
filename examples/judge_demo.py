"""Phase 3 demo: LLM-as-a-Judge with a fake provider.

Demonstrates the judge pipeline without touching a real LLM. To use a real
OpenAI-compatible API, swap ``FakeJudgeProvider`` for ``OpenAIJudgeProvider``
(with ``OPENAI_API_KEY`` set, or Ollama/vLLM ``base_url``).
"""

import asyncio

from agentmeter import (
    JudgeProvider,
    JudgeResult,
    LLMJudgeEvaluator,
    Runner,
    TestCase,
)


class FakeJudgeProvider(JudgeProvider):
    """Pretends to reason; always returns a fixed structured verdict."""

    async def judge(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        return JudgeResult(
            passed=True,
            score=0.95,
            reason="Agent 正确理解了用户意图并完成了查询。",
            evidence=["调用了 weather 工具"],
        )


async def main() -> None:
    async def fake_agent(message: str) -> str:
        return "北京今天 21℃，晴"

    testcase = TestCase(
        name="intent_test",
        input="请帮我查询北京今天的天气",
        agent=fake_agent,
        evaluators=[
            LLMJudgeEvaluator(
                provider=FakeJudgeProvider(),
                criteria="判断 Agent 是否正确理解了用户查询天气的意图",
                expected_behavior="调用天气工具并返回天气信息",
            )
        ],
    )

    result = await Runner().run(testcase)

    for item in result.results:
        print(f"{item.evaluator}: {item.verdict.value} score={item.score}")
        print(f"  reason: {item.reason}")
        print(f"  evidence: {item.metadata.get('evidence')}")
    print(f"PASSED={result.passed}")

    assert result.passed


if __name__ == "__main__":
    asyncio.run(main())
