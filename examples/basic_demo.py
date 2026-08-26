"""Minimal AgentMeter demo: a fake agent tested end to end."""

import asyncio

from agentmeter import OutputContainsEvaluator, Runner, TestCase


async def fake_agent(message: str) -> str:
    return "你好，北京！"


async def main() -> None:
    testcase = TestCase(
        name="hello_test",
        input="请向北京打招呼",
        agent=fake_agent,
        evaluators=[OutputContainsEvaluator("北京")],
    )

    result = await Runner().run(testcase)

    print(f"Trace: {result.trace.trace_id}")
    for event in result.trace.events:
        print(f"  [{event.type}] {event.content}")
    print(f"Final output: {result.trace.final_output}")
    for item in result.results:
        print(f"{item.evaluator}: passed={item.passed} score={item.score} reason={item.reason}")
    print(f"PASSED={result.passed} SCORE={result.score}")

    assert result.passed


if __name__ == "__main__":
    asyncio.run(main())
