"""Phase 2 demo: agent with tool calls, evaluated deterministically.

Runs a fake agent (implemented as a custom adapter that records tool
events) and evaluates it with tool, argument, and output evaluators.

Pipeline: Agent -> Trace -> Tool Evaluator -> Argument Evaluator -> Output
Evaluator -> Result
"""

import asyncio

from agentmeter import (
    AgentAdapter,
    AgentMessageEvent,
    OutputContainsEvaluator,
    Runner,
    TestCase,
    ToolArgumentEvaluator,
    ToolCallCountEvaluator,
    ToolCallEvent,
    ToolResultEvent,
    Trace,
    UserMessageEvent,
)


class FakeWeatherAgent(AgentAdapter):
    """A hand-rolled agent that calls ``weather`` then answers."""

    async def run(self, message: str) -> Trace:
        trace = Trace(input=message)
        trace.add_event(UserMessageEvent(content=message))

        call = ToolCallEvent(name="weather", arguments={"city": "Beijing"})
        trace.add_event(call)
        trace.add_event(ToolResultEvent(call_id=call.call_id, name="weather", result={"temp": 21}))

        trace.final_output = "北京今天 21℃，晴"
        trace.add_event(AgentMessageEvent(content=trace.final_output))
        return trace


async def main() -> None:
    testcase = TestCase(
        name="weather_agent_test",
        input="北京天气怎么样？",
        agent=FakeWeatherAgent(),
        evaluators=[
            ToolCallCountEvaluator("weather", 1),
            ToolArgumentEvaluator("weather", expected="Beijing", field="city"),
            OutputContainsEvaluator("北京"),
        ],
    )

    result = await Runner().run(testcase)

    for event in result.trace.events:
        print(f"  [{event.type}] {event}")
    print(f"Final output: {result.trace.final_output}")
    for item in result.results:
        print(f"{item.evaluator}: {item.verdict.value} score={item.score} — {item.reason}")
    print(f"PASSED={result.passed} SCORE={result.score}")

    assert result.passed


if __name__ == "__main__":
    asyncio.run(main())
