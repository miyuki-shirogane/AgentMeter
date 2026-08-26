"""Test runner.

The runner orchestrates the full pipeline:

    TestCase
      → execute agent (adapter) → Trace
      → run each evaluator       → EvaluationResult[]
      → aggregate                → TestRunResult

Errors are contained per unit: an exception raised by the agent, or by an
evaluator, is turned into an ERROR verdict rather than crashing the whole
run. ERROR is never converted into FAIL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentmeter.core.results import AggregateResult, EvaluationResult, TestRunResult
from agentmeter.core.verdict import Verdict

if TYPE_CHECKING:
    from agentmeter.core.models import TestCase


class Runner:
    """Executes test cases and aggregates evaluation results."""

    async def run(self, testcase: TestCase) -> TestRunResult:
        try:
            trace = await testcase.agent.run(testcase.input)
        except Exception as exc:  # noqa: BLE001 - surface any agent failure as ERROR
            return TestRunResult(
                testcase_name=testcase.name,
                trace=None,
                results=[
                    EvaluationResult(
                        evaluator="agent",
                        verdict=Verdict.ERROR,
                        score=0.0,
                        reason=f"agent raised {type(exc).__name__}: {exc}",
                        metadata={"error_type": type(exc).__name__},
                    )
                ],
            )

        results: list[EvaluationResult] = []
        for evaluator in testcase.evaluators:
            try:
                results.append(await evaluator.evaluate(trace))
            except Exception as exc:  # noqa: BLE001 - surface any evaluator failure as ERROR
                results.append(
                    EvaluationResult(
                        evaluator=type(evaluator).__name__,
                        verdict=Verdict.ERROR,
                        score=0.0,
                        reason=f"evaluator raised {type(exc).__name__}: {exc}",
                        metadata={"error_type": type(exc).__name__},
                    )
                )
        return TestRunResult(
            testcase_name=testcase.name,
            trace=trace,
            results=results,
        )

    async def run_many(
        self,
        testcase: TestCase,
        runs: int,
        *,
        required_pass_rate: float | None = None,
    ) -> AggregateResult:
        """Run ``testcase`` ``runs`` times and aggregate the statistics.

        Each run produces an independent :class:`TestRunResult`. Runs are
        executed sequentially; every run re-invokes the adapter, so
        framework adapters must guarantee a fresh execution context per run.
        """
        run_results = [await self.run(testcase) for _ in range(runs)]
        return AggregateResult.from_runs(
            testcase.name,
            run_results,
            required_pass_rate=required_pass_rate,
        )
