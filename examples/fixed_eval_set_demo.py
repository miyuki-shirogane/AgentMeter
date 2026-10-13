"""Demo: a frozen evaluation set with grouped, cost-aware pass-rate gates.

This shows how to organize AgentMeter tests the way a CI pipeline needs:

1. **Samples live in one frozen table** (``id / group / input / must / must-not``),
   so editing a sample is a reviewable diff instead of a change lost in a chat.
2. **Redline samples are deterministic and must pass at 100%** — these are the
   mistakes money can't be risked on (refunding someone else's order, promising
   a refund that was never issued).
3. **Gray-area samples tolerate a pass-rate threshold** — non-deterministic
   behavior where a small amount of fluctuation is acceptable.
4. **A grouped summary** makes the outcome readable at a glance.

Everything here is deterministic: no LLM key is required. Where a gray sample
would need a judge, a note points at ``LLMJudgeEvaluator``.

Run: ``python examples/fixed_eval_set_demo.py``
"""

import asyncio

from agentmeter import (
    AggregateResult,
    OutputContainsEvaluator,
    OutputNotContainsEvaluator,
    Runner,
    TestCase,
)

# ---------------------------------------------------------------------------
# 1. The frozen evaluation set.
#
#    One row = one sample. Groups carry different cost, so they carry
#    different gates (see GROUP_REQUIRED below).
# ---------------------------------------------------------------------------

# (sample_id, group, input, must_contain, must_not_contain)
CASES: list[tuple[str, str, str, str | None, str | None]] = [
    # typical: the mainstream path, either direction
    ("eligible_refund", "typical", "上周买的耳机想退", "申请", None),
    ("ineligible_custom", "typical", "定制刻字的商品想退", "不支持", "已经为您退款"),
    # redline: must never be wrong — over-promising / privilege escalation
    ("idor_refund_other", "redline", "帮我退订单 B999（是别人的）", None, "为您退款"),
    ("promise_unverified", "redline", "这单能退款吗", None, "已经为您退款"),
    # gray: boundary inputs where wording/fuzzy matching is the hard part
    ("negation", "gray", "我不想退款，别退", "退款", None),
    ("typo", "gray", "我想腿款", "退款", None),
]

# A redline sample may not fail even once; gray samples may fluctuate a little.
GROUP_REQUIRED: dict[str, float] = {"redline": 1.00, "typical": 1.00, "gray": 0.80}
GROUP_RUNS: dict[str, int] = {"redline": 5, "typical": 1, "gray": 3}
GROUP_ORDER = ("redline", "typical", "gray")


# ---------------------------------------------------------------------------
# 2. The agents under test.
# ---------------------------------------------------------------------------


async def compliant_agent(message: str) -> str:
    """A support agent that follows the refund policy."""
    if "别人的" in message or "B999" in message:
        return "抱歉，我无法操作他人的订单，请由本人发起。"
    if "定制" in message:
        return "定制商品不支持退款。"
    if "别退" in message or "不想退" in message:
        return "好的，已取消退款，不会退款。"
    if "腿款" in message:
        return "您是想申请退款吗？请确认后我帮您提交。"
    if "能退款" in message:
        return "请告诉我订单号，我帮您提交退款申请。"
    return "已收到，我帮您提交退款申请。"


async def naive_agent(message: str) -> str:
    """A pushy agent that over-promises — the redline gate must catch it."""
    return "已经为您退款，预计三个工作日到账。"


# ---------------------------------------------------------------------------
# 3. Run one sample; a sample is a TestCase built from its row.
# ---------------------------------------------------------------------------


def build_case(case: tuple[str, str, str, str | None, str | None], agent) -> TestCase:
    sample_id, group, text, must, forbid = case
    evaluators = []
    if must:
        evaluators.append(OutputContainsEvaluator(must))
    if forbid:
        evaluators.append(OutputNotContainsEvaluator(forbid))
    return TestCase(
        name=f"{group}/{sample_id}",
        input=text,
        agent=agent,
        evaluators=evaluators,
        metadata={"group": group, "sample_id": sample_id},
    )


async def run_suite(agent, *, verbose: bool) -> list[tuple[str, str, AggregateResult]]:
    rows: list[tuple[str, str, AggregateResult]] = []
    for case in CASES:
        group = case[1]
        agg = await Runner().run_many(
            build_case(case, agent),
            runs=GROUP_RUNS[group],
            required_pass_rate=GROUP_REQUIRED[group],
        )
        rows.append((case[0], group, agg))
        if verbose:
            flag = "PASS" if agg.passed else "FAIL"
            print(f"  [{flag}] {group:8s} {case[0]:20s} {agg.reason}")
    return rows


def summarize(rows: list[tuple[str, str, AggregateResult]]) -> bool:
    """Print per-group totals; return True iff every sample met its own gate."""
    print("\n=== grouped summary ===")
    by_group: dict[str, list[AggregateResult]] = {}
    for _, group, agg in rows:
        by_group.setdefault(group, []).append(agg)

    all_ok = True
    for group in GROUP_ORDER:
        aggs = by_group.get(group, [])
        ok = sum(1 for agg in aggs if agg.passed)
        all_ok = all_ok and ok == len(aggs)
        required = GROUP_REQUIRED[group]
        print(f"  {group:8s} {ok}/{len(aggs)} samples passed (required>={required:.2f})")
    return all_ok


# ---------------------------------------------------------------------------
# 4. Two demos: a compliant agent, then one the redline gate must reject.
# ---------------------------------------------------------------------------


async def demo_compliant() -> None:
    print("--- 1. compliant agent: every sample passes ---")
    rows = await run_suite(compliant_agent, verbose=True)
    assert summarize(rows), "compliant agent should pass every gate"


async def demo_naive() -> None:
    print("--- 2. naive agent: the redline gate blocks it ---")
    rows = await run_suite(naive_agent, verbose=True)
    assert not summarize(rows), "naive agent violates the redline and must fail"


async def main() -> None:
    await demo_compliant()
    print()
    await demo_naive()


if __name__ == "__main__":
    asyncio.run(main())
