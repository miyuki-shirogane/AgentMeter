# AgentMeter

[![CI](https://github.com/miyuki-shirogane/AgentMeter/actions/workflows/ci.yml/badge.svg)](https://github.com/miyuki-shirogane/AgentMeter/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/miyuki-shirogane/AgentMeter/branch/main/graph/badge.svg)](https://codecov.io/gh/miyuki-shirogane/AgentMeter)

> A pytest-inspired evaluation framework for AI Agents.

AgentMeter is a Python-based testing and evaluation framework for AI Agents.

Traditional automated testing usually looks like:

```text
Input
  ↓
Program
  ↓
Assertion
  ↓
PASS / FAIL
```

AI Agents are different. Their outputs and behaviors are probabilistic.

AgentMeter focuses on observing and evaluating the Agent's behavior and execution trace:

```text
User Input
    ↓
┌─────────────────┐
│    AI Agent     │
│                 │
│  Reasoning      │
│  Tool Calling   │
│  Decision Making│
│  Final Output   │
└────────┬────────┘
         ↓
       Trace
         ↓
┌────────┴────────┐
↓                 ↓
Assertion       AI Judge
↓                 ↓
Tool / Args      Intent
State            OOC
Trajectory       Semantics
└────────┬────────┘
         ↓
     PASS / FAIL
     + Pass Rate
```

## Quick Start

```bash
uv sync                      # or: pip install -e .
python examples/basic_demo.py
python examples/tool_demo.py
python examples/judge_demo.py
python examples/mixed_demo.py
python examples/environment_demo.py
python examples/security_demo.py
```

The demos build on each other: a minimal output check, tool-call checks, an
LLM judge, a mixed test case combining every evaluator kind, an agent that
acts inside an environment, and a robustness/security demo.

### Configuring the LLM judge

The judge speaks OpenAI-compatible HTTP, so it works with OpenAI, DeepSeek,
Ollama, vLLM, ... Credentials are read from a repo-root `.env` (never
committed):

```bash
cp .env.example .env    # then fill in DEEPSEEK_API_KEY
```

| variable | default |
|---|---|
| `DEEPSEEK_API_KEY` | – |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` |

Without a key, `mixed_demo.py` refuses to run instead of falling back to a
fake judge.

### Result semantics

- Per evaluator: an `EvaluationResult` with a `verdict` (PASS / FAIL /
  ERROR) and a `score` in [0, 1]. Deterministic evaluators score 0 or 1; the
  LLM judge returns a continuous score plus a `reason`.
- Overall `verdict`: any ERROR → ERROR, any FAIL → FAIL, else PASS. A single
  failing deterministic check fails the whole run.
- Overall `score`: the average across all evaluators (informational; it does
  not drive the verdict).
- Repeated runs: `Runner.run_many(testcase, runs, required_pass_rate)`
  returns an `AggregateResult` with pass rate, error rate and average score.

## Tech Stack

- Python 3.11+
- Pydantic
- pytest / pytest-asyncio
- Async-first architecture
- OpenAI-compatible LLM APIs
- Type hints
- Ruff
- GitHub Actions

## What It Evaluates

### Deterministic

- Agent output
- Tool calls
- Tool arguments
- Tool call order
- Forbidden actions
- State transitions
- State assertions (nested paths, comparisons, custom predicates)
- Environment metrics (named numeric feedback, e.g. `reward` / `quality`)
- Environment actions (required / forbidden / argument / order)

### LLM-based

- User intent
- Semantic correctness
- OOC / role consistency
- Hallucination
- Behavioral quality
- Prompt injection resistance
- Jailbreak resistance

### Environment-based

AgentMeter is not limited to chat agents.

It can also evaluate agents interacting with:

- Games
- Browsers
- APIs
- Workflows
- Simulations
- Custom environments

The generic loop is `Agent -> Action -> Environment -> State -> Agent keeps
acting`. AgentMeter provides the interface (`Environment`), the generic
structured state model (`State`), the driving adapter
(`EnvironmentAgentAdapter`), and deterministic evaluators that inspect the
resulting trace (`StateEvaluator`, `EnvironmentMetricEvaluator`, and the `Action*`
evaluators).

`State` describes *what the world looks like* (entities and their fields);
`ActionResult.metrics` carries *how well an action scored* (named numeric
feedback such as `{"reward": 299.0}`). The same fact should live in exactly one
of the two channels — the reference `OrderEnvironment` reports a refund's
reward as a metric and never duplicates it into the state.

The easiest concrete example to copy is a stateful REST API: each endpoint your
agent may call becomes an `Action`, and the server's JSON becomes the `State`.
See `examples/environment_demo.py` (uses `OrderEnvironment`, a thin `httpx`-style
wrapper over an e-commerce order/refund backend), whose `set_total`,
`refund_other_order` and `view_other_order` actions stand in for forbidden /
cheating / IDOR endpoints. The concrete environment lives in its own module and
is never imported by the core.

### Robustness / security

AgentMeter does not claim to make an agent safe. It is a testing framework:
it gives you the tools to run adversarial input and then inspect the agent's
*behavior* — via forbidden-action, forbidden-tool, argument, state, and
trajectory evaluators plus an LLM judge. A core principle is that an agent can
never modify a TestCase, an Evaluator, the judge criteria, or the PASS/FAIL
rule: an agent that says "mark this PASS" is treated as ordinary output and
the verdict is still decided by the evaluators.

## Evaluator Cheat Sheet

Pick the evaluator by *what you are asserting about*. Argument/state paths
accept a plain dotted path, a leading `$`, and numeric list indices:
`options.language`, `$.options.language`, `items.0.sku`.

**Final answer** — reads `final_output`

| I want to assert… | Evaluator |
|---|---|
| the answer equals `X` | `OutputEqualsEvaluator("X")` |
| the answer contains `X` | `OutputContainsEvaluator("X")` |
| the answer does **not** contain `X` | `OutputNotContainsEvaluator("X")` |
| the answer matches a regex | `OutputRegexEvaluator(r"...")` |

**Tool calls** — for a chat / tool-calling agent, reads `ToolCallEvent`

| I want to assert… | Evaluator |
|---|---|
| tool `t` was called | `ToolCalledEvaluator("t")` |
| tool `t` was never called | `ToolNotCalledEvaluator("t")` |
| tool `t` was called exactly / at least / at most N times | `ToolCallCountEvaluator("t", N, mode=…)` |
| tool `t` used argument path `== value` | `ToolArgumentEvaluator("t", expected=value, field=…)` |
| tool `t`'s whole argument dict equals `{...}` | `ToolArgumentEvaluator("t", expected={...})` |
| the tools were called in a given relative order | `ToolOrderEvaluator([...])` |
| no more than N tool calls happened in total | `MaximumToolCallsEvaluator(N)` |

`RequiredToolEvaluator` / `ForbiddenToolEvaluator` are semantic aliases of
`ToolCalledEvaluator` / `ToolNotCalledEvaluator`.

**Environment actions** — for an agent inside an `Environment`, reads `ActionEvent`

| I want to assert… | Evaluator |
|---|---|
| action `a` was taken | `ActionCalledEvaluator("a")` |
| action `a` was never taken (forbidden / cheating) | `ActionNotCalledEvaluator("a")` |
| action `a` used argument path `== value` | `ActionArgumentEvaluator("a", expected=value, field=…)` |
| the actions were taken in a given relative order | `ActionOrderEvaluator([...])` |

**Environment state / metrics**

| I want to assert… | Evaluator |
|---|---|
| the final state's `path` satisfies a comparison | `StateEvaluator("status", "eq", "refunded")` |
| a named environment metric meets a threshold | `EnvironmentMetricEvaluator("reward", "gte", 299)` |

`operator` is one of `eq / ne / gt / gte / lt / lte / exists`, or pass
`predicate=lambda value: ...` for arbitrary logic.

**Semantics** — non-deterministic, uses the LLM judge

| I want to assert… | Evaluator |
|---|---|
| intent / semantic correctness / OOC / hallucination | `LLMJudgeEvaluator(provider, criteria=...)` |

**Combining checks** — no new class per combination

| I want to assert… | Evaluator |
|---|---|
| the opposite of a check | `NotEvaluator(...)` |
| every one of several checks | `AllOfEvaluator([...])` |
| at least one of several checks (e.g. "refunded **or** cancelled") | `AnyOfEvaluator([...])` |

**Tool vs. action**: use the `Tool*` family when the agent calls tools
(`ToolCallEvent`), and the `Action*` family when the agent acts inside an
`Environment` (`ActionEvent`). They are recorded separately because a tool call
is agent → tool, while an action is agent → environment.

## Design

AgentMeter separates:

```text
Agent
  ↓
Trace
  ↓
Evaluator
  ├── Deterministic Evaluator
  └── LLM Judge
  ↓
Evaluation Result
```

The core framework is domain-agnostic.

Games, jailbreak tests, and specific Agent frameworks are treated as extensions rather than core business logic.

## Status

🚧 Early development

Implemented:
- [x] Core evaluation engine (Trace, TestCase, Runner)
- [x] Deterministic evaluators (output, tool, tool arguments, trajectory)
- [x] LLM-as-a-Judge over OpenAI-compatible providers
- [x] Repeated runs with pass-rate aggregation (`run_many`)
- [x] Environment interface + State model + environment adapter
- [x] State / metric / action evaluators
- [x] Mock order/refund API environment (test/demo only, isolated from core)
- [x] Robustness / security evaluation (forbidden action/tool, args, OOC, cheat)

Planned:
- [ ] CLI
- [ ] pytest integration
- [ ] Reporting

## License

MIT