# AgentMeter

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
- Rewards
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
resulting trace (`StateEvaluator`, `RewardEvaluator`, and the `Action*`
evaluators).

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
- [x] State / reward / action evaluators
- [x] Mock order/refund API environment (test/demo only, isolated from core)
- [x] Robustness / security evaluation (forbidden action/tool, args, OOC, cheat)

Planned:
- [ ] CLI
- [ ] pytest integration
- [ ] Reporting

## License

MIT