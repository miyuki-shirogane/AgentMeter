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
```

The demos build on each other: a minimal output check, tool-call checks, an
LLM judge, and a mixed test case combining every evaluator kind.

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
- Rewards

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

Planned:
- [ ] Environment evaluation
- [ ] Security testing (prompt injection / jailbreak)
- [ ] CLI
- [ ] pytest integration
- [ ] Reporting

## License

MIT