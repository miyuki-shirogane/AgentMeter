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

The project is being developed incrementally, starting from the core evaluation engine and gradually adding:

- LLM Judge
- Environment evaluation
- Security testing
- CLI
- pytest integration
- Reporting

## License

MIT