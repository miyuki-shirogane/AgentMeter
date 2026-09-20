# AgentMeter

[![CI](https://github.com/miyuki-shirogane/AgentMeter/actions/workflows/ci.yml/badge.svg)](https://github.com/miyuki-shirogane/AgentMeter/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/miyuki-shirogane/AgentMeter/branch/main/graph/badge.svg)](https://codecov.io/gh/miyuki-shirogane/AgentMeter)

[English](README.md) | 简体中文

> 一个受 pytest 启发的 AI Agent 评测框架。

AgentMeter 是一个基于 Python 的 AI Agent 测试与评测框架。

传统的自动化测试通常长这样：

```text
Input
  ↓
Program
  ↓
Assertion
  ↓
PASS / FAIL
```

AI Agent 不一样：它的输出和行为是概率性的。

AgentMeter 关注的是**观察并评测 Agent 的行为与执行轨迹**：

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

## 快速开始

```bash
uv sync                      # 或者：pip install -e .
python examples/basic_demo.py
python examples/tool_demo.py
python examples/judge_demo.py
python examples/mixed_demo.py
python examples/environment_demo.py
python examples/security_demo.py
```

这些 demo 是层层递进的：最小化的输出断言 → 工具调用断言 → LLM 裁判 →
把各类 evaluator 混在一个用例里 → 在环境中行动的 Agent → 鲁棒性/安全演示。

### 配置 LLM 裁判

裁判走的是 OpenAI 兼容的 HTTP 接口，所以 OpenAI、DeepSeek、Ollama、vLLM……
都能用。凭据从仓库根目录的 `.env` 读取（**永不提交**）：

```bash
cp .env.example .env    # 然后填上 DEEPSEEK_API_KEY
```

| 变量 | 默认值 |
|---|---|
| `DEEPSEEK_API_KEY` | – |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` |

没有 key 时，`mixed_demo.py` 会**直接拒绝运行**，而不是退化成假裁判。

### 结果语义

- **单个 evaluator**：产出 `EvaluationResult`，包含 `verdict`（PASS / FAIL /
  ERROR）与 [0, 1] 区间的 `score`。确定性 evaluator 的 score 只有 0 或 1；
  LLM 裁判会返回连续分数外加 `reason`。
- **整体 `verdict`**：任一 ERROR → ERROR，任一 FAIL → FAIL，否则 PASS。
  一个确定性断言失败，整轮就算失败。
- **整体 `score`**：所有 evaluator 的平均分（仅供参考，**不驱动判定**）。
- **重复运行**：`Runner.run_many(testcase, runs, required_pass_rate)`
  返回 `AggregateResult`，含通过率、错误率与平均分。

## 技术栈

- Python 3.11+
- Pydantic
- pytest / pytest-asyncio
- 异步优先架构
- OpenAI 兼容的 LLM API
- 类型注解
- Ruff
- GitHub Actions

## 能评测什么

### 确定性

- Agent 的最终输出
- 工具调用
- 工具参数
- 工具调用顺序
- 禁止的动作
- 状态断言（最终取值：嵌套路径、比较运算、自定义谓词）
- 状态历史（增量 / 迁移 / 单调性 / 不可变性）
- 环境指标（带名字的数值反馈，如 `reward` / `quality`）
- 环境动作（必需 / 禁止 / 参数 / 顺序）

### 基于 LLM

- 用户意图
- 语义正确性
- OOC / 角色一致性
- 幻觉
- 行为质量
- 抗提示注入
- 抗越狱

### 基于环境

AgentMeter 不局限于聊天型 Agent。

它同样能评测与下列对象交互的 Agent：

- 游戏
- 浏览器
- API
- 工作流
- 仿真环境
- 自定义环境

通用循环是 `Agent -> Action -> Environment -> State -> Agent keeps
acting`。AgentMeter 提供接口（`Environment`）、通用结构化状态模型
（`State`）、驱动适配器（`EnvironmentAgentAdapter`），以及一批检查所产生
轨迹的确定性 evaluator（`StateEvaluator`、`EnvironmentMetricEvaluator`，
以及 `Action*` 系列）。

`State` 描述**世界长什么样**（实体及其字段）；`ActionResult.metrics`
承载**这一步干得好不好**（带名字的数值反馈，如 `{"reward": 299.0}`）。
同一条事实应当只落在两者之一 —— 参考实现 `OrderEnvironment` 把退款的
reward 报成 metric，**绝不重复塞进 state**。

最容易照抄的具体例子就是一个有状态的 REST API：Agent 能调的每个端点变成
一个 `Action`，服务端返回的 JSON 变成 `State`。见
`examples/environment_demo.py`（使用 `OrderEnvironment`，一个包在电商
下单/退款后端外层的、httpx 风格的薄封装）：它的 `set_total`、
`refund_other_order`、`view_other_order` 分别代表禁止 / 作弊 / IDOR 端点。
具体环境独立成模块，**core 永不 import 它**。

### 鲁棒性 / 安全

AgentMeter 不声称能让 Agent 变安全。它是个测试框架：给你工具去跑对抗性输入，
然后检查 Agent 的**行为** —— 通过禁止动作、禁止工具、参数、状态、轨迹这些
evaluator，外加一个 LLM 裁判。一条核心原则是：Agent 永远改不了 TestCase、
Evaluator、裁判标准或 PASS/FAIL 规则 —— Agent 说"把结果标成 PASS"只会被
当成普通输出，判定依然由 evaluator 决定。

## Evaluator 速查表

按**你要断言什么**来挑 evaluator。参数/状态路径支持普通点号路径、前置 `$`、
以及数字列表下标：`options.language`、`$.options.language`、`items.0.sku`。

**最终回答** —— 读 `final_output`

| 我想断言…… | 用哪个 Evaluator |
|---|---|
| 回答等于 `X` | `OutputEqualsEvaluator("X")` |
| 回答包含 `X` | `OutputContainsEvaluator("X")` |
| 回答**不**包含 `X` | `OutputNotContainsEvaluator("X")` |
| 回答匹配正则 | `OutputRegexEvaluator(r"...")` |

**工具调用** —— 用于聊天 / 工具调用型 Agent，读 `ToolCallEvent`

| 我想断言…… | 用哪个 Evaluator |
|---|---|
| 工具 `t` 被调用了 | `ToolCalledEvaluator("t")` |
| 工具 `t` 从未被调用 | `ToolNotCalledEvaluator("t")` |
| 工具 `t` 被调用了恰好 / 至少 / 至多 N 次 | `ToolCallCountEvaluator("t", N, mode=…)` |
| 工具 `t` 的参数路径 `== value` | `ToolArgumentEvaluator("t", expected=value, field=…)` |
| 工具 `t` 的整个参数字典等于 `{...}` | `ToolArgumentEvaluator("t", expected={...})` |
| 工具调用顺序符合给定相对次序 | `ToolOrderEvaluator([...])` |
| 工具调用总数不超过 N | `MaximumToolCallsEvaluator(N)` |

`RequiredToolEvaluator` / `ForbiddenToolEvaluator` 是
`ToolCalledEvaluator` / `ToolNotCalledEvaluator` 的语义别名。

**环境动作** —— 用于在 `Environment` 内行动的 Agent，读 `ActionEvent`

| 我想断言…… | 用哪个 Evaluator |
|---|---|
| 动作 `a` 被执行了 | `ActionCalledEvaluator("a")` |
| 动作 `a` 从未被执行（禁止 / 作弊） | `ActionNotCalledEvaluator("a")` |
| 动作 `a` 的参数路径 `== value` | `ActionArgumentEvaluator("a", expected=value, field=…)` |
| 动作执行顺序符合给定相对次序 | `ActionOrderEvaluator([...])` |

**环境状态 —— 最终取值**

| 我想断言…… | 用哪个 Evaluator |
|---|---|
| 最终状态的 `path` 满足某个比较 | `StateEvaluator("status", "eq", "refunded")` |
| 某个环境指标达到阈值 | `EnvironmentMetricEvaluator("reward", "gte", 299)` |

**环境状态 —— 随时间变化** —— 能抓住"数值被短暂改到非法值、随后又改回来"
这类最终值检查看不见的情况

| 我想断言…… | 用哪个 Evaluator |
|---|---|
| `path` 的每一次上报变更都合法（如"total 从不超过 5000"） | `StateChangeEvaluator("total", "lte", 5000)` |
| `path` 只沿允许的迁移前进 | `StateTransitionEvaluator("status", [("draft","paid"),("paid","refunded")])` |
| 数值 `path` 从不反向 | `StateMonotonicEvaluator("processed")` |
| `path` 全程从未被改动 | `StateUnchangedEvaluator("owner")` |

`StateUnchangedEvaluator` 比"禁止某个动作"更强：Agent 常常能**借合法端点**
达到非法状态（用优惠券而不是管理员端点把价格压下去），而"状态从未改变"这条
断言不依赖它用了哪个端点。

`operator` 取值为 `eq / ne / gt / gte / lt / lte / exists`，也可以传
`predicate=lambda value: ...` 做任意逻辑。

**语义** —— 非确定性，使用 LLM 裁判

| 我想断言…… | 用哪个 Evaluator |
|---|---|
| 意图 / 语义正确性 / OOC / 幻觉 | `LLMJudgeEvaluator(provider, criteria=...)` |

**组合断言** —— 不需要为每种组合新增类

| 我想断言…… | 用哪个 Evaluator |
|---|---|
| 某个断言的取反 | `NotEvaluator(...)` |
| 若干断言全部成立 | `AllOfEvaluator([...])` |
| 若干断言至少一个成立（如"已退款**或**已取消"） | `AnyOfEvaluator([...])` |

**工具族 vs 动作族**：Agent 调用**工具**（`ToolCallEvent`）时用 `Tool*`
系列；Agent 在 `Environment` 内**行动**（`ActionEvent`）时用 `Action*` 系列。
两者分开记录，因为工具调用是 Agent → 工具，而动作是 Agent → 环境。

## 设计

AgentMeter 把这几层分开：

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

core 框架与具体领域无关。

游戏、越狱测试、以及具体的 Agent 框架都被当作**扩展**，而不是核心业务逻辑。

## 状态

🚧 早期开发中

已实现：
- [x] 核心评测引擎（Trace、TestCase、Runner）
- [x] 确定性 evaluator（输出、工具、工具参数、轨迹）
- [x] 基于 OpenAI 兼容服务的 LLM-as-a-Judge
- [x] 重复运行与通过率聚合（`run_many`）
- [x] 环境接口 + State 模型 + 环境适配器
- [x] 状态 / 指标 / 动作 evaluator
- [x] 状态历史 evaluator（增量 / 迁移 / 单调性 / 不可变性）
- [x] 组合 evaluator（`NotEvaluator` / `AllOfEvaluator` / `AnyOfEvaluator`）
- [x] Mock 订单/退款 API 环境（仅用于测试/demo，与 core 隔离）
- [x] 鲁棒性 / 安全评测（禁止动作/工具、参数、OOC、作弊）

计划中：
- [ ] CLI
- [ ] pytest 集成
- [ ] 报告

## 许可证

MIT
