# 读懂 Trace：AgentMeter 的数据枢纽

> AgentMeter 的核心很"空"：只关心 **Agent → Trace → Evaluator** 这条管线。而 `Trace` 是其中的数据枢纽——**所有被测对象的行为都被翻译成它，所有 evaluator 的断言都从它读起**。本文把它讲透：它是什么、长什么样、从哪来、到哪去、怎么读。
>
> 配套可运行案例：`examples/basic_demo.py`、`examples/environment_demo.py`、`examples/fixed_eval_set_demo.py`

## Trace 是什么

一句话：**一次运行的"行车记录仪"**。不管被测的是一个聊天 Agent、一个会在环境里反复行动的任务 Agent，还是别的什么，只要它跑一次，我们就产出一条 `Trace`，把这次运行里发生的一切按时间顺序记下来。

它是框架里唯一的"通用货币"：

- **生产者**：Adapter（`adapters/`）把各式各样的 Agent 翻译成 `Trace`；
- **消费者**：Evaluator（`evaluators/`）只认 `Trace`，从里面读出证据做判定。

正因为中间统一成 `Trace`，evaluator 才不用关心被测对象到底是哪种 Agent。

## Trace 的字段

`Trace` 是一个 Pydantic 模型（`core/trace.py`），字段固定：

```
Trace
├── trace_id      str          这次运行的唯一 ID（UUID）
├── input         str          输入
├── events        list[...]    有序的事件流（核心）
├── final_output  str | None   最终输出
└── metadata      dict         自定义附加字段
```

- `trace_id`：便于把一份结果和一次运行对上。
- `input` / `final_output`：一头一尾，最常被断言的两个点。
- `events`：真正的重头戏，下面单独讲。
- `metadata`：留给框架之外的信息——比如你把某类模型的原始分数、概率、样本来源塞在这里。

## events：事件流

`events` 是一个**有序列表**，每一项也是一个对象，靠 `type` 字段区分种类，不同种类带不同属性：

| `type` | 含义 | 主要字段 |
|---|---|---|
| `user_message` | 用户（或测试台）输入 | `content` |
| `agent_message` | Agent 产出 | `content` |
| `tool_call` | 请求调用工具 | `call_id` / `name` / `arguments` |
| `tool_result` | 工具返回 | `call_id` / `name` / `result` |
| `action` | 在环境里的动作 | `action_id` / `name` / `arguments` |
| `environment_event` | 环境反馈 | `source` / `data` |
| `state_snapshot` | 状态快照 | `action_id` / `state` |
| `state_change` | 状态增量 | `action_id` / `changes` |
| `metric` | 命名的数值反馈 | `name` / `value` |
| `custom_event` | 自定义事件 | `name` / `data` |

几个约定值得记住：

- `tool_call` / `tool_result` 靠 `call_id` 配对，**不靠先后顺序**——工具可能交错或重复调用。
- `action` 是 Agent → **环境**，`tool_call` 是 Agent → **工具**，两者分开记录。
- `state_snapshot` / `state_change` 靠 `action_id` 关联到产生它的那个动作。
- `metric` 是"命名的数值"通道，一条 Trace 可以带多个。

## 两个真实例子

### 例 1：简单聊天型

```
Trace(
  input="请向北京打招呼",
  final_output="你好，北京！",
  events=[
    {"type": "user_message",  "content": "请向北京打招呼"},
    {"type": "agent_message", "content": "你好，北京！"},
  ],
)
```

只有一进一出两个事件。

### 例 2：环境型（一条 Trace 装整条轨迹）

```
[1]  user_message      {'content': '帮我下一单然后申请退款'}
[2]  state_snapshot    {'action_id': None, 'state': {..., 'status': 'draft', 'total': 0.0}}
[3]  action            {'name': 'add_item', 'arguments': {'sku': 'iphone', 'qty': 1}}
[4]  environment_event {'observation': 'POST .../items -> added iphonex1'}
[5]  state_change      {'changes': {'items': ['iphone'], 'total': 299.0}}
[6]  state_snapshot    {'state': {..., 'status': 'draft', 'total': 299.0}}
[7]  action            {'name': 'checkout', 'arguments': {}}
[8]  environment_event {'observation': 'POST .../checkout -> paid'}
[9]  state_change      {'changes': {'status': 'paid'}}
[10] state_snapshot    {'state': {..., 'status': 'paid', 'total': 299.0}}
[11] agent_message     {'content': 'order refunded'}
```

从"收到请求"到"加购 → 下单 → 收尾"，**整条轨迹都在同一个 Trace 的 `events` 里**，evaluator 可以逐条回放地检查。

## Trace 从哪来：Adapter

`Trace` 由 **Adapter** 生产，接口只有一个方法（`adapters/base.py`）：

```python
class AgentAdapter(ABC):
    @abstractmethod
    async def run(self, message: str) -> Trace: ...
```

框架自带几种：

- **`CallableAgentAdapter`**：把任意 `message -> str` 的函数包成 Agent，产出 `user_message` + `agent_message`。你传 `agent=my_func` 时自动套用（`core/models.py`）。
- **`EnvironmentAgentAdapter`**：跑"观察状态 → 决策 → 落子"的循环，把每一步翻译成 `action` / `environment_event` / `state_change` / `state_snapshot` / `metric`。

要接入一个新东西，**只需写一个 Adapter**，把它的行为翻译成 `Trace` 即可——框架内核一个字都不用改。

## 自己写 Adapter：把真实系统翻译成 Trace

真实项目里，Agent 往往**没有**一个 `run()` 主动把 Trace 交给你。常见形态是：

- 应用被包装成一个 **HTTP 接口**，你只有一头（输入）一尾（输出）；
- 中间动作（工具调用、决策、模型概率……）散落在别处，比如数据库的 `ai_logs` 表。

所以关键的心智转变是：**Trace 不是 Agent"给"你的，而是 Adapter 从多个来源"拼装"出来的。**

```
Agent ─┬─ HTTP 响应（输入 / 输出）        ┐
       ├─ 数据库 ai_logs（中间动作）      ├─→ Adapter 拼装 → Trace
       └─ 其它（文件 / 日志 / 接口）      ┘
```

框架不关心数据从哪来，Adapter 负责收集、对齐、翻译。

### 先解决三个问题

假设 Agent 部署成 `POST /chat`，每次请求往 `ai_logs` 写记录：

```
conversation_id | step | type        | payload                               
C001            | 1    | user_msg    | {"text": "..."}                        
C001            | 2    | tool_call   | {"tool": "query_order", "args": {...}} 
C001            | 3    | tool_result | {"result": {...}}                      
C001            | 4    | decision    | {"label": "block", "probs": {...}}     
C001            | 5    | agent_msg   | {"text": "..."}                        
```

要拼出 Trace，必须先拿到三个"键"：

1. **关联键**：哪些行属于这一次运行？→ 需要一个 run id（如 HTTP 响应回一个 `conversation_id`，或请求时传入）。
2. **排序键**：行的先后顺序？→ 用 `step` 或 `created_at`。
3. **映射键**：每行变成什么事件？→ 看 `type` 列。

三个缺一个，Trace 就拼不起来。

### 装配代码（示意）

```python
class HttpAgentAdapter(AgentAdapter):
    def __init__(self, base_url, db):
        self._http = httpx.AsyncClient()
        self._base_url = base_url
        self._db = db

    async def run(self, message: str) -> Trace:
        # ① 输入口：发请求，拿"关联键 + 最终输出"
        resp = await self._http.post(f"{self._base_url}/chat", json={"message": message})
        data = resp.json()
        run_id = data["conversation_id"]          # 关联键
        final_output = data["reply"]

        # ② 中间动作：从数据库捞本次运行的记录
        rows = await self._db.query(
            "SELECT step, type, payload FROM ai_logs "
            "WHERE conversation_id = %s ORDER BY step",
            run_id,
        )

        # ③ 拼装 Trace
        trace = Trace(input=message, final_output=final_output)
        trace.add_event(UserMessageEvent(content=message))
        for row in rows:
            trace.add_event(self._translate(row))     # 逐行翻译
        trace.add_event(AgentMessageEvent(content=final_output))
        return trace

    def _translate(self, row):
        kind = row["type"]
        if kind == "tool_call":
            return ToolCallEvent(name=row["payload"]["tool"],
                                 arguments=row["payload"]["args"])
        if kind == "decision":
            # 决策（含概率）→ CustomEvent，也可按需拆成 MetricEvent
            return CustomEvent(name="decision", data=row["payload"])
        if kind == "tool_result":
            return ToolResultEvent(call_id=..., name=..., result=row["payload"]["result"])
        ...
```

**`type` 列就是映射表的入口**：按业务约定，把每行翻译成对应的事件类型。

### 会踩的坑

- **日志滞后**：异步写日志时，HTTP 返回那一刻 DB 可能还没写完。要么轮询到本次 `run_id` 标记完成，要么等待一小段时间再查。
- **拿不到关联键**：若响应不回 id、请求也不传，就只能靠时间窗猜——很脆，尽量让应用回一个 id。
- **直接连 DB 会把测试框架耦合到你的库**：更干净的做法是让应用多暴露一个 `GET /traces/{run_id}`，Adapter 只调接口、不碰 DB。
- **日志不全**：那 Trace 就不全。**只能记录"能观察到的东西"**，观察不到的就记不了——这是诚实的边界。

### 顺带：概率型决策模型怎么进来

如果 Agent 内部调了一个"输出概率"的决策模型，它的输入/输出（含概率）通常就落在 `ai_logs` 的 `decision` 行里。Adapter 取出来——概率挂进 `CustomEvent` / `MetricEvent`，选中的类挂 `final_output`——Trace 里就同时有了"行为"和"概率"。这就是把一个概率型模型接入 AgentMeter 的真实形态。

## Trace 到哪去：Evaluator

`Runner` 干的事就两步（`runner/runner.py`）：

```python
trace = await testcase.agent.run(testcase.input)   # ① 让 Adapter 生产 Trace
for evaluator in testcase.evaluators:              # ② 把 Trace 喂给每个 evaluator
    result = await evaluator.evaluate(trace)
```

Evaluator 的契约同样只有一个方法（`evaluators/base.py`）：

```python
class Evaluator(ABC):
    @abstractmethod
    async def evaluate(self, trace: Trace) -> EvaluationResult: ...
```

产出的 `EvaluationResult` 含 `verdict`（PASS / FAIL / ERROR）、`score`（[0, 1]）和 `reason`。整体判定规则：**任一 ERROR → ERROR，任一 FAIL → FAIL，否则 PASS**。

## 读取 Trace 的常用方法

`Trace` 自带一批遍历/查询方法，写 evaluator 时不用手撸过滤：

| 想拿什么 | 方法 |
|---|---|
| 最终输出 | `trace.final_output` |
| 所有工具调用 / 名字 | `trace.tool_calls()` / `trace.tool_call_names()` |
| 某次调用的结果 | `trace.result_for_call(call_id)` |
| 所有环境动作 / 名字 | `trace.actions()` / `trace.action_names()` |
| 状态快照 / 变化 | `trace.state_snapshots()` / `trace.state_changes()` |
| 某个 metric 的最新值 | `trace.metric("reward")` |
| 最终状态 | `trace.final_state` |
| 原始事件流 | `trace.events` |

## 实战：写一个读 Trace 的 evaluator

假设要断言"订单最后真的被退款了"——注意，**看状态，不看 Agent 嘴上说了什么**：

```python
from agentmeter import Evaluator, EvaluationResult, Verdict, Trace

class RefundCompletedEvaluator(Evaluator):
    async def evaluate(self, trace: Trace) -> EvaluationResult:
        state = trace.final_state or {}
        ok = state.get("status") == "refunded"
        return EvaluationResult(
            evaluator="RefundCompleted",
            verdict=Verdict.PASS if ok else Verdict.FAIL,
            score=1.0 if ok else 0.0,
            reason=f"final status={state.get('status')!r}",
        )
```

把它放进 `TestCase.evaluators`，`Runner` 就会把每条 Trace 喂给它。

## 序列化与回放

`Trace` 是 Pydantic 模型，能直接落盘成 JSON：

```python
payload = trace.model_dump()          # dict
as_json = trace.model_dump_json()     # str
```

这也是"证据可回放"的基础：把一份 Trace 存下来，事后能还原"当时 Agent 到底做了什么"，而不只是"最后判了 PASS 还是 FAIL"。

## 参考

- 可运行案例：`examples/basic_demo.py`（最简）、`examples/environment_demo.py`（环境型）、`examples/fixed_eval_set_demo.py`（分组门禁）
- 环境 / 动作 / 安全测试：`docs/environment-and-security.md`
- 固定评测集 + CI：`docs/fixed-eval-set-and-ci.md`
- Evaluator 速查：`README.zh-CN.md`「Evaluator 速查表」
