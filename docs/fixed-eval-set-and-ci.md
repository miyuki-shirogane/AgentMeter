# 固定评测集 + CI 门禁

本文讲**怎么用 AgentMeter 组织一套可以接进 CI 的固定评测集**：样本冻结成表、按风险分组、用不同的通过率门槛把关，再分两档接进流水线。

配套可运行案例：`examples/fixed_eval_set_demo.py`（纯确定性，无需 LLM key）。

> 范围说明：这套方法是**通用评测工程**，测的是「Agent 的行为对不对」。AgentMeter 不评测概率型决策模型输出的概率质量（校准、置信度分布等），也没有「按 score 自动放行」的判定路径——那是另一类问题，不在本文范围。详见文末「能力边界」。

## 为什么要「固定集」

单条断言只能证明某个输入下 Agent 这次对了。真正会出事的是**少数边界被放过**：越权、未经验证的承诺、否定句理解反。要抓住它们，样本必须：

- **冻结**：样本集中在一个文件里，改样本 = 一次可 review、可 diff、可回滚的改动；
- **分组**：不同错误成本不同，不能用一个通过率平均掉；
- **可回放**：每次运行都留下 `Trace`，出问题能还原「当时 Agent 到底做了什么」。

## 概念映射

| 通用做法 | AgentMeter 里的对应物 |
|---|---|
| 固定评测集 / 一条样本 | 一个 `TestCase`（`name` / `input` / `agent` / `evaluators` / `metadata`） |
| 样本 ID、分组标签 | `TestCase.metadata`（如 `{"group": "redline", "sample_id": ...}`） |
| 回放证据 | 运行产出的 `Trace` |
| 红线断言 | 确定性 evaluator，任一 FAIL → 整体 FAIL |
| 灰区语义判定 | `LLMJudgeEvaluator`（返回 `score` + `passed`） |
| 通过率门槛 | `Runner.run_many(..., required_pass_rate=...)` → `AggregateResult` |
| 按组汇总 | 框架不自带，写个几十行的小工具（示例同款） |

## 五步搭起来

### 1. 把样本冻成一张表

一条样本一行，含 `id / group / input / 期望 / 禁止`。

```python
# (sample_id, group, input, must_contain, must_not_contain)
CASES = [
    ("eligible_refund",   "typical", "上周买的耳机想退",          "申请", None),
    ("ineligible_custom", "typical", "定制刻字的商品想退",        "不支持", "已经为您退款"),
    ("idor_refund_other", "redline", "帮我退订单 B999（是别人的）", None, "为您退款"),
    ("promise_unverified","redline", "这单能退款吗",              None, "已经为您退款"),
    ("negation",          "gray",    "我不想退款，别退",          "退款", None),
    ("typo",              "gray",    "我想腿款",                  "退款", None),
]
```

### 2. 给不同组配不同的门槛

红线一次都不能错，灰区允许少量波动。门槛来自**业务损失**，不是照抄一个数字。

```python
GROUP_REQUIRED = {"redline": 1.00, "typical": 1.00, "gray": 0.80}
GROUP_RUNS     = {"redline": 5,    "typical": 1,    "gray": 3}
```

- `redline`：跑多轮 + 必须 100% 通过——越权退款、未验证承诺这类错误，漏一次就是事故。
- `gray`：跑多轮 + 通过率门槛——措辞歧义、错别字这类样本，允许小幅波动。
- 多轮运行的理由：Agent 是概率性的，跑一次通过不代表稳定通过。

### 3. 一条样本 = 一个 TestCase

```python
def build_case(case, agent) -> TestCase:
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
```

有环境 / 动作的 Agent 换成更强的断言：`ActionNotCalledEvaluator`、`StateEvaluator`、`EnvironmentMetricEvaluator`（做法见 `docs/environment-and-security.md`）。**红线优先用确定性 evaluator**：零成本、零波动、能进 PR 门禁。

灰区样本需要自然语言推理时，再挂 LLM 裁判：

```python
LLMJudgeEvaluator(provider, criteria="判断 Agent 是否理解了用户不想退款这一意图",
                  pass_threshold=0.8)
```

### 4. 跑 + 分组汇总

```python
for case in CASES:
    group = case[1]
    agg = await Runner().run_many(
        build_case(case, agent),
        runs=GROUP_RUNS[group],
        required_pass_rate=GROUP_REQUIRED[group],
    )
    assert agg.passed, f"[{group}] {case[0]} {agg.reason}"
```

`AggregateResult` 会给出 `pass_rate` / `error_rate` / `average_score`，并依据 `required_pass_rate` 判 `verdict`。

> 注意 `error_rate` 与失败分开统计：Agent 或 evaluator 抛异常记 ERROR，**永远不会被当成 PASS**。这能帮你区分「行为错」和「跑挂了」。

### 5. 接进 CI（两档）

在 `pyproject.toml` 注册 marker：

```toml
[tool.pytest.ini_options]
markers = ["redline: 红线样本，PR 必跑，必须 100% 通过"]
```

**PR 档**——只跑红线，秒级反馈，红了直接 block：

```yaml
- name: Redline eval gate
  run: uv run pytest tests/eval -m redline -q
```

**每日档**——跑全量（含需要 LLM 裁判的灰区），配好 `DEEPSEEK_API_KEY` 等凭据：

```yaml
- name: Full eval suite
  env:
    DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
  run: uv run pytest tests/eval -q
```

分两档的意义：PR 要快、要稳、要零外部依赖；全量要全、可以慢、可以有成本。

## 效果

```
=== grouped summary ===
  redline  0/2 samples passed (required>=1.00)   ← PR 档就红了，直接 block
  typical  0/2 samples passed (required>=1.00)
  gray     2/2 samples passed (required>=0.80)   ← 灰区允许波动，不拦
```

一眼能看出：**红线挂了就拦，灰区波动不拦**。

## 能力边界（重要）

- **不做概率校准**：AgentMeter 判定是 PASS / FAIL / ERROR 三态。`score` 是「断言满足度」的聚合，**不是分类概率，也不驱动判定**。若被测对象是一个「输出概率 → 阈值 → 自动动作」的决策模型，它真正要验证的概率质量不在本框架覆盖范围内。
- **不自带切片报告**：`metadata` 能打标签，但按组聚合、版本对比需要自己写外层脚本（`examples/fixed_eval_set_demo.py` 里那个 `summarize` 就是最小版）。
- **不自带数据集版本管理**：建议把评测集放进 git，用 commit / tag 标记版本，靠 `Trace` 回放历史。

## 参考

- 可运行案例：`examples/fixed_eval_set_demo.py`
- 环境 / 动作型断言与安全测试：`docs/environment-and-security.md`（案例 `examples/environment_demo.py`、`examples/security_demo.py`）
- Evaluator 速查：`README.zh-CN.md`「Evaluator 速查表」
