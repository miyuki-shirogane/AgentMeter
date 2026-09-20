# GitHub Actions CI/CD 配置说明

本项目使用 GitHub Actions 实现自动化测试、代码质量检查与打包验证。

## Workflow 文件

### 1. `ci.yml` — 主 CI 流程

- **触发时机**：推送到 `main` 分支，或对 `main` 发起 PR
- **测试矩阵**：Python 3.11、3.12
- **包含三个 job**：
  - `Test and Coverage`：安装依赖（`uv sync --extra dev`）→ 运行全量测试 → 覆盖率必须 **≥ 85%** → 上传 `coverage.xml` 产物 → 上传 Codecov
  - `Code Quality Checks`：`ruff check`（阻断）+ `ruff format --check`（仅提示，不阻断）
  - `Build Package`：`uv build` 验证 wheel / sdist 可正常构建，并上传产物

### 2. `pr-check.yml` — PR 门禁

- **触发时机**：对 `main` 创建 / 更新 / 重新打开 / 转为 ready 的 PR
- **功能**：
  - 强制所有测试通过
  - 强制覆盖率 ≥ 85%
  - 自动在 PR 中留言（成功 / 失败状态，fork PR 跳过以免 token 权限报错）
  - 提示目标分支的保护规则配置

## 启用门禁功能（必须配置）

要让 CI 成为真正的「门禁」，需在 GitHub 仓库设置中配置分支保护规则：

1. 进入 **仓库 → Settings → Branches**
2. 添加保护规则（Branch name pattern: `main`），勾选：
   - Require a pull request before merging（至少 1 个 reviewer）
   - Require status checks to pass before merging
     - 选择 required checks：`Test and Coverage`、`Code Quality Checks`、`PR Test Gate`
   - Require branches to be up to date before merging
   - Do not allow bypassing the above settings
3. 保存

### 效果

- 测试失败 → **无法合并**
- 覆盖率 < 85% → **无法合并**
- `ruff check` 不通过 → **无法合并**
- 全部通过 → **可以合并**

## 状态检查

创建 PR 后可在以下位置查看 CI 状态：

1. PR 页面底部 —— 所有 checks 的状态
2. **Actions** 标签页 —— 详细运行日志
3. PR 评论 —— 自动添加的成功 / 失败消息

## 本地复现 CI

提交代码前建议在本地跑一遍同样的命令：

```bash
# 安装（含 dev 依赖）
uv sync --extra dev

# 代码质量
uv run ruff check .
uv run ruff format --check .

# 测试 + 覆盖率（与 CI 阈值一致）
uv run pytest tests/ --cov=src/agentmeter --cov-report=term --cov-fail-under=85

# 打包验证
uv build
```

## 环境变量与密钥（可选）

在 **仓库 → Settings → Secrets and variables → Actions** 中可配置：

- `CODECOV_TOKEN` —— 上传覆盖率到 Codecov 所需（未配置时 Codecov 步骤不会使 CI 失败）

> 说明：`.env` 与 `uv.lock` 均在 `.gitignore` 中。CI 不读取任何真实密钥，测试用例本身不依赖网络或 `DEEPSEEK_API_KEY`。
> 若希望 CI 依赖完全可复现，建议将 `uv.lock` 纳入版本控制，并把 workflow 改为 `uv sync --frozen`。
