# GitHub Stars Manager [![Stars Manager CI](https://github.com/gandli/github_stars_manager/actions/workflows/stars_manager.yml/badge.svg)](https://github.com/gandli/github_stars_manager/actions/workflows/stars_manager.yml)

一个用于批量获取 GitHub 已加星仓库，并通过 Zhipu（OpenAI 风格 API）进行分类、打标签与摘要生成的脚本。支持按加星时间升序处理、断点续跑、去重合并，并输出到 JSON / CSV / Markdown。

## 功能特点
- 拉取加星仓库，按 `starred_at` 升序处理，便于从早到晚逐批分析。
- 调用 ZHIPU 模型生成类别、标签、摘要，并进行本地规则校正与去重。
- 每次执行直接合并输出到 `outputs/results_all.json`、`outputs/results_all.csv`、`outputs/results_all.md`。
- 分类与关键词规则可配置（通过环境变量或外部 JSON 文件），避免硬编码。
- 提供 GitHub Actions 工作流，支持定时和手动执行并自动提交结果。

## 环境要求
- Python 3.10+
- 依赖：`requests`、`openai`（可选：`pandas`、`tenacity`）

## 安装与运行

### 使用 pip
```powershell
python -m pip install --upgrade pip
pip install requests openai
```

### 使用 uv（推荐）
```powershell
# 安装 uv（Windows PowerShell）
iwr -useb https://astral.sh/uv/install.ps1 | iex

# 初始化项目（如需）并添加依赖
uv init
uv add requests openai

# 同步安装并运行脚本
uv sync
uv run python stars_manager.py --batch-size 10
```

### 配置 .env（零依赖加载）
在项目根目录创建 `.env`：
```env
GH_TOKEN=ghp_xxx
API_KEY=xxx
BASE_URL=https://open.bigmodel.cn/api/paas/v4/
MODEL=glm-4.5-flash

# 可选：分类配置
CATEGORIES_FILE=c:/path/to/categories.json
CATEGORY_RULES_FILE=c:/path/to/category_rules.json
DEFAULT_CATEGORY=开发工具
```

脚本会在启动时调用内置的 `load_env()` 从 `.env` 读取变量。

### 运行示例
```powershell
python stars_manager.py --batch-size 20 --sleep 0.3 --model glm-4.5-flash
# 或显式覆盖基础地址
python stars_manager.py --batch-size 20 --base-url https://open.bigmodel.cn/api/paas/v4/
```

## 输出文件
- `outputs/results_all.json`：所有分析结果（合并、去重）。
- `outputs/results_all.csv`：同上，CSV 格式。
- `outputs/results_all.md`：同上，Markdown 表格。

字段：`repo_full_name`、`owner`、`html_url`、`description`、`topics`、`category`、`tags`、`summary`、`starred_at`、`analyzed_at`。

## 去重与断点续跑
- 去重键：`repo_full_name + '|' + starred_at`
- 每次运行只处理未出现过的唯一键，并直接合并到 `results_all.*`。
- 已存在 `results_all.json` 时，脚本会基于其内容判断哪些是“已处理”。

## 分类配置（可选）
- 为避免硬编码，支持通过环境变量或文件自定义分类与规则：
  - `CATEGORIES`（逗号分隔）：如 `Web应用,移动应用,桌面应用,...`
  - `CATEGORIES_FILE`（JSON 数组）：分类列表文件。
  - `CATEGORY_RULES_FILE`（JSON 对象）：`{分类: [关键词...]}`，用于本地关键词匹配映射。
  - `DEFAULT_CATEGORY`（字符串）：无法匹配时的默认分类（默认：`开发工具`）。
- CLI 覆盖参数：
  - `--categories-file`、`--category-rules-file`、`--default-category`、`--base-url`、`--model`、`--batch-size`、`--sleep`。

## GitHub Actions（CI/CD）
已提供工作流：`.github/workflows/stars_manager.yml`

- 触发：
  - `workflow_dispatch`（手动，支持输入 `batch_size`、`sleep`、`model`）
  - `schedule`（每天 `UTC 03:00` 自动运行）
- 机密与变量（仓库 Settings → Secrets and variables → Actions）：
  - Secrets：`GH_STAR_TOKEN`（读取 Star 列表的 PAT）、`API_KEY`（智谱 API Key）
  - Variables（可选）：`BASE_URL`、`MODEL`
- 执行：安装依赖 → 运行脚本 → 若 `outputs/` 变更则自动提交（`Update results_all via CI`）。

## 常见问题
- JSON 解析异常：脚本启用了 `response_format={"type": "json_object"}` 并包含健壮清理逻辑，尽量保证 AI 返回纯 JSON。
- 速率限制：可调低 `--batch-size` 或增加 `--sleep`。
- 无描述仓库：提示与降级分析已处理；标签和摘要尽量保持简洁。

## 开发提示
- 代码风格保持简洁，改动尽量小而集中。
- 若扩展依赖，请更新 `任务.md` 和本 README。