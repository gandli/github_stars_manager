"""
脚本概览
——
本脚本用于：
1) 通过 GitHub API 获取当前用户已加星的仓库列表（包含加星时间 starred_at）。
2) 调用 ZHIPU/OpenAI 风格的对话模型接口，为每个仓库生成分类、标签与简短摘要。
3) 将分析结果合并输出到 JSON/CSV/Markdown 三种格式的文件中，便于后续浏览与统计。

运行环境与配置：
- 必需环境变量：`GH_TOKEN`（GitHub 访问令牌，用于读取加星仓库）。
- 可选环境变量：
  - `API_KEY`（ZHIPU/OpenAI 风格接口密钥，缺失时将使用本地关键词规则进行回退并给出说明）。
  - `BASE_URL`（模型接口基础地址，main 中也支持通过命令行参数 `--base-url` 覆盖）。
  - `MODEL`（模型名称，默认 `glm-4`，也可由 CI 的 env/vars 或命令行传入）。
  - `CATEGORIES_FILE`、`CATEGORY_RULES_FILE`（用于加载自定义分类及关键词规则）。
  - `DEFAULT_CATEGORY`（无法匹配时的默认分类，默认“开发工具”）。

输出结果：
- 最终合并输出路径：`outputs/results_all.json`、`outputs/results_all.csv`、`outputs/results_all.md`。

在 GitHub Actions 中，本脚本由工作流传入 `batch_size`、`sleep`、`model` 等参数，
也会依赖 `env` 中的 `GH_TOKEN` / `API_KEY` / `BASE_URL` / `MODEL` 等变量（已支持 secrets/vars 回退）。
"""

import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests
from openai import OpenAI


# --------- .env loader (zero-dependency) ---------
def load_env(path: str = ".env") -> None:
    """
    从指定的 `.env` 文件加载环境变量到当前进程（不覆盖已存在的值）。
    参数：
    - path：`.env` 文件路径，默认当前目录下的 `.env`。
    返回：无。
    """
    # 从 .env 文件加载键值对到当前进程的环境变量（不覆盖已存在的值）。
    # 格式示例：
    # GH_TOKEN=ghp_xxx
    # API_KEY=xxx
    # BASE_URL=https://open.bigmodel.cn/api/paas/v4/
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


# --------- GitHub stars fetcher ---------
GITHUB_API = "https://api.github.com"


def get_github_headers(token: str) -> Dict[str, str]:
    """
    构造访问 GitHub API 的请求头。
    参数：
    - token：GitHub 访问令牌（GH_TOKEN）。
    返回：包含认证与 Accept、User-Agent 的请求头字典。
    """
    # 构造访问 GitHub API 所需的请求头：
    # - 使用 Bearer 令牌进行认证
    # - 设置 Accept 为 star+json 以在 /user/starred 接口返回 starred_at 字段
    # - 自定义 User-Agent 便于跟踪
    return {
        "Authorization": f"Bearer {token}",
        # Include starred_at in response for /user/starred
        "Accept": "application/vnd.github.v3.star+json",
        "User-Agent": "github-stars-manager",
    }


def fetch_starred_repos_sorted_asc(
    token: str,
    page_start: int = 1,
    per_page: int = 100,
    needed_unprocessed: int = 10,
    processed_keys: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    功能：从 GitHub 读取当前用户的加星仓库列表，按照加星时间升序遍历，返回“尚未处理”的若干条项目。

    参数说明：
    - token：GitHub 访问令牌（GH_TOKEN）。
    - page_start：起始分页页码，默认 1。
    - per_page：每页数量，默认 100，上限由 GitHub API 限制。
    - needed_unprocessed：需要返回的未处理条数上限（满足即停止）。
    - processed_keys：已处理项目的唯一键集合，用于跳过重复项（键格式 "full_name|starred_at"）。

    返回：包含若干仓库信息的字典列表，每项含 repo_full_name/html_url/description/topics/starred_at 等。
    """
    headers = get_github_headers(token)
    results: List[Dict[str, Any]] = []
    page = page_start
    processed_keys = processed_keys or set()

    while len(results) < needed_unprocessed:
        params = {
            "per_page": per_page,
            "page": page,
            "sort": "created",  # creation of the star event
            "direction": "asc",
        }
        resp = requests.get(
            f"{GITHUB_API}/user/starred", headers=headers, params=params, timeout=30
        )
        if resp.status_code != 200:
            raise RuntimeError(f"GitHub API error: {resp.status_code} {resp.text}")

        batch = resp.json()
        if not batch:
            break  # no more pages

        for item in batch:
            # item structure when using star+json: { starred_at, repo: {...} }
            repo = item.get("repo", {})
            full_name = repo.get("full_name")
            starred_at = item.get("starred_at")
            if not full_name or not starred_at:
                continue
            unique_key = f"{full_name}|{starred_at}"
            if unique_key in processed_keys:
                continue

            results.append(
                {
                    "repo_full_name": full_name,
                    "owner": full_name.split("/")[0] if "/" in full_name else "",
                    "html_url": repo.get("html_url"),
                    "description": repo.get("description"),
                    "topics": repo.get("topics", []),  # may be missing
                    "starred_at": starred_at,
                }
            )
            if len(results) >= needed_unprocessed:
                break

        page += 1
        time.sleep(0.3)

    return results


# --------- Zhipu AI analysis (OpenAI-style) ---------
BASE_URL = os.environ.get("BASE_URL")
# 模型接口基础地址（可选）。在 main 中会被命令行参数或环境变量覆盖，
# 若均未设置则默认使用 `https://open.bigmodel.cn/api/paas/v4/`。


# --------- Categories configuration (flexible) ---------
DEFAULT_ALLOWED_CATEGORIES = [
    # 默认允许的分类集合，可通过环境变量或文件覆盖。
    "Web应用",
    "移动应用",
    "桌面应用",
    "数据库",
    "AI/机器学习",
    "开发工具",
    "安全工具",
    "游戏",
    "设计工具",
    "效率工具",
    "教育学习",
    "社交网络",
    "数据分析",
]

DEFAULT_CATEGORY_RULES: Dict[str, List[str]] = {
    # 本地关键词到分类的映射，用于在无 API 或 API 失败时做规则回退。
    "Web应用": [
        "web",
        "http",
        "rest",
        "frontend",
        "backend",
        "website",
        "spa",
        "vue",
        "react",
        "svelte",
        "nextjs",
        "nuxt",
    ],
    "移动应用": [
        "android",
        "ios",
        "mobile",
        "apk",
        "react native",
        "flutter",
        "cordova",
    ],
    "桌面应用": ["desktop", "electron", "qt", "gtk", "win32", "macos", "wxwidgets"],
    "数据库": [
        "database",
        "db",
        "sql",
        "nosql",
        "postgres",
        "mysql",
        "mongodb",
        "redis",
        "cassandra",
        "sqlite",
    ],
    "AI/机器学习": [
        "machine learning",
        "ml",
        "ai",
        "deep learning",
        "transformer",
        "llm",
        "pytorch",
        "tensorflow",
        "keras",
    ],
    "开发工具": [
        "dev",
        "developer",
        "sdk",
        "library",
        "framework",
        "build",
        "compile",
        "cli",
        "lint",
        "ci",
        "testing",
        "tool",
    ],
    "安全工具": [
        "security",
        "vuln",
        "pentest",
        "penetration",
        "exploit",
        "auth",
        "encryption",
        "ssl",
        "xss",
        "cve",
    ],
    "游戏": ["game", "gaming", "unity", "unreal", "godot"],
    "设计工具": [
        "design",
        "ui",
        "ux",
        "figma",
        "sketch",
        "graphics",
        "svg",
        "illustration",
    ],
    "效率工具": [
        "productivity",
        "todo",
        "note",
        "task",
        "calendar",
        "automation",
        "workflow",
    ],
    "教育学习": [
        "education",
        "learn",
        "learning",
        "tutorial",
        "course",
        "teaching",
        "docs",
        "examples",
    ],
    "社交网络": ["social", "network", "chat", "messaging", "community", "forum", "sns"],
    "数据分析": [
        "data analysis",
        "analytics",
        "bi",
        "pandas",
        "numpy",
        "visualization",
        "chart",
        "plot",
    ],
}


def load_categories(
    file_path: Optional[str] = None, env_key: str = "CATEGORIES"
) -> List[str]:
    """
    从文件或环境变量加载允许的分类列表。
    - 若指定 `file_path`（或通过 `CATEGORIES_FILE` 环境变量传入）存在，则读取 JSON 数组。
    - 否则从环境变量 `CATEGORIES` 读取逗号分隔的字符串。
    - 若上述均为空，则使用内置的 `DEFAULT_ALLOWED_CATEGORIES`。
    """
    cats: List[str] = []
    if not file_path:
        file_path = os.environ.get("CATEGORIES_FILE")
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    cats = [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            cats = []
    if not cats:
        env_val = os.environ.get(env_key)
        if env_val:
            cats = [x.strip() for x in env_val.split(",") if x.strip()]
    if not cats:
        cats = DEFAULT_ALLOWED_CATEGORIES
    return cats


def load_category_rules(file_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    加载分类关键词规则：形如 {"分类": ["关键词1", "关键词2", ...]} 的 JSON 对象。
    - 若指定 `file_path`（或通过 `CATEGORY_RULES_FILE` 环境变量传入）存在，则读取该文件。
    - 否则返回内置的 `DEFAULT_CATEGORY_RULES`。
    """
    if not file_path:
        file_path = os.environ.get("CATEGORY_RULES_FILE")
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    rules: Dict[str, List[str]] = {}
                    for cat, words in data.items():
                        if isinstance(words, list):
                            rules[str(cat)] = [str(w).lower() for w in words]
                    return rules
        except Exception:
            pass
    return DEFAULT_CATEGORY_RULES


def classify_category_by_keywords(
    repo: Dict[str, Any], category_rules: Dict[str, List[str]]
) -> Optional[str]:
    """
    基于关键词规则对仓库进行简单分类。
    参数：
    - repo：仓库信息字典（包含名称、描述、topics 等）。
    - category_rules：{分类: [关键词...]} 的映射。
    返回：命中的分类名称；若未命中返回 None。
    """
    # 基于简单的关键词匹配进行分类：
    # 将仓库名称、简介与 topics 合并为一个小写字符串，
    # 若某个分类的关键词列表中的词命中该文本，则返回该分类。
    text = " ".join(
        [
            str(repo.get("repo_full_name") or ""),
            str(repo.get("description") or ""),
            " ".join(repo.get("topics", []) or []),
        ]
    ).lower()
    for cat, words in category_rules.items():
        for w in words:
            if w in text:
                return cat
    return None


def normalize_category(
    candidate: Optional[str],
    repo: Dict[str, Any],
    allowed_categories: List[str],
    category_rules: Dict[str, List[str]],
    default_category: str = "开发工具",
) -> str:
    """
    规范化模型返回的分类到允许集合中。
    参数：
    - candidate：模型给出的分类候选。
    - repo：仓库信息字典。
    - allowed_categories：允许的分类列表。
    - category_rules：本地关键词规则映射，用于回退。
    - default_category：无法确定时使用的默认分类。
    返回：最终确定的分类名称。
    """
    # 规范化分类：
    # 1) 若模型给出的分类 candidate 在允许列表中，则直接返回。
    # 2) 否则尝试按本地关键词规则进行映射。
    # 3) 若仍无法确定，则返回默认分类。
    if candidate in allowed_categories:
        return candidate
    mapped = classify_category_by_keywords(repo, category_rules)
    if mapped and mapped in allowed_categories:
        return mapped
    return default_category


def parse_json_content(content: Optional[str]) -> Dict[str, Any]:
    """
    解析模型返回的 JSON 内容（鲁棒处理）：
    - 去除代码块围栏（例如 ```json ... ```）。
    - 若直接解析失败，尝试截取第一个 "{" 与最后一个 "}" 之间的子串再解析。
    - 返回字典；解析失败返回空字典。
    """
    if not content:
        return {}
    text = content.strip()
    # remove code fences
    if text.startswith("```"):
        # remove leading ```json or ```
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        # remove trailing ```
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # first attempt
    try:
        return json.loads(text)
    except Exception:
        # try to locate JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return {}
        return {}


def analyze_with_zhipu(
    api_key: Optional[str],
    repo: Dict[str, Any],
    model: str = "glm-4",
    timeout: int = 60,
    allowed_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    调用 ZHIPU/OpenAI 风格接口为仓库生成分类、标签与摘要。
    - 若 `api_key` 缺失或调用失败，则返回回退结果（Uncategorized 与空标签、摘要给出错误说明）。
    - 使用 `BASE_URL` 作为 API 基础地址（可通过命令行或环境变量设置）。
    - 返回结构示例：{"category": "开发工具", "tags": ["cli", ...], "summary": "..."}
    """
    base_result = {
        "category": "Uncategorized",
        "tags": [],
        "summary": "",
    }

    if not api_key:
        base_result["summary"] = "API_KEY 未设置，跳过分析。"
        return base_result

    cats = allowed_categories or DEFAULT_ALLOWED_CATEGORIES
    prompt = (
        "你是一位资深开源项目分类助手。"
        "请根据仓库名称、简介和主题，给出简洁的中文摘要，并从以下固定分类中严格选择一个作为类别："
        f"{', '.join(cats)}。"
        "提供3-6个标签。只返回严格的 JSON（不要包含反引号或额外文本）："
        "category（以上列表之一，字符串），tags（字符串数组），summary（不超过120字）。"
    )

    repo_ctx = {
        "name": repo.get("repo_full_name"),
        "url": repo.get("html_url"),
        "description": repo.get("description"),
        "topics": repo.get("topics", []),
    }
    try:
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是专业的开源项目分类与摘要助手。"},
                {
                    "role": "user",
                    "content": f"{prompt}\n\n仓库信息: {json.dumps(repo_ctx, ensure_ascii=False)}",
                },
            ],
            temperature=0.2,
            top_p=0.7,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        parsed = parse_json_content(content)
        category = parsed.get("category") or base_result["category"]
        tags = parsed.get("tags") or []
        summary = parsed.get("summary") or ""
        # normalize types
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list):
            tags = []
        # Normalization is handled by caller with configured rules
        return {"category": category, "tags": tags, "summary": summary}
    except Exception as e:
        base_result["summary"] = f"ZHIPU 调用异常: {e}"
        return base_result


# --------- Output writers ---------
def ensure_output_dir(path: str) -> None:
    """
    确保输出目录存在；若不存在则创建。
    参数：
    - path：目录路径。
    返回：无。
    """
    # 确保输出目录存在，不存在则创建。
    os.makedirs(path, exist_ok=True)


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    将结果写入 JSON 文件。
    参数：
    - path：输出文件路径。
    - rows：结果行列表（字典）。
    返回：无。
    """
    # 将结果写入 JSON 文件，采用 UTF-8 编码并保留中文。
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    将结果写入 CSV 文件。
    参数：
    - path：输出文件路径。
    - rows：结果行列表（字典）。
    返回：无。
    """
    # 将结果写入 CSV 文件，列包含：仓库名/所有者/链接/简介/主题/分类/标签/摘要/加星时间/分析时间。
    # 其中 `topics` 与 `tags` 字段以分号拼接为字符串，便于表格查看。
    cols = [
        "repo_full_name",
        "owner",
        "html_url",
        "description",
        "topics",
        "category",
        "tags",
        "summary",
        "starred_at",
        "analyzed_at",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            r = row.copy()
            r["topics"] = ";".join(r.get("topics", []) or [])
            r["tags"] = ";".join(r.get("tags", []) or [])
            writer.writerow(r)


def write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    将结果写入 Markdown 表格文件。
    参数：
    - path：输出文件路径。
    - rows：结果行列表（字典）。
    返回：无。
    """
    # 将结果写入 Markdown 表格，便于在 GitHub 上直接浏览。
    lines = []
    lines.append("# GitHub Stars 分析结果")
    lines.append("")
    lines.append("| 仓库 | 类别 | 标签 | 摘要 | 加星时间 |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        repo_link = f"[{r.get('repo_full_name')}]({r.get('html_url')})"
        category = r.get("category") or ""
        tags = ", ".join(r.get("tags", []) or [])
        summary = (r.get("summary") or "").replace("\n", " ")
        starred_at = r.get("starred_at") or ""
        lines.append(
            f"| {repo_link} | {category} | {tags} | {summary} | {starred_at} |"
        )
    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# --------- Merge & dedupe ---------
def unique_key(row: Dict[str, Any]) -> str:
    """
    构造用于合并去重的唯一键。
    参数：
    - row：结果行字典，需包含 `repo_full_name` 与 `starred_at`。
    返回：唯一键字符串 `repo_full_name|starred_at`。
    """
    # 为每条记录生成用于去重的唯一键：`repo_full_name|starred_at`。
    return f"{row.get('repo_full_name')}|{row.get('starred_at')}"


def merge_json_files(output_dir: str, merged_path: str) -> List[Dict[str, Any]]:
    """
    合并指定输出目录下的所有 JSON 结果文件到 `merged_path`：
    - 先读取已存在的合并文件（如有）加载到字典映射。
    - 再遍历目录中的其他 JSON 分片，按唯一键覆盖（保留最新）。
    - 写回有序的合并结果并返回列表。
    """
    all_rows: Dict[str, Dict[str, Any]] = {}
    # Load existing merged if present
    if os.path.exists(merged_path):
        try:
            with open(merged_path, "r", encoding="utf-8") as f:
                for row in json.load(f):
                    all_rows[unique_key(row)] = row
        except Exception:
            pass

    for name in os.listdir(output_dir):
        if not name.endswith(".json") or name == os.path.basename(merged_path):
            continue
        path = os.path.join(output_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for row in json.load(f):
                    all_rows[unique_key(row)] = row
        except Exception:
            continue

    rows = list(all_rows.values())
    rows.sort(key=lambda r: r.get("starred_at") or "")
    write_json(merged_path, rows)
    return rows


def merge_all_formats(output_dir: str) -> None:
    """
    合并目录中的 JSON 分片为总结果，并同步生成 CSV 与 Markdown。
    参数：
    - output_dir：输出目录路径。
    返回：无。
    """
    # 将目录中现有的 JSON 分片合并为 `results_all.json`，并同步生成 CSV 与 Markdown。
    json_path = os.path.join(output_dir, "results_all.json")
    rows = merge_json_files(output_dir, json_path)
    csv_path = os.path.join(output_dir, "results_all.csv")
    md_path = os.path.join(output_dir, "results_all.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)


def merge_with_new_rows(output_dir: str, new_rows: List[Dict[str, Any]]) -> None:
    """
    将本次分析得到的 `new_rows` 直接合并到 `results_all.*`，无需生成中间分片文件：
    - 读取现有合并文件（如有），按唯一键覆盖。
    - 写回 JSON/CSV/Markdown 三种格式的最终结果。
    """
    merged_json = os.path.join(output_dir, "results_all.json")
    all_rows: Dict[str, Dict[str, Any]] = {}
    # Load existing merged if present
    if os.path.exists(merged_json):
        try:
            with open(merged_json, "r", encoding="utf-8") as f:
                for row in json.load(f):
                    all_rows[unique_key(row)] = row
        except Exception:
            pass

    # Add new rows (override by unique key)
    for row in new_rows:
        all_rows[unique_key(row)] = row

    rows = list(all_rows.values())
    rows.sort(key=lambda r: r.get("starred_at") or "")
    write_json(merged_json, rows)
    write_csv(os.path.join(output_dir, "results_all.csv"), rows)
    write_markdown(os.path.join(output_dir, "results_all.md"), rows)


# --------- Main batch execution ---------
def main() -> None:
    """
    脚本入口：解析参数、加载配置，执行批量分析并写出结果文件。
    步骤：
    1) 解析命令行参数与 .env；
    2) 校验 `GH_TOKEN`，解析 `API_KEY` 与 `BASE_URL`；
    3) 加载分类与规则；
    4) 读取历史合并用于去重；
    5) 拉取待分析仓库并调用模型；
    6) 合并写出 JSON/CSV/MD 结果。
    返回：无。
    """
    # 主流程：解析参数 -> 加载 .env -> 读取凭据与接口地址 -> 加载分类配置 ->
    # 读取已处理的键 -> 从 GitHub 获取待处理项 -> 调用模型分析 -> 合并写出结果。
    parser = argparse.ArgumentParser(description="GitHub Star 项目分类、标签与摘要生成")
    parser.add_argument("--batch-size", type=int, default=10, help="每次处理的项目数量")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", "glm-4"),
        help="ZHIPU AI 模型名称",
    )
    parser.add_argument("--per-page", type=int, default=100, help="GitHub API 每页大小")
    parser.add_argument("--sleep", type=float, default=0.6, help="每次分析间隔秒数")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="ZHIPU/OpenAI 风格 API 基础地址，优先于环境变量",
    )
    parser.add_argument(
        "--categories-file",
        type=str,
        default=os.environ.get("CATEGORIES_FILE"),
        help="分类列表文件（JSON数组），可覆盖默认分类",
    )
    parser.add_argument(
        "--category-rules-file",
        type=str,
        default=os.environ.get("CATEGORY_RULES_FILE"),
        help="分类关键词规则文件（JSON对象），用于本地映射",
    )
    parser.add_argument(
        "--default-category",
        type=str,
        default=os.environ.get("DEFAULT_CATEGORY", "开发工具"),
        help="无法匹配时的默认分类",
    )
    args = parser.parse_args()

    load_env()
    # 支持从本地 .env 加载环境变量，方便本地调试或 CI 统一配置。

    GH_TOKEN = os.environ.get("GH_TOKEN")
    if not GH_TOKEN:
        print("错误：未设置 GH_TOKEN。请在 .env 或环境变量中配置。", file=sys.stderr)
        sys.exit(1)
    API_KEY = os.environ.get("API_KEY")
    # 若缺少 API_KEY，将在后续分析中采用关键词规则回退，并在摘要中提示。

    # Resolve BASE_URL from CLI or environment (supports .env)
    global BASE_URL
    BASE_URL = (
        args.base_url
        or os.environ.get("BASE_URL")
        or BASE_URL
        or "https://open.bigmodel.cn/api/paas/v4/"
    )
    # 解析最终使用的接口基础地址：优先命令行 -> 环境变量 -> 预设常量 -> 默认值。

    # Load configurable categories and rules
    allowed_categories = load_categories(args.categories_file)
    category_rules = load_category_rules(args.category_rules_file)
    # 加载可选的分类列表与关键词规则（可由外部文件或环境变量提供）。

    ensure_output_dir(args.output_dir)
    # 确保输出目录存在。

    # Determine already processed keys from merged JSON if available
    merged_json = os.path.join(args.output_dir, "results_all.json")
    processed_keys: set = set()
    if os.path.exists(merged_json):
        try:
            with open(merged_json, "r", encoding="utf-8") as f:
                for row in json.load(f):
                    processed_keys.add(unique_key(row))
        except Exception:
            pass
    # 读取历史合并文件以构建已处理集合，避免重复分析。

    to_process = fetch_starred_repos_sorted_asc(
        token=GH_TOKEN,
        per_page=args.per_page,
        needed_unprocessed=args.batch_size,
        processed_keys=processed_keys,
    )
    # 从 GitHub 拉取按时间升序的加星仓库，过滤掉已处理项，返回本次批次待分析的列表。

    if not to_process:
        print("没有可处理的新项目，或者已到列表末尾。")
        # 如已有合并文件则保持；如存在旧的 part 文件，兼容性合并一次
        if os.path.exists(os.path.join(args.output_dir, "results_all.json")):
            return
        merge_all_formats(args.output_dir)
        return

    analyzed_rows: List[Dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for repo in to_process:
        # 逐个仓库进行模型分析，得到初始分类、标签与摘要。
        analysis = analyze_with_zhipu(
            API_KEY,
            repo,
            model=args.model,
            allowed_categories=allowed_categories,
        )
        # 对模型返回的分类进行规范化（仅保留允许列表，或用关键词规则映射），无法确定则用默认分类。
        normalized_cat = normalize_category(
            analysis.get("category"),
            repo,
            allowed_categories,
            category_rules,
            default_category=args.default_category,
        )
        row = {
            **repo,
            "category": normalized_cat,
            "tags": analysis.get("tags", []),
            "summary": analysis.get("summary", ""),
            "analyzed_at": now_iso,
        }
        analyzed_rows.append(row)
        time.sleep(args.sleep)

    # 直接合并到 results_all.*
    merge_with_new_rows(args.output_dir, analyzed_rows)
    print(f"批次完成：{len(analyzed_rows)} 项。已更新合并文件：outputs/results_all.*")


if __name__ == "__main__":
    main()
