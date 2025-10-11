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
    """Fetch starred repos, ascending by starred_at, returning next unprocessed items.

    processed_keys: set of unique keys to skip (e.g., f"{full_name}|{starred_at}").
    Stops once it collects needed_unprocessed items.
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


# --------- Categories configuration (flexible) ---------
DEFAULT_ALLOWED_CATEGORIES = [
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
    "Web应用": [
        "web", "http", "rest", "frontend", "backend", "website", "spa",
        "vue", "react", "svelte", "nextjs", "nuxt",
    ],
    "移动应用": ["android", "ios", "mobile", "apk", "react native", "flutter", "cordova"],
    "桌面应用": ["desktop", "electron", "qt", "gtk", "win32", "macos", "wxwidgets"],
    "数据库": ["database", "db", "sql", "nosql", "postgres", "mysql", "mongodb", "redis", "cassandra", "sqlite"],
    "AI/机器学习": ["machine learning", "ml", "ai", "deep learning", "transformer", "llm", "pytorch", "tensorflow", "keras"],
    "开发工具": ["dev", "developer", "sdk", "library", "framework", "build", "compile", "cli", "lint", "ci", "testing", "tool"],
    "安全工具": ["security", "vuln", "pentest", "penetration", "exploit", "auth", "encryption", "ssl", "xss", "cve"],
    "游戏": ["game", "gaming", "unity", "unreal", "godot"],
    "设计工具": ["design", "ui", "ux", "figma", "sketch", "graphics", "svg", "illustration"],
    "效率工具": ["productivity", "todo", "note", "task", "calendar", "automation", "workflow"],
    "教育学习": ["education", "learn", "learning", "tutorial", "course", "teaching", "docs", "examples"],
    "社交网络": ["social", "network", "chat", "messaging", "community", "forum", "sns"],
    "数据分析": ["data analysis", "analytics", "bi", "pandas", "numpy", "visualization", "chart", "plot"],
}

def load_categories(file_path: Optional[str] = None, env_key: str = "CATEGORIES") -> List[str]:
    """Load allowed categories from file (JSON array) or env (comma-separated)."""
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
    """Load category keyword rules from JSON mapping {category: [keywords...]}."""
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


def classify_category_by_keywords(repo: Dict[str, Any], category_rules: Dict[str, List[str]]) -> Optional[str]:
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
    if candidate in allowed_categories:
        return candidate
    mapped = classify_category_by_keywords(repo, category_rules)
    if mapped and mapped in allowed_categories:
        return mapped
    return default_category


def parse_json_content(content: Optional[str]) -> Dict[str, Any]:
    """Robustly parse JSON from model content.
    - Strips code fences like ```json ... ```
    - If parsing fails, attempts to extract substring between first '{' and last '}'
    - Returns dict; empty on failure
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
    """Call ZHIPU AI to get category/tags/summary for a repo.
    If api_key is None or call fails, return a fallback analysis.
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
    os.makedirs(path, exist_ok=True)


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
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
    return f"{row.get('repo_full_name')}|{row.get('starred_at')}"


def merge_json_files(output_dir: str, merged_path: str) -> List[Dict[str, Any]]:
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
    json_path = os.path.join(output_dir, "results_all.json")
    rows = merge_json_files(output_dir, json_path)
    csv_path = os.path.join(output_dir, "results_all.csv")
    md_path = os.path.join(output_dir, "results_all.md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)


def merge_with_new_rows(output_dir: str, new_rows: List[Dict[str, Any]]) -> None:
    """Merge analyzed rows directly into results_all.* without creating part files."""
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

    GH_TOKEN = os.environ.get("GH_TOKEN")
    if not GH_TOKEN:
        print(
            "错误：未设置 GH_TOKEN。请在 .env 或环境变量中配置。", file=sys.stderr
        )
        sys.exit(1)
    API_KEY = os.environ.get("API_KEY")

    # Resolve BASE_URL from CLI or environment (supports .env)
    global BASE_URL
    BASE_URL = (
        args.base_url
        or os.environ.get("BASE_URL")
        or BASE_URL
        or "https://open.bigmodel.cn/api/paas/v4/"
    )

    # Load configurable categories and rules
    allowed_categories = load_categories(args.categories_file)
    category_rules = load_category_rules(args.category_rules_file)

    ensure_output_dir(args.output_dir)

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

    to_process = fetch_starred_repos_sorted_asc(
        token=GH_TOKEN,
        per_page=args.per_page,
        needed_unprocessed=args.batch_size,
        processed_keys=processed_keys,
    )

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
        analysis = analyze_with_zhipu(
            API_KEY,
            repo,
            model=args.model,
            allowed_categories=allowed_categories,
        )
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
    print(
        f"批次完成：{len(analyzed_rows)} 项。已更新合并文件：outputs/results_all.*"
    )


if __name__ == "__main__":
    main()
