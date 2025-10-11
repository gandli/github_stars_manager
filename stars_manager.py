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
ZHIPU_BASE_URL = os.environ.get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")


# --------- Fixed categories ---------
ALLOWED_CATEGORIES = [
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


def classify_category_by_keywords(repo: Dict[str, Any]) -> Optional[str]:
    text = " ".join(
        [
            str(repo.get("repo_full_name") or ""),
            str(repo.get("description") or ""),
            " ".join(repo.get("topics", []) or []),
        ]
    ).lower()

    def has_any(words: List[str]) -> bool:
        return any(w in text for w in words)

    if has_any(["web", "http", "rest", "frontend", "backend", "website", "spa", "vue", "react", "svelte", "nextjs", "nuxt"]):
        return "Web应用"
    if has_any(["android", "ios", "mobile", "apk", "react native", "flutter", "cordova"]):
        return "移动应用"
    if has_any(["desktop", "electron", "qt", "gtk", "win32", "macos", "wxwidgets"]):
        return "桌面应用"
    if has_any(["database", "db", "sql", "nosql", "postgres", "mysql", "mongodb", "redis", "cassandra", "sqlite"]):
        return "数据库"
    if has_any(["machine learning", "ml", "ai", "deep learning", "transformer", "llm", "pytorch", "tensorflow", "keras"]):
        return "AI/机器学习"
    if has_any(["dev", "developer", "sdk", "library", "framework", "build", "compile", "cli", "lint", "ci", "testing", "tool"]):
        return "开发工具"
    if has_any(["security", "vuln", "pentest", "penetration", "exploit", "auth", "encryption", "ssl", "xss", "cve"]):
        return "安全工具"
    if has_any(["game", "gaming", "unity", "unreal", "godot"]):
        return "游戏"
    if has_any(["design", "ui", "ux", "figma", "sketch", "graphics", "svg", "illustration"]):
        return "设计工具"
    if has_any(["productivity", "todo", "note", "task", "calendar", "automation", "workflow"]):
        return "效率工具"
    if has_any(["education", "learn", "learning", "tutorial", "course", "teaching", "docs", "examples"]):
        return "教育学习"
    if has_any(["social", "network", "chat", "messaging", "community", "forum", "sns"]):
        return "社交网络"
    if has_any(["data analysis", "analytics", "bi", "pandas", "numpy", "visualization", "chart", "plot"]):
        return "数据分析"
    return None


def normalize_category(candidate: Optional[str], repo: Dict[str, Any]) -> str:
    if candidate in ALLOWED_CATEGORIES:
        return candidate
    # Try heuristic mapping
    mapped = classify_category_by_keywords(repo)
    if mapped:
        return mapped
    # Default fallback
    return "开发工具"


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
        base_result["summary"] = "ZHIPU_API_KEY 未设置，跳过分析。"
        return base_result

    prompt = (
        "你是一位资深开源项目分类助手。"
        "请根据仓库名称、简介和主题，给出简洁的中文摘要，并从以下固定分类中严格选择一个作为类别："
        f"{', '.join(ALLOWED_CATEGORIES)}。"
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
        client = OpenAI(api_key=api_key, base_url=ZHIPU_BASE_URL)
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
        category = normalize_category(category, repo)
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


# --------- Main batch execution ---------
def main() -> None:
    parser = argparse.ArgumentParser(description="GitHub Star 项目分类、标签与摘要生成")
    parser.add_argument("--batch-size", type=int, default=10, help="每次处理的项目数量")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--model", type=str, default=os.environ.get("ZHIPU_MODEL", "glm-4"), help="ZHIPU AI 模型名称")
    parser.add_argument("--per-page", type=int, default=100, help="GitHub API 每页大小")
    parser.add_argument("--sleep", type=float, default=0.6, help="每次分析间隔秒数")
    args = parser.parse_args()

    load_env()

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print(
            "错误：未设置 GITHUB_TOKEN。请在 .env 或环境变量中配置。", file=sys.stderr
        )
        sys.exit(1)
    zhipu_api_key = os.environ.get("ZHIPU_API_KEY")

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
        token=github_token,
        per_page=args.per_page,
        needed_unprocessed=args.batch_size,
        processed_keys=processed_keys,
    )

    if not to_process:
        print("没有可处理的新项目，或者已到列表末尾。")
        # 仍尝试合并，保证最终文件存在
        merge_all_formats(args.output_dir)
        return

    analyzed_rows: List[Dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for repo in to_process:
        analysis = analyze_with_zhipu(zhipu_api_key, repo, model=args.model)
        row = {
            **repo,
            **analysis,
            "analyzed_at": now_iso,
        }
        analyzed_rows.append(row)
        time.sleep(args.sleep)

    # Write part files named by timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_part = os.path.join(args.output_dir, f"results_part_{ts}.json")
    csv_part = os.path.join(args.output_dir, f"results_part_{ts}.csv")
    md_part = os.path.join(args.output_dir, f"results_part_{ts}.md")
    write_json(json_part, analyzed_rows)
    write_csv(csv_part, analyzed_rows)
    write_markdown(md_part, analyzed_rows)

    # Merge & dedupe into results_all.*
    merge_all_formats(args.output_dir)
    print(
        f"批次完成：{len(analyzed_rows)} 项。输出：\n- {json_part}\n- {csv_part}\n- {md_part}\n并已更新合并文件：results_all.*"
    )


if __name__ == "__main__":
    main()
