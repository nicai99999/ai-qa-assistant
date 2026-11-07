"""评测脚本 v2：关键词命中 + 新旧 collection 对比。"""

import json
import logging
from pathlib import Path

# 修正了导入路径，使用标准的包名
from src.retriever import retrieve
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

EVAL_PATH = Path(r"D:\ai-qa-assistant\data\eval_set_v2.json")
MIN_KEYWORD_HITS = 2


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def keyword_hits(hits: list[dict], keywords: list[str]) -> int:
    """有多少条召回块，包含了至少 2 个关键词。"""
    count = 0
    for hit in hits:
        text = hit.get("text") or ""
        # 如果该块中包含了 keywords 列表里的词，计数+1
        matched = sum(1 for kw in keywords if kw in text)
        if matched >= MIN_KEYWORD_HITS:
            count += 1
    return count


def top1_ok(hits: list[dict], keywords: list[str]) -> bool:
    """Top-1 召回块是否包含了至少 2 个关键词。"""
    if not hits:
        return False
    text = hits[0].get("text") or ""
    return sum(1 for kw in keywords if kw in text) >= MIN_KEYWORD_HITS


def run_one(name: str, retriever_fn, questions: list[dict], k: int) -> dict:
    print(f">>> 执行：{name}")
    answerable = [q for q in questions if q["answerable"]]
    top1 = topk = 0

    for item in answerable:
        hits = retriever_fn(item["question"], k=k)
        # Top-K 命中：只要召回的一堆块里，有一个块达标就行
        if keyword_hits(hits, item["expected_keywords"]) > 0:
            topk += 1
        # Top-1 命中：必须严格是排第一的那块达标
        if top1_ok(hits, item["expected_keywords"]):
            top1 += 1

    total = len(answerable)
    r1 = top1 / total if total else 0.0
    rk = topk / total if total else 0.0
    print(f"\n【{name}】Top1 {top1}/{total}={r1:.1%}   TopK {topk}/{total}={rk:.1%}")
    return {"name": name, "rate1": r1, "ratek": rk, "total": total}


def main() -> None:
    setup_logging()
    questions = load_eval_set()

    print("=" * 64)
    print(f"评测集 v2：{len(questions)} 题（关键词匹配，至少命中 {MIN_KEYWORD_HITS} 个）")
    print("=" * 64)

    # ★ 删除了“相邻块扩展”的测试，保留了新旧库的对比
    results = [
        run_one(
            "v1 基线 K=9",
            lambda q, k: retrieve(q, k=k, collection="rag_notes"),
            questions, 9
        ),
        run_one(
            "v2 基线 K=9",
            lambda q, k: retrieve(q, k=k, collection="rag_notes_v2"),
            questions, 9
        ),
        run_one(
            "v2 基线 K=5",
            lambda q, k: retrieve(q, k=k, collection="rag_notes_v2"),
            questions, 5
        ),
    ]

    print("\n" + "=" * 64)
    print("汇总")
    print("=" * 64)
    print(f"{'方案':<24}{'Top-1':>10}{'Top-K':>10}")
    for r in results:
        print(f"{r['name']:<24}{r['rate1']:>9.1%}{r['ratek']:>10.1%}")


if __name__ == "__main__":
    main()