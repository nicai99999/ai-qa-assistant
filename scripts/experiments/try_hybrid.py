"""D1 实验：向量 vs 混合检索，在评测集上对比命中率。

用法（在项目根目录）：
    uv run python scripts/experiments/try_hybrid.py

这个脚本的定位：
    它不是产品代码，是「对照实验」。
    产出 4 行数字，用来回答一个问题：加了 BM25 之后，检索到底有没有变好？

    长期价值在于「可复现」：以后想调 RRF 的 k、停用词表、BM25 的 top_n，
    改完重跑一次，就知道是变好还是变坏。没有它只能凭感觉。

注意路径层级：本文件在 scripts/experiments/ 下，
所以项目根是 parents[2]，不是 parents[1]。
"""

import json
import sys
from pathlib import Path

# scripts/experiments/try_hybrid.py
#   parents[0] = experiments
#   parents[1] = scripts
#   parents[2] = 项目根
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logging_setup import setup_logging
from src.retriever import retrieve
from hybrid_retriever import retrieve_hybrid

EVAL_PATH = ROOT / "data" / "eval_set_v2.json"
MIN_KEYWORD_HITS = 2


def keyword_ok(text: str, keywords: list[str]) -> bool:
    """一块里命中至少 2 个关键词，才算这块能回答问题。"""
    return sum(1 for kw in keywords if kw in (text or "")) >= MIN_KEYWORD_HITS
    # 先判断文本是不是空，如果不是就返回 text；如果是空就返回 ""，然后再判断 kw 在不在 text 中
    # 1 for kw in keywords if kw in (text or "")生成器表达式
    # sum类似于计数器，每当遍历出一个符合条件的词就出一个1 , sum就将所有的1累计起来


def run(name: str, fn, questions: list[dict], k: int) -> None:
    """跑一组配置，打印 Top-1 / Top-K 命中率。"""
    top1 = topk = total = 0

    for item in questions:
        if not item["answerable"]:
            continue
        total += 1
        hits = fn(item["question"], k=k)

        # ==========================================
        # 指标 1：Top-K 召回率（只要找出来了就算对）
        # ==========================================
        # hits 是检索器返回的 Top-K 个结果列表（比如前 3 条）。
        # any(...) 会遍历 hits 里的每一条结果 h：
        #   1. h.get("text")：取出该结果的文本内容。
        #   2. keyword_ok(...)：检查这段文本是否命中了期望的关键词。
        # 只要这 K 条结果里【有任意一条】命中了关键词，any 就会返回 True。
        # 此时 topk 加 1，表示系统成功在 Top-K 范围内找到了答案。
        if any(keyword_ok(h.get("text"), item["expected_keywords"]) for h in hits):
            topk += 1

        # ==========================================
        # 指标 2：Top-1 准确率（必须第一名才是对的）
        # ==========================================
        # hits 判断：必须先确保检索结果不为空，防止 hits[0] 报越界错误（IndexError）。
        # hits[0]：死死盯住排名【第 1 名】的那一条结果。
        # keyword_ok(...)：检查这唯一的第一名，是否命中了期望的关键词。
        # 只有第一名完全正确，top1 才会加 1。
        if hits and keyword_ok(hits[0].get("text"), item["expected_keywords"]):
            top1 += 1

    print(
        f"{name:<22} Top1 {top1}/{total}={top1 / total:.1%}    "
        f"TopK {topk}/{total}={topk / total:.1%}"
    )


def main() -> None:
    setup_logging()

    if not EVAL_PATH.exists():
        raise SystemExit(f"评测集不存在：{EVAL_PATH}")

    questions = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    answerable = sum(1 for q in questions if q["answerable"])

    print("=" * 72)
    print(f"评测集：{len(questions)} 题（可答 {answerable} 题，关键词命中 ≥{MIN_KEYWORD_HITS} 个）")
    print("=" * 72)

    # 前两行是主对比：同样输出 5 块，看混合比基线好多少
    run("基线 向量 K=5", retrieve, questions, 5)
    run("混合 RRF K=5", retrieve_hybrid, questions, 5)

    # 后两行是控制变量：向量多给一倍候选能不能自己追上来
    # 如果追不上，说明混合的提升不能归因于「看得多」
    run("基线 向量 K=10", retrieve, questions, 10)
    run("混合 RRF K=10", retrieve_hybrid, questions, 10)


if __name__ == "__main__":
    main()
