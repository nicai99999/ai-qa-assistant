"""Day 19 试跑：RAG 问答 + 幻觉观察 + Top-K 对比。

用法：
    uv run python try_rag_qa.py
"""

import logging

from rag_qa import ask_with_rag
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

# 前两个是正常问题（文档里有答案），后两个是超纲问题（文档里没有）
# 前几个是正常问题（你的面试文档里有非常详细的答案）
# 后几个是超纲问题（文档里完全没有提及）
CASES = [
    ("正常1", "RAG的完整链路是什么？"),
    ("正常2", "如果对话历史太长，有哪几种处理方案？"),

    ("超纲1", "LangChain的创始人是谁？"),
    ("超纲2", "这个项目目前部署在哪个云服务器上？"),
]


def show(label: str, question: str, k: int = 3) -> None:
    result = ask_with_rag(question, k=k)

    print("\n" + "=" * 60)
    print(f"[{label}] 问：{question}")
    print(f"召回：{len(result.hits)} 条，最高相似度：{result.top_score:.4f}")
    for i, hit in enumerate(result.hits, start=1):
        score = hit.get("score") or 0.0
        source = (hit.get("metadata") or {}).get("source", "?")
        text = (hit.get("text") or "")[:40].replace("\n", " ")
        print(f"  {i}. score={score:.4f} 来源={source} | {text}")
    print(f"答：{result.answer}")
    print(f"判定：{'✅ 说了未提及' if result.is_no_answer else '⚠️ 没有说不确定'}")


def main() -> None:
    setup_logging()

    print("\n########## 实验 1：正常问题 vs 超纲问题 ##########")
    for label, question in CASES:
        show(label, question)

    print("\n########## 实验 3：Top-K 对比（同一问题，K=1/3/5）##########")
    question = CASES[0][1]
    for k in (1, 3, 5):
        result = ask_with_rag(question, k=k)
        scores = [f"{h.get('score', 0):.4f}" for h in result.hits]
        print(f"\nK={k}  召回={len(result.hits)}  分数={scores}")
        print(f"  答案：{result.answer[:80]}")


if __name__ == "__main__":
    main()