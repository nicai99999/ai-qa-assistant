"""Day 26 对比实验：改写前 vs 改写后。

用法：
    uv run python try_query_rewrite.py
"""

import logging

from src.retriever import retrieve
from llm import get_llm
from logging_setup import setup_logging
from memory_v2 import ConversationStoreV2
from query_rewrite import rewrite_query

logger = logging.getLogger(__name__)

SETUP_Q = "RAG 的完整链路是什么？"

PRONOUN_CASES = [
    "它有哪些步骤？",
    "它有什么优点？",
    "这个方法有什么缺点？",
]


def main() -> None:
    setup_logging()

    # 先建一个真实会话，拿到历史
    store = ConversationStoreV2(mode="window", llm=get_llm())
    session = "rewrite-test"

    # 用一次真实调用把历史填进去
    from rag_memory import ask_with_store

    ask_with_store(SETUP_Q, session, store, k=5)
    turns = store.history(session)
    print(f"历史已建立：{len(turns)} 轮")

    print("\n" + "=" * 70)
    print("对比：原问题检索 vs 改写后检索")
    print("=" * 70)

    for question in PRONOUN_CASES:
        rewritten = rewrite_query(question, turns)

        hits_before = retrieve(question, k=5)
        hits_after = retrieve(rewritten, k=5)

        top_before = hits_before[0]["score"] if hits_before else 0.0
        top_after = hits_after[0]["score"] if hits_after else 0.0

        print(f"\n原问题：{question}")
        print(f"改写后：{rewritten}")
        print(f"改写前 Top-1：{top_before:.4f}  ({hits_before[0]['id'] if hits_before else '-'})")
        print(f"改写后 Top-1：{top_after:.4f}  ({hits_after[0]['id'] if hits_after else '-'})")
        print(f"提升：{(top_after - top_before) * 100:+.1f} 个百分点")

    # 端到端跑一次，确认改写版能答对
    print("\n" + "=" * 70)
    print("端到端：带改写问 '它有哪些步骤？'")
    print("=" * 70)

    from rag_rewrite import ask_with_rewrite

    result = ask_with_rewrite("它有哪些步骤？", session, store, k=5)
    print(f"改写为：{result.rewritten_query}")
    print(f"回答：{result.answer}")


if __name__ == "__main__":
    main()