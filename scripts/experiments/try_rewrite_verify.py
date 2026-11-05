"""验证：改写后第 2、3 问到底答对没有。

用法：
    uv run python try_rewrite_verify.py
"""

from logging_setup import setup_logging
from llm import get_llm
from memory_v2 import ConversationStoreV2
from rag_memory import ask_with_store
from rag_rewrite import ask_with_rewrite


def main() -> None:
    setup_logging()

    store = ConversationStoreV2(mode="window", llm=get_llm())
    session = "verify-rewrite"

    # 先建立上下文
    ask_with_store("RAG 的完整链路是什么？", session, store, k=5)

    for question in ["它有哪些步骤？", "它有什么优点？", "这个方法有什么缺点？"]:
        result = ask_with_rewrite(question, session, store, k=5)
        top = result.hits[0] if result.hits else None

        print("\n" + "=" * 64)
        print(f"问：{question}")
        print(f"改写：{result.rewritten_query}")
        if top:
            print(f"Top-1：{top['id']}  相似度 {top['score']:.4f}")
        print(f"答：{result.answer}")


if __name__ == "__main__":
    main()