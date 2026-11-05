"""Day 24 试跑：多轮对话 + token 增长观察。

用法：
    uv run python try_multi_turn.py
"""

import logging
import sys

from src.rag_qa import ask_with_history, format_answer
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

SESSION = "test-session"

# 前两轮建立"记忆"，第 3 轮用代词提问，观察检索是否退化
SCRIPT = [
    "我叫小明，请记住。",
    "RAG 的完整链路是什么？",
    "它有哪些步骤？",              # ← 代词问题，预计检索会退化
    "SSE 和 WebSocket 有什么区别？",
    "为什么大模型要用流式输出？",
]


def main() -> None:
    setup_logging()

    for index, question in enumerate(SCRIPT, start=1):
        print("\n" + "=" * 60)
        print(f"第 {index} 轮｜问：{question}")
        print("-" * 60)

        result = ask_with_history(question, session_id=SESSION, k=5)
        print(format_answer(result))

    print("\n" + "=" * 60)
    print("最后问：我叫什么名字？")
    print("-" * 60)
    result = ask_with_history("我叫什么名字？", session_id=SESSION, k=5)
    print(result.answer)

    # 清空会话再问一次，验证"记忆确实被清掉了"
    from memory import store

    store.clear(SESSION)
    print("\n" + "=" * 60)
    print("（已清空会话）再问：我叫什么名字？")
    print("-" * 60)
    result = ask_with_history("我叫什么名字？", session_id=SESSION, k=5)
    print(result.answer)


if __name__ == "__main__":
    main()