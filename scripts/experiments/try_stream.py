"""Day 22 试跑：流式输出 + 首字延迟。

用法：
    uv run python try_stream.py
"""

import logging

from src.rag_qa import ask_with_rag_stream
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

QUESTIONS = [
    "SSE 和 WebSocket 有什么区别？",
    "请详细解释 RAG 的完整链路，每一步都要说明作用",
]


def main() -> None:
    setup_logging()

    for question in QUESTIONS:
        print("\n" + "=" * 60)
        print(f"问：{question}")
        print("-" * 60)
        result = ask_with_rag_stream(question)
        print("-" * 60)
        print(f"首字延迟：{result.ttft_ms:.0f} ms")
        print(f"总耗时：{result.total_ms:.0f} ms")
        print(f"回答字数：{len(result.answer)}")


if __name__ == "__main__":
    main()