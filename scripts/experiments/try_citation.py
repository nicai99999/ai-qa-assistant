"""Day 20 试跑：带引用的 RAG 问答 + 引用准确性实验。

用法：
    uv run python try_citation.py
"""

import logging

from src.rag_qa import (
    RAG_PROMPT_A,
    ask_with_citation,
    extract_cited_indices,
    format_answer,
)
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

QUESTIONS = [
    # === 正常问题：文档里有极其详细的答案（测试检索命中率和上下文拼接）===
    "RAG的完整链路包含哪些步骤？",
    # "如果对话历史太长，有哪几种处理方案？",
    # "SSE和WebSocket有什么区别？",

    # # === 进阶问题：需要模型综合上下文归纳（测试大模型的阅读理解能力）===
    # "为什么说LangGraph是图，它和普通的Chain有什么区别？",
    # "怎么防止Agent调用危险的接口或遭遇Prompt注入？",
    #
    # # === 超纲问题：文档里绝对没有提到（测试防幻觉和NO_ANSWER拦截）===
    # "LangChain的创始人是谁？"
]


def main() -> None:
    setup_logging()

    print("\n" + "=" * 70)
    print("实验 1 + 2：带引用的问答")
    print("=" * 70)

    for question in QUESTIONS:
        result = ask_with_citation(question)
        cited = extract_cited_indices(result.answer)

        print("\n" + "-" * 70)
        print(f"问：{question}")
        print(format_answer(result))
        print(f"\n模型实际引用的编号：{cited}")
        print(f"本次上下文可用编号：{[c.index for c in result.citations]}")

        if cited:
            invalid = [n for n in cited if n not in [c.index for c in result.citations]]
            if invalid:
                print(f"⚠️ 引用了不存在的编号：{invalid}")

    print("\n" + "=" * 70)
    print("实验 3：有引用 vs 无引用，答案长度对比")
    print("=" * 70)

    question = QUESTIONS[0]

    with_cited = ask_with_citation(question)
    without_cited = ask_with_rag_plain(question)

    print(f"\n带引用 答案字数：{len(with_cited.answer)}")
    print(f"不带引用答案字数：{len(without_cited.answer)}")
    print(f"\n【带引用】\n{with_cited.answer}")
    print(f"\n【不带引用】\n{without_cited.answer}")


def ask_with_rag_plain(question: str):
    """不带引用的版本，用于对比。"""
    from src.rag_qa import ask_with_rag

    return ask_with_rag(question, prompt_template=RAG_PROMPT_A)


if __name__ == "__main__":
    main()