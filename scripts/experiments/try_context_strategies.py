"""Day 25 对比实验：三种上下文管理策略。

实验设计：
    第 1 轮埋一个关键信息（姓名 + 方向），
    中间问 9 个无关的知识问题，
    第 11 轮再问那个关键信息。
    → 看哪种策略能记住，以及各自花了多少 token。
"""

import logging
import sys

from memory_v2 import ConversationStoreV2
from rag_memory import ask_with_store
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

ANCHOR_Q = "我叫小明，我在准备 Java+AI 方向的实习。请记住。"
FILLERS = [
    "RAG 的完整链路是什么？",
    "SSE 和 WebSocket 有什么区别？",
    "什么是 ReAct？",
    "CoT 和 ReAct 的区别是什么？",
    "多轮对话怎么实现的？",
    "为什么大模型要用流式输出？",
    "什么是 Function Calling？",
    "LangChain 里 Retriever 是什么？",
    "向量检索和关键词检索的区别？",
    "什么是指令遵循（instruction following）？",
    "为什么长上下文会变慢？",
    "Rerank 是什么？",
    "什么是 Prompt 注入？",
    "多智能体（Multi-Agent）有什么优缺点？",
]
RECALL_Q = "我叫什么名字？我在准备什么方向的实习？"


def run_one(mode: str) -> dict:
    print("\n" + "#" * 70)
    print(f"# 策略：{mode}")
    print("#" * 70)

    from llm import get_llm

    store = ConversationStoreV2(mode=mode, llm=get_llm() if mode == "summary" else None)
    session = f"sess-{mode}"

    tokens: list[int] = []

    # 第 1 轮：埋信息
    result = ask_with_store(ANCHOR_Q, session, store, k=5)
    tokens.append(len(result.prompt) // 2)  # 粗估，真值看日志

    # 中间 9 轮：无关知识
    for question in FILLERS:
        result = ask_with_store(question, session, store, k=5)
        tokens.append(len(result.prompt) // 2)

    # 第 11 轮：召回测试
    result = ask_with_store(RECALL_Q, session, store, k=5)
    answer = result.answer

    remembered = ("小明" in answer) and ("Java" in answer or "AI" in answer)

    print("\n" + "-" * 70)
    print(f"第 11 轮回答：{answer}")
    print(f"记住了吗：{'✅ 是' if remembered else '❌ 否'}")
    print(f"当前历史轮数：{len(store.history(session))}")
    print(f"当前摘要长度：{len(store.summary(session))} 字")
    print(f"Prompt 长度序列（字符）：{tokens}")
    print(f"摘要内容：{store.summary(session) or '（无）'}")

    return {
        "mode": mode,
        "remembered": remembered,
        "history_turns": len(store.history(session)),
        "summary_len": len(store.summary(session)),
        "last_prompt_len": len(result.prompt),
    }


def main() -> None:
    setup_logging()
    results = [run_one(mode) for mode in ("window", "summary", "priority")]

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"{'策略':<12}{'记住关键信息':>12}{'历史轮数':>10}{'摘要字数':>10}{'末轮Prompt':>12}")
    for r in results:
        print(
            f"{r['mode']:<12}{'✅' if r['remembered'] else '❌':>10}"
            f"{r['history_turns']:>10}{r['summary_len']:>10}{r['last_prompt_len']:>12}"
        )


if __name__ == "__main__":
    main()