"""流式问答：一个字一个字地吐出来。

和阻塞版的区别：
    阻塞版：llm.invoke() → 等全部生成完 → 一次性返回
    流式版：llm.stream() → 每生成一小段就 yield 出来

调用方（Streamlit）拿到生成器后可以边收边渲染，
用户看到的是打字机效果，而不是干等。

注意：检索仍然是同步阻塞的，这段延迟省不掉。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
# 用于类型注解。明确告诉 IDE 和开发者，这个函数返回的是一个迭代器（生成器）

from src.retriever import TOP_K, retrieve
from llm import get_llm
from logging_setup import log_llm_call
from rag_qa import NO_ANSWER, build_citations, build_context

logger = logging.getLogger(__name__)

STREAM_PROMPT = """你是一个严谨的问答助手。

资料：
{context}

问题：{question}

要求：
1. 只使用资料中的信息回答，不要编造。
2. 每个结论后面用 [1]、[2] 标注依据来自第几段资料。
3. 如果资料中没有相关信息，只回答：资料中未提及

回答："""


def stream_answer(
    question: str,
    session_id: str,
    store_obj,
    collector: dict,
        # 普通函数可以用 return 把耗时、引用等信息打包返回。但生成器只能 yield 字符串，无法 return 复杂数据。
        # 所以这里传入一个外部的空字典（引用传递），把需要额外传递的数据塞进去，调用方在外面就能读到了
    k: int = TOP_K,
) -> Iterator[str]:
    """生成器：边生成边 yield 文本片段。

    Args:
        collector: 一个空字典，用于把检索结果/耗时带出去给调用方
                   （生成器没法 return 值，只能靠外部容器传出来）。
    """
    question = (question or "").strip()
    if not question:
        yield "问题不能为空。"
        # 不像 return 立刻摧毁整个函数，而是把这个字符串“抛”给调用它的地方（比如 Web 界面或终端）。
        # 抛出去之后，函数暂停。等前端接收完毕，调用者再次请求数据时，代码执行到下一行的 return。
        return

    # ---------- 阶段 1：检索（同步阻塞） ----------
    t0 = time.perf_counter()
    hits = retrieve(question, k=k)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    # elapsed_ms = (time.perf_counter() - start) * 1000
    # 如果 retrieve_ms 只有 50ms（说明数据库查得极快），而生成花了 4950ms，那瓶颈在大模型生成太慢。你需要换用更快的模型（比如 deepseek-flash），或者开启流式输出。
    # 如果 retrieve_ms 高达 2000ms，说明你的向量数据库卡了。可能是网络慢、数据量太大没有建好索引，或者你调用的云端 Embedding 接口（阿里云）响应太慢。
    collector["retrieve_ms"] = retrieve_ms
    collector["hits"] = hits
    # 把检索耗时和检索结果塞进 collector 字典里

    if not hits:
        store_obj.append(session_id, question, NO_ANSWER)
        yield NO_ANSWER
        return

    context = build_context(hits)
    collector["citations"] = build_citations(hits)

    prompt = STREAM_PROMPT.format(context=context, question=question)

    # ---------- 阶段 2：流式生成 ----------
    llm = get_llm()
    pieces: list[str] = []

    start = time.perf_counter()
    first_token_ms: float | None = None

    for chunk in llm.stream(prompt):
        text = chunk.content
        if not text:
            continue

        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - start) * 1000
            collector["first_token_ms"] = first_token_ms
            # 记录首字延迟（TTFT）

        pieces.append(str(text))
        # 把字收进列表（为了最后存历史）
        yield str(text)
        # 把字吐给调用方（前端）。前端拿到一个字符，立刻渲染一个字符，这就形成了打字机效果。

    end = time.perf_counter()
    total_ms = (end - start) * 1000
    collector["total_ms"] = total_ms

    answer = "".join(pieces).strip()

    collector["total_tokens"] = 0
    # 如果 stream 的 chunk 带 usage，可以累加；多数情况下拿不到，
    # 这里退化为按字符数粗估：
    collector["total_tokens"] = len(context) // 2 + len(answer) // 2

    store_obj.append(session_id, question, answer)

    logger.info(
        "流式完成 检索=%.0fms 首字=%.0fms 生成=%.0fms 答长=%d",
        retrieve_ms,
        first_token_ms or 0.0,
        total_ms,
        len(answer),
    )