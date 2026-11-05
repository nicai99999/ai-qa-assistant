"""带上下文管理的 RAG 问答。

和 Day 24 的 ask_with_history 的区别：
    支持"摘要 + 最近几轮"的组合，
    所以既能记住很久以前的事实，又不让 token 无限膨胀。
"""

from __future__ import annotations

import logging

from src.retriever import TOP_K, retrieve
from llm import get_llm
from logging_setup import log_llm_call
from rag_qa import (
    NO_ANSWER,
    RagResult,
    build_citations,
    build_context,
)

logger = logging.getLogger(__name__)

MEMORY_PROMPT = """你是一个严谨的问答助手。

对话记忆：
{memory}

资料：
{context}

问题：{question}

要求：
1. 只使用资料中的信息回答知识性问题，不要编造。
2. 每个结论后面用 [1]、[2] 标注依据来自第几段资料。
3. 如果资料中没有相关信息，只回答：资料中未提及
4. 如果问题是在询问用户自己说过的信息（姓名、偏好、之前提过的事），
   直接根据"对话记忆"回答，并在末尾标注（来自对话记忆）。

回答："""


def _format_turns(turns: list) -> str:
    lines: list[str] = []
    for turn in turns:
        lines.append(f"用户：{turn.question}")
        answer = turn.answer if len(turn.answer) <= 100 else turn.answer[:100] + "…"
        lines.append(f"助手：{answer}")
    return "\n".join(lines)


def build_memory_block(store_obj, session_id: str) -> str:
    """把"摘要 + 最近几轮"拼成一段记忆文本。"""
    parts: list[str] = []

    summary = store_obj.summary(session_id)
    if summary:
        parts.append(f"【早期对话摘要】\n{summary}")

    turns = store_obj.history(session_id)
    if turns:
        parts.append(f"【最近对话】\n{_format_turns(turns)}")

        summary_len = len(summary) if summary else 0
        logger.debug("记忆块：摘要=%d字 历史=%d轮", summary_len, len(turns))

    return "\n\n".join(parts) if parts else "（无）"


def ask_with_store(
    question: str,
    session_id: str,
    store_obj,
    k: int = TOP_K,
) -> RagResult:
    """用指定 store 做一次带上下文管理的 RAG 问答。"""
    import time

    question = (question or "").strip()
    if not question:
        raise ValueError("question 不能为空")

    memory = build_memory_block(store_obj, session_id)

    hits = retrieve(question, k=k)
    if not hits:
        store_obj.append(session_id, question, NO_ANSWER)
        return RagResult(answer=NO_ANSWER)

    context = build_context(hits)
    citations = build_citations(hits)

    prompt = MEMORY_PROMPT.format(
        memory=memory, context=context, question=question
    )

    llm = get_llm()
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed_ms = (time.perf_counter() - start) * 1000

    answer = str(response.content).strip()
    log_llm_call(logger, elapsed_ms, getattr(response, "usage_metadata", None))

    store_obj.append(session_id, question, answer)

    return RagResult(
        answer=answer,
        hits=hits,
        context=context,
        prompt=prompt,
        citations=citations,
    )