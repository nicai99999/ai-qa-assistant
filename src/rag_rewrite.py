"""带查询改写的 RAG 问答。

链路：历史 → 改写问题 → 用改写后的问题检索 → 生成 → 存历史（存原问题）
"""

from __future__ import annotations

import logging
import time

from src.retriever import TOP_K, retrieve
from llm import get_llm
from logging_setup import log_llm_call
from query_rewrite import rewrite_query
from rag_memory import build_memory_block
from rag_qa import (
    NO_ANSWER,
    RagResult,
    build_citations,
    build_context,
)

logger = logging.getLogger(__name__)

REWRITE_PROMPT_TEMPLATE = """你是一个严谨的问答助手。

对话记忆：
{memory}

资料：
{context}

问题：{question}

要求：
1. 只使用资料中的信息回答知识性问题，不要编造。
2. 每个结论后面用 [1]、[2] 标注依据来自第几段资料。
3. 如果资料中没有相关信息，只回答：资料中未提及
4. 如果问题是在询问用户自己说过的信息，直接根据"对话记忆"回答，
   并在末尾标注（来自对话记忆）。
   **如果答案来自资料，不要标注（来自对话记忆）。**

回答："""


def ask_with_rewrite(
    question: str,
    session_id: str,
    store_obj,
    k: int = TOP_K,
    use_rewrite: bool = True,
) -> RagResult:
    """带查询改写的 RAG 问答。

    和 ask_with_store 的唯一区别：
        检索用的是"改写后的问题"，而不是原问题。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question 不能为空")

    turns = store_obj.history(session_id)

    # 1. 改写（可关闭，用于做 A/B 对比）
    rewritten = rewrite_query(question, turns) if use_rewrite else question
    if rewritten != question:
        logger.info("查询改写：%r → %r", question, rewritten)

    # 2. 用改写后的问题检索
    hits = retrieve(rewritten, k=k)
    if not hits:
        store_obj.append(session_id, question, NO_ANSWER)
        return RagResult(answer=NO_ANSWER)

    # 3. 生成（记忆块用完整记忆，问题用原问题）
    memory = build_memory_block(store_obj, session_id)
    context = build_context(hits)
    citations = build_citations(hits)

    prompt = REWRITE_PROMPT_TEMPLATE.format(
        memory=memory, context=context, question=question
    )

    llm = get_llm()
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed_ms = (time.perf_counter() - start) * 1000

    answer = str(response.content).strip()
    log_llm_call(logger, elapsed_ms, getattr(response, "usage_metadata", None))

    logger.info("问答完成 召回=%d", len(hits))

    # 4. 历史里存原问题（用户真实说的）
    store_obj.append(session_id, question, answer)

    result = RagResult(
        answer=answer,
        hits=hits,
        context=context,
        prompt=prompt,
        citations=citations,
    )
    result.rewritten_query = rewritten
    return result