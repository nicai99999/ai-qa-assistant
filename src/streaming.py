"""流式问答（完整链路：路由 → 改写 → 检索 → 证据校验 → 记忆 → 生成）。

和阻塞版的区别：
    阻塞版：llm.invoke() → 等全部生成完 → 一次性返回
    流式版：llm.stream() → 每生成一小段就 yield 出来

调用方（Streamlit）拿到生成器后可以边收边渲染，
用户看到的是打字机效果，而不是干等。

注意：检索仍然是同步阻塞的，这段延迟省不掉。
分流：
    chitchat  → 直接生成（不检索）
    personal  → 只读记忆（不检索）
    knowledge → 完整 RAG 链路
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
# 用于类型注解。明确告诉 IDE 和开发者，这个函数返回的是一个迭代器（生成器）

from src.retriever import TOP_K, retrieve
from evidence_check import check_evidence
from llm import get_llm
from logging_setup import log_llm_call
from query_rewrite import rewrite_query
from rag_memory import build_memory_block
from rag_qa import NO_ANSWER, build_citations, build_context
from router import classify

logger = logging.getLogger(__name__)

MAX_QUESTION_LEN = 2000

# 证据校验策略：score | coverage | judge | staged
# 默认用 "score"：2026-11-09 实测（16 题评测集）中，
#   score 策略（阈值 0.40）达到 识别率 80% / 保留率 100% / 调和 0.89
#   staged 策略反而误拒 4 道可答题（保留率只有 64%）
#   原因：重叠区可答题的关键词覆盖率普遍偏低，被 MIN_COVERAGE 拦掉
# 如果以后语料变化导致 score 不够用，可以改回 "staged" 或 "judge" 重新对比。
EVIDENCE_STRATEGY = "score"
USE_LLM_ROUTE = False       # 规则够用；打开后短问句会多一次模型调用

STREAM_PROMPT = """你是一个严谨的问答助手。

对话记忆：
{memory}

资料：
{context}

问题：{question}

要求：
1. 只使用资料中的信息回答，不要编造。
2. 每个结论后面用 [1]、[2] 标注依据来自第几段资料。
3. 如果资料中没有相关信息，只回答：资料中未提及
4. 如果问题是在询问用户自己说过的信息，直接根据"对话记忆"回答，
   并在末尾标注（来自对话记忆）。

回答："""

MEMORY_ONLY_PROMPT = """你是一个助手，只根据对话记忆回答。

对话记忆：
{memory}

问题：{question}

要求：
1. 只用记忆里的信息回答，不要编造。
2. 如果记忆里没有，回答：对话中没有提到过。
3. 不要检索，也不要引用资料编号。

回答："""

CHITCHAT_PROMPT = """你是一个友好的助手。用户在进行日常寒暄。

用户：{question}

要求：简短、自然、有礼貌地回应，一两句话即可。不要展开技术内容。

回答："""


def _stream_llm(llm, prompt: str, collector: dict, holder: dict) -> Iterator[str]:
    """把一段 prompt 流式生成出来，回答全文通过 holder 带出。

    三个分支（闲聊/个人/知识）都要生成，所以抽出来避免重复。
    """
    pieces: list[str] = []
    # 收进列表，为了最后存历史

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
        # 收进列表，为了最后存历史

        yield str(text)
        # 吐给调用方。前端拿到一个片段就渲染一个，形成打字机效果。

    collector["total_ms"] = (time.perf_counter() - start) * 1000
    holder["answer"] = "".join(pieces).strip()


def stream_answer(
    question: str,
    session_id: str,
    store_obj,
    collector: dict,
    k: int = TOP_K,
    where: dict | None = None,
    use_rewrite: bool = True,
    evidence_strategy: str = EVIDENCE_STRATEGY,
    use_llm_route: bool = USE_LLM_ROUTE,
) -> Iterator[str]:
    """生成器：边生成边 yield 文本片段。

    Args:
        collector: 普通函数能用 return 打包返回，生成器不行。
                   所以传一个外部字典（引用传递），把耗时/引用/token 塞进去，
                   调用方在外面读。
        where: 可选的检索过滤条件，如 {"source": "xxx.pdf"}。
               不传 = 在整个知识库检索（默认）。
        use_rewrite: 是否用历史改写问题。关掉可做 A/B 对比。
        evidence_strategy: 证据校验策略，见 evidence_check.check_evidence。
    """
    question = (question or "").strip()

    # 空问题要立刻结束。少了 return 就会继续往下走，拿空串去检索
    if not question:
        yield "问题不能为空。"
        return

    if len(question) > MAX_QUESTION_LEN:
        yield f"问题过长（{len(question)} 字），请控制在 {MAX_QUESTION_LEN} 字以内。"
        return

    llm = get_llm()
    holder: dict = {}

    # ---------- 阶段 0：路由 ----------
    # 必须在检索之前——否则"你好"照样会检索一遍
    route = classify(question, llm=llm, use_llm=use_llm_route)
    collector["route"] = route.as_dict()

    # ---------- 分支 1：闲聊 ----------
    if route.intent == "chitchat":
        prompt = CHITCHAT_PROMPT.format(question=question)
        yield from _stream_llm(llm, prompt, collector, holder)
        store_obj.append(session_id, question, holder.get("answer", ""))
        return

    # ---------- 分支 2：个人信息 ----------
    if route.intent == "personal":
        memory = build_memory_block(store_obj, session_id)
        prompt = MEMORY_ONLY_PROMPT.format(memory=memory, question=question)
        yield from _stream_llm(llm, prompt, collector, holder)
        store_obj.append(session_id, question, holder.get("answer", ""))
        return

    # ---------- 分支 3：知识问题（原 RAG 链路） ----------
    # ---------- 阶段 1：改写 ----------
    # 代词句（"它有哪些步骤"）没有实词，直接检索会退化。
    # 先用对话历史把它补成完整问句，再拿去检索。
    turns = store_obj.history(session_id)
    rewritten = rewrite_query(question, turns) if use_rewrite else question
    collector["rewritten"] = rewritten
    collector["where"] = where

    # ---------- 阶段 2：检索（同步阻塞） ----------
    t0 = time.perf_counter()
    hits = retrieve(rewritten, k=k, where=where)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    collector["retrieve_ms"] = retrieve_ms
    collector["hits"] = hits

    if not hits:
        store_obj.append(session_id, question, NO_ANSWER)
        yield NO_ANSWER
        return

    # ---------- 阶段 3：证据校验 ----------
    # 检索总会返回 Top-K，哪怕库里没有相关内容。
    # 这里判断"资料是否真的能回答问题"，不能就拒答，避免硬编。
    verdict = check_evidence(question, hits, strategy=evidence_strategy, llm=llm)
    collector["evidence"] = verdict.as_dict()

    if not verdict.ok:
        message = f"{NO_ANSWER}（{verdict.reason}）"
        store_obj.append(session_id, question, message)
        yield message
        return

    context = build_context(hits)
    collector["citations"] = build_citations(hits)

    # ---------- 阶段 4：拼记忆块 ----------
    # 没有这一步，多轮对话就是假的——模型看不到历史。
    memory = build_memory_block(store_obj, session_id)

    prompt = STREAM_PROMPT.format(
        memory=memory, context=context, question=question
    )

    # ---------- 阶段 5：流式生成 ----------
    yield from _stream_llm(llm, prompt, collector, holder)

    answer = holder.get("answer", "")

    # 流式接口通常拿不到精确 usage，按字符数粗估；界面要标"估算"
    collector["total_tokens"] = len(context) // 2 + len(answer) // 2

    store_obj.append(session_id, question, answer)

    logger.info(
        "流式完成 路由=%s 改写=%r 检索=%.0fms 答长=%d",
        route.intent, rewritten, retrieve_ms, len(answer),
    )