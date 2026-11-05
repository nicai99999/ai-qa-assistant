"""查询改写：用对话历史把代词问句补全成独立问句。"""

from __future__ import annotations

import logging
import time                                    # ← 新增

from llm import get_llm
from logging_setup import log_llm_call         # ← 新增

logger = logging.getLogger(__name__)

REWRITE_HISTORY_TURNS = 4

REWRITE_PROMPT = """根据下面的对话历史，把用户的最新问题改写成一个**独立完整**的问题。

要求：
1. 把所有代词（它、这个、那个、这些）替换成具体对象
2. 保留原问题的意图，**不要回答问题**
3. **不要展开或补充答案内容**，只补全指代对象
4. 只输出改写后的问题，不要任何解释、不要引号

对话历史：
{history}

最新问题：{question}

改写后的问题："""


def _format_history(turns: list) -> str:
    lines: list[str] = []
    for turn in turns[-REWRITE_HISTORY_TURNS:]:
        lines.append(f"用户：{turn.question}")
        answer = turn.answer if len(turn.answer) <= 80 else turn.answer[:80] + "…"
        lines.append(f"助手：{answer}")
    return "\n".join(lines)


def rewrite_query(question: str, turns: list) -> str:
    """用历史把问题改写得更完整。任何异常情况都退回原问题。"""
    question = (question or "").strip()
    if not question:
        return question

    # 护栏 1：没有历史就不改写
    if not turns:
        logger.debug("无历史，跳过改写")
        return question

    history = _format_history(turns)
    prompt = REWRITE_PROMPT.format(history=history, question=question)

    # 计时 + 调用（失败就退回）
    start = time.perf_counter()
    try:
        result = get_llm().invoke(prompt)
    except Exception:
        logger.warning("改写失败，退回原问题：%r", question, exc_info=True)
        # exc_info=True = 把异常堆栈一起打出来，这是排错的关键。
        return question

    # ★ 关键：改写调用的 token 也要记，否则成本统计有洞
    elapsed_ms = (time.perf_counter() - start) * 1000
    log_llm_call(logger, elapsed_ms, getattr(result, "usage_metadata", None))

    rewritten = str(result.content).strip().strip('"').strip()
    # 去除首尾空白字符 , 去除首尾" , 防止"内部有空白字符

    # 护栏 2：空或过长就退回
    if not rewritten or len(rewritten) > 200:
        logger.debug("改写结果异常（长度 %d），退回原问题", len(rewritten))
        return question

    logger.info("改写：%r → %r", question, rewritten)
    return rewritten