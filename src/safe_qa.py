"""健壮性包装层：把任何异常翻译成用户能看懂的话，并统一记日志。

为什么需要这一层：
    上层业务代码不应该关心"是 401 还是超时"。
    所有异常在这里收口，业务层只拿到一句人话。

设计选择：
    不按异常类名 except（openai 的异常类在不同版本会改名），
    而是**检查类名 + 错误文本里的特征**，这样跨版本更稳。
"""

from __future__ import annotations

import logging
import time

from llm import get_llm
from memory_v2 import ConversationStoreV2
from rag_rewrite import ask_with_rewrite

logger = logging.getLogger(__name__)

MAX_QUESTION_LEN = 2000

FALLBACK = "系统开小差了，请稍后重试。"


def friendly_message(exc: Exception) -> str:
    """把异常翻译成用户能看懂的话。

    判断顺序有讲究：先判断特征更明确的（认证/额度），
    最后才是兜底。
    """
    name = type(exc).__name__.lower()
    # 拿到异常的对象名（比如 TimeoutError 变成 timeouterror）。
    # type(exc)：获取这个对象所属的类。因为 exc 是 TimeoutError 类的实例，所以 type(exc) 返回的就是 TimeoutError 这个类本身。
    # 类本身也是一个对象，它有一个内置属性叫 __name__，存的就是这个类的名字

    text = str(exc).lower()
    # 拿到异常的具体报错文本

    if "authentication" in name or "401" in text or "invalid api key" in text:
        return "API Key 无效或已过期，请检查 .env 配置。"
    # 先判断特征最明显的（Key 错了、没钱了），再判断网络问题，最后用 FALLBACK 兜底
    if "insufficient" in text or "balance" in text or "quota" in text:
        return "账户额度不足，请充值后再试。"

    if "rate" in name or "429" in text or "rate limit" in text:
        return "请求过于频繁，请稍后重试。"

    if "timeout" in name or "timed out" in text:
        return "模型响应超时，请稍后重试。"

    if "connection" in name or "network" in text or "unreachable" in text:
        return "网络连接失败，请检查网络或代理。"

    return FALLBACK


def validate_question(question: str) -> str:
    """校验输入，不合格就抛 ValueError（由 safe_ask 统一转成提示）。"""
    # 前置过滤器。在真正花钱调用 API 之前，先把垃圾数据拦下来
    question = (question or "").strip()
    # (question or "").strip()：防御性写法。如果传进来 None，None or "" 会变成 ""，然后再 .strip() 就安全了，绝不会因为 None.strip() 报错

    if not question:
        raise ValueError("问题不能为空。")

    if len(question) > MAX_QUESTION_LEN:
        raise ValueError(
            f"问题过长（{len(question)} 字），请控制在 {MAX_QUESTION_LEN} 字以内。"
        )

    return question


def safe_ask(
    question: str,
    session_id: str,
    store_obj: ConversationStoreV2,
    k: int = 5,
) -> str:
    """安全问答入口：任何异常都不会向上抛。

    日志会记录三项：耗时、token（由下层记录）、成功/失败。
    """
    start = time.perf_counter()
    question_preview = len(question)
    # 为了日志脱敏。如果用户输入了 5000 字，日志里只记前 30 个字，防止日志文件瞬间被撑爆

    try:
        cleaned = validate_question(question)

        result = ask_with_rewrite(cleaned, session_id, store_obj, k=k)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "调用成功 session=%s 耗时=%.0fms 问=%r 召回=%d 答长=%d",
            session_id,
            elapsed_ms,
            question_preview,
            len(result.hits),
            len(result.answer),
        )
        return result.answer

    except ValueError as exc:
        # 输入问题：不算系统错误，用 warning
        logger.warning("输入不合法：%s | 原始问题=%r", exc, question_preview)
        return str(exc)

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "调用失败 session=%s 耗时=%.0fms 问=%r 异常=%s: %s",
            session_id,
            elapsed_ms,
            question_preview,
            type(exc).__name__,
            exc,
            exc_info=True,
            # 在日志里把完整的报错堆栈（Traceback）打印出来”。在排查代码 Bug 时，能直接看到是哪一行代码崩的
        )
        return friendly_message(exc)

def safe_agent_invoke(agent, question: str) -> str:
    """agent 层的兜底：工具没挡住异常时，这里接住。"""
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        return str(result["messages"][-1].content)
    except Exception as exc:
        logger.error(
            "agent 调用失败：%s: %s", type(exc).__name__, exc, exc_info=True
        )
        return friendly_message(exc)