"""意图路由：先判断问题类型，再决定走哪条链路。

为什么要路由：
    所有问题都走 RAG 是有代价的——
    "你好" 和 "我叫什么" 都会先检索一遍（150ms + token），
    而且"我叫什么"会被证据校验拒答（库里本来就没有），体验差。

三类意图：
    knowledge 知识问题  → 检索 + 证据校验 + 生成（默认）
    personal  个人信息  → 只读记忆，不检索
    chitchat  闲聊      → 直接生成，不检索

降级设计：
    规则优先（0 成本）。规则认不出时默认走 knowledge（损失最小的方向）。
    可选的 LLM 分类只用在"短问句"上，避免给每个问题都加一次调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PERSONAL_KEYWORDS = (
    "我叫", "我是", "我的", "我喜欢", "我要", "我准备",
    "记住", "我今年", "我住在", "我会", "我在",
)

CHITCHAT_WORDS = (
    "你好", "您好", "hi", "hello", "谢谢", "多谢", "再见", "拜拜",
    "在吗", "早上好", "晚安", "辛苦了", "好的",
)

# 只放"实质疑问词"。不要放"吗"/"？"——
# 否则"在吗"会被判成疑问句，规则失效。
QUESTION_MARKERS = ("什么", "怎么", "为什么", "如何", "哪", "多少", "是否")

CHITCHAT_MAX_LEN = 8
LLM_ROUTE_MAX_LEN = 12

INTENTS = ("knowledge", "personal", "chitchat")

ROUTE_PROMPT = """判断用户这句话属于哪一类，只输出一个词。

类别：
- knowledge：询问知识、技术、业务问题
- personal：关于用户自己的信息（姓名、身份、偏好、计划）
- chitchat：寒暄、道谢、告别、无实质内容的闲聊

用户：{question}

类别："""


@dataclass
class Route:
    """路由结果。"""

    intent: str
    confidence: float
    by: str          # "rule" | "llm" | "fallback"

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 2),
            "by": self.by,
        }


def classify_by_rule(question: str) -> Route | None:
    """规则分类。认不出来返回 None，交给上层兜底。"""
    q = (question or "").strip()
    if not q:
        return None

    low = q.lower()

    # 个人信息：最先判，因为它优先级最高（问名字不该去查库）
    if any(word in q for word in PERSONAL_KEYWORDS):
        return Route("personal", 0.9, "rule")

    # 闲聊：必须同时满足"短"+"有问候词"+"没有实质疑问词"
    if len(q) <= CHITCHAT_MAX_LEN:
        has_greeting = any(word in low for word in CHITCHAT_WORDS)
        has_question = any(marker in q for marker in QUESTION_MARKERS)
        if has_greeting and not has_question:
            return Route("chitchat", 0.85, "rule")

    return None


def classify_by_llm(question: str, llm) -> Route:
    """让模型分类。失败会抛异常，由上层捕获。"""
    response = llm.invoke(ROUTE_PROMPT.format(question=question))
    text = str(response.content).strip().lower()

    for intent in INTENTS:
        if intent in text:
            return Route(intent, 0.7, "llm")

    # 模型答了个别的词，当作认不出
    return Route("knowledge", 0.3, "llm")


def classify(question: str, llm=None, use_llm: bool = False) -> Route:
    """对外入口：规则优先，可选 LLM 兜底，最后默认知识路径。

    降级链：规则 → LLM → knowledge
    """
    route = classify_by_rule(question)
    if route is not None:
        logger.info("路由 rule intent=%s", route.intent)
        return route

    q = (question or "").strip()
    if use_llm and llm is not None and 0 < len(q) <= LLM_ROUTE_MAX_LEN:
        try:
            route = classify_by_llm(q, llm)
            logger.info("路由 llm intent=%s", route.intent)
            return route
        except Exception as exc:                 # noqa: BLE001
            logger.warning("LLM 路由失败，降级到默认：%s: %s", type(exc).__name__, exc)

    logger.info("路由 fallback intent=knowledge")
    return Route("knowledge", 0.5, "fallback")