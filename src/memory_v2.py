"""会话记忆 v2：三种上下文管理策略。

策略对比：
    window   滑动窗口——只留最近 N 轮，最早的直接丢。
             优点：实现最简单、成本最低。
             缺点：丢掉的可能是关键信息。

    summary  摘要压缩——超限的老对话压成一段摘要，只保留事实。
             优点：token 最省，长期信息不丢。
             缺点：多一次模型调用，有信息损失风险。

    priority 优先级截断——给每轮打重要性分，超限时先淘汰低分的。
             优点：关键信息不会被时间挤掉。
             缺点：打分靠规则，规则不准就失效。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_TURNS = 10          # window / priority 的最大保留轮数
SUMMARY_TRIGGER = 6     # summary 策略：超过这么多轮开始压缩
KEEP_RECENT = 4         # summary 策略：压缩后保留最近几轮原文

# 简单的"重要性"关键词——命中就给高分
IMPORTANT_KEYWORDS = ("我叫", "我是", "我的", "我喜欢", "我要", "我准备", "记住")


@dataclass
class Turn:
    """一轮对话。"""

    question: str
    answer: str
    importance: int = 0


def score_importance(question: str) -> int:
    """给问题打重要性分数。命中关键词记 2 分，否则 0 分。"""
    return 2 if any(k in question for k in IMPORTANT_KEYWORDS) else 0
    # 结果A  if  条件  else  结果B
    # any(...)：这是 Python 的“存在性判断”函数。
    # 只要括号里生成的那一堆布尔值中有一个是 True，any() 就会立刻返回 True。如果全是 False，它才返回 False


class ConversationStoreV2:
    """支持三种上下文管理策略的会话存储。"""

    def __init__(self, mode: str = "window", max_turns: int = MAX_TURNS, llm=None) -> None:
        if mode not in {"window", "summary", "priority"}:
            raise ValueError(f"未知策略：{mode}")

        self.mode = mode
        self.max_turns = max_turns
        self.llm = llm
        self._store: dict[str, list[Turn]] = {}
        self._summary: dict[str, str] = {}

    # ---------- 对外接口 ----------

    def append(self, session_id: str, question: str, answer: str) -> None:
        turns = self._store.setdefault(session_id, [])
        turns.append(Turn(question, answer, score_importance(question)))
        self._trim(session_id)

    def history(self, session_id: str) -> list[Turn]:
        return list(self._store.get(session_id, []))

    def summary(self, session_id: str) -> str:
        return self._summary.get(session_id, "")

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        self._summary.pop(session_id, None)

    # ---------- 三种策略 ----------

    def _trim(self, session_id: str) -> None:
        turns = self._store[session_id]

        if self.mode == "window":
            if len(turns) > self.max_turns:
                self._store[session_id] = turns[-self.max_turns :]

        elif self.mode == "summary":
            if len(turns) > SUMMARY_TRIGGER:
                old = turns[:-KEEP_RECENT]
                recent = turns[-KEEP_RECENT:]
                if old:
                    self._summary[session_id] = self._summarize(
                        self._summary.get(session_id, ""), old
                    )
                self._store[session_id] = recent

        elif self.mode == "priority":
            if len(turns) > self.max_turns:
                important_idx = [i for i, t in enumerate(turns) if t.importance >= 2]
                # 此处给遍历turns给turn索引
                # [  # <- 这是 turns（列表）
                #         #         Turn(question="你好", answer="你好呀"),       # <- 第 1 轮问答
                #         #         Turn(question="你是谁", answer="我是AI助手")    # <- 第 2 轮问答
                #         #     ],
                slots = self.max_turns - len(important_idx)
                recent_idx = list(range(len(turns)))[-slots:] if slots > 0 else []
                # 此处根据 turns 的总长度，生成了一串代表“最近位置”的数字序列。
                # 与上面的索引是相对应的
                keep = sorted(set(important_idx) | set(recent_idx))
                # |：求并集
                self._store[session_id] = [turns[i] for i in keep]

    def _summarize(self, old_summary: str, turns: list[Turn]) -> str:
        """把若干轮老对话压成一段摘要。"""
        if self.llm is None:
            raise RuntimeError("summary 策略需要传入 llm")

        text = "\n".join(
            f"用户：{t.question}\n助手：{t.answer[:60]}" for t in turns
        )
        prompt = (
            "把下面的对话压缩成 3 条以内的要点，只保留事实信息"
            "（用户的身份、偏好、目标、明确要求），不要保留寒暄和无关内容。\n\n"
            f"已有摘要：{old_summary or '（无）'}\n\n"
            f"新增对话：\n{text}\n\n"
            "摘要："
        )
        result = self.llm.invoke(prompt)
        return str(result.content).strip()