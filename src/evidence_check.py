"""证据校验：判断召回的资料是否足够回答问题。

为什么需要：
    RAG 的幻觉主要不是模型"喜欢编"，而是检索永远会返回 Top-K——
    哪怕库里根本没有相关内容。模型拿到 5 条不相关的块，
    仍会硬编一个看起来合理的答案。

三种信号（成本从低到高）：
    1. max_score        检索最高相似度。最便宜，但区分度有限。
    2. keyword_coverage 问题里的实词有多少出现在召回块里。本地计算，区分度较好。
    3. llm_judge        让模型只看资料判断"能否回答"。最准，但多一次调用。

四种策略：
    "score"     只信相似度
    "coverage"  只信覆盖率
    "judge"     只信模型判断
    "staged"    三段式（推荐，0 成本）：
                  相似度 >= HIGH  → 放行（这一段没有不可答样本）
                  相似度 <  LOW   → 拒答（这一段没有可答样本）
                  中间区          → 看覆盖率

阈值必须按你自己的语料校准：
    uv run python scripts/experiments/try_evidence.py
    看表 B 标了 ← 的那一行，把数值填到下面的常量里。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hybrid_retriever import _tokenize

logger = logging.getLogger(__name__)

# ============ 阈值（依据 2026-11-09 实测，16 题评测集）============
# 可答题最高相似度：0.721 0.707 0.696 0.648 0.588 0.577 0.573 0.537 0.480 0.477 0.455
# 不可答题最高相似度：0.571 0.285 0.245 0.224 0.133
#
# 阈值扫描结论（scripts/experiments/try_evidence.py 表 B）：
#   0.30 ~ 0.45 全部是 识别率 80% / 保留率 100% / 调和 0.89
#   0.50 开始误拒可答题（保留率掉到 73%）
#
# 在 0.30~0.45 这个平台区里取中间值 0.40，而不是取边界 0.30：
#   取 0.30 → 距不可答最高分 0.285 只有 0.015，换语料极易翻车
#   取 0.40 → 下边距 0.115、上边距 0.055，两边都有余量
LOW_SCORE = 0.40        # score 策略的主阈值：低于它直接拒答

# staged 策略用的高低档（当前数据下 staged 不如单阈值，保留以备语料变化）
HIGH_SCORE = 0.60       # 高于此值直接放行
MIDDLE_LOW = 0.45       # 中间区下沿
MIN_COVERAGE = 0.34     # 中间区要求的最低关键词覆盖率

# ============ 信号 3 的 prompt ============
JUDGE_PROMPT = """你是资料审核员。判断下面提供的资料能否回答用户的问题。

用户问题：{question}

提供的资料：
{context}

判断标准：
- 资料里明确包含回答问题所需的事实或结论 → 回答"能"
- 资料只是话题相关、但没有答案本身 → 回答"不能"
- 只输出一个词：能 或 不能。不要解释。

判断："""


@dataclass
class Verdict:
    """校验结果。"""

    ok: bool
    reason: str
    max_score: float
    coverage: float
    judge: bool | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "max_score": round(self.max_score, 3),
            "coverage": round(self.coverage, 3),
            "judge": self.judge,
        }


# ============ 信号 2 ============

def keyword_coverage(question: str, hits: list[dict]) -> float:
    """问题里的实词，有多大比例出现在了召回块中。

    只算"是否出现在任意一块里"，不按出现次数加权——
    避免长块靠篇幅天然覆盖更多词。
    """
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return 0.0

    joined = " ".join((h.get("text") or "") for h in hits).lower()
    hit_count = sum(1 for token in q_tokens if token in joined)
    return hit_count / len(q_tokens)


# ============ 信号 3 ============

def llm_judge(question: str, hits: list[dict], llm) -> bool:
    """让模型只根据资料判断能否回答问题。返回 True 表示"能"。"""
    if not hits or llm is None:
        return False

    context = "\n\n".join(
        f"[{i}] {h.get('text', '')}" for i, h in enumerate(hits, start=1)
    )

    try:
        response = llm.invoke(JUDGE_PROMPT.format(question=question, context=context))
        text = str(response.content).strip()
    except Exception as exc:                     # noqa: BLE001
        # 校验层自己不能把主流程打挂，保守判为"不能回答"
        logger.warning("llm_judge 调用失败：%s: %s", type(exc).__name__, exc)
        return False

    # 注意："不能"里含"能"，必须先判否定
    if "不能" in text or "无法" in text or "不足" in text:
        return False
    return "能" in text


# ============ 对外入口 ============

def check_evidence(
    question: str,
    hits: list[dict],
    strategy: str = "staged",
    llm=None,
) -> Verdict:
    """综合判断资料是否足够回答问题。

    Args:
        strategy: "score" | "coverage" | "judge" | "staged"
        llm:      只有 strategy="judge" 时才需要
    """
    if not hits:
        return Verdict(False, "没有检索到任何资料", 0.0, 0.0)

    max_score = max((h.get("score") or 0.0) for h in hits)
    coverage = keyword_coverage(question, hits)

    if strategy == "score":
        ok = max_score >= LOW_SCORE
        reason = "证据充分" if ok else "召回内容与问题相关性过低"

    elif strategy == "coverage":
        ok = coverage >= MIN_COVERAGE
        reason = "证据充分" if ok else "召回内容没有覆盖问题的关键信息"

    elif strategy == "judge":
        judge = llm_judge(question, hits, llm)
        ok = judge
        reason = "模型判断资料可回答" if ok else "模型判断资料不足以回答"
        logger.info("证据校验 strategy=judge ok=%s score=%.3f coverage=%.3f",
                    ok, max_score, coverage)
        return Verdict(ok, reason, max_score, coverage, judge)

    elif strategy == "staged":
        if max_score >= HIGH_SCORE:
            ok, reason = True, "证据充分（高相似度）"
        elif max_score < MIDDLE_LOW:
            ok, reason = False, "召回内容与问题相关性过低"
        elif coverage >= MIN_COVERAGE:
            ok, reason = True, "证据充分（中间区，关键词覆盖足够）"
        else:
            ok, reason = False, "召回内容没有覆盖问题的关键信息"

    else:
        raise ValueError(f"未知策略：{strategy}")

    logger.info(
        "证据校验 strategy=%s ok=%s score=%.3f coverage=%.3f reason=%s",
        strategy, ok, max_score, coverage, reason,
    )
    return Verdict(ok, reason, max_score, coverage)
