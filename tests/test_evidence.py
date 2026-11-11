"""证据校验的测试：验证三种信号、四种策略、以及与 streaming 的集成。

全部离线运行——用假 hits 和假 LLM，不调用任何外部 API。

运行：
    uv run pytest tests/test_evidence.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# ============================================================
# 测试替身
# ============================================================


def make_hit(text: str, score: float, source: str = "a.txt", chunk: str = "0") -> dict:
    return {
        "id": f"{source}::{chunk}",
        "text": text,
        "score": score,
        "metadata": {"source": source},
    }


class FakeJudge:
    """假的 LLM，invoke 返回固定内容。"""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.answer)


class FakeStreamLLM(FakeJudge):
    """假的 LLM，同时支持 stream（用于验证生成阶段有没有被触发）。"""

    def stream(self, prompt: str):
        self.prompts.append(prompt)
        for piece in ["假", "答案"]:
            yield SimpleNamespace(content=piece)


class FakeStore:
    """假的历史存储。

    参数名用 _session_id / _question：故意不用，但保留位置以匹配真实接口。
    """

    def __init__(self) -> None:
        self.appended: list[tuple[str, str]] = []

    def history(self, _session_id: str):
        return []

    def summary(self, _session_id: str) -> str:
        return ""

    def append(self, _session_id: str, question: str, answer: str) -> None:
        self.appended.append((question, answer))


# ============================================================
# 信号 2：关键词覆盖率
# ============================================================


def test_coverage_full_when_text_contains_terms():
    from evidence_check import keyword_coverage

    hits = [make_hit("向量检索的原理是先编码再算相似度", 0.5)]
    assert keyword_coverage("向量检索是什么", hits) == pytest.approx(1.0)
    # assert 某个条件
    # 「我保证这个条件是真的。」
    #
    # 条件是真的 → 继续往下跑
    # 条件是假的 → 立刻停下，报错 AssertionError
    # 测试文件就是一堆这样的"保证"。全都不报错 = 全部通过


def test_coverage_zero_when_text_unrelated():
    from evidence_check import keyword_coverage

    hits = [make_hit("今天天气很好", 0.5)]
    assert keyword_coverage("向量检索是什么", hits) == 0.0


def test_coverage_empty_hits():
    from evidence_check import keyword_coverage

    assert keyword_coverage("任意问题", []) == 0.0

def test_coverage_partial():
    """覆盖率应该是比例，不是布尔值。

    问题分词后是 {"向量检索", "分块"} 两个词（"和"是停用词被过滤），
    只命中其中一个 → 覆盖率 0.5。
    """
    from evidence_check import keyword_coverage

    hits = [make_hit("向量检索的原理是先编码", 0.5)]
    assert keyword_coverage("向量检索和分块", hits) == pytest.approx(0.5)


# ============================================================
# 信号 3：LLM 判断
# ============================================================


def test_judge_accepts_positive_answer():
    # 定义一个测试（名字必须以 test_ 开头，pytest 才会找到它）
    from evidence_check import llm_judge

    llm = FakeJudge("能")
    assert llm_judge("问题", [make_hit("资料", 0.5)], llm) is True
    # 我保证：给"任意问题"和空列表时，覆盖率算出来是 0.0


def test_judge_rejects_negative_answer():
    # 字符串包含判断的顺序错误
    """关键：'不能' 里含 '能'，不能被误判成 True。"""
    from evidence_check import llm_judge

    llm = FakeJudge("不能")
    assert llm_judge("问题", [make_hit("资料", 0.5)], llm) is False


def test_judge_handles_api_error():
    """校验层自己不能把主流程打挂。"""

    from evidence_check import llm_judge

    class Broken:
        def invoke(self, prompt):
            raise RuntimeError("模拟接口故障")

    # noqa: F821 - Broken 定义在下方便于阅读，这里直接使用
    assert llm_judge("问题", [make_hit("资料", 0.5)], Broken()) is False


# ============================================================
# 四种策略
# ============================================================


def test_score_strategy_passes_high_score():
    from evidence_check import check_evidence

    verdict = check_evidence("随便问问", [make_hit("不相关内容", 0.70)], strategy="score")
    assert verdict.ok is True


def test_score_strategy_rejects_low_score():
    from evidence_check import check_evidence

    verdict = check_evidence("随便问问", [make_hit("不相关内容", 0.20)], strategy="score")
    assert verdict.ok is False
    assert "相关性过低" in verdict.reason


def test_staged_middle_zone_low_coverage_rejected():
    from evidence_check import check_evidence

    verdict = check_evidence("向量检索是什么", [make_hit("今天天气很好", 0.50)], strategy="staged")
    assert verdict.ok is False
    assert "关键信息" in verdict.reason


def test_staged_middle_zone_high_coverage_passes():
    from evidence_check import check_evidence

    verdict = check_evidence(
        "向量检索是什么", [make_hit("向量检索的原理是编码后算相似度", 0.50)], strategy="staged"
    )
    assert verdict.ok is True


def test_staged_does_not_call_llm():
    """staged 策略是 0 成本的：传 llm=None 也不能报错。"""
    from evidence_check import check_evidence

    verdict = check_evidence(
        "向量检索是什么", [make_hit("向量检索的内容", 0.50)], strategy="staged", llm=None
    )
    assert verdict.ok is True
    assert verdict.judge is None


def test_empty_hits_rejected_for_all_strategies():
    from evidence_check import check_evidence

    for strategy in ("score", "coverage", "staged"):
        assert check_evidence("问题", [], strategy=strategy).ok is False


def test_unknown_strategy_raises():
    from evidence_check import check_evidence

    with pytest.raises(ValueError):
        # pytest.raises(...)（保证会报错）
        # with 是 Python 的语法，意思是"在这个范围内，请按我指定的方式处理异常"。
        check_evidence("问题", [make_hit("资料", 0.5)], strategy="不存在的策略")


def test_verdict_as_dict_is_serializable():
    """collector 里存的是 dict，要能进 JSON。"""
    import json

    from evidence_check import check_evidence

    verdict = check_evidence("向量检索是什么", [make_hit("向量检索的内容", 0.5)])
    json.dumps(verdict.as_dict())           # 只要这行不抛异常，测试就通过
    # verdict.as_dict() 把verdict拆成普通字典
    # {"ok": True, "reason": "证据充分", "max_score": 0.5, "coverage": 0.83, "judge": None}
    # json.dumps(...) 把字典打包装箱成一段文本


# ============================================================
# 与 streaming 的集成
# ============================================================


def test_streaming_rejects_weak_evidence_and_skips_generation(monkeypatch):
    """集成验证：证据不足时直接拒答，且不调用生成。"""
    import streaming

    monkeypatch.setattr(streaming, "retrieve", lambda q, **kw: [make_hit("无关内容", 0.10)])
    # monkeypatch（猴子补丁）的作用是：把某个模块里的函数，临时换成别的函数。

    llm = FakeStreamLLM("能")
    monkeypatch.setattr(streaming, "get_llm", lambda: llm)
    # 临时把 streaming 模块里的 get_llm 函数，替换成 lambda: llm 这个假函数。
    # lambda: llm 完全等于：
    # def 临时函数():
    #     return llm
    monkeypatch.setattr(streaming, "rewrite_query", lambda q, turns: q, raising=False)
    monkeypatch.setattr(streaming, "build_memory_block", lambda store, sid: "", raising=False)

    # 测试要验证的是"闸门逻辑"（证据不足就不生成），不是"模型答得好不好"。
    # 所以把这三个全换成假的：

    collector: dict = {}
    chunks = list(streaming.stream_answer("一个库外问题", "s1", FakeStore(), collector))

    assert len(chunks) == 1, f"应该只产出一条拒答消息，实际 {chunks}"
    assert "未提及" in chunks[0]
    assert llm.prompts == [], "证据不足时不应调用生成"
    assert collector["evidence"]["ok"] is False


def test_streaming_generates_when_evidence_strong(monkeypatch):
    """集成验证：证据充分时正常生成。"""
    import streaming

    monkeypatch.setattr(
        streaming, "retrieve", lambda q, **kw: [make_hit("向量检索的原理", 0.75)]
    )

    llm = FakeStreamLLM("能")
    monkeypatch.setattr(streaming, "get_llm", lambda: llm)
    monkeypatch.setattr(streaming, "rewrite_query", lambda q, turns: q, raising=False)
    monkeypatch.setattr(streaming, "build_memory_block", lambda store, sid: "", raising=False)

    collector: dict = {}
    chunks = list(streaming.stream_answer("向量检索是什么", "s1", FakeStore(), collector))

    assert "".join(chunks) == "假答案"
    assert collector["evidence"]["ok"] is True