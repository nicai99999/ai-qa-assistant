"""冒烟测试：把已知 bug 变成会失败的用例。

全部离线运行——不调用任何模型/embedding API，所以不花 token。

运行：
    uv run pytest -q

每个用例名后面的注释标注了它防的是哪个 bug。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# ============================================================
# 测试替身
# ============================================================


class FakeTurn:
    def __init__(self, question: str, answer: str) -> None:
        self.question = question
        self.answer = answer


class FakeStore:
    """替代 ConversationStoreV2，不需要模型。"""

    def __init__(self, turns: list[FakeTurn] | None = None) -> None:
        self._turns = turns or []
        self.appended: list[tuple[str, str]] = []

    def history(self, session_id: str) -> list[FakeTurn]:
        return list(self._turns)

    def summary(self, session_id: str) -> str:
        return ""

    def append(self, session_id: str, question: str, answer: str) -> None:
        self.appended.append((question, answer))


class FakeLLM:
    """记录收到的 prompt，返回固定分片。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def stream(self, prompt: str):
        self.prompts.append(prompt)
        for piece in ["假", "答案"]:
            yield SimpleNamespace(content=piece)

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(
            content="假答案",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )


def make_hit(text: str = "命中内容", source: str = "a.txt", chunk: str = "0", score: float = 0.9):
    return {
        "id": f"{source}::{chunk}",
        "text": text,
        "score": score,
        "metadata": {"source": source},
    }


@pytest.fixture()
def fake_retrieve(monkeypatch):
    """替换 streaming.retrieve，并记录调用参数。"""
    import streaming

    calls: dict = {"queries": [], "kwargs": []}

    def _fake(question, **kwargs):
        calls["queries"].append(question)
        calls["kwargs"].append(kwargs)
        return [make_hit()]

    monkeypatch.setattr(streaming, "retrieve", _fake, raising=False)
    return calls


@pytest.fixture()
def fake_llm(monkeypatch):
    """替换 streaming.get_llm。"""
    import streaming

    llm = FakeLLM()
    monkeypatch.setattr(streaming, "get_llm", lambda: llm, raising=False)
    return llm


# ============================================================
# streaming.py —— 已知 bug 的回归测试
# ============================================================


def test_empty_question_stops_immediately(fake_retrieve, fake_llm):
    """BUG: 空问题 yield 后缺 return，会继续往下检索。"""
    import streaming

    chunks = list(streaming.stream_answer("", "s1", FakeStore(), {}))

    assert len(chunks) == 1, f"空问题应只返回一条提示，实际返回 {len(chunks)} 条：{chunks}"
    assert "不能为空" in chunks[0]
    assert fake_retrieve["queries"] == [], "空问题不应触发检索"


def test_where_is_not_hardcoded(fake_retrieve, fake_llm):
    """BUG: retrieve 里硬编码 where={'source': '01_...'}，把检索锁死在一份文档。"""
    import streaming

    list(streaming.stream_answer("任意问题", "s1", FakeStore(), {}))

    where = fake_retrieve["kwargs"][0].get("where")
    assert where is None, f"默认不应限定文档，实际 where={where}"


def test_where_is_forwarded_when_given(fake_retrieve, fake_llm):
    """正常行为：调用方传 where 时应原样透传。"""
    import streaming

    list(streaming.stream_answer("任意问题", "s1", FakeStore(), {}, where={"source": "a.txt"}))

    assert fake_retrieve["kwargs"][0].get("where") == {"source": "a.txt"}


def test_memory_block_reaches_prompt(fake_retrieve, fake_llm):
    """BUG: store_obj 只写不读，多轮对话模型看不到历史。"""
    import streaming

    store = FakeStore([FakeTurn("历史问句标记", "历史答案标记")])
    list(streaming.stream_answer("现在的问题", "s1", store, {}))

    assert fake_llm.prompts, "模型没有被调用"
    prompt = fake_llm.prompts[0]
    assert "历史问句标记" in prompt, "prompt 里没有历史内容，多轮对话是假的"


def test_rewrite_applied_before_retrieve(fake_retrieve, fake_llm, monkeypatch):
    """BUG: 没接查询改写，代词问题会退化。"""
    import streaming

    monkeypatch.setattr(
        streaming, "rewrite_query", lambda q, turns: "改写后的问题", raising=False
    )
    list(streaming.stream_answer("它有哪些步骤", "s1", FakeStore(), {}))

    assert fake_retrieve["queries"] == ["改写后的问题"], (
        f"检索应使用改写后的问题，实际：{fake_retrieve['queries']}"
    )


def test_stream_prompt_has_memory_slot():
    """BUG: STREAM_PROMPT 缺 {memory} 占位符。"""
    import streaming

    assert "{memory}" in streaming.STREAM_PROMPT
    assert "{context}" in streaming.STREAM_PROMPT
    assert "{question}" in streaming.STREAM_PROMPT


def test_stream_stores_history_after_finish(fake_retrieve, fake_llm):
    """正常行为：流式结束后要把这一轮写进历史。"""
    import streaming

    store = FakeStore()
    list(streaming.stream_answer("问一句", "s1", store, {}))

    assert store.appended, "流式结束后没有写入历史"
    assert store.appended[0][0] == "问一句"


# ============================================================
# memory_v2 —— 纯逻辑
# ============================================================


def test_window_truncates():
    from memory_v2 import ConversationStoreV2

    store = ConversationStoreV2(mode="window", max_turns=3)
    for i in range(6):
        store.append("s", f"q{i}", f"a{i}")

    turns = store.history("s")
    assert len(turns) == 3
    assert turns[-1].question == "q5"


def test_priority_keeps_important():
    from memory_v2 import ConversationStoreV2

    store = ConversationStoreV2(mode="priority", max_turns=3)
    store.append("s", "我叫小明", "好的")       # 高分（含"我叫"）
    for i in range(5):
        store.append("s", f"普通问题{i}", f"答案{i}")

    questions = [t.question for t in store.history("s")]
    assert "我叫小明" in questions, "优先策略应保留高重要性轮次"


def test_score_importance():
    from memory_v2 import score_importance

    assert score_importance("我叫小明") >= 2
    assert score_importance("什么是 RAG") == 0


# ============================================================
# rag_qa —— 格式化与引用
# ============================================================


def test_build_context_numbering():
    from rag_qa import build_context

    ctx = build_context([make_hit("第一段"), make_hit("第二段")])
    assert "[1]" in ctx and "[2]" in ctx
    assert ctx.index("[1]") < ctx.index("[2]")


def test_build_citations_splits_id():
    from rag_qa import build_citations

    citations = build_citations([make_hit(source="doc.txt", chunk="7")])
    assert citations[0].source == "doc.txt"
    assert citations[0].chunk == "7"
    assert citations[0].index == 1


def test_extract_cited_indices():
    from rag_qa import extract_cited_indices

    assert extract_cited_indices("结论 A[1]，结论 B[2][1]") == [1, 2]
    assert extract_cited_indices("没有引用") == []


# ============================================================
# safe_qa —— 异常翻译
# ============================================================


@pytest.mark.parametrize(
    "exc, expected",
    [
        (Exception("Error code: 401 - invalid api key"), "API Key"),
        (Exception("insufficient balance"), "额度不足"),
        (Exception("429 rate limit exceeded"), "频繁"),
        (TimeoutError("request timed out"), "超时"),
        (ConnectionError("network is unreachable"), "网络"),
        (Exception("something weird"), "稍后重试"),
    ],
)
def test_friendly_message(exc, expected):
    from safe_qa import friendly_message

    assert expected in friendly_message(exc)


def test_validate_question_rejects_bad_input():
    from safe_qa import validate_question

    with pytest.raises(ValueError):
        validate_question("")

    with pytest.raises(ValueError):
        validate_question("x" * 3000)

    assert validate_question("  你好  ") == "你好"
