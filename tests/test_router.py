"""意图路由的测试。

前 2 个是示例（已写好）。后 4 个 TODO 请你按 docstring 描述补上断言。
写完再看文末的参考答案。

运行：
    uv run pytest tests/test_router.py -v
"""

from __future__ import annotations

from router import classify


# ---------- 示例 1：规则能识别个人信息 ----------
def test_rule_detects_personal():
    from router import classify_by_rule

    route = classify_by_rule("我叫小明")
    assert route is not None
    assert route.intent == "personal"
    assert route.by == "rule"


# ---------- 示例 2：认不出时默认知识路径 ----------
def test_fallback_defaults_to_knowledge():
    from router import classify

    route = classify("RAG 的完整链路是什么")
    assert route.intent == "knowledge"
    assert route.by == "fallback"


# ---------- TODO 1 ----------
def test_rule_detects_greeting():
    from router import classify
    """「你好」应被判成 chitchat。

    要断言的：
      - route 不是 None
      - route.intent == "chitchat"
    """
    route  = classify("你好")
    assert route is not None
    assert route.intent == "chitchat"


# ---------- TODO 2 ----------
def test_greeting_with_question_is_knowledge():
    from router import classify
    """「你好，RAG 是什么」应该走知识路径，不是闲聊。

    为什么：里面有实质疑问词「什么」，说明用户真在问问题。
    要断言的：
      - route.intent == "knowledge"
    """
    route = classify("你好，RAG是什么")
    assert route.intent == "knowledge"


# ---------- TODO 3 ----------
def test_empty_question_falls_back():
    from router import classify
    """空字符串应走 fallback，且 intent 是 knowledge。

    为什么：`classify_by_rule("")` 会返回 None，所以 `classify` 走兜底。
    要断言的：
      - route.intent == "knowledge"
      - route.by == "fallback"
    """
    route = classify("")
    assert route.intent == "knowledge"
    assert route.by == "fallback"


# ---------- TODO 4 ----------
def test_route_as_dict_is_serializable():
    import json
    from router import classify
    """Route 对象要能被转成 JSON（因为要存进 collector）。

    提示：`route.as_dict()` 得到字典，`json.dumps(...)` 不抛异常即通过。
    """
    route = classify("我叫什么名字")
    json.dumps(route.as_dict())