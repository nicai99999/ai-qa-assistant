"""Day 29 测试：模型会不会选对工具。

用法：
    uv run python try_agent.py
"""

import logging

from agent import build_agent
from logging_setup import setup_logging

logger = logging.getLogger(__name__)

CASES = [
    ("现在几点了？", "get_current_time"),
    ("知识库里都有哪些文档？", "list_documents"),
    ("RAG 的完整链路是什么？", "search_knowledge_base"),
    ("帮我查一下 SSE 和 WebSocket 的区别", "search_knowledge_base"),
    ("你好", "（不调用工具）"),
    ("现在几点，另外知识库里一共有几份文档？", "get_current_time + list_documents"),
]


def extract_tool_calls(messages) -> list[tuple[str, dict]]:
    """从消息序列里抠出所有工具调用。"""
    calls: list[tuple[str, dict]] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            # getattr(message, "tool_calls", None) or []：不是所有消息都有 tool_calls 属性（比如用户的提问就没有），所以安全地获取，如果没有就返回空列表
            # getattr(obj, name, default) —— 安全摸奖
                # obj：你要摸的对象（这里是 message，比如一条 AI 消息或用户消息）。
                # name：你要摸的属性名（这里是 "tool_calls"）。
                # default：如果摸不到这个属性，默认返回什么（这里是 None）。
            # 如果最终 A or B , A为None "假" , 就返回B
            calls.append((call.get("name", "?"), call.get("args", {})))
    return calls


def main() -> None:
    setup_logging()
    agent = build_agent()

    correct = 0

    for question, expected in CASES:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result["messages"]
        calls = extract_tool_calls(messages)

        actual = " + ".join(c[0] for c in calls) if calls else "（不调用工具）"
        hit = actual == expected
        correct += hit

        answer = messages[-1].content
        print("\n" + "=" * 66)
        print(f"问：{question}")
        print(f"期望：{expected}")
        print(f"实际：{actual}  {'✅' if hit else '❌'}")
        for name, args in calls:
            print(f"   调用 {name}，参数 {args}")
        print(f"答：{str(answer)[:150]}")

    print("\n" + "=" * 66)
    print(f"工具选择正确率：{correct}/{len(CASES)}")


if __name__ == "__main__":
    main()