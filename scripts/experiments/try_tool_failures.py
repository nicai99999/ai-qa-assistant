"""Day 30 实验：工具失败时 agent 会怎样。

用法：
    uv run python try_tool_failures.py
"""

import logging
import time

from langchain.agents import create_agent
from langchain_core.tools import tool

from llm import get_llm
from logging_setup import setup_logging
from tools import TOOLS

logger = logging.getLogger(__name__)


# ---------- 故意做坏的三个工具 ----------

@tool
def always_raise(query: str) -> str:
    """检索知识库（这个工具当前故障，调用必定报错）。"""
    raise RuntimeError("模拟：数据库连接失败")


@tool
def slow_tool(seconds: int = 3) -> str:
    """一个很慢的工具，用于测试超时。参数 seconds 是睡眠秒数。"""
    time.sleep(seconds)
    return f"已等待 {seconds} 秒"


@tool
def strict_tool(query: str) -> str:
    """一个参数校验很严的工具，空字符串会被拒绝。"""
    if not query.strip():
        return "参数错误：query 不能为空，请提供具体查询内容。"
    return f"收到查询：{query}"


BROKEN_TOOLS = [always_raise, slow_tool, strict_tool]


def run_case(title: str, question: str, tools) -> None:
    print("\n" + "=" * 66)
    print(f"【{title}】")
    print(f"问：{question}")

    agent = create_agent(
        model=get_llm(),
        tools=tools,
        system_prompt="你可以调用工具。工具报错时要如实告诉用户。",
    )

    start = time.perf_counter()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        elapsed = (time.perf_counter() - start) * 1000

        messages = result["messages"]
        # 表里放置刚刚发生的所有事情：包含用户的话、AI 决定调工具的消息、工具返回的原始数据、AI 最终的总结话。它是一本完整的“流水账”。
        calls = [
            c.get("name")
            for m in messages
            for c in (getattr(m, "tool_calls", None) or [])
        ]
        # 把整个聊天记录扫一遍，把所有被调用过的工具名称抓出来，放进一个列表里

        # calls = []
        # for m in messages:                               # 遍历所有历史消息
        #     tool_calls = getattr(m, "tool_calls", None)  # 尝试拿工具调用列表
        #     if not tool_calls:
        #         tool_calls = []                          # 如果没拿到，变成空列表防报错
        #     for c in tool_calls:                         # 遍历这个消息里的所有工具
        #         calls.append(c.get("name"))              # 把工具名提取出来

        answer = messages[-1].content
        # 在 Agent 的机制里，最后一条消息永远是大模型经过所有工具运算后，写给用户的最终自然语言回复

        print(f"工具调用：{calls or '（无）'}")
        print(f"耗时：{elapsed:.0f} ms")
        print(f"答：{str(answer)[:300]}")

    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"❌ agent 崩溃：{type(exc).__name__}: {exc}")
        print(f"耗时：{elapsed:.0f} ms")


def main() -> None:
    setup_logging()

    run_case("场景1 工具内部抛异常", "用 always_raise 工具查一下 RAG", BROKEN_TOOLS)
    run_case("场景2 工具很慢（3 秒）", "用 slow_tool 等 3 秒", BROKEN_TOOLS)
    run_case("场景3 参数为空", "用 strict_tool 查询空字符串", BROKEN_TOOLS)
    run_case("场景4 正常工具对照", "现在几点了？", TOOLS)


if __name__ == "__main__":
    main()