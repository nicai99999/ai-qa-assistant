"""把工具挂到模型上，构成一个能自主调用工具的 agent。

LangChain v1 用 create_agent（旧的 AgentExecutor / initialize_agent 已废弃）。
"""

from __future__ import annotations

import logging

from langchain.agents import create_agent

from llm import get_llm
from tools import TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个严谨的助手，可以调用工具来回答问题。

规则：
1. 需要外部信息时先调用工具，不要凭记忆回答。
2. 知识性问题必须先查知识库，查到什么说什么。
3. 如果工具返回"没有找到"，如实告诉用户，不要编造。
4. 闲聊类问题直接回答，不需要调用工具。
"""


def build_agent():
    """构建 agent。返回的对象用 .invoke({"messages": [...]}) 调用。"""
    return create_agent(
        model=get_llm(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )