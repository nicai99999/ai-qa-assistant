import logging
# 引入 Python 标准库的日志模块
import os
import time
from datetime import datetime

from langchain.agents import create_agent

from llm import get_llm
from logging_setup import setup_logging, log_llm_call
from tools import get_current_time, add_numbers

logger = logging.getLogger(__name__)
# 获取一个名为 __main__ 的 logger 实例。放在函数外面（模块级）是规范做法，这样全局都能用这个 logger，不用反复获取。

def main():
    setup_logging()
    # 执行自定义的日志配置,这个必须最先执行，不然之后的日志可能不会按你的预期输出。
    print("AI 问答助手已启动，输入 exit 或 quit 退出。")

    # 3. 启动时调一次 get_llm()，拿到 llm 对象
    llm = get_llm()
    agent = create_agent(
        llm,
        tools=[get_current_time,add_numbers],
        system_prompt="你是一个时间管理大师"
    )

    while True:
        user_input = input("你 > ").strip()

        # 5. 保留 exit / quit 处理
        if user_input.lower() in ("exit", "quit"):
            print("再见。")
            break

        # 5. 保留空输入处理
        if not user_input:
            continue

        # 4. 换成 llm.invoke 并打印
        print("\n思考中...")

        # agent,只能使用以下的参数写法即字典，不能使用列表
        messages = {
            "messages": [
                {"role": "user", "content": user_input}
            ]
        }

        # --- 整合的计时与日志代码开始 ---
        start = time.perf_counter()
        # 开始计时
        # perf_counter() 是高精度性能计时器，专用于测代码执行时间，极其精准。
        response = agent.invoke(messages)
        elapsed = (time.perf_counter() - start) * 1000
        # 再次读取当前时间，减去 start，算出这趟调用花了多少秒。
        # * 1000：把秒转成毫秒（ms）。因为大模型响应通常以毫秒计，看毫秒更直观。

        last_message = response["messages"][-1]

        # 注意：usage_metadata 可能只在真实的 AIMessage 对象上有
        # 这里做个防错，防止某些版本的 Agent 把 metadata 吞了
        usage = getattr(last_message, "usage_metadata", None) or {}

        log_llm_call(logger, elapsed, usage)
        # elapsed：耗时（毫秒）。
        # response.usage_metadata：LangChain 提供的 token 用量（包含输入 token、输出 token、
        # --- 整合的计时与日志代码结束 ---
        print(f"AI: {last_message.content}\n")


if __name__ == "__main__":
    main()
