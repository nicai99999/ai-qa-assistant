"""llm 实例。整个进程共用一个，避免重复建连接池。"""
from functools import lru_cache
from langchain.chat_models import init_chat_model
from config import settings



@lru_cache(maxsize=1)
# @lru_cache(maxsize=1)
# lru_cache：Python 标准库的缓存装饰器。
# maxsize=1：只缓存1 个结果
# 第一次调用：执行函数，缓存结果
# 后续调用：直接返回缓存，不再执行
def get_llm():
    """全局单例。第一次调用创建，后续复用。"""
    return init_chat_model(
        settings.model,
        model_provider="deepseek",
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        extra_body={"thinking" : {"type" : "disabled"}}
    )