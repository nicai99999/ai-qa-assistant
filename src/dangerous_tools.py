"""危险工具：默认不执行，必须显式确认。

为什么单独放一个文件：
    危险工具不应该和普通工具一起注册给 agent。
    这个文件的工具只在"需要验证防护逻辑"时才挂上去。
"""

from __future__ import annotations

import datetime
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# 记录"被请求过但未确认"的删除操作，方便审计
PENDING_DELETES: list[str] = []


@tool
def delete_document(filename: str, confirm: bool = False) -> str:
    """从知识库中删除一份文档。

    ⚠️ 这是危险操作，会永久删除数据。

    什么时候用：用户明确要求删除某份文档时调用。
    不要用于：任何非明确删除的场景。

    Args:
        filename: 要删除的文档文件名（必须与 list_documents 返回的名字完全一致）。
        confirm: 是否已获得用户明确同意。第一次调用必须传 False。
    """
    filename = (filename or "").strip()

    # TODO 1: 如果没有文件名，返回参数错误
    if not filename:
        return "参数错误：filename 不能为空。请先用 list_documents 查看有哪些文档。"

    # TODO 2: 如果 confirm 为 False：
    #           - 把 filename 记进 PENDING_DELETES
    #           - 记一条 warning 日志
    #           - 返回一句"需要确认"的话，并告诉模型下一步该怎么做
    if not confirm:
        PENDING_DELETES.append(filename)
        logger.warning("拦截一次未确认的删除请求：%s", filename)
        return (
            f"⚠️ 未执行删除。「{filename}」是危险操作，需要用户明确同意。\n"
            f"请向用户复述：将永久删除「{filename}」，确认吗？\n"
            f"用户明确同意后，再用 confirm=True 重新调用。\n"
            f"用户不同意则不要再次调用。"
        )

    # TODO 3: confirm 为 True 时，不要真的删（用"模拟删除"），并返回结果
    # 真实项目里这里才真删；本实验只做模拟，避免丢数据
    logger.info("（模拟）执行删除：%s", filename)
    if filename in PENDING_DELETES:
        PENDING_DELETES.remove(filename)
    return f"已删除「{filename}」（本次为模拟删除，数据未真正移除）。"



DANGEROUS_TOOLS = [delete_document]