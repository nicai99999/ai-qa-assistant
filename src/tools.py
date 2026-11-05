"""给模型调用的工具集。

三个原则：
1. 工具只做一件事，粒度要小
2. description 写清"什么时候该用它"（模型靠它选工具）
3. 出错不要抛异常，返回一句错误描述让模型自己判断
"""

from __future__ import annotations

import logging
from datetime import datetime
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ===== 示例：已经写好的，照它的样子写后面两个 =====

@tool
def get_current_time() -> str:
    """获取当前的日期和时间。

    什么时候用：用户问"现在几点""今天几号""当前时间"时调用。
    不要用于：询问历史事件时间、文档里的时间。
    """
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


# ===== 你要写的 1：查知识库 =====

@tool
def search_knowledge_base(query: str) -> str:
    """在本地知识库中检索与问题相关的资料片段。

    什么时候用：用户问知识性问题（RAG、SSE、Agent 等技术概念）时调用。
    不要用于：闲聊、问时间、问知识库里有哪些文档。
    """
    # ★ 改动 1：参数校验，防止空字符串浪费 API 调用
    if not query.strip():
        return "参数错误：query 不能为空。请提供有效的问题。"

    try:
        # ★ 注意：如果你的运行方式报错，这里可能需要改成 from ai_qa_assistant.retriever import retrieve
        from src.retriever import retrieve

        hits = retrieve(query, k=5)
        if not hits:
            return "知识库中没有找到相关内容。"

        lines: list[str] = []
        for index, hit in enumerate(hits, start=1):
            source = (hit.get("metadata") or {}).get("source", "?")
            # hits.append({
            #                 "id": res["ids"][0][i],
            #                 "text": res["documents"][0][i],
            #                 "score": similarity,  # 这里存的是相似度，不是距离！
            #                 "metadata": res["metadatas"][0][i]
            #                     #  'metadatas': [
            #                     #     [{'source': '02_网络协议笔记.txt'}, {'source': '02_网络协议笔记.txt'}, {'source': '02_网络协议笔记.txt'}]],
            #             })
            score = hit.get("score") or 0.0
            # 块最长 399 字，不截断；k=5 时总量约 1900 字，token 可接受
            text = (hit.get("text") or "").strip().replace("\n", " ")
            lines.append(f"[{index}] (来源 {source}, 相似度 {score:.2f}) {text}")

        logger.info("工具检索 query=%r 召回=%d", query, len(hits))
        return "\n\n".join(lines)

    except Exception as exc:
        logger.warning("工具检索失败：%s", exc, exc_info=True)
        # ★ 改动 2：错误信息带上异常类型，帮大模型理解具体是什么错（如 ConnectionError, Timeout 等）
        return f"检索失败（{type(exc).__name__}）：{exc}。请告知用户稍后重试。"


@tool
def list_documents() -> str:
    """列出本地知识库里所有已入库的文档文件名。

    什么时候用：用户问"知识库里有哪些文档""都有什么资料"时调用。
    不要用于：问文档内容（那要用 search_knowledge_base）。
    """
    try:
        import chromadb

        from src.retriever import COLLECTION_NAME, DB_PATH

        col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
        data = col.get(include=["metadatas"])
        # col.get() 返回的不是 metadatas 列表本身，而是包含了 metadatas 的整个大字典
        # 数据库 API 的设计规定：include 这个参数，只接收“清单（列表）”，不接收单个物品

        # ["metadatas"] 是传给它的值（这是一个只包含一个元素的列表）。

        # data = {
        #     "ids": [...],          # 顺带返回的
        #     "documents": None,
        #     "metadatas": [         # 👈 你真正想要的数据在这里面！
        #         {"source": "笔记1.txt"},
        #         {"source": "笔记2.txt"}
        #     ],
        #     "included": ["metadatas"]
        # }

        # {'ids': [['02_网络协议笔记.txt::0', '02_网络协议笔记.txt::2', '02_网络协议笔记.txt::1']], 'embeddings': None,
        #  'documents': [
        #      ['TCP 三次握手\n第一次：客户端发送 SYN=1，序号 seq=x\n第二次：服务端返回 SYN=1，ACK=1，seq=y，ack=x+1\n第三次：客户端发送 ACK=1，ack=y+1',
        #       '。\n如果只有两次，服务端无法确认客户端的接收能力，也无法防止历史连接请求造成的错误。',
        #       '为什么是三次而不是两次：为了确认双方的收发能力都正常']], 'uris': None,
        #  'included': ['metadatas', 'documents', 'distances'], 'data': None,       # 数据库返回的结果
        #  'metadatas': [
        #     [{'source': '02_网络协议笔记.txt'}, {'source': '02_网络协议笔记.txt'}, {'source': '02_网络协议笔记.txt'}]],
        #  'distances': [[0.0, 0.37685900926589966, 0.38716739416122437]]}

        metadatas = data.get("metadatas") or []
        sources: set[str] = set()

        for meta in metadatas:
            if not isinstance(meta, dict):
            #       isinstance(对象, 类型)
                continue
            source = meta.get("source")
            if isinstance(source, str) and source:
                sources.add(source)

        if not sources:
            return "知识库目前是空的。"

        names = sorted(sources)
        return f"知识库共 {len(names)} 份文档：\n" + "\n".join(f"- {s}" for s in names)

    except Exception as exc:
        logger.warning("列出文档失败：%s", exc, exc_info=True)
        # ★ 改动 3：错误信息带上异常类型及明确提示
        return f"读取文档列表失败（{type(exc).__name__}）：{exc}。请告知用户稍后重试。"


# ===== 注册给 agent 用 =====

TOOLS = [get_current_time, search_knowledge_base, list_documents]