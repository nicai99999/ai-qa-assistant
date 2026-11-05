"""打印库里的块，帮你写评测题。

用法：
    uv run python scripts/build_eval_set.py
"""

import random

import chromadb

from retriever import COLLECTION_NAME, DB_PATH


def main() -> None:
    col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    data = col.get(include=["documents", "metadatas"])
    # include  库里存的是 1024 维的向量（embeddings）。一个块有 1024 个浮点数，如果把库里的几千个块的向量全拉出来，内存直接爆掉
    # {
    #     "ids": [ "笔记.txt::0", "笔记.txt::1", "笔记.txt::2", ... ],
    #     "documents": [
    #         "TCP 三次握手的第一次：客户端发送 SYN...",
    #         "UDP 与 TCP 的区别：1. UDP 无连接...",
    #         "证书链的验证是 HTTPS 安全的核心...",
    #         # ... 所有块的真实文本
    #     ],
    #     "metadatas": [
    #         {"source": "02_网络协议笔记.txt"},
    #         {"source": "02_网络协议笔记.txt"},
    #         {"source": "02_网络协议笔记.txt"},
    #         # ... 所有块对应的元数据
    #     ]
    # }
    ids: list[str] = data["ids"]
    docs: list[str] = data["documents"]

    index = list(range(len(ids)))
    # 代表了库里的所有文档块的序号
    random.seed(42)          # 固定种子，每次跑看到的是同一批
    random.shuffle(index)
    # 随机打乱
    # 直接取前 15 个（index[:15] 也就是 0 到 14），那这 15 块可能全是来自同一个文件（比如 笔记.txt::0 到 ::14），它们的内容高度相似，评测题就会非常单一。

    print(f"库里共 {len(ids)} 个块，下面展示 15 个。\n")

    for n, i in enumerate(index[:15], start=1):
        print(f"===== 样本 {n} =====")
        print("id:", ids[i])
        print("内容:", (docs[i] or "")[:200].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()