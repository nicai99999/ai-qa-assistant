"""重建索引：用更大的块重切，写入新 collection。

动机：Day 22 实测发现所有块都 <120 字符，平均约 50-60 字，
      导致标题块与正文块分离、答案被切成多块，Top-1 卡在 54.5%。

做法：chunk_size=400 / overlap=80 重切，写入 rag_notes_v2。

【手写】规则：docstring 和结构给你，标 TODO 的那一行自己补。
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from embedding import make_embed_client, embed_texts
from src.retriever import DB_PATH

logger = logging.getLogger(__name__)

DATA_DIR = Path(r"D:\ai-qa-assistant\data")

NEW_COLLECTION = "rag_notes_v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def load_texts(data_dir: Path) -> list[tuple[str, str]]:
    """读取 data/ 下所有 pdf / txt / md，返回 [(文件名, 全文), ...]。"""
    docs: list[tuple[str, str]] = []

    for path in sorted(data_dir.iterdir()):
        suffix = path.suffix.lower()
        text = ""

        if suffix == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            continue

        if text.strip():
            docs.append((path.name, text))
            logger.info("读入 %s，%d 字符", path.name, len(text))

    return docs


def split_texts(docs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """切块，返回 [(文件名, 块文本), ...]。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )

    chunks: list[tuple[str, str]] = []
    for name, text in docs:
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append((name, piece))

    logger.info("共切成 %d 块", len(chunks))
    return chunks


def embed_documents(texts: list[str]) -> list[list[float]]:
    """批量向量化。

    ⚠️ TODO：这里接入你 embedding.py 里的向量化函数。
    预期接口：接收 list[str]，返回 list[list[float]]，顺序一一对应。

    如果 embedding.py 里只有一个"单条"函数，就循环调用，
    或者直接复用它的 OpenAI 客户端：
        client.embeddings.create(model=EMBEDDING_MODEL, input=texts).data[i].embedding
    """

    list = embed_texts(make_embed_client() , texts)
    if len(list) != len(texts):
        raise NotImplementedError("调用你 embedding.py 里的向量化函数")
    return list


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    docs = load_texts(DATA_DIR)
    chunks = split_texts(docs)
    if not chunks:
        raise SystemExit("没有读到任何文本")

    lengths = sorted(len(t) for _, t in chunks)
    logger.info(
        "块长：最短 %d，中位 %d，平均 %d，最长 %d",
        lengths[0],
        lengths[len(lengths) // 2],
        sum(lengths) // len(lengths),
        lengths[-1],
    )

    texts = [t for _, t in chunks]
    vectors = embed_documents(texts)
    if len(vectors) != len(texts):
        raise SystemExit(f"向量数 {len(vectors)} 和块数 {len(texts)} 不一致")

    client = chromadb.PersistentClient(path=DB_PATH)
    # 已存在就先删掉，保证重跑幂等
    try:
        client.delete_collection(NEW_COLLECTION)
        logger.info("已删除旧 collection %s", NEW_COLLECTION)
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=NEW_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{name}::{i}" for i, (name, _) in enumerate(chunks)]
    metadatas = [{"source": name} for name, _ in chunks]

    col.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
    logger.info("写入 %s，共 %d 块", NEW_COLLECTION, col.count())


if __name__ == "__main__":
    main()