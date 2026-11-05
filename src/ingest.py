"""增量入库：把一个文件解析、切块、向量化、写入知识库。

和 reindex.py 的区别：
    reindex 是"全量重建"（删库重来，适合初始化）
    ingest  是"增量添加"（只处理一个文件，适合用户上传）

覆盖式更新：
    写入前先按 source 删除旧块，
    这样重复上传同一份文件不会产生重复内容。
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.retriever import COLLECTION_NAME, DB_PATH, _chroma_client
from embedding import embed_texts, make_embed_client

logger = logging.getLogger(__name__)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def extract_text(filename: str, data: bytes) -> str:
    """从文件字节里提取纯文本。"""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        # 是PDF时：用 io.BytesIO(data) 把内存里的字节数据包装成一个“虚拟文件”，让 PdfReader 以为自己在读一个真实的硬盘文件
        # 在内存里凭空捏造了一个虚拟文件。
        # 把 data（从网页上传上来的文件二进制数据）塞进这个虚拟文件里，它就有了类似硬盘文件的 read() 和 seek() 方法。
        # PdfReader 拿到它之后，才能正常读写

        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="ignore")
    # data 是一坨纯纯的二进制字节（bytes）
    # decode 直接把这坨二进制字节(data)按照 utf-8 的规则“翻译”成人类能看懂的字符串

    raise ValueError(f"不支持的文件类型：{suffix}（仅支持 pdf / txt / md）")


def split_into_chunks(text: str) -> list[str]:
    """按段落/句子切块。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]
    # if c.strip()只将非空字符留下


def ingest(filename: str, data: bytes) -> int:
    """入库一个文件，返回写入的块数。

    Raises:
        ValueError: 文件类型不支持，或没有提取到文字。
    """
    text = extract_text(filename, data)
    if not text.strip():
        raise ValueError(
            "没有从文件里提取到文字。如果是扫描件（图片型 PDF），"
            "需要先做 OCR 才能入库。"
        )

    chunks = split_into_chunks(text)
    if not chunks:
        raise ValueError("切块结果为空。")

    vectors = embed_texts(make_embed_client(), chunks)
    if len(vectors) != len(chunks):
        raise ValueError(f"向量数 {len(vectors)} 与块数 {len(chunks)} 不一致")

    col = _chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 覆盖式更新：先删掉这个文件的旧块
    try:
        col.delete(where={"source": filename})
        # 在 metadatas 里，我们之前存过 {"source": filename}。这里直接让数据库（把所有来源等于这个文件名的数据全删了）
        logger.info("已清除 %s 的旧块", filename)
    except Exception:
        # 首次上传时没有旧块，忽略
        pass

    ids = [f"{filename}::{uuid4().hex[:12]}" for _ in chunks]
    # 生成唯一 ID：因为增量添加的文件可能会和之前的文件块重名，不能简单用 ::0 ::1 了，
    # 必须用 uuid4().hex[:12]（生成一个 12 位的随机乱码），保证绝对不冲突
    metadatas = [{"source": filename} for _ in chunks]
    # 像流水线一样，切出多少段文字，就复印多少张相同的“来源标签”，一一对应地贴上去

    col.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)

    logger.info("入库完成 file=%s 块数=%d", filename, len(chunks))
    return len(chunks)