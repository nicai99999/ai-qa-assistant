"""文档加载：把 PDF / txt / md 读成 list[Document]。

RAG 链路第 ① 步。上层只认 Document，不关心里面是 PDF 还是 txt。
"""
from pathlib import Path

import pypdf
# 专门用来解析 PDF 文件的第三方库

from langchain_core.documents import Document
# langChain 的“标准数据容器”


def _load_text(path: str) -> list[Document]:
    """纯文本类：直接读，只需要注意编码。"""
    text = Path(path).read_text(encoding="utf-8")     # ← 坑点 1：必须指定 utf-8
    # 读文件所有内容,不指定 utf-8 就会直接报 UnicodeDecodeError
    # windows默认使用GBK，使用会报错
    return [Document(page_content=text, metadata={"source": Path(path).name})]
    # 这里是列表！ 即使是 TXT，我们也把它当成一个只含有一个元素的列表返回。因为上层统一要求是 list[Document]。
    # metadata={"source": Path(path).name}：记录文件名


def _load_pdf(path: str) -> list[Document]:
    """PDF：一页一个 Document。"""
    reader = pypdf.PdfReader(path)
    # 初始化 PDF 阅读器。
    total = len(reader.pages)
    # 获取 PDF 总页数，后面会用到。
    docs: list[Document] = []
    # docs = []：准备一个空列表，用来装每一页的内容。
    for i, page in enumerate(reader.pages):            # ← 坑点 2：页码从 0 开始
    # 遍历 PDF 的每一页。enumerate 获取索引 i（页码）和页面对象 page。
        text = page.extract_text() or ""               # ← 坑点 3：图片型 PDF 会返回空
        # extract_text() 尝试抠出文字。
        # 如果是图片型 PDF（扫描件），extract_text() 会返回 None。如果不加 or ""，
        #   后面计算字符长度时就会报 TypeError: object of type 'NoneType' has no len()。
        docs.append(
            Document(
            # 它专门用来装载文本和它的“背景信息”，方便后面的切分（Chunking）和向量化（Embedding）
                page_content=text,
                # 从文件里读出来的、真实的纯文本字符串
                metadata={
                # 元数据：这相当于这个文档块的“身份证”或“标签”
                    "source": Path(path).name,
                    "page": i,
                    "total_pages": total,               # ← Day 20 出处要用
                },
            )
        )
        # 把一页文本的内容和相关属性（元数据）打包成一个“标准文档对象”，然后塞进一个列表里。
    return docs


def load_document(path: str) -> list[Document]:
    """统一入口：按后缀分派。以后加格式只改这里。"""
    suffix = Path(path).suffix.lower()
    # 获取文件后缀（如 .pdf, .txt），并转成小写（防止 .PDF 匹配不到）
    if suffix in (".txt", ".md"):
        return _load_text(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    raise ValueError(f"不支持的文档格式：{suffix}（支持 .txt / .md / .pdf）")


def inspect_documents(docs: list[Document]) -> None:
    """质量检查：发现'PDF 提出来是空的'这种事故。"""
    total_chars = sum(len(d.page_content) for d in docs)
    # Document(
    #             # 它专门用来装载文本和它的“背景信息”，方便后面的切分（Chunking）和向量化（Embedding）
    #                 page_content=text,
    #                 # 从文件里读出来的、真实的纯文本字符串
    #                 metadata={
    #                 # 元数据：这相当于这个文档块的“身份证”或“标签”
    #                     "source": Path(path).name,
    #                     "page": i,
    #                     "total_pages": total,               # ← Day 20 出处要用
    #                 },
    #             )
    # 统计所有文档的总字符数。如果解析出来 total_chars = 0，你一眼就能看出“出事故了”（比如把图片型 PDF 放进去了）。
    print(f"共 {len(docs)} 个 Document，总字符数 {total_chars}")
    for d in docs[:3]:
    # docs[:3]：只打印前 3 个 Document，防止控制台被刷屏。看看它们的元数据和字数是否正常。
        print(f"  - {d.metadata}  字符数={len(d.page_content)}")
    head = docs[0].page_content[:100]
    print(f"第一段前 100 字：{head!r}")
    # 重点 !r：{head!r} 是 f-string 里的转换符，意思是调用 repr()。


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parents[2] / "data"
    # 当前脚本的路径
    # .resolve().parents[2]：向上找两级目录。
    # 脚本在 src/ai_qa_assistant/ 里，向上两级就是项目根目录 ai-qa-assistant。
    #
    # 拼上 "data"，定位到数据文件夹。
    for f in sorted(base.glob("*")):
    # 遍历 data/ 下的所有文件。
        if f.suffix.lower() not in (".txt", ".md", ".pdf"):
            continue
        print(f"\n===== {f.name} =====")
        inspect_documents(load_document(str(f)))