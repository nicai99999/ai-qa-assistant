"""Day 17：把分块结果向量化，存进本地 Chroma。

【AI生成】产物。今天必须读懂 + 完成三个改动任务（见执行卡下午段）。
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
# 这里不仅是用 OpenAI，更是用这个 SDK 来调用阿里云百炼（DashScope）。因为百炼兼容了 OpenAI 的接口格式。
import chromadb
# 本地轻量级向量数据库，用来存向量。

from splitter import split_text

# ============ 配置区（改这里，不要改下面） ============
PROJECT_ROOT = Path(r"D:\ai-qa-assistant")
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
# CHROMA_DIR：向量库文件将来会保存在本地硬盘的 chroma_db/ 文件夹里（持久化）。
# 把“项目根目录”和“chroma_db 文件夹名”拼接在一起，转换成完整的绝对路径字符串。
COLLECTION_NAME = "rag_notes_v2"
DOC_PATH = str(PROJECT_ROOT / "data" / "06_面试纲要汇总.txt")
# 找一个叫 rag_notes.txt 的文件，那就是我们要读取和切分的原始资料。
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
BATCH_SIZE = 8          # 一批发几条。（阿里云百炼 text-embedding-v4 单请求上限 10 条）
# 重要工程细节。调用向量化 API 时，不要一条一条发，网络开销太大；也不要几千条一起发，会被 API 拒绝（百炼限制单次最多 10 条）。所以设置为 8 是最佳实践。
REBUILD = False         # 改动任务③：True = 先删集合再建
# 一个开关。如果设为 True，每次运行都会把以前的旧数据删掉重建（防止数据重复叠加）。第一次运行或者修改了文档内容后，建议设为 True

#当 REBUILD = True 时：代码相当于先把数据库里的旧表清空，然后再把这次最新切分、最新向量化的数据存进去。保证库里永远只有最新且唯一的一份。

# 当 REBUILD = False 时：代码不删旧数据，直接在后面追加（适合你每天都有全新的、不重样的文档需要入库的增量场景）。

# 什么时候必须设为 True？（工程实践）
# 你修改了原文档内容（比如笔记里加了新知识点），旧向量作废了。
# 你修改了切分参数（比如把 CHUNK_SIZE 从 300 改成了 500），旧的切块大小不对了。
# 你换了 Embedding 模型（比如从 1024 维换成了 1536 维），旧的向量维度都对不上了。
# 你单纯只是测试跑错了，想洗掉脏数据重新来。

# ====================================================



def make_embed_client() -> OpenAI:
    # 创建并返回一个专门用来把“文本”变成“向量（数字数组）”的 AI 客户端实例。
    # get_llm() 是为了拿到一个“聊天模型”（负责对话）；
    # make_embed_client() 是为了拿到一个“嵌入模型”（负责把文字转成高维坐标）。

    """构造 embedding 客户端。DeepSeek 不行（实测 404），所以是第三方 OpenAI 兼容服务。"""
    load_dotenv(PROJECT_ROOT / ".env")
    return OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),       # ← 换成你实际的变量名
        base_url=os.environ["EMBEDDING_BASE_URL"],
        # 指定请求要发到哪个服务器。
        timeout=60, # 超时设置（防卡死）
        max_retries=2, # 最大重试次数（防网络抖动）
    )


def embed_texts(emb_client: OpenAI, texts: list[str]) -> list[list[float]]:
# 文本向量化
    """把一批文本向量化。返回 list[list[float]]，长度和输入一一对应。"""

    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")
    # 告诉阿里云的服务器，我要用哪个模型来转。默认用 text-embedding-v4
    # 优先从环境变量里找；如果找不到（或者返回 None），就用后面的默认值兜底，绝不让程序崩溃。
    dim = int(os.environ.get("EMBEDDING_DIM", "1024"))
# 规定生成的向量维度。为什么加 int()？因为环境变量读出来的全是字符串（比如 "1024"），传参给 API 时必须转成数字（整数）
    # 为了让ai读懂人类文字，会将文字映射到超级高维（1024）的数学空间
        #1024维的空间内，意思相近的句子，他们的1024维坐标距离就会非常近
        # 用 Chroma 去查东西，底层其实就是在这 1024 维空间里，算哪个坐标离你提问的坐标最近。
    vectors: list[list[float]] = []
    # 准备一个大筐（外层列表），用来装一堆“坐标”。
    # list[list[float]]：因为你要传进去很多块文本（texts），每一块都会变成一个 1024 维的列表。所以最终的结果是一个“列表的列表”。
    # 层装的是“块”，内层装的是“该块对应的 1024 个数字坐标”。


    for start in range(0, len(texts), BATCH_SIZE):
        # 它生成的是一个纯粹的数字序列。
        # 假设 len(texts) = 10，BATCH_SIZE = 8。
        # range(0, 10, 8) 生成的数字是：0, 8。
        # 所以循环执行两次：第一次 start = 0，第二次 start = 8。
        # 一次取BATCH_SIZE个向量
    # range(start, stop, step)
    # API 有硬性限制：比如阿里云的 text-embedding-v4，一次请求最多只允许传 10 条 文本。如果你有 100 个文本块，直接全塞进去，API 会直接报错 HTTP 400 Bad Request。
    # 防止网络超时：即便 API 允许传 100 条，数据量太大也会导致网络传输极慢、甚至超时崩溃。分批发，每次都在几秒内完成，非常稳定。
    # 失败重试成本低：如果一批 8 条数据发送失败了，SDK 重试只要把这 8 条重发一次就行。如果你把 100 条全塞在一个请求里，一旦失败，整个 100 条都要重算。
        batch = texts[start : start + BATCH_SIZE]
        # 取出texts中的page_content
        resp = emb_client.embeddings.create(model=model, input=batch, dimensions=dim)
        # resp 是一个对象（Object）。它内部封装了服务器返回的复杂数据
        # resp里只有一个data[Embedding(embedding[])]
        # 这是真正走网络、花钱的地方。把 batch 文本发给阿里云的服务器。服务器计算完返回 resp
        # ★ 关键：不要假设返回顺序，按 index 排序
        items = sorted(resp.data, key=lambda d: d.index)
        # items是resp.data排序，内容为[对象(index , embedding)]
        # sorted(..., key=lambda d: d.index)，强行把它们按 0, 1, 2 的顺序重新排好。
        # resp.data的内容
        #         [
        #         Embedding(index=0, embedding=[0.12, -0.45, 0.88, ...1024个数字]),
        #         Embedding(index=1, embedding=[-0.98, 0.33, 0.11, ...1024个数字])
        #         ]
        # 因为阿里云服务器是多线程并发处理的，文本1 特别长，处理得慢；文本2 特别短，瞬间处理完了。
        # 服务器返回给你的顺序变成了：[向量2, 向量0, 向量1]（乱序的）。
        # 如果你不排序，直接 extend 进大列表。最后你拿着 [向量2, 向量0, 向量1] 去和你的 [文本0, 文本1, 文本2] 做绑定。
        # 结果就是：以后用户在搜索“文本0”的内容，系统却给他返回了“文本2”的答案。 整个 RAG 系统直接变成智障。

        vectors.extend([it.embedding for it in items])
        # it.embedding 就是那条 1024 个浮点数的列表。
        # extend 把排好序的向量塞进大筐。
        # vectors = [
        #     [0.12, -0.45, 0.88, ...],   # 第 0 个元素：对应 "苹果" 的 1024 维向量
        #     [-0.98, 0.33, 0.11, ...],   # 第 1 个元素：对应 "香蕉" 的 1024 维向量
        #     [0.55, 0.22, -0.77, ...],   # 第 2 个元素：对应 "橘子" 的 1024 维向量
        #     [0.01, 0.99, 0.44, ...],    # 第 3 个元素：对应 "西瓜" 的 1024 维向量
        #     [-0.5, -0.1, 0.66, ...]     # 第 4 个元素：对应 "葡萄" 的 1024 维向量
        # ]
        print(f"  已向量化 {start + len(batch)}/{len(texts)} 条")
    return vectors


def get_collection(chroma: chromadb.ClientAPI):
    """拿到集合。改动任务③会用到 REBUILD 开关。"""
    if REBUILD:
        try:
            chroma.delete_collection(name=COLLECTION_NAME)
            # chroma.delete_collection(...)：这是 Chroma 官方的底层 API，把名为 rag_notes 的整个集合（包含里面的所有数据、索引）瞬间抹除。
            print(f"[REBUILD] 已删除旧集合 {COLLECTION_NAME}")
        except Exception as e:
            print(f"[REBUILD] 没有旧集合可删（{type(e).__name__}），继续")
            # try-except 容错：如果是第一次运行，数据库里根本没有 rag_notes 这个集合，调用 delete_collection 会报错崩掉。
            # 用 try-except 捕获它，打一行日志，程序继续往下走。
    return chroma.get_or_create_collection(
        # 如果有这个集合，就把它拿出来；如果没有，就新建一个空集合。这样写就不用写
        name=COLLECTION_NAME,
        # ★ 显式指定 cosine。默认是 l2，对文本 embedding 不合适（见下午 14:30 那段）
        configuration={"hnsw": {"space": "cosine"}},
        # "hnsw"：告诉数据库使用 HNSW（Hierarchical Navigable Small World） 算法来构建索引。
        # "space": "cosine"：将衡量向量之间相似度的函数，从默认的 L2（欧几里得距离） 改为 Cosine（余弦相似度）。
    )


def main() -> None:
    print("=" * 60)
    print(f"文档：{DOC_PATH}")

    chunks = split_text(DOC_PATH, CHUNK_SIZE, CHUNK_OVERLAP)

    # chunks = text_splitter.split_documents([doc]) 底层为。 前者text_splitter为对象，后者为方法
    # doc = Document(page_content=text, metadata={"source": Path(path).name})
    # 自动从 Document 对象中提取 page_content 进行切割，它还会自动把原 Document 的 metadata（元数据）原封不动地复制给切割出来的每一个小块。

    # chunks最终还是一个对象列表
    # Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='Q33：Agent 和 Chain 的区别'),
    # Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•Chain：流程固定，按预设顺序执行\n•Agent：模型自主决定下一步（用不用工具、用哪个）\nQ34：LangChain 里常见组件的作用'),
    # Document(metadata={'s
    print(f"分块参数：chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"分块数：{len(chunks)}")
    if not chunks:
        print("！！分块数为 0，先检查文档内容，别往下走")
        sys.exit(1)

    emb_client = make_embed_client()

    # ---- 1) 向量化 ----
    # 先获取切块后的文本
    # 再用emded_texts方法将文本向量化并获取

    print("-" * 60)
    print("开始向量化……")
    texts = [c.page_content for c in chunks]
    # 将chunks中每一个对象中的page_content都取出，赋值给texts
    vectors = embed_texts(emb_client, texts)
    dim = len(vectors[0])
    # 前面那个 dim（在 embed_texts 里）：是“期望值”（我要什么）。
    # 后面那个 dim（在 main 里）：是“实际值”（我拿到了什么）。
    print(f"向量维度：{dim}")

    # ---- 2) 入库 ----
    # 先连接库
    # 再获取库
    print("-" * 60)
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    # 只是告诉 Chroma，“我要把数据存在 CHROMA_DIR 这个文件夹里”，并在内存里建立好连接。如果这个文件夹不存在，它会帮你建个空文件夹
    collection = get_collection(chroma)
    # 在刚连上的数据库里，获取或创建一个名叫 rag_notes 的“集合（Collection）

    # ✅ 用 enumerate 在遍历的同时获取下标 i
    ids = [f"{c.metadata.get('source', 'unknown')}::{i}" for i, c in enumerate(chunks)]
    # 最终ids为[字符串] 形式为 '02_网络协议笔记.txt::0'
    # chunks最终还是一个对象列表
    # 每个对象 Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='Q33：Agent 和 Chain 的区别'),
    collection.add(
        ids=ids,
        # 入库前，将内容来历和每块的索引拼接到一起，以 ::分隔
        embeddings=vectors,
        documents=texts,
        metadatas=[dict(c.metadata) for c in chunks],
    )
    # 执行 collection.add(...) 时，Chroma 在底层就像拉链一样，把下标相同的元素强行缝合在一起，变成数据库里的一条记录（Row）：
    # 向量库中每一个行的内容即为
    # 行号(Index)	ids (主键)	embedding (向量)	    document (原文)	        metadata (元数据)
    # 0	        "笔记.txt::0"	[0.12, -0.45, ...]	"TCP 三次握手是..."	    {"source": "笔记.txt"}
    # 1	        "笔记.txt::1"	[-0.98, 0.33, ...]	"为什么是三次..."	        {"source": "笔记.txt"}

    print(f"本批写入：{len(ids)} 条")

    # ---- 3) 入库自检（三件套） ----
    print("=" * 60)
    print("入库自检")
    print(f"  集合名    ：{collection.name}")
    print(f"  总条数    ：{collection.count()}")
    print(f"  向量维度  ：{dim}")
    peek = collection.peek(limit=1)
    print(f"  抽样 metadata：{peek['metadatas'][0]}")
    print(f"  抽样文档前 30 字：{peek['documents'][0][:30]!r}")
    print("=" * 60)


if __name__ == "__main__":
    main()