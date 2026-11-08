"""混合检索：BM25（关键词）+ 向量（语义），用 RRF 融合。

为什么需要两路：
    向量检索擅长"语义相近"（同义、换述、跨语言），
    但它会丢精确词——错误码、型号、人名、API 名。
    BM25 擅长精确词匹配（TF-IDF 家族），但换个说法就查不到。
    两者互补，融合后召回更稳。

为什么用 RRF 融合，而不是把分数相加：
    两路的分数不在一个量纲上：
      向量侧是余弦相似度（0-1）
      BM25 侧是无上界的词频得分（可能 3.5，也可能 40）
    直接相加没有意义。
    RRF 只看"排名"，不看分数——这是它能跨异构检索器融合的原因。

    RRF 公式：score(d) = Σ_i  1 / (k + rank_i(d))
    k 取 60 是原论文的默认值，作用是压低头部差异、让尾部也有贡献。
"""

from __future__ import annotations

import logging
from functools import lru_cache
# 导入 lru_cache。这是一个缓存装饰器，用来避免重复计算。

import chromadb
import jieba
# 导入 jieba，中文分词的神器。

from rank_bm25 import BM25Okapi
# 从 rank_bm25 库导入 BM25Okapi 算法类，这是底层计算词频的引擎。

from src.retriever import COLLECTION_NAME, DB_PATH, retrieve

logger = logging.getLogger(__name__)

# ---------- 分词配置：BM25 成败的关键 ----------

# 停用词：虚词和疑问词，它们不携带检索意图，只会污染分数
STOPWORDS = set(
    # 定义一个集合 STOPWORDS，使用 set() 是为了提高查找速度
    """
    的 了 和 与 及 或 是 在 有 我 你 他 她 它 们 这 那 个 之 也 都 就 而
    什么 怎么 为什么 如何 哪个 哪些 多少 区别 介绍 说明 一下 可以 需要
    以及 但是 因为 所以 如果 就是 不是 还是 已经 进行 通过 对于 关于
    """.split()
)
# 一个多行字符串，包含了所有不需要的废话词。.split() 会按空格把它们切成列表，set() 再把列表转成集合。

# 领域词：保证 jieba 不把它们切碎（否则 SSE 会变成 S/S/E）
DOMAIN_WORDS = [
    "SSE", "WebSocket", "RAG", "BM25", "RRF", "Chroma", "LangChain", "LangGraph",
    "Embedding", "FunctionCalling", "Prompt", "Agent", "Tool", "TopK",
    "向量检索", "混合检索", "分块", "重排", "召回", "引用", "上下文", "记忆",
    "工具调用", "流式输出", "多轮对话", "意图路由", "证据校验", "重分块",
]

for _word in DOMAIN_WORDS:
    jieba.add_word(_word)
    # 把词强行塞进 jieba 的默认词典，强制它以后遇到这些词不许切碎。


def _tokenize(text: str) -> list[str]:
    """分词 + 去停用词 + 去单字符噪声。"""
    tokens: list[str] = []
    for word in jieba.cut(text):
        # jieba.cut 把长文本切成词，for 循环一个一个拿出来。
        word = word.strip()
        if not word or word in STOPWORDS:
            continue
            # 如果切出来是空的，或者在停用词集合里。直接跳过

        # 丢掉单个汉字/小写字母（"S"、"的"），保留 SSE 这类大写缩写
        if len(word) == 1 and not word.isupper():
            # 如果词的长度只有 1，且它不是全大写（isupper() 为假）
            continue
        tokens.append(word.lower())
        # 把通过层层筛选的词，转成小写后，塞进结果列表
    return tokens

RRF_K = 60      # RRF 平滑常数
TOP_N = 10      # 每一路先各取多少条


@lru_cache(maxsize=1)
def _load_corpus():
    """从 Chroma 一次性加载所有块，建 BM25 索引。

    返回 (ids, texts, sources, bm25)。
    用 lru_cache 保证只建一次——建索引要遍历所有块并分词，很慢。
    """
    col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    # collection.add(
    #         ids=ids,
            # ids默认是主键，所以默认返回，其他内容如果不写入include，可能默认不会返回
    #         # 入库前，将内容来历和每块的索引拼接到一起，以 ::分隔
    #         embeddings=vectors,
    #         documents=texts,
    #         metadatas=[dict(c.metadata) for c in chunks],
    #     )
    data = col.get(include=["documents", "metadatas"])
    # 升级了 Chroma 库，或者换个了底层实现，它可能就不默认返回 documents 了
    # 拒绝拿没用的数据（省内存），如果不写include可能所有数据全部返回
    # 以及如果你需要把向量取出就必须写include=["embeddings"]，否则无法取出

    ids = list(data["ids"])
    texts = [(d or "") for d in data["documents"]]
    sources = [((m or {}).get("source") or "?") for m in data["metadatas"]]

    tokenized = [_tokenize(t) for t in texts]
    # 对所有文档文本调用分词函数，把文档变成二维列表（每个被切分的块是一个词列表）。

    bm25 = BM25Okapi(tokenized)
    # 用所有分词好的文档，实例化 BM25 引擎。它会在这时计算好所有词的 IDF 等底层数值。

    logger.info("BM25 索引建立完成：%d 块", len(ids))
    return ids, texts, sources, bm25


def search_bm25(question: str, n: int = TOP_N) -> list[dict]:
    # 定义 BM25 搜索函数，接收问题，默认返回 10 条。
    """BM25 关键词检索。

    返回结构和其他检索器一致：id / text / score / metadata。
    """
    ids, texts, sources, bm25 = _load_corpus()

    tokens = _tokenize(question)
    # 把用户的提问分词，比如 "SSE 怎么用" 变成 ["sse", "怎么用"]。

    scores = bm25.get_scores(tokens)
    # 核心一步。用分好的词在 BM25 引擎里算分，返回一个包含所有文档分数的数组。

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    # range(len(scores)) 生成原始的下标序列。
    # :分隔"参数"和"函数体"

    # 没有把 texts 数组和 ids 数组打乱。
    # 只生成了一个排好序的下标列表。
    # 根据 [15.5, 0.0, 5.2] 的排序，order 变成了：[0, 2, 1]（因为 15.5 第一，5.2 第二，0.0 第三）。

    hits: list[dict] = []
    for i in order:
        if scores[i] <= 0:
            continue          # 分数为 0 说明一个词都没命中，不要
        hits.append(
            {
                "id": ids[i],
                "text": texts[i],
                "score": float(scores[i]),
                "metadata": {"source": sources[i]},
            }
        )
    logger.debug("BM25 query=%r 命中=%d", question, len(hits))
    return hits


def rrf_fuse(*hit_lists: list[dict], top_k: int = 5, k: int = RRF_K) -> list[dict]:
    # 可以传任意数量的参数进来，我会把它们全部打包成一个元组（Tuple）
    # *hit_lists 表示接收任意数量的列表作为参数。
    """多路检索结果按排名融合（Reciprocal Rank Fusion）。

    注意：融合后的 score 是 RRF 分（约 0.0x），只用于排序，不可当相似度解释。
    所以把原始分数保留在 score 里，融合分放 rrf_score。
    """
    rrf_scores: dict[str, float] = {}
    originals: dict[str, dict] = {}

    for hits in hit_lists:
        # vector_hits = retrieve(question, k=n)
        # bm25_hits = search_bm25(question, n=n)
        #
        #fused = rrf_fuse(vector_hits, bm25_hits, top_k=k)
        for rank, hit in enumerate(hits, start=1):
            hid = hit["id"]
            rrf_scores[hid] = rrf_scores.get(hid, 0.0) + 1.0 / (k + rank)
            # ！如果文档第一次出现，基础分是 0.0；如果之前已经出现过，就把两路的分数相加。
            originals.setdefault(hid, hit)
            # hid是键 ， hit是值
            # 保证同样的键只存一次，用 setdefault 保证只存第一次出现的，不覆盖

    ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    # rrf_scores.items()将rrf_scores变成元组

    fused: list[dict] = []
    for hid, score in ranked:
        item = dict(originals[hid])
        # 拿出了里面那个完整的字典（也就是当初的 hit）
        # hit是之前存入的四个值，
        # 把原始hit浅拷贝一份（防止修改影响原对象）
        item["rrf_score"] = score
        # 把计算出来的 RRF 分数挂到数据上。
        fused.append(item)
        # 将hit+score合并成为item然后存入fused

    return fused


def retrieve_hybrid(question: str, k: int = 5, n: int = TOP_N) -> list[dict]:
    """向量 + BM25 双路召回，RRF 融合后返回前 k 条。"""
    vector_hits = retrieve(question, k=n)
    bm25_hits = search_bm25(question, n=n)

    fused = rrf_fuse(vector_hits, bm25_hits, top_k=k)

    logger.info(
        "混合检索 向量=%d BM25=%d 融合=%d",
        len(vector_hits), len(bm25_hits), len(fused),
    )
    return fused