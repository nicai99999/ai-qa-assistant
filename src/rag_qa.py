"""RAG 问答链路：检索 → 拼上下文 → 生成回答。

适配本项目的真实接口：
  - retriever.retrieve(question, k) -> list[dict]，key 为 id/text/score/metadata
  - score 是相似度（1 - 余弦距离），越大越相关
  - llm.get_llm() 返回全局单例 ChatModel

设计要点：
1. 检索 / 拼装 / 生成三段分离，便于单独测试。
2. Prompt 抽成模块常量，方便做实验对比（今天要用 3 个变体）。
3. 检索为空时直接返回，不浪费一次模型调用。
4. 返回 hits 明细，Day 20 做引用时直接用。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from retriever import TOP_K, retrieve
from llm import get_llm
from logging_setup import log_llm_call

logger = logging.getLogger(__name__)

# 资料中查不到时的统一回复，便于上层判断
NO_ANSWER = "资料中未提及"

# 版本 A：基础约束（默认用这个）
RAG_PROMPT_A = """你是一个严谨的问答助手。请严格根据下面提供的资料回答问题。

资料：
{context}

问题：{question}

要求：
1. 只使用资料中的信息回答，不要编造。
2. 如果资料中没有相关信息，直接回答"资料中未提及"，不要猜测。
3. 回答要简洁，不要大段复述资料原文。

回答："""

# 版本 B：强制先判断（实验用）
RAG_PROMPT_B = """你是一个严谨的问答助手。

资料：
{context}

问题：{question}

请先判断下面这段资料是否足以回答问题。
如果不足，只输出：资料中未提及
如果足够，直接给出答案，不要解释你的判断过程。

回答："""

# 版本 C：强制标注依据（实验用）
RAG_PROMPT_CITED = """你是一个严谨的问答助手。请严格根据下面提供的资料回答问题。

资料：
{context}

问题：{question}

要求：
1. 每个结论后面用 [1]、[2] 这样的编号标注依据来自第几段资料。
2. 只使用资料中的信息回答，不要编造。
3. 只能引用资料里真实存在的编号，不要自己编号。
4. 如果资料中没有相关信息，只回答：资料中未提及

回答："""


@dataclass
class Citation:
    """一条引用信息。"""

    index: int          # 上下文里的编号，1 开始
    source: str         # 来源文件名
    chunk: str          # 第几块（从 id 里拆出来）
    score: float        # 相似度，越大越相关
    snippet: str        # 内容摘要（前 60 字）

    def as_line(self) -> str:
        """渲染成一行给用户看。"""
        return f"[{self.index}] {self.source}（第 {self.chunk} 块，相似度 {self.score:.3f}）"


@dataclass
class RagResult:
    """一次 RAG 问答的结果。"""

    answer: str
    hits: list[dict] = field(default_factory=list)
    # default_factory=list：注意，这里传的 list 没有加括号！它传的是“生成列表的工厂类（函数）
    # 在后台执行一次 list()，造出一个全新的空列表给 doc1
    context: str = ""
    prompt: str = ""
    citations: list[Citation] = field(default_factory=list)

    @property
    # 私有字段 + Getter	private int age;
    # public int getAge() { return age; }	person.getAge()
    def is_no_answer(self) -> bool:
        """模型是否明确表示资料中没有答案。"""
        return NO_ANSWER in self.answer
    # in：用来判断前面的子串是否包含在后面的长串里。

    @property
    def top_score(self) -> float:
        """召回结果中最高相似度，检索质量的一个粗指标。"""
        if not self.hits:
            return 0.0
        return max(hit.get("score") or 0.0 for hit in self.hits)

    # hits.append({
    #                 "id": res["ids"][0][i],
    #                 "text": res["documents"][0][i],
    #                 "score": similarity,  # 这里存的是相似度，不是距离！
    #                 "metadata": res["metadatas"][0][i]
    #             })

    # hit.get("score") or 0.0？
    # 短路求值（兜底）：or 0.0 是极其巧妙的写法。如果 .get("score") 拿到了真实分数（比如 0.8），
    # 0.8 or 0.0 返回 0.8；如果拿到了 None，None or 0.0 就会返回 0.0。这就保证传给 max() 的绝对是一堆纯正的浮点数，绝不会因为 None 导致比较报错。

def _split_id(hit_id: str) -> tuple[str, str]:

    # hits.append({
    #                 "id": res["ids"][0][i],
            # 'ids': [['02_网络协议笔记.txt::0', '02_网络协议笔记.txt::2', '02_网络协议笔记.txt::1']],
    #                 "text": res["documents"][0][i],
    #                 "score": similarity,  # 这里存的是相似度，不是距离！
    #                 "metadata": res["metadatas"][0][i]
    #             })
    """从 hit 的 id 里拆出 (来源, 第几块)。

    你的 id 形如 '02_网络协议笔记.txt::0'，按 '::' 拆。
    格式变了就改这里，别改上面的逻辑。
    """
    if not hit_id:
        return "?", "?"
    if "::" in hit_id:
        source, chunk = hit_id.rsplit("::", 1)
        # rsplit 的全称是right split（从右往左切）
        # 从右向左找，最多只按:: 切一刀，切出两个部分
        return source, chunk
    return hit_id, "?"

def build_context(hits: list[dict]) -> str:
    """把检索到的片段拼成带编号的上下文。

    编号（[1]、[2]）有两个用途：
    1. 让模型能引用（版本 C 的 Prompt 要用）
    2. Day 20 做"答案 + 来源"时直接对应
    """
    if not hits:
        return ""

    parts: list[str] = []
    for index, hit in enumerate(hits, start=1):
        # start=1 的意思就是：强制让枚举出来的索引（index）从 1 开始，而不是默认的 0
        text = (hit.get("text") or "").strip()
        if text:
            parts.append(f"[{index}] {text}")
            # "[1] TCP 三次握手"
    return "\n\n".join(parts)
    # 遍历 parts 列表，把里面的字符串一个个拿出来，用 \n\n 粘在一起，最后返回一个完整的长字符串


def build_citations(hits: list[dict]) -> list[Citation]:
    """从 hits 生成引用列表。编号必须和 build_context 里的一致。"""
    citations: list[Citation] = []
    for index, hit in enumerate(hits, start=1):
        source, chunk = _split_id(hit.get("id") or "")
        if source == "?":
            # id 拆不出来时退回 metadata 里的 source
            source = (hit.get("metadata") or {}).get("source", "?")

        snippet = (hit.get("text") or "").strip().replace("\n", " ")[:60]
        citations.append(
            Citation(
                index=index,
                source=source,
                chunk=chunk,
                score=hit.get("score") or 0.0,
                snippet=snippet,
            )
        )
    return citations


def extract_cited_indices(answer: str) -> list[int]:
    """从答案里解析出模型实际引用了哪几个编号。

    用于实验 2：核对"模型说的 [1] 到底对不对"。
    返回去重后的升序列表。
    """
    found = re.findall(r"\[(\d+)\]", answer or "")
    # 如果模型回答：“TCP需要三次握手[1]，因为要确认收发能力[3]。”，findall 会提取出所有的数字，返回一个字符串列表：['1', '3']
    # re：Python 的正则表达式模块
    # r"\[(\d+)\]"：这是正则表达式的模式（Pattern）。
    # \[：匹配左方括号 [（因为方括号在正则里有特殊含义，所以要加斜杠转义）。
    # (\d+)：匹配一个或多个数字（\d 代表数字，+ 代表连续）。加括号 () 代表只提取括号里的数字。
    # \]：匹配右方括号 ]。
    return sorted({int(n) for n in found})


def format_answer(result: RagResult, show_sources: bool = True) -> str:
    """把结果渲染成「答案 + 来源列表」。"""
    lines = [result.answer]

    if show_sources and result.citations:
        lines.append("")
        lines.append("来源：")
        for citation in result.citations:
            lines.append(f"  {citation.as_line()}")

    return "\n".join(lines)


def ask_with_rag(
    question: str,
    k: int = TOP_K,
    prompt_template: str = RAG_PROMPT_A,
) -> RagResult:
    """执行一次完整的 RAG 问答。

    Args:
        question: 用户问题。
        k: 召回的片段数。
        prompt_template: Prompt 模板，实验时可以换 RAG_PROMPT_B / RAG_PROMPT_C。

    Returns:
        RagResult，包含答案、召回的片段、拼好的上下文和实际使用的 Prompt。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question 不能为空")

    # 1. 检索
    hits = retrieve(question, k=k)
    if not hits:
        logger.warning("未检索到任何片段，question=%r", question)
        return RagResult(answer=NO_ANSWER)

    # 2. 拼上下文
    context = build_context(hits)
    citations = build_citations(hits)
    logger.debug("召回 %d 个片段，上下文长度 %d 字符", len(hits), len(context))

    # 3. 生成（带耗时和 token 记录）
    prompt = prompt_template.format(context=context, question=question)
    llm = get_llm()

    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed_ms = (time.perf_counter() - start) * 1000
    # 计算一段代码执行所花费的时间，单位是“毫秒（ms）

    answer = str(response.content).strip()
    log_llm_call(logger, elapsed_ms, getattr(response, "usage_metadata", None))

    return RagResult(answer=answer, hits=hits, context=context, prompt=prompt, citations=citations,)
    # 实例化时只执行类的构造函数（__init__），不会执行类里的其他业务方法。
    # 执行 return RagResult(answer=answer, hits=hits, ...) 时，会将hits 等内容直接塞进 result 对象的内存里。需要时可直接取出
# 内存里的 result 对象长这样：
# result = {
#     "answer": "TCP需要三次握手...",
#     "hits": [{"score": 0.87}, {"score": 0.61}, {"score": 0.58}],
#     "context": "[1] TCP三次握手...",
#     "prompt": "你是..."
# }

def ask_with_citation(question: str, k: int = TOP_K) -> RagResult:
    """执行一次带引用的 RAG 问答。

    和 ask_with_rag 的唯一区别：默认用 RAG_PROMPT_CITED，
    要求模型在答案里标 [1][2]。
    """
    return ask_with_rag(question, k=k, prompt_template=RAG_PROMPT_CITED)


def ask_with_rag_stream(
    question: str,
    k: int = TOP_K,
    prompt_template: str = RAG_PROMPT_CITED,
) -> RagResult:
    """流式版本的 RAG 问答：边生成边打印，并返回首字延迟。

    和非流式版本的区别只有一个：把 llm.invoke() 换成 llm.stream()。
    invoke 一次给全部；stream 是一段一段给（每段是一个 AIMessageChunk）。

    新增的可观测指标：
        首字延迟（time to first token）—— 从发请求到第一个字出现的时间
        总耗时 —— 全部生成完的时间

    这两个数字是"为什么流式体验更好"的量化依据：首字延迟远小于总耗时。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("question 不能为空")

    hits = retrieve(question, k=k)
    #向量化大模型，最开始返回的为res的格式为键值对 {'ids': , 'documents': , 'included': , 'metadatas': , 'distances': }
    # {'ids': [['02_网络协议笔记.txt::0', '02_网络协议笔记.txt::2', '02_网络协议笔记.txt::1']], 'embeddings': None,
        #  'documents': [
        #      ['TCP 三次握手\n第一次：客户端发送 SYN=1，序号 seq=x\n第二次：服务端返回 SYN=1，ACK=1，seq=y，ack=x+1\n第三次：客户端发送 ACK=1，ack=y+1',
        #       '。\n如果只有两次，服务端无法确认客户端的接收能力，也无法防止历史连接请求造成的错误。',
        #       '为什么是三次而不是两次：为了确认双方的收发能力都正常']], 'uris': None,
        #  'included': ['metadatas', 'documents', 'distances'], 'data': None,
        #  'metadatas': [
        #     [{'source': '02_网络协议笔记.txt'}, {'source': '02_网络协议笔记.txt'}, {'source': '02_网络协议笔记.txt'}]],
        #  'distances': [[0.0, 0.37685900926589966, 0.38716739416122437]]}

# 优化res去除不必要的内容，最终将优化为hits
    # hits.append({
    #                 "id": res["ids"][0][i],
    # 'ids': [['02_网络协议笔记.txt::0', '02_网络协议笔记.txt::2', '02_网络协议笔记.txt::1']],
    #                 "text": res["documents"][0][i],
    #                 "score": similarity,  # 这里存的是相似度，不是距离！
    #                 "metadata": res["metadatas"][0][i]
    #             })
    if not hits:
        logger.warning("未检索到任何片段，question=%r", question)
        return RagResult(answer=NO_ANSWER)

    context = build_context(hits)
    citations = build_citations(hits)
    prompt = prompt_template.format(context=context, question=question)

    llm = get_llm()

    start = time.perf_counter()
    first_token_at: float | None = None # 记录第一个字输出的时间
    pieces: list[str] = [] # 准备一个空列表，用来把大模型一点一点吐出来的文本碎片收集起来

# llm.invoke()，是阻塞式的（如等 3 秒后一次性给 100 个字）。

    for chunk in llm.stream(prompt): # chunk为一个token（词元）
        # llm.stream() 返回的是一个生成器（Generator）。大模型每算出一个字（或一个词），就立刻通过网络塞给你一个 chunk（AIMessageChunk 对象）

        # 在英文里：一个 Token 可能是一个完整的单词（如 "apple"），也可能是半个单词（如 "un" + "break" + "able"）。

        # 在中文里：一个 Token 通常是一个词（如 "人工智能"）或一个单字（如 "我"）。有时候，一个非常罕见的字（比如生僻字）甚至会被拆成几个字节的 Token。

        # print(chunk)
        # AIMessageChunk(content='你', ...)
        # AIMessageChunk(content='好', ...)
        # AIMessageChunk(content='，', ...)
        # AIMessageChunk(content='世', ...)
        # AIMessageChunk(content='界', ...)

        if first_token_at is None:
            # 这里检查的不是 chunk，而是 first_token_at。
            first_token_at = time.perf_counter()
            # 记录首字延迟（TTFT）
        # 首字延迟（Time To First Token，简称 TTFT）指的是：按下“发送”按钮的那一刻，到大模型吐出“第一个字”的那一刻，中间这段时间。
        # “第一个字和第二个字之间的延迟”，在专业术语里叫 “字间延迟”（Inter-token latency）。

            # 极其重要的工程指标。
            # 当循环第一次执行时，first_token_at 必定是 None，此时立刻按下第二块秒表。
            # 为什么要单独记这个？因为用户觉得“快不快”，取决于第一块秒表到第二块秒表的时间（首字延迟，TTFT）

        text = chunk.content
        if text:
            pieces.append(str(text))
            print(text, end="", flush=True)
            # 打字机效果”的物理实现！
            # end=""：告诉 Python 打印完这个字后不要换行，下一个字紧挨着它。
            # flush=True：默认情况下，Python 会把输出先放在内存缓冲区，等攒够一行或程序结束才显示。加了 flush=True，强行清空缓冲区，让字瞬间出现在用户的屏幕上。

    end = time.perf_counter()
    # 循环结束，大模型全部说完，按下最后一块秒表
    print()  # 换行
    # 补一个换行符，防止污染下一行的终端输出

    answer = "".join(pieces).strip()
    ttft_ms = ((first_token_at or end) - start) * 1000
    # 首字延迟（Time To First Token）。用“第一块秒表时间 - 开始时间”乘 1000 换算成毫秒
    # 大模型一个字都没吐出来（比如触发了安全拦截），first_token_at 会一直是 None。None or end 会返回 end。不会报错

    total_ms = (end - start) * 1000
    # 从发请求到全部说完，总共花了多少毫秒

    logger.info("首字延迟 %.0f ms，总耗时 %.0f ms，回答长度 %d", ttft_ms, total_ms, len(answer))

    result = RagResult(
        answer=answer,
        hits=hits,
        context=context,
        prompt=prompt,
        citations=citations,
    )
    # 把大模型的回答、检索到的文档、生成的 Prompt、以及引用信息，全部打包成一个对象。

    # 把两个延迟指标挂到结果上，方便脚本读取
    result.ttft_ms = ttft_ms
    result.total_ms = total_ms
    return result



# ===== 多轮对话（Day 24）=====

HISTORY_PROMPT = """你是一个严谨的问答助手。

对话历史：
{history}

资料：
{context}

问题：{question}

要求：
1. 只使用资料中的信息回答，不要编造。
2. 每个结论后面用 [1]、[2] 标注依据来自第几段资料。
3. 如果资料中没有相关信息，只回答：资料中未提及
4. 可以参考对话历史理解用户指的是什么，但答案必须来自资料。
5. 如果问题是在询问对话中**用户自己说过的信息**（姓名、偏好、之前提过的事），
   直接根据对话历史回答，并在末尾标注（来自对话历史）。

回答："""


def _format_history(turns: list) -> str:
    """把历史轮次拼成文本。没有历史时返回"（无）"。"""
    if not turns:
        return "（无）"

    lines: list[str] = []
    for turn in turns:
        # [  # <- 这是 turns（列表）
        #     #         Turn(question="你好", answer="你好呀"),       # <- 第 1 轮问答
        #     #         Turn(question="你是谁", answer="我是AI助手")    # <- 第 2 轮问答
        #     #     ]

        lines.append(f"用户：{turn.question}")
        # Turn(question="你好", answer="你好呀"),

        # 助手回答太长会撑爆 token，这里截断到 100 字
        answer = turn.answer if len(turn.answer) <= 100 else turn.answer[:100] + "…"
        # 结果A  if  条件  else  结果B
        lines.append(f"助手：{answer}")
    return "\n".join(lines)


def ask_with_history(
    question: str,
    session_id: str = "default",
    k: int = TOP_K,
) -> RagResult:
    """带会话记忆的 RAG 问答。

    和 ask_with_citation 的区别：
        1) Prompt 里多插了一段对话历史
        2) 回答完成后把这一轮写进 store

    注意（今天的已知局限）：
        检索仍然只用当前问题，不用历史。
        所以代词问题（"它是什么"）检索效果差 —— Day 26 解决。
    """
    from memory_v2 import ConversationStoreV2
    mode = "priority"
    store = ConversationStoreV2(mode=mode, llm=get_llm() if mode == "summary" else None)

    question = (question or "").strip()
    if not question:
        raise ValueError("question 不能为空")

    history = store.history(session_id)

    hits = retrieve(question, k=k)
    if not hits:
        answer = NO_ANSWER
        store.append(session_id, question, answer)
        return RagResult(answer=answer)

    context = build_context(hits)
    citations = build_citations(hits)

    prompt = HISTORY_PROMPT.format(
        history=_format_history(history),
        context=context,
        question=question,
    )

    llm = get_llm()
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed_ms = (time.perf_counter() - start) * 1000

    answer = str(response.content).strip()
    log_llm_call(logger, elapsed_ms, getattr(response, "usage_metadata", None))

    store.append(session_id, question, answer)

    return RagResult(
        answer=answer,
        hits=hits,
        context=context,
        prompt=prompt,
        citations=citations,
    )