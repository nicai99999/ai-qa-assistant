"""Day 16：文档分块（Chunking）实验。

复用 Day 15 的 loader.load_document()，不重写加载逻辑。
"""

from loader import load_document          # ← 复用 Day 15 的产出
from langchain_text_splitters import RecursiveCharacterTextSplitter   # ← v1 的正确路径

# 统一把参数放到顶部，方便后续做 200/500/1000 的对比实验
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
END_MARKS = "。！？；!?;\n"      # 句末标点

def summarize(chunks: list, chunk_size: int, chunk_overlap: int) -> dict:
    """算出一组参数的统计指标，供对比表使用。"""
    lengths = [len(c.page_content) for c in chunks]
    # 硬断句：上一块的结尾不是句末标点 → 说明切在了句子中间（粗略指标）
    hard_cut = sum(
        1 for i in range(len(chunks) - 1)
        if chunks[i].page_content and chunks[i].page_content[-1] not in END_MARKS
    )
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "块数": len(chunks),
        "平均块长": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "最长块": max(lengths) if lengths else 0,
        "最短块": min(lengths) if lengths else 0,
        "硬断句数": hard_cut,
    }
def split_text(path, chunk_size, chunk_overlap) :
    # TODO(你写)：用 load_document 读出 docs，取 doc = docs[0]
    docs = load_document(r"D:\ai-qa-assistant\data\06_面试纲要汇总.txt")
    # docs是一个集合[对象(page_content=text , metadata = {})]
    doc = docs[0] # 取出第一个对象
    # Document(page_content=text, metadata={"source": Path(path).name})
    #!
    # print(doc.page_content[0:100])
    # # TODO(你写)：打印总字符数 len(doc.page_content)
    # print(len(doc.page_content))
#!
    # TODO(你写)：构造 splitter，用 split_documents([doc]) 得到 chunks
    # RecursiveCharacterTextSplitter(
        # 递归降级”。它的目标是：尽最大努力保留原文的语义结构（段落、句子），实在没办法了才硬切。
    #     self,
    #     separators: list[str] | None = None,
        # 分隔符列表。默认是 ["\n\n", "\n", " ", ""]。它会优先按“双换行（段落）”切，切完太大就按“单换行”切，再大就按“空格”切，最后按“字符”切。
    #     keep_separator: bool | Literal['start', 'end'] = True,
        # 切分时是否保留分隔符,'start' 或 'end'：把分隔符放在块的开头或结尾
    #     is_separator_regex: bool = False,
        # is_separator_regex：separators 里的字符串是不是正则表达式。默认 False 表示当作普通字符串处理。
    #     kwargs: Any = {}
        # 接收所有其他参数。如 chunk_size、chunk_overlap 就是被它一网打尽，传给父类的。
    # )
    # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        # CharacterTextSplitter 不能保证块不超限，所以 RAG 里默认用 RecursiveCharacterTextSplitter。
    text_splitter = RecursiveCharacterTextSplitter(
        # 递归降级”。它的目标是：尽最大努力保留原文的语义结构（段落、句子），实在没办法了才硬切。
        # separators = None,
        # 分隔符列表。默认是 ["\n\n", "\n", " ", ""]。它会优先按“双换行（段落）”切，切完太大就按“单换行”切，再大就按“空格”切，最后按“字符”切。
        separators=["\n\n", "。", "！", "？", "\n", " ", ""],
        keep_separator= True,
        # 切分时是否保留分隔符,'start' 或 'end'：把分隔符放在块的开头或结尾
        is_separator_regex= False,
        # is_separator_regex：separators 里的字符串是不是正则表达式。默认 False 表示当作普通字符串处理。
        # 接收所有其他参数。如 chunk_size、chunk_overlap 就是被它一网打尽，传给父类的。
        # Set a really small chunk size, just to show.
        chunk_size=CHUNK_SIZE,
        # chunk_size：块的最大大小，其中大小由 length_function 决定。
        chunk_overlap=CHUNK_OVERLAP,
        # 块之间的目标重叠量。重叠块有助于缓解上下文在块之间被分割时的信息丢失
        # 好处	                                        代价
        # 切断处的上下文被救回来	                        总块数变多 —— 同一份文档 200/0 可能 48 块，200/50 变成 60 块左右
        # 检索命中率上升（关键信息不容易漏）	                存储和向量化成本上升（多出来的块都要算 embedding、都要存）
        # 答案更少因为"半句话"而跑偏	                    检索结果更容易重复（相邻块内容重叠，可能一起被召回，挤占 top_k 名额）
        #                                               重叠太大（比如接近 chunk_size）会产生大量重复内容，等于变相浪费上下文
        length_function=len,
        # 确定块大小的函数
        # 分隔符列表（默认为 ["\n\n", "\n", " ", ""]）是否应被解释为正则表达式
    )

    chunks = text_splitter.split_documents([doc])
    # doc = Document(page_content=text, metadata={"source": Path(path).name})
    # 自动从 Document 对象中提取 page_content 进行切割，它还会自动把原 Document 的 metadata（元数据）原封不动地复制给切割出来的每一个小块。
    # print(chunks)

    # page_content='批次 1｜Day 14–21 之后（RAG 周结束）\n这批现在就能答（因为你项目里真的做了）。做完 RAG 后逐条过一遍'),

    # Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='。\n1.1 项目类\nQ1：介绍一下你的项目\n•骨架：做什么 → 技术栈 → 核心链路 → 你负责的部分 → 一个亮点数字\n•❌ 别说"我跟着教程做的"'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='•❌ 别说"我跟着教程做的"\nQ2：为什么用 LangChain，不直接调 API'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•骨架：① 屏蔽不同模型平台的差异（换模型只改配置）② 消息对象化，序列化反序列化不用手写 ③ 提供 Tool/Memory/Retriever 等可复用组件 ④ 代码量和出错点都显著减少'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='•加分："我先手写过一遍原生 HTTP 调用，所以知道框架到底省了什么'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='。"\nQ3：你的项目怎么跑起来的'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•骨架：uv 管理依赖 → uv run -m ai_qa_assistant → 读 .env 拿 key → 初始化 ChatModel → 命令行循环\n1.2 RAG 核心（★ 最高频）'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='1.2 RAG 核心（★ 最高频）\nQ4：RAG 的完整链路'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★★\n•骨架：文档解析 → 分块 → Embedding 向量化 → 存入向量库 → 用户问题向量化 → 相似度检索 Top-K → 拼进 Prompt → 模型生成 → 返回答案+来源'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='Q5：为什么要分块'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•骨架：① 模型上下文有上限 ② 整篇塞进去噪音太多、检索粒度太粗 ③ 分块后能精确定位到相关段落\nQ6：分块大小怎么定？重叠多少'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？重叠多少？ ★★\n•骨架：说你实际试过的参数 + 观察到的差异\n•关键："我试了 200 和 500 两档，200 时召回更精确但上下文容易断裂，500 时上下文完整但容易混入无关内容'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='。最后选了 ___，因为 ___'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='。"\n•重叠作用：防止一句话被从中间切断，导致语义丢失\nQ7：为什么要向量检索？和关键词检索（BM25）区别'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？和关键词检索（BM25）区别？ ★★\n•向量：比"语义相似"，能处理同义词/换述\n•关键词：比"字面匹配"，精确但不懂同义\n•加分："两者可以混合，关键词补精确性，向量补语义'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='。"\nQ8：Top-K 取多少？召回不准怎么办'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？召回不准怎么办？ ★★\n•骨架：K 太小漏信息，太大引入噪音且费 token\n•优化手段：调 K、加相似度阈值过滤、改分块、加 Rerank 重排、混合检索\nQ9：为什么选这个向量库'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•骨架：Chroma 本地跑、零部署、适合学习和单机；Milvus 适合大规模；pgvector 适合已有 PG\n•❌ 别说"因为教程用的这个"\nQ10：怎么评估 RAG 效果好不好'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★★（能答=加分）\n•骨架：建评测集（N 个问题 + 标准答案）→ 跑 → 算命中率/正确率 → 改一个参数再跑 → 对比'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='•关键："我把命中率从 __% 提到 __%"，这个数字必须是你自己测的'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='。\nQ11：模型胡编（幻觉）怎么办'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★\n•骨架：① Prompt 里明确"资料中没有就说不知道" ② 让答案带引用来源，可核对 ③ 提高检索质量（召回准了幻觉就少）\nQ12：Embedding 是什么'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•骨架：把文本映射成高维向量，语义相近的文本向量距离近；维度由模型决定'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='批次 2｜Day 22–28 之后（流式 + 记忆）\n2.1 流式\nQ13：流式输出怎么实现的'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★★\n•骨架：调模型时开 stream=True → 服务端按 SSE 增量返回（每个 chunk 是 delta）→ 逐段推给前端 → 前端边收边渲染'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='•加分："非流式和流式首字延迟差别很大，我实测 ___ 秒 vs ___ 秒'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='。"\nQ14：为什么流式体验更好？\n•骨架：首字延迟低（用户立刻看到反馈），总耗时其实差不多\nQ15：SSE 和 WebSocket 的区别'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★\n•SSE：单向（服务端→客户端）、基于 HTTP、自动重连、轻量\n•WebSocket：双向、独立协议、更重\n•结论：大模型输出是单向推送，SSE 更合适'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='2.2 记忆（★ 能引出开源框架话题）\nQ16：多轮对话怎么实现的'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★★\n•骨架：会话 ID 标识用户 → 把历史消息（role + content）存起来 → 每次请求把历史一起发给模型\nQ17：历史太长怎么办'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★★（这题答好能拉开差距）\n•五种方案（按成本从低到高）：\na.滑动窗口：只留最近 N 条，简单便宜，适合短会话\nb.摘要压缩：早期对话总结成摘要，再拼最近几轮'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='c.优先级截断：给消息打重要性标签，满了先淘汰低分的\nd.向量记忆：历史事实入向量库，按需检索注入\ne.工具结果裁剪：只留模型实际用到的部分\n•加分："我的项目用了 ___，因为 ___'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='。我了解到 Claude Code 这类工具走的是 ___ 思路，区别在于 ___。"\nQ18：你项目里具体用了哪种？为什么？\n•必须答你自己实现的那种，并说清代价（比如滑动窗口会丢早期信息）'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='批次 3｜Day 29–35 之后（工具 + 界面）\n3.1 工具调用\nQ19：工具调用（Function Calling）的完整流程'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★★\n•骨架：① 把工具的描述（名字、参数 schema）给模型 ② 模型判断需要调用，返回工具名 + 参数 ③ 你的代码执行 ④ 结果回传给模型 ⑤ 模型给最终答案'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='Q20：有多个工具时，模型怎么决定用哪个'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•骨架：靠工具的名称和描述（description 写得好，模型才选得准）+ 模型自己推理\nQ21：工具调用失败了怎么办'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★\n•骨架：① 把错误信息明确回传给模型，让它自己决定重试还是换方案 ② 加超时 ③ 兜底话术，不能让 agent 崩\nQ22：为什么要限制 Agent 能调用哪些工具？ ★★\n•骨架：安全边界'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='。比如不能让模型调用删除数据、转账的接口；Prompt 注入可能诱导它调危险工具\nQ23：什么是 Prompt 注入？怎么防'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？怎么防？\n•骨架：攻击者在输入里藏指令，让模型绕过原设定；防法：输入过滤、权限隔离、危险工具需人工确认\n3.2 工程\nQ24：界面怎么做的'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•骨架：Streamlit，输入框 + 流式输出 + 文件上传；数据从哪来（会话状态 / 后端接口）\nQ25：token 成本怎么控制'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★\n•骨架：① 记录每次调用的 token ② 上下文管理（别无限堆历史）③ 按任务选合适模型 ④ 缓存重复问题'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='批次 4｜Day 36–42 之后（部署 + 简历）\nQ26：项目怎么部署的'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•骨架：Dockerfile 打包 → 云服务器 → 环境变量传 key（不打进镜像）→ 开安全组端口\nQ27：Dockerfile 每一行在干什么'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•骨架：基础镜像 → 装依赖 → 拷代码 → 暴露端口 → 启动命令\nQ28：遇到过什么坑？怎么解决的'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★★（几乎必问）\n•准备 2 个真实故事，每个讲清：现象 → 排查过程 → 根因 → 修法\n•推荐素材（你真实踩过的）：\n￮base_url 写错导致 404'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='￮base_url 写错导致 404\n￮三个模块 0 字节导致 import 失败\n￮分页插件没注册导致静默失效\n￮LangChain 旧版 API 跑不起来\nQ29：如果重做，你会改什么'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•骨架：说一个具体的（比如"一开始先把项目骨架跑通再写业务""会更早建评测集"）'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='批次 5｜12 月面试期（Agent 概念，你从面经整理的重点）\n这批现在不背，12 月开始。但可以先混个眼熟'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='。\nQ30：ReAct 是什么？完整流程？ ★★★\n•Thought（想）→ Action（调用工具）→ Observation（看结果）→ 循环，直到得出答案\nQ31：CoT 和 ReAct 的区别'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★\n•CoT：只在"想"的层面推理，不行动\n•ReAct：推理 + 行动交替\n•一句话：CoT 是"想清楚再说"，ReAct 是"边想边试"\nQ32：Multi-Agent 有哪些架构？优缺点'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★（面试要逐个分析）\n•常见形态：协作式（多个 agent 分工，如规划者+执行者）、监督式（supervisor 派活给 worker）、层级式（多层管理）'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='•优点：职责拆分、可并行、单项更专精\n•缺点：通信开销大、状态一致性难、调试困难、成本高（调多次模型）\n•关键是能说出"什么时候不该用多 agent"（简单任务单 agent 就够）'),
    # Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='Q33：Agent 和 Chain 的区别'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？\n•Chain：流程固定，按预设顺序执行\n•Agent：模型自主决定下一步（用不用工具、用哪个）\nQ34：LangChain 里常见组件的作用'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★★\n•ChatModel：调模型\n•PromptTemplate：提示词模板化\n•OutputParser：结构化输出\n•Tool：给模型的工具\n•Retriever：检索'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='•Retriever：检索\n•Memory / Checkpointer：会话历史\nQ35：LangGraph 是什么'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？为什么用图不用链？ ★★\n•图 = 节点（步骤）+ 边（流转）+ 状态（共享数据）\n•为什么用图：Agent 需要循环和条件分支，链是直线做不到\nQ36：RAG 在 Agent 体系里是什么位置'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？ ★（你的原洞察很准）\n•RAG 是 Agent 的一种工具/一个检索步骤，本身不等于 Agent\n•模型没有私域知识，所以 RAG 仍是主流解决方案\nQ37：Skill 是什么？为什么企业爱用'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？为什么企业爱用？\n•骨架：把常见任务封装成可复用的技能模块；开发快、门槛低、复用性强\nQ38：你的记忆怎么设计的'), Document(metadata={'source': '06_面试纲要汇总.txt'}, page_content='？（引出开源框架） ★★★\n•先说自己的方案 → 再说"我看过 Claude Code / Hermes 的做法是 ___"\n•核心三问：记忆怎么存/什么时候清、上下文怎么压缩、提示词怎么拼')]

    # doc = Document(page_content=text, metadata={"source": Path(path).name})

    # chunks最终还是一个对象列表
    return chunks
chunks = split_text(CHUNK_SIZE,
CHUNK_OVERLAP ,
END_MARKS  )
# split_documents() 会把原文档的 metadata 深拷贝一份给每个块。 所以"这个块来自哪个文件"的信息天然保留下来了

# TODO(你写)：打印 len(chunks)
print(f"\n=== 参数: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP} ===")
print(f"总计切出：{len(chunks)} 块\n")

# 参考实现：打印部分
for i, chunk in enumerate(chunks):
    ratio = len(chunk.page_content) / CHUNK_SIZE * 100
    print(f"[块 {i:02d}] 长度={len(chunk.page_content)}  ({ratio:.0f}% of {CHUNK_SIZE})来源={chunk.metadata.get('source')}")
    # 02d
    # d：代表 Decimal（十进制整数）。告诉 Python，我要格式化的是一个整数。
    # 2：代表最小宽度为 2。
    # 0：代表如果数字不够 2 位，用 0 在左边补齐（默认是用空格补齐）。
    print(f"  开头：{chunk.page_content[:50]}……")
    print("-" * 60)

# 注意：summarize 要放到循环外面，否则每打印一个块就会把整份文档的统计重复打一遍
print("\n=== 统计汇总 ===")
print(summarize(chunks, CHUNK_SIZE, CHUNK_OVERLAP))