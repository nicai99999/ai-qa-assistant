"""Streamlit 界面：AI 问答助手（多会话 · 流式 · 引用溯源）。

用法：
    uv run streamlit run app.py
"""

import time

import streamlit as st

from llm import get_llm
from logging_setup import setup_logging
from memory_v2 import ConversationStoreV2
from retriever import retrieve
from safe_qa import friendly_message
from streaming import stream_answer
from ingest import ingest

# set_page_config：设置浏览器标签页的标题为“AI 问答助手”，图标为 💬，页面宽屏布局
st.set_page_config(page_title="AI 问答助手", page_icon="💬", layout="wide")

# ==========================================
# 🎨 自定义 CSS 样式注入（美化颜色、间距、圆角）
# ==========================================
st.markdown("""
<style>
    /* 侧边栏背景色微调 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    /* 聊天气泡间距微调 */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    /* 指标卡片背景与圆角 */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    /* 折叠面板圆角 */
    .stExpander {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
# @st.cache_resource（重中之重）：
# Streamlit 的运行机制是：你只要在网页上点一下按钮，整个 Python 脚本就会从头到尾重新跑一遍。
# 如果不加这个装饰器，你每发一条消息，程序就会重新创建一个数据库连接、重新初始化大模型、清空之前的记忆（因为内存重新分配了）。
# cache_resource 告诉 Streamlit：“把 store 缓存下来，不管网页怎么刷新，永远用内存里的这一个实例。”
def init_store() -> ConversationStoreV2:
    """全局只建一次。

    cache_resource 适合"不能被序列化、只能存在一份"的东西：
    数据库连接、模型客户端、会话存储。
    （cache_data 适合纯数据结果，比如一张表。）
    """
    setup_logging()
    store = ConversationStoreV2(mode="priority", llm=get_llm())
    try:
        retrieve("预热", k=1)
    except Exception:
        pass
    return store


store = init_store()
# 调用上面的函数。因为加了缓存，这行代码在程序第一次运行时创建 store，后续刷新网页只是直接拿现成的


# ---------- 初始化状态 ----------
if "histories" not in st.session_state:
    st.session_state.histories = {}
    # .messages 是一个列表（List） , 整个网页只有一个聊天框
    # .histories 是一个字典（Dict） , 装的是多个会话

# 初始化当前会话 ID（带时间戳生成唯一 ID）
if "current_sid" not in st.session_state:
    sid = f"s{int(time.time())}"

    st.session_state.histories[sid] = []
    st.session_state.current_sid = sid
    # 把刚刚生成的sid放入缓存中
    # st.session_state：是 Streamlit 专门用来跨页面刷新保存数据
    # current_sid 口袋里放的小标签

if "stats" not in st.session_state:
    st.session_state.stats = {"tokens": 0, "turns": 0, "elapsed_ms": 0.0}


# ---------- 侧边栏 ----------
with st.sidebar:
    # 所有写在里面的代码，都会渲染在网页左侧的边栏里
    st.header("📁 会话管理")

    sids = list(st.session_state.histories.keys())

    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        # st.button 渲染一个按钮。
        new_sid = f"s{int(time.time() * 1000)}"
        st.session_state.histories[new_sid] = []
        st.session_state.current_sid = new_sid
        st.rerun()
        # 强制刷新整个页面，让界面立刻变成刚新建的空白状态

    chosen = st.selectbox(
    # 调用 Streamlit 的下拉框组件
        "切换会话",
        options=sids,
        # 下拉框里的所有选项。sids 是一个列表（比如 ["s1700001", "s1700002"]）
        index=sids.index(st.session_state.current_sid),
        # 设置下拉框的默认选中项。如果不写这行，下拉框默认会选第一个。但用户可能正在看第 3 个会话
        format_func=lambda s: f"{s}（{len(st.session_state.histories[s]) // 2} 轮）",
    )
    if chosen != st.session_state.current_sid:
        st.session_state.current_sid = chosen
        # 一旦确认用户切换了会话，立刻把系统指针（current_sid）更新为新的选择。
        st.rerun()

    if st.button("🗑️ 清空当前会话", use_container_width=True):
        sid = st.session_state.current_sid
        st.session_state.histories[sid] = []
        # 清空前端（网页）上显示的聊天气泡
        store.clear(sid)
        # 调用之前写的记忆类，清空后端的真实聊天记录
        st.rerun()

    st.divider()
    # 在网页上画一条水平分割线（就像 HTML 里的 <hr> 标签）

    st.header("📚 知识库")

    uploaded = st.file_uploader(
        # st.file_uploader , Streamlit 提供的文件上传组件
        "上传文档（pdf / txt / md）",
        # 这是显示在网页上的提示文字（Label）
        type=["pdf", "txt", "md"],
        # 安全过滤与限制。这告诉组件：“只允许用户选择这 3 种后缀的文件”
    )

    if uploaded is not None and st.button(
        f"📥 入库「{uploaded.name}」", use_container_width=True
    ):
        with st.spinner("解析并向量化中…"):
            try:
                count = ingest(uploaded.name, uploaded.getvalue())
                st.success(f"✅ 已入库 {count} 个片段")
            except ValueError as exc:
                # 预期内的错误（扫描件、格式不支持）
                st.warning(f"⚠️ {str(exc)}")
            except Exception as exc:
                st.error(f"❌ 入库失败：{type(exc).__name__}: {exc}")
                # type(exc).__name__拿到异常的类型名 , {exc}具体的错误详情

    st.divider()

    st.header("📊 实时统计")
    stats = st.session_state.stats
    #           st.session_state.histories	                     st.session_state.stats
    # 数据结构	字典 (Dict)：{ "会话ID": [消息列表] }	            字典 (Dict)：{"tokens": 0, "turns": 0, "elapsed_ms": 0.0}
    # 业务含义	业务内容（聊天记录）。存的是用户和 AI 具体的对话文字。	聚合指标（统计数据）。存的是数字，用于性能监控。
    # 作用域范围	单会话级。每个 session_id 拥有自己独立的一组聊天记录。	全局级。不管你开多少个会话，所有的 Token 和耗时都在累加。

    # 使用 columns 将统计指标横向排布，节省空间
    st.metric("Tokens", stats["tokens"] or "—")
    avg = stats["elapsed_ms"] / stats["turns"] if stats["turns"] else 0.0
    st.metric("平均耗时", f"{avg:.0f} ms")
    st.metric("对话轮数", stats["turns"])
    # st.metric 用于渲染显眼的数据指标卡片


# ---------- 主区 ----------
current = st.session_state.current_sid
history = st.session_state.histories[current]

st.title("💬 AI 问答助手")
st.caption("本地知识库 · 流式输出 · 引用溯源 · 多会话")
# st.title 和 st.caption：渲染大字标题和小字说明

for message in history:
    # 渲染历史记录（把数据变成气泡）
    with st.chat_message(message["role"]):
        # 根据角色自动生成不同的头像和气泡样式
        st.markdown(message["content"])
        # 把具体的文字内容渲染到气泡里。支持 Markdown 格式

if prompt := st.chat_input("问我任何关于知识库的问题"):
    # st.chat_input(...)：在页面最底部生成一个输入框。
    # :=：这是 Python 的海象运算符。它相当于一句话干两件事：把用户输入的内容赋值给 prompt，同时判断 prompt 是否非空。如果用户没输入，这段代码就跳过。

    history.append({"role": "user", "content": prompt})
    # 用户一旦按下回车，立刻把用户的话加到前端记忆里
    with st.chat_message("user"):
        st.markdown(prompt)
        # 并立刻画出一个用户的聊天气泡

    with st.chat_message("assistant"):
        # 画出一个 AI 的聊天气泡
        collector: dict = {}
        started = time.perf_counter()

        try:
            answer = st.write_stream(
                # 来源在检索阶段就确定了，但 st.write_stream 会独占那个渲染区， 只能等它结束再渲染别的
                stream_answer(prompt, current, store, collector)
                # 把用户的问题、会话 ID、记忆库传给 stream_answer。这里真正触发了：改写 -> 检索 -> 生成 -> 记录日志 -> 容错处理
            )
        except Exception as exc:
            st.error(friendly_message(exc))
            answer = ""

        wall_ms = (time.perf_counter() - started) * 1000

        # 生成器结束后，来源已经带出来了
        citations = collector.get("citations") or []
        if citations:
            with st.expander(f"📚 来源（{len(citations)} 条）"):
                # Streamlit 里的一个“折叠面板（手风琴）”组件
                # 用户点一下才会展开显示里面的内容
                for c in citations:
                    st.markdown(f"- {c.as_line()}")
                    # 把具体的文字内容渲染到气泡里。支持 Markdown 格式

        # 整合并美化耗时指标（使用小字和代码块样式，让视觉更集中）
        backend_ms = collector.get("retrieve_ms", 0) + collector.get("total_ms", 0)
        st.caption(
            f"⏱️ **后端** {backend_ms:.0f} ms（检索 {collector.get('retrieve_ms', 0):.0f} + "
            f"生成 {collector.get('total_ms', 0):.0f}）· **前端总** {wall_ms:.0f} ms · "
            f"框架开销 {wall_ms - backend_ms:.0f} ms"
        )

    history.append({"role": "assistant", "content": answer or ""})
    # 最后，把 AI 的回答也存入前端记忆。这样当用户问下一个问题时，界面重新加载，刚才的对话气泡依然会显示在屏幕上

    # 累计统计（token 由 stream_answer 通过 collector 带出，若未实现则跳过）
    stats = st.session_state.stats
    stats["turns"] += 1
    stats["elapsed_ms"] += wall_ms
    stats["tokens"] += collector.get("total_tokens", 0) or 0