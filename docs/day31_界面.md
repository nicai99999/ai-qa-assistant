## 1. 目标

把命令行跑通的 RAG 能力做成能在浏览器里操作的页面，
为"演示 / 录屏 / README 截图"做准备。

---

## 2. 实现

| 文件 | 内容 |
|---|---|
| `app.py` | Streamlit 单页应用：聊天区 + 输入框 + 侧边栏 |
| 依赖 | `uv add streamlit` |

### 结构

```
侧边栏
├── 会话 ID 展示
├── 记忆策略展示（priority）
├── 🗑️ 清空对话按钮
└── 说明文案

主区
├── 标题 + 副标题
├── 历史消息渲染（st.chat_message 循环）
└── 输入框（st.chat_input）
      ↓
    safe_ask() → 追加到 session_state → 重新渲染
```

---

## 3. 两个关键机制（这是今天真正学的东西）

### ① 历史必须放 `st.session_state`

**Streamlit 的执行模型：每次交互，整个脚本从头重跑一遍。**

'''python
# ❌ 普通变量：每次交互被重置
messages = []

# ✅ session_state：跨交互保留
if "messages" not in st.session_state:
    st.session_state.messages = []
'''

**后果对比**：放普通变量里，每问一句历史就清空，看起来像"失忆"。

### ② 模型和存储必须 `@st.cache_resource`

'''python
@st.cache_resource
def init_store() -> ConversationStoreV2:
    setup_logging()
    return ConversationStoreV2(mode="priority", llm=get_llm())
'''

**不放缓存会怎样**：每次交互都重建 llm 客户端和会话存储，
既慢（重新建连接池）又会导致"上一轮的记忆丢了"（store 是新的实例）。

`cache_resource` vs `cache_data`：

| 装饰器 | 适合存 | 例子 |
|---|---|---|
| `cache_resource` | 对象实例，不可序列化 | 模型客户端、数据库连接、store |
| `cache_data` | 数据值，可序列化 | DataFrame、查询结果 |

---

## 4. 验证结果

| 检查项 | 结果 |
|---|---|
| `uv run streamlit run app.py` 能打开页面 | ✅ |
| 能提问并拿到带来源的回答 | ✅ |
| 多轮对话历史保留（不被清空） | ✅ |
| 侧边栏清空按钮有效 | ✅ |
| 清空后再问"我刚才问了什么"→ 答不出 | ___（待确认） |
| 截图存 `docs/screenshot_v1.png` | ___（待补） |

---

## 5. 踩过的坑

| 现象 | 原因 | 解 |
|---|---|---|
| 每问一句历史就清空 | 历史存在普通变量里 | 改放 `st.session_state` |
| 页面一直转圈 | `get_llm()` 没缓存，每次重跑都重建客户端 | 加 `@st.cache_resource` |
| 答案显示成一坨 | 用了 `st.write()` | 改 `st.markdown()` |

---

## 6. 为什么用 Streamlit 而不是 Vue

| 方案 | 出可演示版本的时间 |
|---|---|
| Streamlit | **1 天** |
| Vue 3 + 前后端分离 | 约 1 周 |

**取舍逻辑**：时间预算固定，把一周花在前端上，就少一周做 RAG 优化。
而简历上"Top-1 从 54.5% 提到 72.7%"比"会写 Vue 页面"值钱得多。

**结论**：先用 Streamlit 出闭环，拿到 offer 后再补前端。

**面试话术**：

> "我用 Streamlit 做界面，因为它是时间预算下的最优解——
> 一天能出可演示版本，而 Vue 要一周。我把时间花在了 RAG 优化上，
> 那部分有量化结果。前端是我的已知短板，后续会补。"

---

## 7. 已知局限

- 无鉴权，任何人打开页面都能用（本地/内网演示可以，公网部署前必须加）
- 单会话：所有访客共用 `web-session`，多人同时用会串历史
- 无错误重试：一次失败要用户手动重发

---

## 8. 下一步（Day 32）

流式显示：把"等整块生成完"改成打字机效果。
