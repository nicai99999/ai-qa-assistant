"""D2 实验：三种证据信号对比。

用法（在项目根目录）：
    uv run python scripts/experiments/try_evidence.py

三种信号：
    1. max_score        检索最高相似度        连续值，需扫阈值
    2. keyword_coverage 问题实词在召回块里的覆盖率  连续值，需扫阈值
    3. llm_judge        让模型只看资料判断能否回答  布尔，直接算准确率

输出四张表：
    A. 每题三信号原始数据
    B. 前两个信号的阈值扫描 + 最佳点
    C. 三个信号的效果对比
    D. 重叠区详情（判断第二信号能否切开）

注意：llm_judge 会调 16 次模型，约 20-40 秒，会花少量 token。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm import get_llm
from logging_setup import setup_logging
from src.retriever import retrieve
from hybrid_retriever import _tokenize

EVAL_PATH = ROOT / "data" / "eval_set_v2.json"
TOP_K = 5

# ---------- 信号 3：让模型判断资料是否足够 ----------

JUDGE_PROMPT = """你是资料审核员。判断下面提供的资料能否回答用户的问题。

用户问题：{question}

提供的资料：
{context}

判断标准：
- 如果资料里明确包含回答问题所需的事实或结论，回答"能"
- 如果资料只是话题相关、但没有答案本身，回答"不能"
- 只输出一个词：能 或 不能。不要解释。

判断："""


def llm_judge(question: str, hits: list[dict], llm) -> bool:
    """返回 True 表示"资料足够回答"。"""
    if not hits:
        return False

    context = "\n\n".join(
        f"[{i}] {h.get('text', '')}" for i, h in enumerate(hits, start=1)
    )
    prompt = JUDGE_PROMPT.format(question=question, context=context)

    try:
        response = llm.invoke(prompt)
        text = str(response.content).strip()
    except Exception as exc:                      # noqa: BLE001
        print(f"    ⚠️ llm_judge 调用失败：{type(exc).__name__}: {exc}")
        return False

    # "不能"里含"能"，所以先判否定
    if "不能" in text or "无法" in text or "不足" in text:
        return False
    return "能" in text


# ---------- 信号 2：关键词覆盖率 ----------

def keyword_coverage(question: str, hits: list[dict]) -> float:
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return 0.0
    joined = " ".join((h.get("text") or "") for h in hits).lower()
    return sum(1 for t in q_tokens if t in joined) / len(q_tokens)
    # 返回的内容即为，关键词覆盖率

# ---------- 汇总所有题 ----------

def collect(questions: list[dict], llm) -> list[dict]:
    rows: list[dict] = []
    for index, item in enumerate(questions, start=1):
        question = item["question"]
        hits = retrieve(question, k=TOP_K)

        judge = llm_judge(question, hits, llm)
        print(f"  [{index}/{len(questions)}] judge={'能' if judge else '不能'}  {question[:26]}")

        rows.append(
            {
                "question": question,
                "answerable": bool(item["answerable"]),
                "score": max((h.get("score") or 0.0) for h in hits) if hits else 0.0,
                "coverage": keyword_coverage(question, hits),
                "judge": judge,
            }
        )
    return rows
    # 返回每个问题从数据库中检索的5条内容中的5个核心数据
    # 问题 可回答性 检索中满足内容的5条最高得分 关键词覆盖率 大模型判断


# ---------- 表 A ----------

def table_a(rows: list[dict]) -> None:
    print("\n" + "=" * 84)
    print("表 A：每题原始数据")
    print("=" * 84)
    print(f"{'相似度':>7}  {'覆盖率':>6}  {'模型判断':>8}  {'实际可答':>8}  问题")
    for r in sorted(rows, key=lambda x: x["score"], reverse=True):
        print(
            f"{r['score']:>7.3f}  {r['coverage']:>6.2f}  "
            # 去字典r中寻找score(存入时添加了'所以获取时也添加) 对应的值
            f"{'能' if r['judge'] else '不能':>8}  "
            f"{'是' if r['answerable'] else '否':>8}  {r['question'][:30]}"
        )


# ---------- 表 B：阈值扫描 ----------

def sweep(name: str, key: str, rows: list[dict]) -> tuple[float, float, float, float]:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]

    print("\n" + "=" * 84)
    print(f"表 B：{name} 阈值扫描")
    print("=" * 84)
    print(f"{'阈值':>6}  {'不可答识别率':>12}  {'可答保留率':>10}  {'调和平均':>8}")
    # 生成一个格式的表头
    # f-string 允许直接把字符串放在大括号里
    # f-string 也能对写死的字符串（常量）进行格式化控制，看到大括号自动解析

    best = (0.0, 0.0, 0.0, -1.0)
    # 初始 F1 设为 -1.0，保证后面无论算出来多差，都能覆盖它
    for i in range(0, 101, 5):
        # 一共 21 个数字，循环会执行 21 次。
        t = i / 100
        # 生成 0.00 到 1.00 之间，间隔为 0.05 的阈值（分数线）
        rejected = sum(1 for r in unans if r[key] < t)
        # key由外部传入
        # score_best = sweep("max_score", "score", rows)
        #                                  ↑ 这里传进去的是字符串 "score"
        # cov_best = sweep("keyword_coverage", "coverage", rows)
        kept = sum(1 for r in ans if r[key] >= t)

        reject_rate = rejected / len(unans) if unans else 0.0
        # 遇到不会的题，成功拒绝回答的比例
        keep_rate = kept / len(ans) if ans else 0.0
        f1 = (
        # 计算 F1（识别率和保留率的调和平均）
        # 对短板极其敏感。 只要有一个指标（比如识别率或保留率）接近 0，整体得分就会瞬间被拉到 0 附近。
            2 * reject_rate * keep_rate / (reject_rate + keep_rate)
            if (reject_rate + keep_rate) > 0
            else 0.0
        )
        mark = ""
        if f1 > best[3]:
            # 如果算出的新 F1 分数比之前旧F1分高(即best[3]) ， 才会保存目前的新得分
            best = (t, reject_rate, keep_rate, f1)
            mark = "  ←"
        print(f"{t:>6.2f}  {reject_rate:>11.0%}  {keep_rate:>9.0%}  {f1:>7.2f}{mark}")

    print(f"最佳：阈值 {best[0]:.2f}  识别率 {best[1]:.0%}  保留率 {best[2]:.0%}  调和 {best[3]:.2f}")
    return best


# ---------- 表 C：三信号对比 ----------

def table_c(rows: list[dict], score_best, cov_best) -> None:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]

    judge_reject = sum(1 for r in unans if not r["judge"]) / len(unans) if unans else 0.0
    judge_keep = sum(1 for r in ans if r["judge"]) / len(ans) if ans else 0.0
    judge_f1 = (
        2 * judge_reject * judge_keep / (judge_reject + judge_keep)
        if (judge_reject + judge_keep) > 0
        else 0.0
    )

    print("\n" + "=" * 84)
    print("表 C：三种信号效果对比")
    print("=" * 84)
    print(f"{'信号':<18} {'不可答识别率':>12} {'可答保留率':>10} {'调和平均':>8} {'成本':>8}")
    print(
        f"{'max_score':<18} {score_best[1]:>11.0%} {score_best[2]:>9.0%} "
        f"{score_best[3]:>7.2f} {'0':>8}"
    )
    print(
        f"{'keyword_coverage':<18} {cov_best[1]:>11.0%} {cov_best[2]:>9.0%} "
        f"{cov_best[3]:>7.2f} {'0':>8}"
    )
    print(
        f"{'llm_judge':<18} {judge_reject:>11.0%} {judge_keep:>9.0%} "
        f"{judge_f1:>7.2f} {'1 次调用':>8}"
    )


# ---------- 表 D：重叠区 ----------

def table_d(rows: list[dict], low: float, high: float) -> None:
    print("\n" + "=" * 84)
    print(f"表 D：重叠区（{low:.3f} ~ {high:.3f}）—— 第二信号能不能切开")
    print("=" * 84)

    overlap = [r for r in rows if low <= r["score"] <= high]
    if not overlap:
        print("没有重叠区，单用相似度即可。")
        return

    print(f"{'相似度':>7}  {'覆盖率':>6}  {'模型判断':>8}  {'可答':>4}  问题")
    for r in sorted(overlap, key=lambda x: x["score"], reverse=True):
        print(
            f"{r['score']:>7.3f}  {r['coverage']:>6.2f}  "
            f"{'能' if r['judge'] else '不能':>8}  "
            f"{'是' if r['answerable'] else '否':>4}  {r['question'][:28]}"
        )

    cov_ans = [r["coverage"] for r in overlap if r["answerable"]]
    cov_unans = [r["coverage"] for r in overlap if not r["answerable"]]
    if cov_ans and cov_unans:
        print(f"\n  重叠区可答覆盖率：  {[round(c, 2) for c in cov_ans]}")
        # round(c, 2)：把浮点数保留 2 位小数（让打印出来好看，比如把 0.853621 变成 0.85）
        print(f"  重叠区不可答覆盖率：{[round(c, 2) for c in cov_unans]}")
        if min(cov_ans) > max(cov_unans):
            # 重叠区可答覆盖率：  [0.85, 0.9, 0.88]
            # 重叠区不可答覆盖率：[0.2, 0.15, 0.35]
            # 0.85 > 0.35 成立 ， 可以分开
            print(f"  → coverage 能切开，中间区用 >= {max(cov_unans) + 0.01:.2f}")
        else:
            print("  → coverage 切不开")


# ---------- main ----------

def main() -> None:
    setup_logging()
    questions = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    llm = get_llm()

    print(f"评测集：{len(questions)} 题。开始采集三种信号（含 {len(questions)} 次模型判断）…\n")
    rows = collect(questions, llm)

    table_a(rows)
    score_best = sweep("max_score", "score", rows)
    cov_best = sweep("keyword_coverage", "coverage", rows)
    table_c(rows, score_best, cov_best)

    ans = [r["score"] for r in rows if r["answerable"]]
    unans = [r["score"] for r in rows if not r["answerable"]]
    if ans and unans:
        table_d(rows, min(ans), max(unans))


if __name__ == "__main__":
    main()
