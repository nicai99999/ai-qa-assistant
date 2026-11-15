"""D3 实验：意图路由准确率。

用法（在项目根目录）：
    uv run python scripts/experiments/try_route.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logging_setup import setup_logging
from router import classify_by_rule

DATA = ROOT / "data" / "route_set.json"


def main() -> None:
    setup_logging()
    items = json.loads(DATA.read_text(encoding="utf-8"))

    by_intent: dict[str, int] = {}
    for item in items:
        by_intent[item["intent"]] = by_intent.get(item["intent"], 0) + 1

    print(f"评测集：{len(items)} 条", " / ".join(f"{k} {v}" for k, v in by_intent.items()))

    correct = 0
    wrong: list[tuple[str, str, str]] = []

    for item in items:
        route = classify_by_rule(item["question"])
        actual = route.intent if route else "knowledge"
        if actual == item["intent"]:
            correct += 1
        else:
            wrong.append((item["question"], item["intent"], actual))

    total = len(items)
    print(f"\n规则路由：正确 {correct}/{total} = {correct / total:.1%}")

    if wrong:
        print("\n错例：")
        for q, expect, actual in wrong:
            print(f"  {q:<34} 期望 {expect:<9} 实际 {actual}")

    hard = [i for i in items if i.get("hard")]
    if hard:
        print(f"\n其中 {len(hard)} 条是难例（没有 '我叫'、不是标准问候语等）")
        print("→ 这些就是规则的真实边界，也是 LLM 兜底存在的理由")


if __name__ == "__main__":
    main()