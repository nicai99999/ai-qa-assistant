"""Day 27 异常测试：5 种场景，验证都不崩。

用法：
    uv run python try_error_scenarios.py
"""

import logging

from llm import get_llm
from logging_setup import setup_logging
from memory_v2 import ConversationStoreV2
from safe_qa import friendly_message, safe_ask, validate_question

logger = logging.getLogger(__name__)


def show_case(title: str, result: str) -> None:
    print("\n" + "-" * 60)
    print(f"【{title}】")
    print(f"返回：{result}")


def main() -> None:
    setup_logging()

    store = ConversationStoreV2(mode="window", llm=get_llm())
    session = "err-test"

    # ---------- 场景 1、2：输入校验（真实调用） ----------
    show_case("场景1 空问题", safe_ask("", session, store))
    show_case("场景2 超长输入", safe_ask("啊" * 3000, session, store))

    # ---------- 场景 3-7：异常翻译（构造假异常，测映射） ----------
    cases = [
        ("场景3 API Key 错误", Exception("Error code: 401 - invalid api key provided")),
        ("场景4 账户额度不足", Exception("insufficient balance, please top up")),
        ("场景5 请求限流", Exception("429 rate limit exceeded")),
        ("场景6 请求超时", TimeoutError("request timed out after 30s")),
        ("场景7 网络不通", ConnectionError("network is unreachable")),
        ("场景8 未知异常", Exception("something weird happened")),
    ]
    for title, exc in cases:
        show_case(title, friendly_message(exc))

    # ---------- 场景 9：真实的错误 key（可选，需要手动改 .env） ----------
    print("\n" + "=" * 60)
    print("场景9 真实错误 Key：需要手动把 .env 的 key 改错一位，重跑本脚本")
    print("=" * 60)

    # ---------- 一致性检查 ----------
    print("\n" + "=" * 60)
    print("校验函数自测")
    print("=" * 60)
    try:
        validate_question("")
    except ValueError as exc:
        print(f"空问题被拦下：{exc}")
    try:
        validate_question("x" * 3000)
    except ValueError as exc:
        print(f"超长输入被拦下：{exc}")
    print(f"正常输入通过：{validate_question('你好')!r}")


if __name__ == "__main__":
    main()