"""python -m ai_qa_assistant 的入口。"""
from logging_setup import setup_logging
from llm import get_llm
from memory_v2 import ConversationStoreV2
from safe_qa import safe_ask


def main() -> None:
    setup_logging()
    store = ConversationStoreV2(mode="priority", llm=get_llm())
    session = "cli"
    print("AI 问答助手已启动，输入 exit 退出。")

    while True:
        question = input("你 > ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        print(f"助手 > {safe_ask(question, session, store)}")


if __name__ == "__main__":
    main()