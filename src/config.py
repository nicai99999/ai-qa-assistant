"""集中管理配置。所有环境相关的值都从这里读。"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
# frozen=True 防止运行时被误改——配置应该"读完就定死"。
class Settings:
# 换环境只改这里。
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-flash"
    temperature: float = 0.0
    timeout: int = 30
    max_tokens: int = 1000

    @classmethod
    # @classmethod 是“类方法”。它绑定在“类”本身上，而不是绑定在“实例对象”上。它不需要创建对象（new）就能直接调用。
    def from_env(cls) -> "Settings":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY，请检查 .env")
        return cls(api_key=api_key)


settings = Settings.from_env()
# 当 Python 执行这个文件（比如 import config）时，会从上到下瞬间跑完上面所有的步骤：
# 导入 config 时就立刻创建一个全局 settings。
# 好处：其他地方 from .config import settings，直接用。
# 意义：
# 配置集中：换模型只改 config.py
# 敏感信息不散落：api_key 只在 config.py 出现