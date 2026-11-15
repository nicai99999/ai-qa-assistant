"""日志配置。

关键点：显式指定 UTF-8。
容器里 locale 常常是 POSIX（C），sys.stderr 默认编码是 ASCII，
写中文日志会抛 UnicodeEncodeError，导致整个请求失败。
"""

import io
import logging
import sys


def _utf8_stream(stream):
    """把流强制成 UTF-8。

    优先 reconfigure；不行就套一层 TextIOWrapper，
    并用 errors="replace" 保证永远不会因为编码问题抛异常。
    """
    try:
        # 尝试直接重新配置流的编码（Python 3.7+ 支持）
        stream.reconfigure(encoding="utf-8", errors="replace")
        return stream
    except Exception:
        pass

    # 如果 reconfigure 失败，退而求其次，获取底层的二进制 buffer 进行包装
    buf = getattr(stream, "buffer", None)
    if buf is not None:
        try:
            # 套一层 TextIOWrapper，强制用 utf-8 编码。
            # errors="replace" 保证遇到无法编码的字符时用 ? 替代，绝不抛出异常导致程序崩溃
            return io.TextIOWrapper(
                buf, encoding="utf-8", errors="replace", line_buffering=True
            )
        except Exception:
            pass
    return stream


def setup_logging() -> None:
    """全局日志配置，程序启动时调一次。"""

    # 创建处理器，强制使用包装后的 UTF-8 流 (sys.stderr)
    handler = logging.StreamHandler(_utf8_stream(sys.stderr))

    # 给 handler 设置详细的日志格式
    handler.setFormatter(
        logging.Formatter(
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            # format：规定每条日志长什么样。
            # %(asctime)s：时间（由后面的 datefmt 决定格式）。
            # [%(levelname)s]：日志级别（外面加了方括号，如 [INFO]）。
            # %(name)s：输出日志的模块名（也就是 getLogger(__name__) 传进去的名字）。
            # %(message)s：你实际写的日志内容。
            # 最终输出效果：14:30:00 [INFO] __main__: 程序启动
            datefmt="%H:%M:%S",
        )
    )

    # force=True：重复调用时也生效。
    # 因为 Streamlit 的运行机制是每次点击都会从头重跑脚本，如果不加 force=True，
    # 第二次调用 basicConfig 会静默失败，导致新配置不生效。
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def log_llm_call(logger: logging.Logger, elapsed_ms: float, usage: dict | None) -> None:
    """记录一次模型调用的耗时和 token 用量。"""
    # logger：传入一个 logger 实例（比如 main.py 里的那个 logger）。
    # elapsed_ms：耗时，浮点数（单位毫秒）。
    # usage：token 用量，类型是字典或 None（因为有些模型可能不返回用量）。

    logger.info("耗时 %.0f ms", elapsed_ms)
    # 延迟格式化：如果日志级别高于 INFO（比如把级别调成 ERROR），这条日志根本不会被输出。
    # 用 f-string 的话，无论输不输出，Python 都要在内存里计算并拼接一次字符串，浪费性能；而用 % 占位符传参，如果不输出，连拼接的动作都不做。

    if not usage:
        return
    # 防御性编程。
    # 如果 usage 是 None（比如调用的模型不支持返回 token 统计），直接提前 return 结束函数，防止后面的 .get() 报错。

    logger.info(
        # 打印 token 用量。
        "token 输入=%s 输出=%s 总计=%s",
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        usage.get("total_tokens"),
    )