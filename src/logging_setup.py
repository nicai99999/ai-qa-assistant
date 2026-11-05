import logging


def setup_logging() -> None:
    """全局日志配置，程序启动时调一次。"""
    logging.basicConfig(
        # basicConfig：对日志系统进行一次性的基础配置。
        level=logging.INFO,
        # level=logging.INFO：设置最低输出级别为 INFO。
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        # format：规定每条日志长什么样。
        # %(asctime)s：时间（由后面的 datefmt 决定格式）。
        # [%(levelname)s]：日志级别（外面加了方括号，如 [INFO]）。
        # %(name)s：输出日志的模块名（也就是 getLogger(__name__) 传进去的名字）。
        # %(message)s：你实际写的日志内容。
        # 最终输出效果：14:30:00 [INFO] __main__: 程序启动
        datefmt="%H:%M:%S",
    )


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