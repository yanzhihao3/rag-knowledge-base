import logging
import sys
import contextvars # 上下文变量，用于在异步/多线程中传递数据
from logging.handlers import RotatingFileHandler

# ContextVar 是"当前执行上下文专属"的变量——每个请求在它自己的上下文里读写，互不干扰。
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
# 创建一个线程/协程安全的变量，用于存储当前请求的ID
# default="-" 是兜底值：还没被请求设置时显示 "-"（ES 子线程日志里的 "-" 就是它）

class RequestIdFilter(logging.Filter):
    """把当前线程的 request_id 注入到每条日志记录中。"""

    def filter(self, record: logging.LogRecord) -> bool:
        # logging 标准库机制：每条日志是一条 LogRecord，
        # Filter 在记录写出前执行，把当前请求的 id 挂到 record.request_id，
        # 格式化时 %(request_id)s 才能取到值。
        # return True 表示放行（返回 False 会丢弃这条日志）。
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    """幂等的日志初始化：INFO 级别 + 控制台 + 轮转文件，统一格式带 request_id。"""
    # 顺序不能反：先设 level，再查 handler。
    # 因为 pytest 环境里 root logger 可能已有其他 handler，
    # 若先查 handler 直接 return，级别会停在 WARNING 没设成 INFO。
    # （真实踩过：第一次跑测试 assert 30 == 20 失败，就是这个原因。）
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return  # 幂等守卫：setup_logging() 可能被调用多次，检测到已有轮转 handler 就 return，
        # 否则 handler 重复添加，日志会打 N 遍。
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(request_id)s | %(message)s"
    )

    # 控制台处理器：开发时直接看
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    # 文件轮转处理器：写 rag.log，5MB 轮转、保留 3 份，防止日志无限膨胀占满磁盘；
    # encoding="utf-8" 保证中文不乱码（Windows 控制台乱码但 rag.log 正常就是它的功劳）
    file_handler = RotatingFileHandler(
        "rag.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    # 两个 handler 都挂 Filter，所以控制台和文件都带 request_id
    filter_ = RequestIdFilter()
    console.addFilter(filter_)
    file_handler.addFilter(filter_)

    logger.addHandler(console)
    logger.addHandler(file_handler)
