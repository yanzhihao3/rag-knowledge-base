import logging
import sys
import contextvars
from logging.handlers import RotatingFileHandler

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """把当前线程的 request_id 注入到每条日志记录中。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    """幂等的日志初始化：INFO 级别 + 控制台 + 轮转文件，统一格式带 request_id。"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return  # 幂等：我们的轮转 handler 已在，避免重复添加
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(request_id)s | %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        "rag.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    filter_ = RequestIdFilter()
    console.addFilter(filter_)
    file_handler.addFilter(filter_)

    logger.addHandler(console)
    logger.addHandler(file_handler)
