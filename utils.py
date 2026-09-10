import time
from functools import wraps
import os
import logging

logger = logging.getLogger(__name__)


def safe_remove_file(file_path: str) -> None:
    """删除物理文件。文件是低危孤儿，失败只告警不阻断。"""
    if not file_path:
        return
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("已删除物理文件: %s", file_path)
    except Exception:
        logger.warning("物理文件删除失败，需手动清理: %s", file_path)


def with_retry(max_retries=3, base_delay=1):
    """
    指数退避重试装饰器
    :param max_retries: 最大重试次数
    :param base_delay: 基础延迟秒数，重试间隔 = base_delay * 2^retry_count
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for retry_count in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if retry_count < max_retries - 1:
                        delay = base_delay * (2 ** retry_count)
                        logger.warning("[重试] %s 第%d次失败，%.1fs后重试: %s", func.__name__, retry_count + 1, delay, e)
                        time.sleep(delay)
                    else:
                        logger.error("[重试耗尽] %s 失败: %s", func.__name__, e)
            raise last_exception
        return wrapper
    return decorator

