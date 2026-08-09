import time
from typing import Dict
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


class TaskStateMachine:
    """
    文档解析任务状态机
    状态流转：pending → processing → completed/failed
    """

    STATE_PENDING = "pending"
    STATE_PROCESSING = "processing"
    STATE_COMPLETED = "completed"
    STATE_FAILED = "failed"
    STATE_RETRYING = "retrying"

    def __init__(self):
        self.states: Dict[str, str] = {}

    def set_state(self, document_id: str, state: str) -> None:
        """设置任务状态"""
        self.states[document_id] = state
        logger.info("[状态机] doc=%s: %s", document_id, state)

    def get_state(self, document_id: str) -> str:
        """获取任务状态，默认是pending"""
        return self.states.get(document_id, self.STATE_PENDING)

    def is_completed(self, document_id: str) -> bool:
        """检查是否已完成"""
        return self.get_state(document_id) == self.STATE_COMPLETED

    def is_failed(self, document_id: str) -> bool:
        """检查是否失败"""
        return self.get_state(document_id) == self.STATE_FAILED

    def reset(self, document_id: str) -> None:
        """重置任务状态"""
        if document_id in self.states:
            del self.states[document_id]


# 全局状态机实例
task_state_machine = TaskStateMachine()
