import time
import sqlite3
from typing import Dict
from functools import wraps
import os
import logging

logger = logging.getLogger(__name__)


class SqliteStateStore:
    """把任务状态持久化到 SQLite，服务重启后状态不丢。"""

    def __init__(self, db_path: str = "rag.db"):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS task_state ("
                " document_id TEXT PRIMARY KEY, state TEXT NOT NULL)"
            )
            conn.commit()

    def load(self) -> Dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT document_id, state FROM task_state").fetchall()
        return {str(document_id): state for document_id, state in rows}

    def save(self, document_id: str, state: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO task_state (document_id, state) VALUES (?, ?)",
                (str(document_id), state),
            )
            conn.commit()

    def delete(self, document_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM task_state WHERE document_id = ?", (str(document_id),))
            conn.commit()


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

    def __init__(self, store=None):
        self.states: Dict[str, str] = {}
        self.store = store
        if store is not None:
            self.states.update(store.load())  # 重启后从持久层恢复已存在的状态

    def set_state(self, document_id: str, state: str) -> None:
        """设置任务状态（内存 + 持久层）"""
        self.states[document_id] = state
        if self.store is not None:
            self.store.save(document_id, state)
        logger.info("[状态机] doc=%s: %s", document_id, state)

    def get_state(self, document_id: str) -> str:
        """获取任务状态，默认是pending；内存没有时从持久层恢复"""
        if document_id not in self.states and self.store is not None:
            self.states.update(self.store.load())
        return self.states.get(document_id, self.STATE_PENDING)

    def is_completed(self, document_id: str) -> bool:
        """检查是否已完成"""
        return self.get_state(document_id) == self.STATE_COMPLETED

    def is_failed(self, document_id: str) -> bool:
        """检查是否失败"""
        return self.get_state(document_id) == self.STATE_FAILED

    def reset(self, document_id: str) -> None:
        """重置任务状态（内存 + 持久层）"""
        if document_id in self.states:
            del self.states[document_id]
        if self.store is not None:
            self.store.delete(document_id)


# 全局状态机实例（挂 SQLite 持久层，重启不丢）
task_state_machine = TaskStateMachine(store=SqliteStateStore())
