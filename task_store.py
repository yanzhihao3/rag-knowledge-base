"""任务状态存储：数据库里的 task 表是异步任务状态的唯一权威来源。

状态机：pending → started → success / failure（重试时回到 started 且 retries+1）
为什么落库而不是只放 Redis：
- 可查询、可审计，API 能直接按 task_id / document_id 查进度；
- Redis 重启或清空后，历史状态仍在；
- 带 department_id，可复用多租户过滤规则。
"""
import uuid
from datetime import datetime
from typing import Optional

from db_api import Session, Task

STATUS_PENDING = "pending"
STATUS_STARTED = "started"
STATUS_SUCCESS = "success"
STATUS_FAILURE = "failure"

ACTIVE_STATUSES = (STATUS_PENDING, STATUS_STARTED)


def new_task_id(document_id: int) -> str:
    """确定性前缀 + 随机后缀：便于人工识别，同时允许失败后重新入队。"""
    return f"doc-{document_id}-{uuid.uuid4().hex[:12]}"


def _to_dict(task: Task) -> dict:
    return {
        "task_id": task.task_id,
        "document_id": task.document_id,
        "knowledge_id": task.knowledge_id,
        "owner_id": task.owner_id,
        "department_id": task.department_id,
        "status": task.status,
        "retries": task.retries,
        "celery_task_id": task.celery_task_id,
        "error_msg": task.error_msg,
        "create_dt": str(task.create_dt) if task.create_dt else None,
        "started_dt": str(task.started_dt) if task.started_dt else None,
        "finished_dt": str(task.finished_dt) if task.finished_dt else None,
    }


def create_task(*, task_id: str, document_id: int, knowledge_id: Optional[int],
                owner_id: int = 0, department_id: int = 0) -> str:
    """创建待执行任务记录（投递前调用），返回 task_id。"""
    with Session() as session:
        session.add(Task(
            task_id=task_id,
            document_id=document_id,
            knowledge_id=knowledge_id,
            owner_id=owner_id,
            department_id=department_id,
            status=STATUS_PENDING,
        ))
        session.commit()
    return task_id


def get_task(task_id: str) -> Optional[dict]:
    with Session() as session:
        task = session.query(Task).filter(Task.task_id == task_id).first()
        return _to_dict(task) if task is not None else None


def get_latest_task(document_id: int) -> Optional[dict]:
    with Session() as session:
        task = (session.query(Task)
                .filter(Task.document_id == document_id)
                .order_by(Task.create_dt.desc())
                .first())
        return _to_dict(task) if task is not None else None


def find_active_task(document_id: int) -> Optional[dict]:
    """查该文档是否已有进行中的任务（防止重复投递）。"""
    with Session() as session:
        task = (session.query(Task)
                .filter(Task.document_id == document_id, Task.status.in_(ACTIVE_STATUSES))
                .order_by(Task.create_dt.desc())
                .first())
        return _to_dict(task) if task is not None else None


def mark_started(task_id: str, celery_task_id: Optional[str] = None) -> None:
    with Session() as session:
        task = session.query(Task).filter(Task.task_id == task_id).first()
        if task is None:
            return
        task.status = STATUS_STARTED
        task.celery_task_id = celery_task_id
        if task.started_dt is None:
            task.started_dt = datetime.now()
        session.commit()


def mark_retrying(task_id: str, retries: int, error_msg: str = "") -> None:
    with Session() as session:
        task = session.query(Task).filter(Task.task_id == task_id).first()
        if task is None:
            return
        task.status = STATUS_STARTED
        task.retries = retries
        task.error_msg = (error_msg or "")[:1000]
        session.commit()


def mark_success(task_id: str) -> None:
    with Session() as session:
        task = session.query(Task).filter(Task.task_id == task_id).first()
        if task is None:
            return
        task.status = STATUS_SUCCESS
        task.error_msg = None
        task.finished_dt = datetime.now()
        session.commit()


def mark_failure(task_id: str, error_msg: str) -> None:
    with Session() as session:
        task = session.query(Task).filter(Task.task_id == task_id).first()
        if task is None:
            return
        task.status = STATUS_FAILURE
        task.error_msg = (error_msg or "")[:1000]
        task.finished_dt = datetime.now()
        session.commit()


def delete_by_document(document_id: int) -> int:
    """文档被删除时，一并清掉它的任务记录。"""
    with Session() as session:
        deleted = session.query(Task).filter(Task.document_id == document_id).delete()
        session.commit()
        return deleted
