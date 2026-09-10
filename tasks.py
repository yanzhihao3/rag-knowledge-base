"""Celery 任务：文档解析入库。

分层重试怎么理解：
- extract_content 内部已有 @with_retry（处理数据库/ES 瞬时抖动，不重新排队）；
- 这里再做任务级重试（处理 worker 崩溃、模型加载失败等）；
- 外层 max_retries 刻意设小，避免与内层相乘造成重试风暴。
"""
import logging

from celery_app import celery_app
from db_api import KnowledgeDocument, Session
from rag_api import RAG
from task_store import (
    create_task,
    find_active_task,
    mark_failure,
    mark_retrying,
    mark_started,
    mark_success,
    new_task_id,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 10


def enqueue_parse_document(document: KnowledgeDocument) -> str:
    """投递文档解析任务；同一文档已有进行中的任务时复用，不重复投递。"""
    active = find_active_task(document.document_id)
    if active is not None:
        logger.info("[task] 文档 %s 已有进行中的任务 %s，跳过重复投递",
                    document.document_id, active["task_id"])
        return active["task_id"]

    task_id = new_task_id(document.document_id)
    create_task(task_id=task_id, document_id=document.document_id,
                knowledge_id=document.knowledge_id,
                owner_id=document.owner_id, department_id=document.department_id)
    parse_document.apply_async(args=[document.document_id, task_id], task_id=task_id)
    logger.info("[task] 已投递解析任务 %s（document_id=%s）", task_id, document.document_id)
    return task_id


@celery_app.task(bind=True, name="tasks.parse_document", max_retries=MAX_RETRIES, acks_late=True)
def parse_document(self, document_id: int, task_id: str):
    """解析 PDF → 向量化 → 写入 ES；状态全程回写 task 表。"""
    with Session() as session:
        doc = (session.query(KnowledgeDocument)
               .filter(KnowledgeDocument.document_id == document_id)
               .first())
        if doc is None:
            mark_failure(task_id, f"文档记录不存在: document_id={document_id}")
            return
        knowledge_id = doc.knowledge_id
        title, file_type, file_path = doc.title, doc.file_type, doc.file_path
        owner_id, department_id = doc.owner_id, doc.department_id

    mark_started(task_id, celery_task_id=self.request.id)
    try:
        RAG().extract_content(
            knowledge_id=knowledge_id,
            document_id=document_id,
            title=title,
            file_type=file_type,
            file_path=file_path,
            owner_id=owner_id,
            department_id=department_id,
        )
        mark_success(task_id)
        logger.info("[task] 文档解析完成: document_id=%s task_id=%s", document_id, task_id)
    except Exception as exc:
        retries = self.request.retries
        if retries >= MAX_RETRIES:
            logger.exception("[task] 文档解析最终失败: document_id=%s", document_id)
            mark_failure(task_id, str(exc))
            raise
        logger.warning("[task] 文档解析第 %d 次失败，稍后重试: document_id=%s err=%s",
                       retries + 1, document_id, exc)
        mark_retrying(task_id, retries + 1, str(exc))
        raise self.retry(exc=exc, countdown=RETRY_BACKOFF_SECONDS * (3 ** retries))
