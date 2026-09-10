"""任务状态存储测试（轻量：临时 SQLite，不需要 Redis / ES / 模型，CI 可跑）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from db_api import Base, Session, Task, engine
from task_store import (
    STATUS_FAILURE,
    STATUS_PENDING,
    STATUS_STARTED,
    STATUS_SUCCESS,
    create_task,
    delete_by_document,
    find_active_task,
    get_latest_task,
    get_task,
    mark_failure,
    mark_retrying,
    mark_started,
    mark_success,
    new_task_id,
)


@pytest.fixture(autouse=True)
def clean_tasks():
    Base.metadata.create_all(engine)
    with Session() as session:
        session.query(Task).delete()
        session.commit()
    yield
    with Session() as session:
        session.query(Task).delete()
        session.commit()


def _new_task(document_id: int = 1) -> str:
    return create_task(task_id=new_task_id(document_id), document_id=document_id,
                       knowledge_id=10, owner_id=1, department_id=1)


def test_task_id_is_unique():
    assert new_task_id(1) != new_task_id(1)


def test_status_transitions():
    task_id = _new_task()
    assert get_task(task_id)["status"] == STATUS_PENDING

    mark_started(task_id, celery_task_id="celery-abc")
    started = get_task(task_id)
    assert started["status"] == STATUS_STARTED
    assert started["celery_task_id"] == "celery-abc"
    assert started["started_dt"] is not None

    mark_retrying(task_id, retries=1, error_msg="boom")
    retried = get_task(task_id)
    assert retried["status"] == STATUS_STARTED
    assert retried["retries"] == 1
    assert retried["error_msg"] == "boom"

    mark_success(task_id)
    done = get_task(task_id)
    assert done["status"] == STATUS_SUCCESS
    assert done["error_msg"] is None
    assert done["finished_dt"] is not None


def test_failure_records_error_and_finish_time():
    task_id = _new_task()
    mark_started(task_id)
    mark_failure(task_id, "ES 连接失败")
    task = get_task(task_id)
    assert task["status"] == STATUS_FAILURE
    assert task["error_msg"] == "ES 连接失败"
    assert task["finished_dt"] is not None


def test_find_active_task_only_returns_unfinished():
    task_id = _new_task(document_id=7)
    assert find_active_task(7)["task_id"] == task_id

    mark_started(task_id)
    assert find_active_task(7) is not None

    mark_success(task_id)
    assert find_active_task(7) is None       # 已完成的任务不算"进行中"


def test_latest_task_returns_newest():
    first = _new_task(document_id=9)
    mark_success(first)
    second = _new_task(document_id=9)
    assert get_latest_task(9)["task_id"] == second
    assert get_task(first)["task_id"] == first


def test_delete_by_document():
    _new_task(document_id=11)
    _new_task(document_id=11)
    assert delete_by_document(11) == 2
    assert get_latest_task(11) is None
