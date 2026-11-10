"""Celery 应用：Redis 作为 broker 与 result backend。

启动 worker（Windows 必须 -P solo，Celery 的 prefork 池在 Windows 不可用）：
    celery -A celery_app worker -l info -P solo
"""
import os

from celery import Celery

from env_loader import load_dotenv

load_dotenv()

REDIS_URL = os.environ.get("RAG_REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "rag",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,
    task_default_queue="rag",
    # 可靠性：任务执行完才 ack；worker 中途挂掉，任务会重新投递
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # 一次只预取一个，避免长任务被单个 worker 全占住
    worker_prefetch_multiplier=1,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    # worker 崩溃后，未确认的任务会被重新投递（默认 1 小时太长，5 分钟又太短）。
    #
    # ★ 这个值必须【大于最长任务耗时】。加了图片 OCR 之后，解析一本 240 页的书要
    #   约 9~10 分钟（实测 564 秒）—— 如果还设成 300 秒，任务会在执行中途被判定
    #   "超时未确认"，于是被重新投递、再解析一遍（幂等所以不会产生脏数据，
    #   但白白多烧一倍的 CPU 和时间）。这里留 30 分钟余量，给更大的文档兜底。
    broker_transport_options={"visibility_timeout": 1800},
    result_backend_transport_options={"visibility_timeout": 1800},
    # 连接断开时取消正在跑的长任务，交给重新投递处理，避免"半死不活"占着 worker
    worker_cancel_long_running_tasks_on_connection_loss=True,
    # 结果保留 1 天，便于排查历史任务
    result_expires=86400,
)
