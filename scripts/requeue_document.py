"""手动重新投递某个文档的解析任务（处理卡住的 unacked 任务）。

使用场景：worker 在解析中途被强杀，任务消息卡在 Redis 的 unacked 里，
数据库里的 task 记录停在 started。这个脚本会：
1. 把该文档进行中的旧任务标记为 failure（解除重复投递拦截）；
2. 重新投递一个新的解析任务，返回新的 task_id。

用法：
    python scripts/requeue_document.py --document-id 3
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from db_api import KnowledgeDocument, Session  # noqa: E402
from task_store import find_active_task, mark_failure  # noqa: E402
from tasks import enqueue_parse_document  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="重新投递文档解析任务")
    parser.add_argument("--document-id", type=int, required=True)
    args = parser.parse_args()

    with Session() as session:
        doc = (session.query(KnowledgeDocument)
               .filter(KnowledgeDocument.document_id == args.document_id)
               .first())
        if doc is None:
            sys.exit(f"文档不存在: document_id={args.document_id}")

        active = find_active_task(args.document_id)
        if active is not None:
            mark_failure(active["task_id"], "手动重新投递：清理卡住的旧任务")
            print(f"已把旧任务标记为失败: {active['task_id']}（状态={active['status']}）")

        task_id = enqueue_parse_document(doc)

    print(f"已重新投递: document_id={args.document_id} task_id={task_id}")


if __name__ == "__main__":
    main()
