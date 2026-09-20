"""清理知识库里的文档数据（ES 分块/摘要 + 任务记录 + 文档记录）。

典型场景：同一个 PDF 被重复上传了好几次，检索池和评测的 gold 集合都被放大，
指标失真。先清干净，再重新上传一份。

**默认不删物理文件**：upload_files 里那份可能是你本地唯一的原始 PDF，
删掉就没法重传了。确认别处有备份后再加 --delete-files。

用法：
    # 只看现状，不删任何东西（默认就是 dry-run）
    python scripts/purge_documents.py --knowledge-id 1

    # 真删：把知识库 1 下所有文档的数据清空（保留 knowledge_database 记录）
    python scripts/purge_documents.py --knowledge-id 1 --apply

    # 只删指定的几个文档
    python scripts/purge_documents.py --document-id 2 --document-id 3 --apply
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml  # noqa: E402


def _check_es_reachable(timeout=3.0) -> bool:
    """在 import es_api 之前先探一下 ES。

    es_api 模块在 import 时就会执行 init_es()，ES 没起来的话它会重试 30 次
    （约 60 秒）才放弃，白等。这里先快速失败，给出人话提示。
    """
    import urllib.error
    import urllib.request

    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["elasticsearch"]
    host = os.environ.get("ES_HOST", cfg["host"])
    port = int(os.environ.get("ES_PORT", cfg["port"]))
    scheme = cfg["scheme"]
    try:
        urllib.request.urlopen(f"{scheme}://{host}:{port}", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # 有响应就算活着（401 之类也算）
    except Exception as e:
        print(f"连不上 Elasticsearch ({scheme}://{host}:{port}): {e}")
        return False


def _resolve_targets(document_ids, knowledge_id):
    from db_api import KnowledgeDocument, Session

    with Session() as session:
        q = session.query(KnowledgeDocument)
        if document_ids:
            q = q.filter(KnowledgeDocument.document_id.in_(document_ids))
        elif knowledge_id is not None:
            q = q.filter(KnowledgeDocument.knowledge_id == knowledge_id)
        else:
            sys.exit("必须指定 --knowledge-id 或 --document-id")
        docs = q.order_by(KnowledgeDocument.document_id).all()
        return [
            {
                "document_id": d.document_id,
                "knowledge_id": d.knowledge_id,
                "title": d.title,
                "file_type": d.file_type,
                "file_path": d.file_path,
            }
            for d in docs
        ]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--knowledge-id", type=int, default=None)
    parser.add_argument("--document-id", type=int, action="append", default=None)
    parser.add_argument("--apply", action="store_true", help="真的执行删除（默认只预览）")
    parser.add_argument("--delete-files", action="store_true",
                        help="同时删除 upload_files 里的物理文件（危险，确认有备份再用）")
    args = parser.parse_args()

    ok_es = _check_es_reachable()
    docs = _resolve_targets(args.document_id, args.knowledge_id)

    if not docs:
        sys.exit("没找到匹配的文档记录，检查 --knowledge-id / --document-id")

    print(f"{'=' * 68}")
    print(f"{'待清理文档' if args.apply else '当前匹配到的文档'}（{len(docs)} 条）")
    print(f"{'=' * 68}")
    for d in docs:
        exists = os.path.exists(d["file_path"] or "")
        print(f"  document_id={d['document_id']}  knowledge_id={d['knowledge_id']}  "
              f"文件{'存在' if exists else '缺失'}  {d['title']}")
        print(f"    {d['file_path']}")

    if not args.apply:
        print("\n这是预览模式，什么都没删。确认无误后加 --apply 执行。")
        if not ok_es:
            print("注意：ES 当前不可用，--apply 需要先启动 ES。")
        return

    if not ok_es:
        sys.exit("\nES 不可用，中止删除。先启动 Elasticsearch 再重跑（ES 起来前删 DB 记录会留下孤儿分块）。")

    # ES 删干净之后才动数据库，顺序和 main.py 的删除接口保持一致：
    # 先删 ES，失败就抛异常，绝不出现"元数据没了、向量还在"的中间态。
    from es_api import delete_document_chunks
    from task_store import delete_by_document as delete_tasks_by_document
    from db_api import KnowledgeDocument, Session

    total_chunks = 0
    for d in docs:
        did = d["document_id"]
        deleted = delete_document_chunks(did)
        total_chunks += deleted
        with Session() as session:
            n_tasks = delete_tasks_by_document(did)
            doc = (session.query(KnowledgeDocument)
                   .filter(KnowledgeDocument.document_id == did).first())
            if doc is not None:
                session.delete(doc)
                session.commit()
        print(f"  已清理 document_id={did}: ES 分块/摘要={deleted} 任务={n_tasks}")

        if args.delete_files:
            fp = d["file_path"]
            if fp and os.path.exists(fp):
                os.remove(fp)
                print(f"    已删除物理文件: {fp}")

    print(f"\n完成：清理 {len(docs)} 个文档，ES 删除 {total_chunks} 条记录。")
    if not args.delete_files:
        print("物理文件保留在 upload_files/（没加 --delete-files）。"
              "重新上传成功后，这些旧文件可以手动清理。")


if __name__ == "__main__":
    main()
