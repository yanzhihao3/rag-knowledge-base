"""同步重灌某个文档：重新解析 PDF → 分块 → 向量化 → 写 ES。

和 tasks.parse_document 调的是同一个 rag.extract_content，区别只是省掉 Celery 队列和
task 表状态回写。用于"改了分块 / 解析逻辑之后快速重建索引"，不必启动 worker。

注意：extract_content 内部第一步就是 delete_document_chunks(document_id) 清旧块，
配合确定性 _id 覆盖写，所以重复执行是幂等的，不会产生重复分块。

用法：
    python scripts/reindex_document.py --document-id 3
    python scripts/reindex_document.py --document-id 3 --dry-run   # 只打印将要做什么
"""
import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _es_stats(es, document_id):
    """返回 (总块数, 页级块数, 子块数)。"""
    resp = es.search(
        index="chunk_info",
        size=0,
        query={"term": {"document_id": document_id}},
        aggs={"kinds": {"terms": {"field": "chunk_id", "size": 1000}}},
    )
    total = resp["hits"]["total"]["value"]
    by_chunk_id = {}
    for b in resp["aggregations"]["kinds"]["buckets"]:
        by_chunk_id[b["key"]] = b["doc_count"]
    pages = by_chunk_id.get(0, 0)
    return total, pages, total - pages


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from db_api import KnowledgeDocument, Session

    with Session() as session:
        doc = (session.query(KnowledgeDocument)
               .filter(KnowledgeDocument.document_id == args.document_id)
               .first())
        if doc is None:
            sys.exit(f"文档不存在: document_id={args.document_id}")
        meta = {
            "knowledge_id": doc.knowledge_id,
            "document_id": doc.document_id,
            "title": doc.title,
            "file_type": doc.file_type,
            "file_path": doc.file_path,
            "owner_id": doc.owner_id,
            "department_id": doc.department_id,
        }

    if not os.path.exists(meta["file_path"] or ""):
        sys.exit(f"物理文件不存在: {meta['file_path']}")

    print(f"文档: document_id={meta['document_id']} knowledge_id={meta['knowledge_id']} "
          f"title={meta['title']}")
    print(f"文件: {meta['file_path']}")

    from rag_api import RAG
    from es_api import es

    before = _es_stats(es, meta["document_id"])
    rag = RAG()
    print(f"分块策略: {rag.chunk_strategy}  chunk_size={rag.chunk_size}  "
          f"chunk_overlap_sentences={getattr(rag, 'chunk_overlap_sentences', None)}  "
          f"chunk_overlap={rag.chunk_overlap}")
    print(f"当前索引: 总 {before[0]}（页级 {before[1]} + 子块 {before[2]}）")

    if args.dry_run:
        print("dry-run：什么都没做。")
        return

    t0 = time.monotonic()
    rag.extract_content(
        meta["knowledge_id"], meta["document_id"], meta["title"],
        meta["file_type"], meta["file_path"],
        owner_id=meta["owner_id"], department_id=meta["department_id"],
    )
    elapsed = time.monotonic() - t0

    after = _es_stats(es, meta["document_id"])
    print(f"重灌完成，用时 {elapsed:.0f}s")
    print(f"新索引: 总 {after[0]}（页级 {after[1]} + 子块 {after[2]}）")
    print(f"对比: 子块 {before[2]} → {after[2]}，总块数 {before[0]} → {after[0]}")


if __name__ == "__main__":
    main()
