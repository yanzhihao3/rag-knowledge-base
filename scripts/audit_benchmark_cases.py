"""评测集体检：检查 50 条用例的"答案块"标注到底严不严谨。

背景：benchmark_report.py 的严格口径（答案块命中）是这样造 gold 的——
**全库里所有包含任一 answer_term 的块**。这个口径有三个风险，本脚本把它们量化：

  1. gold 为空      → 该题永远不可能命中。这不是"检索失败"，是题本身坏了，
                      但它会被记成 6% 的失败率里的一员（假失败）。
  2. gold 太大      → 命中门槛过低。一道题的答案块有 80 个，那"命中"几乎没有信息量。
  3. gold 全是页级  → 页级块（chunk_id=0）文本长、覆盖整页，几乎必然出现在 Top-5，
                      等于白送分；而页级块又会被 chk 子块的内容命中，因为它的文本
                      包含子块全文。

用法（需要 ES 已启动，MySQL/ES 里有评测用的那个知识库）：
    python scripts/audit_benchmark_cases.py                 # 默认知识库 2
    $env:RAG_BENCH_KB_ID='3'; python scripts/audit_benchmark_cases.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml  # noqa: E402

from test.benchmark_cases_rag_book import BENCHMARK_CASES  # noqa: E402


def _load_chunks(kb_id):
    """取该知识库全部块：{_id: (content, chunk_id)}。"""
    import json
    import urllib.request as u

    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["elasticsearch"]
    host = os.environ.get("ES_HOST", cfg["host"])
    port = int(os.environ.get("ES_PORT", cfg["port"]))
    scheme = cfg["scheme"]
    url = f"{scheme}://{host}:{port}/chunk_info/_search"
    body = {
        "size": 10000,
        "_source": False,
        "query": {"term": {"knowledge_id": kb_id}},
        "fields": ["chunk_content", "chunk_id"],
    }
    req = u.Request(url, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
    resp = json.load(u.urlopen(req, timeout=30))
    out = {}
    for h in resp["hits"]["hits"]:
        f = h.get("fields", {})
        content = (f.get("chunk_content") or [""])[0]
        chunk_id = (f.get("chunk_id") or [0])[0]
        out[h["_id"]] = (content, chunk_id)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    kb_id = int(os.environ.get("RAG_BENCH_KB_ID", "2"))
    chunks = _load_chunks(kb_id)
    print(f"知识库 {kb_id}：{len(chunks)} 个块"
          f"（页级 {sum(1 for _, (_, c) in chunks.items() if c == 0)}，"
          f"子块 {sum(1 for _, (_, c) in chunks.items() if c > 0)}）")
    print("=" * 78)

    rows = []
    for item in BENCHMARK_CASES:
        terms = [t.lower() for t in item.get("answer_terms", [])]
        if terms:
            gold = {cid for cid, (content, _) in chunks.items()
                    if any(t in (content or "").lower() for t in terms)}
        else:
            gold = set()
        gold_pages = sum(1 for cid in gold if chunks[cid][1] == 0)
        missing = [t for t in item.get("answer_terms", [])
                   if not any(t.lower() in (c or "").lower() for c, _ in chunks.values())]
        rows.append({
            "query": item["query"],
            "difficulty": item["difficulty"],
            "n_terms": len(item.get("answer_terms", [])),
            "gold": len(gold),
            "gold_pages": gold_pages,
            "missing": missing,
        })

    n = len(rows)
    empty = [r for r in rows if r["gold"] == 0]
    huge = [r for r in rows if r["gold"] > 40]
    mostly_pages = [r for r in rows if r["gold"] and r["gold_pages"] / r["gold"] >= 0.5]
    small = [r for r in rows if 1 <= r["gold"] <= 5]

    print(f"用例总数: {n}")
    print(f"  gold 为空（永远不可能命中）: {len(empty)}")
    print(f"  gold <= 5 个块（标注精准）  : {len(small)}")
    print(f"  gold > 40 个块（门槛过低）  : {len(huge)}")
    print(f"  gold 里页级块占比 >= 50%    : {len(mostly_pages)}")
    print()

    if empty:
        print("── gold 为空的题（必须修标注，否则白扣分）" + "─" * 30)
        for r in empty:
            print(f"  [{r['difficulty']}] answer_terms={r['n_terms']}个  {r['query']}")
            if r["missing"]:
                print(f"      这些词在整个知识库里都找不到: {r['missing']}")
        print()

    if huge:
        print("── gold 过大的题（命中门槛过低，指标虚高）" + "─" * 26)
        for r in sorted(huge, key=lambda x: -x["gold"]):
            print(f"  gold={r['gold']:>4}（页级 {r['gold_pages']:>3}）"
                  f"  [{r['difficulty']}] {r['query']}")
        print()

    print("── 逐题明细（按 gold 从大到小）" + "─" * 40)
    print(f"  {'gold':>5} {'页级':>5} {'难度':<7} 问题")
    for r in sorted(rows, key=lambda x: -x["gold"]):
        print(f"  {r['gold']:>5} {r['gold_pages']:>5} {r['difficulty']:<7} {r['query']}")


if __name__ == "__main__":
    main()
