"""分析 50 道评测题的"答案跨度"：答案落在一个块、同一页的多个块，还是跨多页。

用途：判断哪些题是**父子块（small-to-big）能赢的题**。

  - 单块     答案只落在一个子块里      → 父子块没有任何区别
  - 同页多块 答案散在同一页的多个子块   → 父子块该赢：命中任一块就能带回整页全文
  - 跨页     答案横跨多页             → 父子块也救不全（一页只能带一页的内容）

只用 ES 里已有的块算，不调 LLM、不调模型。

用法：
    python scripts/analyze_answer_span.py
    $env:RAG_BENCH_KB_ID='3'; python scripts/analyze_answer_span.py
"""
import json
import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml  # noqa: E402

from test.benchmark_cases_rag_book import BENCHMARK_CASES  # noqa: E402


def _load_chunks(kb_id):
    """返回 [{_id, content, chunk_id, page_number}]。"""
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["elasticsearch"]
    host = os.environ.get("ES_HOST", cfg["host"])
    port = int(os.environ.get("ES_PORT", cfg["port"]))
    body = {
        "size": 10000,
        "_source": False,
        "query": {"term": {"knowledge_id": kb_id}},
        "fields": ["chunk_content", "chunk_id", "page_number"],
    }
    req = urllib.request.Request(
        f"{cfg['scheme']}://{host}:{port}/chunk_info/_search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=30))
    out = []
    for h in resp["hits"]["hits"]:
        f = h.get("fields", {})
        out.append({
            "_id": h["_id"],
            "content": (f.get("chunk_content") or [""])[0],
            "chunk_id": (f.get("chunk_id") or [0])[0],
            "page_number": (f.get("page_number") or [0])[0],
        })
    return out


def _page_key(chunk_id):
    return chunk_id.split("_chunk_")[0]


def classify(item, chunks):
    terms = [t.lower() for t in item.get("answer_terms", [])]
    if not terms:
        return None

    # 第一步：定位"答案页" = 命中 answer_terms **种类最多**的那一页。
    # 为什么要这么找：answer_terms 里有 "2020"「答案生成」这种词，全库散落很多次。
    # 直接统计"命中过哪些页"会把无关提及算进来 —— 第1题的 "2020" 落在 6 个页面上，
    # 但答案只在第 7 页（那页同时命中 "2020" 和 "Facebook"）。多词共现才是答案所在。
    page_score = {}
    for c in chunks:
        if c["chunk_id"] != 0:
            continue
        text = (c["content"] or "").lower()
        hit_terms = {t for t in terms if t in text}
        if hit_terms:
            page_score[c["page_number"]] = len(hit_terms)
    if not page_score:
        return None
    best_score = max(page_score.values())
    best_pages = sorted(p for p, s in page_score.items() if s == best_score)
    # 命中词数相同就全部算候选（答案可能确实跨页）
    on_best = [c for c in chunks if c["chunk_id"] > 0 and c["page_number"] in best_pages
               and any(t in (c["content"] or "").lower() for t in terms)]
    per_page = {}
    for c in on_best:
        per_page[c["page_number"]] = per_page.get(c["page_number"], 0) + 1

    if not on_best:
        kind = "页内无子块命中"
    elif len(per_page) > 1:
        kind = "跨页"
    elif max(per_page.values()) > 1:
        kind = "同页多块"
    else:
        kind = "单块"
    return {
        "terms": len(terms),
        "terms_on_best": best_score,
        "best_pages": best_pages,
        "n_children": len(on_best),
        "n_pages": len(per_page),
        "max_per_page": max(per_page.values()) if per_page else 0,
        "kind": kind,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    kb_id = int(os.environ.get("RAG_BENCH_KB_ID", "2"))
    chunks = _load_chunks(kb_id)
    n_pages_records = sum(1 for c in chunks if c["chunk_id"] == 0)
    print(f"知识库 {kb_id}: {len(chunks)} 个块（页级 {n_pages_records}，"
          f"子块 {len(chunks) - n_pages_records}）")

    rows = []
    for i, item in enumerate(BENCHMARK_CASES, 1):
        info = classify(item, chunks)
        if info:
            rows.append((i, item, info))

    counts = {}
    for _, _, info in rows:
        counts[info["kind"]] = counts.get(info["kind"], 0) + 1
    print("\n" + "=" * 76)
    print("答案跨度分布（决定父子块有没有用武之地）")
    print("=" * 76)
    for k in ("单块", "同页多块", "跨页"):
        print(f"  {k:<8}: {counts.get(k, 0):>2} 题")
    print("\n  说明：这里的'跨页'= 命中词数并列最高的页不止一页，才可能是真跨页；")
    print("        单块/同页多块 是按'答案页内部被切成几块'来分的 —— 这才是父子块的战场。")

    for kind in ("同页多块", "跨页"):
        subset = [(i, c, inf) for i, c, inf in rows if inf["kind"] == kind]
        if not subset:
            continue
        print(f"\n── {kind}（{len(subset)} 题）" + "─" * 42)
        for i, c, inf in subset:
            print(f"  [第{i:>2}题/{c['difficulty']:<6}] 答案页={inf['best_pages']}"
                  f"（命中 {inf['terms_on_best']}/{inf['terms']} 个词）  "
                  f"该页子块{inf['n_children']}个/页数{inf['n_pages']}   {c['query']}")

    sel = [c for c in BENCHMARK_CASES if c["difficulty"] in ("easy", "medium")][:6]
    print("\n── 准确率评测实际跑的 6 题（前 6 道 easy/medium）" + "─" * 22)
    for item in sel:
        info = classify(item, chunks)
        pos = BENCHMARK_CASES.index(item) + 1
        print(f"  [第{pos:>2}题] {info['kind']:<8} 子块{info['n_children']}个   {item['query']}")


if __name__ == "__main__":
    main()
