"""
RAG 系统 Benchmark 评测脚本
=============================
评测维度：
  1. Top-K 召回率对比：BM25 / Vector / 多路融合(RRF)
  2. Top-1 精准度对比：有无重排对首个结果的影响
  3. 回答准确率对比：有无重排对最终回答的影响

运行方式：
  # 完整评测（需要 ES + 模型 + LLM）
  pytest test/benchmark_report.py -v -s

  # 只跑召回率（不依赖 LLM）
  pytest test/benchmark_report.py -v -s -k "recall"

  # 只跑准确率评测（依赖 LLM）
  pytest test/benchmark_report.py -v -s -k "accuracy"

指标输出示例：
  Top-5 召回率对比:
    BM25:       62.5%
    Vector:     75.0%
    Fusion:     87.5%

  回答准确率:
    无重排: 60.0%
    有重排: 85.0%
    提升:   +25.0%
"""
import pytest
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_api import RAG

# ── Windows GBK safe print ─────────────────────────────────
_OK = "[OK]"
_FAIL = "[FAIL]"

# ============================================================
# 评测用例：基于《大模型RAG实战》标注的 50 条（数据见 benchmark_cases_rag_book.py）
# 难度分级：easy - 关键词可直接匹配；medium - 需要语义理解；hard - 需要深度细节
from test.benchmark_cases_rag_book import BENCHMARK_CASES

KB_ID = 1
TOP_K = 5


def setup_module():
    print(f"\n{'='*60}")
    print(f"  RAG Benchmark 评测")
    print(f"{'='*60}")
    print(f"  知识库 ID:      {KB_ID}")
    print(f"  测试用例数:     {len(BENCHMARK_CASES)}")
    n_easy = sum(1 for c in BENCHMARK_CASES if c["difficulty"] == "easy")
    n_med = sum(1 for c in BENCHMARK_CASES if c["difficulty"] == "medium")
    n_hard = sum(1 for c in BENCHMARK_CASES if c["difficulty"] == "hard")
    print(f"  难度分布:       easy={n_easy}, medium={n_med}, hard={n_hard}")
    print(f"  Top-K:          {TOP_K}")
    print(f"{'='*60}\n")


# ── Helper functions ────────────────────────────────────────

def _is_hit(chunks, keywords):
    text = " ".join([c if isinstance(c, str) else str(c) for c in chunks])
    return any(kw.lower() in text.lower() for kw in keywords)


def _extract_chunks(results):
    chunks = []
    for r in results:
        if isinstance(r, dict):
            cc = r.get("chunk_content", "")
            if isinstance(cc, list) and len(cc) > 0:
                chunks.append(cc[0])
            elif isinstance(cc, str):
                chunks.append(cc)
        elif isinstance(r, str):
            chunks.append(r)
    return chunks


def _extract_hits(hits):
    """从 ES hits 提取 (chunk_id, chunk_content)。"""
    out = []
    for h in hits:
        cc = h.get("fields", {}).get("chunk_content") or [""]
        out.append((h["_id"], cc[0] if isinstance(cc, list) else cc))
    return out


def _load_chunks_by_id(kb_id):
    """加载知识库全部分块 (id -> content)，用于按 answer_terms 定位答案块。"""
    from es_api import es
    resp = es.search(
        index="chunk_info",
        size=10000,
        query={"term": {"knowledge_id": kb_id}},
        fields=["chunk_content"],
        source=False,
    )
    out = {}
    for h in resp["hits"]["hits"]:
        cc = h.get("fields", {}).get("chunk_content") or [""]
        out[h["_id"]] = (cc[0] if isinstance(cc, list) else cc) or ""
    return out


def _answer_chunk_ids(item, chunks_by_id):
    """用 answer_terms 在知识库分块里定位该题的答案块 ID 集合。"""
    terms = [t.lower() for t in item.get("answer_terms", [])]
    if not terms:
        return set()
    return {cid for cid, content in chunks_by_id.items() if any(t in content.lower() for t in terms)}


# ============================================================
# 召回率评测
# ============================================================
class TestRecall:
    """Top-K 召回率对比"""

    def setup_method(self):
        self.rag = RAG()
        self.es_available = False
        self.chunks_by_id = {}
        try:
            from es_api import es
            if es.ping():
                self.es_available = True
        except Exception:
            pass
        try:
            self.chunks_by_id = _load_chunks_by_id(KB_ID)
        except Exception:
            pass

    def _search_bm25(self, query):
        rewritten = self.rag.query_rewrite(query, None)
        res = self.rag._word_search(rewritten, KB_ID)
        return _extract_hits(res["hits"]["hits"][:TOP_K])

    def _search_vector(self, query):
        rewritten = self.rag.query_rewrite(query, None)
        vec = self.rag.get_embedding(rewritten)
        res = self.rag._vector_search(vec, KB_ID)
        return _extract_hits(res["hits"]["hits"][:TOP_K])

    def _search_fusion(self, query):
        results = self.rag.query_document(query, KB_ID, None)[:TOP_K]
        out = []
        for r in results:
            cc = r.get("chunk_content") or [""]
            out.append((r.get("_id", ""), cc[0] if isinstance(cc, list) else cc))
        return out

    def _run_one_method(self, label, search_fn):
        n = len(BENCHMARK_CASES)
        kw_hits = 0
        strict_hits = 0
        mrr_sum = 0.0
        details = []
        for item in BENCHMARK_CASES:
            try:
                top = search_fn(item["query"])  # [(chunk_id, content)]
            except Exception as e:
                print(f"    [ERR] {item['query']}: {e}")
                details.append((item["query"], item["difficulty"], False, False, None))
                continue

            contents = [c for _, c in top]
            kw_hit = _is_hit(contents, item["relevant_keywords"])
            gold = _answer_chunk_ids(item, self.chunks_by_id)
            # 严格口径：答案块本身是否出现在 Top-5，以及它的最佳排名（MRR）
            rank = next((i + 1 for i, (cid, _) in enumerate(top) if cid in gold), None)
            strict_hit = rank is not None

            kw_hits += int(kw_hit)
            strict_hits += int(strict_hit)
            if rank:
                mrr_sum += 1.0 / rank
            details.append((item["query"], item["difficulty"], kw_hit, strict_hit, rank))

        return kw_hits / n, strict_hits / n, mrr_sum / n, details

    def test_es_check(self):
        if not self.es_available:
            pytest.skip("ES 不可用，跳过所有召回率测试")

    @pytest.mark.dependency(name="recall_bm25")
    def test_recall_bm25(self):
        if not self.es_available:
            pytest.skip("ES 不可用")
        kw, strict, mrr, details = self._run_one_method("BM25", self._search_bm25)
        print(f"\n  BM25 关键词Top-{TOP_K}: {kw*100:.1f}%  答案块命中: {strict*100:.1f}%  MRR@{TOP_K}: {mrr:.3f}")
        for q, diff, kh, sh, rank in details:
            print(f"    [{diff}] 关键词{_OK if kh else _FAIL} 答案块{_OK if sh else _FAIL} rank={rank}  {q}")

    @pytest.mark.dependency(name="recall_vector")
    def test_recall_vector(self):
        if not self.es_available:
            pytest.skip("ES 不可用")
        kw, strict, mrr, details = self._run_one_method("Vector", self._search_vector)
        print(f"\n  Vector 关键词Top-{TOP_K}: {kw*100:.1f}%  答案块命中: {strict*100:.1f}%  MRR@{TOP_K}: {mrr:.3f}")
        for q, diff, kh, sh, rank in details:
            print(f"    [{diff}] 关键词{_OK if kh else _FAIL} 答案块{_OK if sh else _FAIL} rank={rank}  {q}")

    @pytest.mark.dependency(name="recall_fusion")
    def test_recall_fusion(self):
        if not self.es_available:
            pytest.skip("ES 不可用")
        kw, strict, mrr, details = self._run_one_method("Fusion", self._search_fusion)
        print(f"\n  Fusion 关键词Top-{TOP_K}: {kw*100:.1f}%  答案块命中: {strict*100:.1f}%  MRR@{TOP_K}: {mrr:.3f}")
        for q, diff, kh, sh, rank in details:
            print(f"    [{diff}] 关键词{_OK if kh else _FAIL} 答案块{_OK if sh else _FAIL} rank={rank}  {q}")

    @pytest.mark.dependency(
        name="recall_summary",
        depends=["recall_bm25", "recall_vector", "recall_fusion"],
    )
    def test_recall_summary(self):
        if not self.es_available:
            pytest.skip("ES 不可用")

        print(f"\n{'='*60}")
        print(f"  Top-{TOP_K} 召回率对比汇总")
        print(f"{'='*60}")

        methods = [
            ("BM25",       self._search_bm25),
            ("Vector",     self._search_vector),
            ("Fusion(RRF)", self._search_fusion),
        ]

        results = {}
        for label, fn in methods:
            kw, strict, mrr, details = self._run_one_method(label, fn)
            results[label] = (kw, strict, mrr, details)

        # 汇总表
        print(f"\n  {'Method':<12} {'关键词命中':>10} {'答案块命中':>10} {'MRR@5':>8}")
        print(f"  {'-'*48}")
        for label, (kw, strict, mrr, details) in results.items():
            print(f"  {label:<12} {kw*100:>9.1f}% {strict*100:>9.1f}% {mrr:>8.3f}")

        # 答案块命中：Fusion 相对 BM25 / Vector 的提升
        b = results.get("BM25", (0, 0, 0, []))
        v = results.get("Vector", (0, 0, 0, []))
        f = results.get("Fusion(RRF)", (0, 0, 0, []))
        if b[1] > 0:
            print(f"\n  答案块命中 融合 vs BM25: {(f[1]-b[1])*100:+.1f}%   融合 vs Vector: {(f[1]-v[1])*100:+.1f}%")
            print(f"  MRR@5     融合 vs BM25: {f[2]-b[2]:+.3f}   融合 vs Vector: {f[2]-v[2]:+.3f}")
        print(f"{'='*60}\n")


# ============================================================
# 回答准确率评测（需要 LLM）
# ============================================================
class TestAccuracy:
    """回答准确率对比：无重排 vs 有重排"""

    def setup_method(self):
        self.rag = RAG()
        self.es_available = False
        try:
            from es_api import es
            if es.ping():
                self.es_available = True
        except Exception:
            pass

    def _ask(self, query, use_rerank):
        """单轮对话"""
        original = self.rag.use_rerank
        self.rag.use_rerank = use_rerank
        try:
            messages, _ = self.rag.chat_with_rag(KB_ID, [{"role": "user", "content": query}])
            for msg in reversed(messages):
                if msg.get("role") == "system":
                    return msg.get("content", "")
            return ""
        finally:
            self.rag.use_rerank = original

    def _score(self, answer, expected_points):
        if not answer:
            return 0.0
        covered = sum(1 for pt in expected_points if pt.lower() in answer.lower())
        return covered / len(expected_points)

    def test_accuracy_comparison(self):
        if not self.es_available:
            pytest.skip("ES 不可用")

        # 选前 6 题跑（避免耗时太长）
        test_cases = [c for c in BENCHMARK_CASES if c["difficulty"] in ("easy", "medium")][:6]

        print(f"\n  回答准确率评测（{len(test_cases)} 题）")
        print(f"  {'='*50}")

        scores_no = []
        scores_yes = []

        for i, item in enumerate(test_cases):
            q = item["query"]
            pts = item["expected_points"]
            print(f"\n  [{i+1}] {q}")

            t0 = time.time()
            a_no = self._ask(q, use_rerank=False)
            t_no = time.time() - t0
            s_no = self._score(a_no, pts)
            scores_no.append(s_no)

            t0 = time.time()
            a_yes = self._ask(q, use_rerank=True)
            t_yes = time.time() - t0
            s_yes = self._score(a_yes, pts)
            scores_yes.append(s_yes)

            print(f"      无重排: {s_no*100:.0f}% ({t_no:.1f}s)")
            print(f"      有重排: {s_yes*100:.0f}% ({t_yes:.1f}s)")

        avg_no = float(np.mean(scores_no)) * 100
        avg_yes = float(np.mean(scores_yes)) * 100
        impr = avg_yes - avg_no

        print(f"\n  {'='*50}")
        print(f"  回答准确率汇总")
        print(f"  {'='*50}")
        print(f"  无重排: {avg_no:.1f}%")
        print(f"  有重排: {avg_yes:.1f}%")
        print(f"  提升:   +{impr:.1f}%")
        print(f"  {'='*50}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
