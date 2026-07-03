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
# 测试数据集（基于 kb_id=1 实际文档内容设计）
# 难度分级：
#   easy   - 关键词可直接匹配
#   medium - 需要语义理解或同义表达
#   hard   - 需要深度理解的细节问题
# ============================================================
BENCHMARK_CASES = [
    # ── easy: BM25 也能搞定的 ──
    {
        "query": "RAG的原理是什么",
        "relevant_keywords": ["检索增强生成", "Retrieval Augmented", "检索", "生成"],
        "expected_points": ["检索", "生成", "增强"],
        "difficulty": "easy",
    },
    {
        "query": "为什么要用提示学习",
        "relevant_keywords": ["提示学习", "Prompting", "微调", "预训练"],
        "expected_points": ["微调", "预训练", "提示"],
        "difficulty": "easy",
    },
    # ── medium: 需要语义泛化 ──
    {
        "query": "RAG系统相比直接微调大模型有什么优势",
        "relevant_keywords": ["知识更新", "训练成本", "微调", "数据泄露", "知识库"],
        "expected_points": ["知识更新", "训练", "成本"],
        "difficulty": "medium",
    },
    {
        "query": "怎么把文档切成小块存入ES",
        "relevant_keywords": ["分块", "chunk", "overlap", "pdfplumber", "索引", "es"],
        "expected_points": ["分块", "chunk", "索引"],
        "difficulty": "medium",
    },
    {
        "query": "哪些因素会影响RAG系统的回答质量",
        "relevant_keywords": ["检索", "召回", "知识库", "覆盖", "分块"],
        "expected_points": ["检索", "知识库", "召回"],
        "difficulty": "medium",
    },
    # ── hard: 需要深度语义匹配 ──
    {
        "query": "在构建RAG系统时检索部分为什么重要",
        "relevant_keywords": ["检索部分", "非常重要", "花费大量时间", "召回", "打磨"],
        "expected_points": ["检索", "召回", "质量"],
        "difficulty": "hard",
    },
    {
        "query": "RETRO论文证明了什么结论",
        "relevant_keywords": ["RETRO", "1/25", "参数量", "Trillions of Tokens"],
        "expected_points": ["RETRO", "参数量"],
        "difficulty": "hard",
    },
    {
        "query": "检索增强生成如何解决大模型的知识更新问题",
        "relevant_keywords": ["知识更新", "RAG", "微调", "继续预训练", "新知识"],
        "expected_points": ["知识更新", "RAG"],
        "difficulty": "hard",
    },
    {
        "query": "ES的密集向量检索是如何配置的",
        "relevant_keywords": ["dense_vector", "embedding", "HNSW", "int8_hnsw", "KNN"],
        "expected_points": ["dense_vector", "KNN", "HNSW"],
        "difficulty": "hard",
    },
    {
        "query": "多路召回合并时RRF算法如何给不同检索结果打分",
        "relevant_keywords": ["RRF", "倒数排序", "分数", "排名", "融合"],
        "expected_points": ["RRF", "排名", "融合"],
        "difficulty": "hard",
    },
]

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


# ============================================================
# 召回率评测
# ============================================================
class TestRecall:
    """Top-K 召回率对比"""

    def setup_method(self):
        self.rag = RAG()
        self.es_available = False
        try:
            from es_api import es
            if es.ping():
                self.es_available = True
        except Exception:
            pass

    def _search_bm25(self, query):
        rewritten = self.rag.query_rewrite(query, None)
        res = self.rag._word_search(rewritten, KB_ID)
        return _extract_chunks([
            {"chunk_content": h["fields"].get("chunk_content", [""])[0]}
            for h in res["hits"]["hits"][:TOP_K]
        ])

    def _search_vector(self, query):
        rewritten = self.rag.query_rewrite(query, None)
        vec = self.rag.get_embedding(rewritten)
        res = self.rag._vector_search(vec, KB_ID)
        return _extract_chunks([
            {"chunk_content": h["fields"].get("chunk_content", [""])[0]}
            for h in res["hits"]["hits"][:TOP_K]
        ])

    def _search_fusion(self, query):
        results = self.rag.query_document(query, KB_ID, None)
        return _extract_chunks(results[:TOP_K])

    def _run_one_method(self, label, search_fn):
        hits = 0
        details = []
        for item in BENCHMARK_CASES:
            try:
                chunks = search_fn(item["query"])
                hit = _is_hit(chunks[:TOP_K], item["relevant_keywords"])
                hits += int(hit)
                details.append((item["query"], item["difficulty"], hit))
            except Exception as e:
                print(f"    [ERR] {item['query']}: {e}")
                details.append((item["query"], item["difficulty"], False))
        recall = hits / len(BENCHMARK_CASES)
        return recall, hits, details

    def test_es_check(self):
        if not self.es_available:
            pytest.skip("ES 不可用，跳过所有召回率测试")

    @pytest.mark.dependency(name="recall_bm25")
    def test_recall_bm25(self):
        if not self.es_available:
            pytest.skip("ES 不可用")
        recall, hits, details = self._run_one_method("BM25", self._search_bm25)
        print(f"\n  BM25 Top-{TOP_K}: {recall*100:.1f}% ({hits}/{len(BENCHMARK_CASES)})")
        for q, diff, hit in details:
            print(f"    [{diff}] {_OK if hit else _FAIL} {q}")

    @pytest.mark.dependency(name="recall_vector")
    def test_recall_vector(self):
        if not self.es_available:
            pytest.skip("ES 不可用")
        recall, hits, details = self._run_one_method("Vector", self._search_vector)
        print(f"\n  Vector Top-{TOP_K}: {recall*100:.1f}% ({hits}/{len(BENCHMARK_CASES)})")
        for q, diff, hit in details:
            print(f"    [{diff}] {_OK if hit else _FAIL} {q}")

    @pytest.mark.dependency(name="recall_fusion")
    def test_recall_fusion(self):
        if not self.es_available:
            pytest.skip("ES 不可用")
        recall, hits, details = self._run_one_method("Fusion", self._search_fusion)
        print(f"\n  Fusion Top-{TOP_K}: {recall*100:.1f}% ({hits}/{len(BENCHMARK_CASES)})")
        for q, diff, hit in details:
            print(f"    [{diff}] {_OK if hit else _FAIL} {q}")

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
            recall, hits, details = self._run_one_method(label, fn)
            results[label] = (recall, hits, details)

        # 汇总表
        print(f"\n  {'Method':<16} {'Recall':>8} {'Easy':>8} {'Med':>8} {'Hard':>8}")
        print(f"  {'-'*52}")
        for label, (recall, hits, details) in results.items():
            by_diff = {"easy": 0, "medium": 0, "hard": 0}
            easy_n = sum(1 for c in BENCHMARK_CASES if c["difficulty"] == "easy")
            med_n = sum(1 for c in BENCHMARK_CASES if c["difficulty"] == "medium")
            hard_n = sum(1 for c in BENCHMARK_CASES if c["difficulty"] == "hard")
            for q, diff, hit in details:
                if hit:
                    by_diff[diff] += 1
            e_str = f"{by_diff['easy']}/{easy_n}" if easy_n else "-"
            m_str = f"{by_diff['medium']}/{med_n}" if med_n else "-"
            h_str = f"{by_diff['hard']}/{hard_n}" if hard_n else "-"
            print(f"  {label:<16} {recall*100:>7.1f}% {e_str:>8} {m_str:>8} {h_str:>8}")

        # BM25 -> Fusion 提升
        b_r = results.get("BM25", (0, 0, []))[0]
        f_r = results.get("Fusion(RRF)", (0, 0, []))[0]
        if b_r > 0:
            imp = (f_r - b_r) / b_r * 100
            print(f"\n  多路融合 vs BM25 提升: {imp:+.1f}%")
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
            messages = self.rag.chat_with_rag(KB_ID, [{"role": "user", "content": query}])
            for msg in reversed(messages):
                if msg["role"] == "system":
                    return msg["content"]
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
