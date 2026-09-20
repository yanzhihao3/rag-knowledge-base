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
import re
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

# 知识库 ID：默认指向当前 MySQL / ES 里实际存在的那个知识库（2026-09 是 2）。
# 换知识库时用环境变量覆盖，不要改代码：
#   $env:RAG_BENCH_KB_ID='3'; pytest test/benchmark_report.py -v -s -k recall
KB_ID = int(os.environ.get("RAG_BENCH_KB_ID", "2"))
TOP_K = 5

# 分块 id 都是确定性拼出来的：页级 doc_<did>_page_<pn>，子块 doc_<did>_page_<pn>_chunk_<k>。
# 所以"这个块属于哪一页"可以直接从 id 推出来，不需要额外的 ES 字段。
_PAGE_KEY_RE = re.compile(r"^(doc_\d+_page_\d+)")


def _page_key(chunk_id):
    """把块 id 归一到「页」：doc_3_page_7_chunk_2 → doc_3_page_7。"""
    m = _PAGE_KEY_RE.match(chunk_id or "")
    return m.group(1) if m else (chunk_id or "")


# 要点覆盖率的判定阈值（见 _point_covered 的说明）
_POINT_COVERAGE_THRESHOLD = 0.4


def _point_covered(answer, point, threshold=_POINT_COVERAGE_THRESHOLD):
    """一个要点是否被回答覆盖 —— 用 2 字滑窗覆盖率，而不是逐字子串匹配。

    为什么必须换掉逐字匹配：expected_points 是人工写的书面短句，模型几乎不可能逐字照抄。
    实测（第 5 题）要点是「给出答案来源 / 可追溯文档 / 增强可信度」，模型的回答是
    「提供每个答案的来源信息……看到每个答案的来源文档……增强了模型的回答的可信度」——
    意思全对，但三个要点逐字一个都对不上，全被判 0 分。这样量出来的"回答准确率"
    其实是"模型有没有恰好用这几个词"，不能用来比较检索策略。

    这里把要点切成 2 字片段，命中比例 >= 阈值就算覆盖。0.4 是启发式的：
    太低会把无关回答判成命中，太高又退回"必须逐字"的老问题。
    更严谨的做法是把要点重新标成"判别性关键词组"，或用强模型做 LLM-as-judge。
    """
    a = (answer or "").lower()
    p = (point or "").lower().strip()
    if not p:
        return True
    if len(p) <= 2:
        return p in a
    grams = [p[i:i + 2] for i in range(len(p) - 1)]
    hit = sum(1 for g in grams if g in a)
    return hit / len(grams) >= threshold


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
        # 页级块和子块分开：页级块的 id 就是它自己的 _page_key
        self.page_ids = {cid for cid in self.chunks_by_id if _page_key(cid) == cid}
        self.child_ids = {cid for cid in self.chunks_by_id if _page_key(cid) != cid}

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
        """跑完一个检索方式，返回四组指标。

        gold 有三种口径，难度和含义都不同：
          gold       全库里包含任一 answer_term 的块（**含页级块**）。历史口径，用来和旧结果对齐。
          gold_child 只看子块。去掉"页级块必然包含本页子块文字"这种白送分。
          gold_page  答案所在的**页**（用页级块判定）。页级块文本不随分块方式改变，
                     所以这个口径跨分块方案可比，是改分块/改检索时唯一站得住的尺子。
        """
        n = len(BENCHMARK_CASES)
        hits = {"kw": 0, "strict": 0, "child": 0, "page": 0}
        mrr = {"strict": 0.0, "child": 0.0, "page": 0.0}
        details = []
        for item in BENCHMARK_CASES:
            try:
                top = search_fn(item["query"])  # [(chunk_id, content)]
            except Exception as e:
                print(f"    [ERR] {item['query']}: {e}")
                details.append((item["query"], item["difficulty"], False, False, False, False, None))
                continue

            contents = [c for _, c in top]
            kw_hit = _is_hit(contents, item["relevant_keywords"])
            gold = _answer_chunk_ids(item, self.chunks_by_id)
            gold_child = gold & self.child_ids
            terms = [t.lower() for t in item.get("answer_terms", [])]
            gold_page = {cid for cid in self.page_ids
                         if terms and any(t in (self.chunks_by_id[cid] or "").lower()
                                          for t in terms)}

            # 严格口径：答案块本身是否出现在 Top-5，以及它的最佳排名（MRR）
            rank = next((i + 1 for i, (cid, _) in enumerate(top) if cid in gold), None)
            rank_child = next((i + 1 for i, (cid, _) in enumerate(top)
                               if cid in gold_child), None)
            rank_page = next((i + 1 for i, (cid, _) in enumerate(top)
                              if _page_key(cid) in gold_page), None)

            hits["kw"] += int(kw_hit)
            hits["strict"] += int(rank is not None)
            hits["child"] += int(rank_child is not None)
            hits["page"] += int(rank_page is not None)
            if rank:
                mrr["strict"] += 1.0 / rank
            if rank_child:
                mrr["child"] += 1.0 / rank_child
            if rank_page:
                mrr["page"] += 1.0 / rank_page
            details.append((item["query"], item["difficulty"], kw_hit,
                            rank is not None, rank_child is not None, rank_page is not None, rank))

        return {
            "kw": hits["kw"] / n,
            "strict": hits["strict"] / n,
            "child": hits["child"] / n,
            "page": hits["page"] / n,
            "mrr": mrr["strict"] / n,
            "child_mrr": mrr["child"] / n,
            "page_mrr": mrr["page"] / n,
            "details": details,
        }

    def test_es_check(self):
        if not self.es_available:
            pytest.skip("ES 不可用，跳过所有召回率测试")

    @staticmethod
    def _print_method(label, m):
        print(f"\n  {label} 关键词Top-{TOP_K}: {m['kw']*100:.1f}%  答案块: {m['strict']*100:.1f}%  "
              f"仅子块: {m['child']*100:.1f}%  答案页: {m['page']*100:.1f}%  "
              f"MRR@{TOP_K}: {m['mrr']:.3f}（子块 {m['child_mrr']:.3f} / 页 {m['page_mrr']:.3f}）")
        for q, diff, kh, sh, ch, ph, rank in m["details"]:
            print(f"    [{diff}] 关键词{_OK if kh else _FAIL} 答案块{_OK if sh else _FAIL} "
                  f"rank={rank}  子块{_OK if ch else _FAIL} 页{_OK if ph else _FAIL}  {q}")

    @pytest.mark.dependency(name="recall_bm25")
    def test_recall_bm25(self):
        if not self.es_available:
            pytest.skip("ES 不可用")
        self._print_method("BM25", self._run_one_method("BM25", self._search_bm25))

    @pytest.mark.dependency(name="recall_vector")
    def test_recall_vector(self):
        if not self.es_available:
            pytest.skip("ES 不可用")
        self._print_method("Vector", self._run_one_method("Vector", self._search_vector))

    @pytest.mark.dependency(name="recall_fusion")
    def test_recall_fusion(self):
        if not self.es_available:
            pytest.skip("ES 不可用")
        self._print_method("Fusion", self._run_one_method("Fusion", self._search_fusion))

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
            results[label] = self._run_one_method(label, fn)

        # 汇总表
        print(f"\n  {'Method':<12} {'关键词':>8} {'答案块':>8} {'仅子块':>8} {'答案页':>8} {'MRR@5':>8}")
        print(f"  {'-'*56}")
        for label, m in results.items():
            print(f"  {label:<12} {m['kw']*100:>7.1f}% {m['strict']*100:>7.1f}% "
                  f"{m['child']*100:>7.1f}% {m['page']*100:>7.1f}% {m['mrr']:>8.3f}")

        # 答案块命中：Fusion 相对 BM25 / Vector 的提升
        b = results["BM25"]
        v = results["Vector"]
        f = results["Fusion(RRF)"]
        print(f"\n  答案块命中 融合 vs BM25: {(f['strict']-b['strict'])*100:+.1f}%   "
              f"融合 vs Vector: {(f['strict']-v['strict'])*100:+.1f}%")
        print(f"  答案页命中 融合 vs BM25: {(f['page']-b['page'])*100:+.1f}%   "
              f"融合 vs Vector: {(f['page']-v['page'])*100:+.1f}%")
        print(f"  MRR@5     融合 vs BM25: {f['mrr']-b['mrr']:+.3f}   "
              f"融合 vs Vector: {f['mrr']-v['mrr']:+.3f}")
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
        """返回 (逐字命中率, 要点覆盖率)。

        逐字那一列保留下来只为对照：它衡量的其实是"模型有没有恰好用上标注里的词"，
        中文改写场景下几乎恒为 0，不能当准确率用。真正用来比较的是要点覆盖率。
        """
        if not answer or not expected_points:
            return 0.0, 0.0
        exact = sum(1 for pt in expected_points if pt.lower() in answer.lower())
        cover = sum(1 for pt in expected_points if _point_covered(answer, pt))
        return exact / len(expected_points), cover / len(expected_points)

    def test_accuracy_comparison(self):
        if not self.es_available:
            pytest.skip("ES 不可用")

        # 选前 6 题跑（避免耗时太长）
        test_cases = [c for c in BENCHMARK_CASES if c["difficulty"] in ("easy", "medium")][:6]

        print(f"\n  回答准确率评测（{len(test_cases)} 题）")
        print(f"  {'='*50}")

        scores_no = []
        scores_yes = []
        exact_no = []
        exact_yes = []

        for i, item in enumerate(test_cases):
            q = item["query"]
            pts = item["expected_points"]
            print(f"\n  [{i+1}] {q}")

            t0 = time.time()
            a_no = self._ask(q, use_rerank=False)
            t_no = time.time() - t0
            e_no, s_no = self._score(a_no, pts)
            scores_no.append(s_no)
            exact_no.append(e_no)

            t0 = time.time()
            a_yes = self._ask(q, use_rerank=True)
            t_yes = time.time() - t0
            e_yes, s_yes = self._score(a_yes, pts)
            scores_yes.append(s_yes)
            exact_yes.append(e_yes)

            print(f"      无重排: 要点{s_no*100:.0f}% (逐字{e_no*100:.0f}%) ({t_no:.1f}s)")
            print(f"      有重排: 要点{s_yes*100:.0f}% (逐字{e_yes*100:.0f}%) ({t_yes:.1f}s)")

        avg_no = float(np.mean(scores_no)) * 100
        avg_yes = float(np.mean(scores_yes)) * 100
        impr = avg_yes - avg_no

        print(f"\n  {'='*50}")
        print(f"  回答准确率汇总")
        print(f"  {'='*50}")
        print(f"  要点覆盖  无重排: {avg_no:.1f}%   有重排: {avg_yes:.1f}%   提升: +{impr:.1f}%")
        print(f"  逐字匹配  无重排: {float(np.mean(exact_no))*100:.1f}%   "
              f"有重排: {float(np.mean(exact_yes))*100:.1f}%   （对照用，中文改写场景下恒低）")
        print(f"  {'='*50}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
