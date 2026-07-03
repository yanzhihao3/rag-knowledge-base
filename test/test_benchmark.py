"""
召回率评测脚本
格式：query +期望关键词（命中任一即算召回）

运行方式：
    pytest test/test_benchmark.py -v -s

    # 或设置知识库ID
    pytest test/test_benchmark.py -v -s --kb_id=2
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_api import RAG


class TestBenchmarkMetrics:
    """Top-K 召回率评测"""

    def setup_method(self):
        self.rag = RAG()
        self.es_available = False
        try:
            from es_api import es
            if es.ping():
                self.es_available = True
        except Exception:
            pass

    def _load_test_dataset(self):
        """测试集：query + 命中的关键词列表"""
        return [
            {
                "query": "RAG的原理是什么",
                "relevant_keywords": ["检索增强", "RAG", "向量检索", "LLM"],
            },
            {
                "query": "PDF怎么上传",
                "relevant_keywords": ["PDF", "上传", "文档", "background"],
            },
            {
                "query": "服务端口是多少",
                "relevant_keywords": ["6010", "端口", "uvicorn", "启动"],
            },
            {
                "query": "向量维度",
                "relevant_keywords": ["512", "向量", "embedding", "维度"],
            },
            {
                "query": "支持哪些文件格式",
                "relevant_keywords": ["PDF", "word", "文档", "格式"],
            },
            {
                "query": "分块大小是多少",
                "relevant_keywords": ["256", "分块", "chunk", "token"],
            },
            {
                "query": "重排模型是什么",
                "relevant_keywords": ["rerank", "重排", "cross-encoder", "bge"],
            },
            {
                "query": "怎么查询知识库",
                "relevant_keywords": ["知识库", "查询", "knowledge_id", "检索"],
            },
        ]

    def _is_hit(self, top5_chunks, relevant_keywords):
        """命中判断：Top-5 内容中包含任一关键词即算命中"""
        text = " ".join(top5_chunks)
        return any(kw in text for kw in relevant_keywords)

    def _get_top5_chunks(self, results):
        """从 query_document 返回结果中提取 chunk_content"""
        chunks = []
        for r in results:
            # r可能是 dict包含 chunk_content 字段
            if isinstance(r, dict):
                cc = r.get("chunk_content", "")
                if isinstance(cc, list):
                    chunks.append(cc[0])
                elif isinstance(cc, str):
                    chunks.append(cc)
            elif isinstance(r, str):
                chunks.append(r)
        return chunks

    @pytest.mark.skipif(not bool(os.environ.get("RUN_INTEGRATION_TESTS")),
                        reason="需要 RUN_INTEGRATION_TESTS=1 环境变量才运行")
    def test_top5_recall(self):
        """Top-5 召回率测试"""
        if not self.es_available:
            pytest.skip("ES不可用，跳过")

        test_dataset = self._load_test_dataset()
        # 从命令行 --kb_id 获取，或默认 1
        kb_id = 1

        k = 5
        hits = 0
        total = len(test_dataset)

        for item in test_dataset:
            try:
                results = self.rag.query_document(item["query"], kb_id)
                top5_chunks = self._get_top5_chunks(results[:k])
                hit = self._is_hit(top5_chunks, item["relevant_keywords"])
                hits += hit

                status = "✓" if hit else "✗"
                print(f"[{status}] Q: {item['query']}")
                if not hit:
                    print(f"    期望关键词: {item['relevant_keywords']}")
                    print(f"    Top-{k}: {[c[:40] for c in top5_chunks]}")
            except Exception as e:
                print(f"[✗] Q: {item['query']} -错误: {e}")

        recall = hits / total
        print(f"\n{'='*50}")
        print(f"Top-{k} 召回率: {recall*100:.1f}% ({hits}/{total})")
        print(f"{'='*50}")

        # 低于 70% 说明检索配置有问题
        assert recall >= 0.7, f"召回率 {recall*100:.1f}% 低于 70%，请检查"

    @pytest.mark.skipif(not bool(os.environ.get("RUN_INTEGRATION_TESTS")),
                        reason="需要 RUN_INTEGRATION_TESTS=1 环境变量才运行")
    def test_recall_breakdown(self):
        """单路召回 vs 双路融合召回率拆解"""
        if not self.es_available:
            pytest.skip("ES不可用，跳过")

        test_dataset = self._load_test_dataset()
        kb_id = 1
        k = 10

        bm25_hits = 0
        vector_hits = 0
        fusion_hits = 0

        for item in test_dataset:
            try:
                rewritten_query = self.rag.query_rewrite(item["query"], None)
                embedding_vector = self.rag.get_embedding(rewritten_query)

                bm25_res = self.rag._word_search(rewritten_query, kb_id)
                vector_res = self.rag._vector_search(embedding_vector, kb_id)

                # BM25 Top-K
                bm25_chunks = []
                for r in bm25_res["hits"]["hits"][:k]:
                    cc = r["fields"].get("chunk_content", [""])[0] if r["fields"] else ""
                    bm25_chunks.append(cc)

                # Vector Top-K
                vector_chunks = []
                for r in vector_res["hits"]["hits"][:k]:
                    cc = r["fields"].get("chunk_content", [""])[0] if r["fields"] else ""
                    vector_chunks.append(cc)

                bm25_hit = self._is_hit(bm25_chunks, item["relevant_keywords"])
                vector_hit = self._is_hit(vector_chunks, item["relevant_keywords"])

                bm25_hits += int(bm25_hit)
                vector_hits += int(vector_hit)

                print(f"[{'✓' if bm25_hit else '✗'}] BM25 | {'✓' if vector_hit else '✗'}] Vector | Q: {item['query']}")
            except Exception as e:
               print(f"[✗] Q: {item['query']} - 错误: {e}")

        total = len(test_dataset)
        print(f"\n{'='*50}")
        print(f"BM25 召回率: {bm25_hits/total*100:.1f}% ({bm25_hits}/{total})")
        print(f"Vector 召回率:  {vector_hits/total*100:.1f}% ({vector_hits}/{total})")
        print(f"{'='*50}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])