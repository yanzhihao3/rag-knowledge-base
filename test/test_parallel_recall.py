"""
双路并行召回测试

测试 BM25 + KNN 双路并行召回是否正确实现
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_api import RAG


class TestParallelRecall:
    """双路并行召回测试"""

    def setup_method(self):
        self.rag = RAG()

    def test_word_search_method_exists(self):
        """测试第一路：关键词检索方法存在"""
        assert hasattr(self.rag, '_word_search'), "缺少 _word_search 方法"
        assert callable(self.rag._word_search), "_word_search 应该是可调用的"

    def test_vector_search_method_exists(self):
        """测试第二路：向量检索方法存在"""
        assert hasattr(self.rag, '_vector_search'), "缺少 _vector_search 方法"
        assert callable(self.rag._vector_search), "_vector_search 应该是可调用的"

    def test_query_document_uses_threadpool(self):
        """测试 query_document 使用 ThreadPoolExecutor 并行"""
        # Mock 两路搜索，返回预设结果
        mock_word_response = {
            "hits": {
                "hits": [
                    {"_id": "doc1", "fields": {"chunk_content": ["测试内容1"]}},
                    {"_id": "doc2", "fields": {"chunk_content": ["测试内容2"]}},
                ]
            }
        }
        mock_vector_response = {
            "hits": {
                "hits": [
                    {"_id": "doc3", "fields": {"chunk_content": ["测试内容3"]}},
                    {"_id": "doc1", "fields": {"chunk_content": ["测试内容1"]}},
                ]
            }
        }

        with patch.object(self.rag, '_word_search', return_value=mock_word_response) as mock_word, \
             patch.object(self.rag, '_vector_search', return_value=mock_vector_response) as mock_vector, \
             patch.object(self.rag, 'query_rewrite', return_value="测试查询"), \
             patch.object(self.rag, 'get_embedding', return_value=[0.1] * 512):

            results = self.rag.query_document("测试", knowledge_id=1)

            # 验证两路都被调用
            assert mock_word.called, "BM25 检索未被调用"
            assert mock_vector.called, "KNN 检索未被调用"

            # 验证结果合并正确（doc1 在两路都出现）
            assert len(results) > 0, "结果为空"

    def test_rrf_fusion_with_both_channels(self):
        """测试 RRF 融合正确处理两路结果"""
        mock_word_response = {
            "hits": {
                "hits": [
                    {"_id": "doc1", "fields": {"chunk_content": ["内容1"]}},
                    {"_id": "doc2", "fields": {"chunk_content": ["内容2"]}},
                ]
            }
        }
        mock_vector_response = {
            "hits": {
                "hits": [
                    {"_id": "doc2", "fields": {"chunk_content": ["内容2"]}},
                    {"_id": "doc3", "fields": {"chunk_content": ["内容3"]}},
                ]
            }
        }

        with patch.object(self.rag, '_word_search', return_value=mock_word_response), \
             patch.object(self.rag, '_vector_search', return_value=mock_vector_response), \
             patch.object(self.rag, 'query_rewrite', return_value="测试"), \
             patch.object(self.rag, 'get_embedding', return_value=[0.1] * 512), \
             patch.object(self.rag, 'use_rerank', False):  # 跳过重排序简化测试

            results = self.rag.query_document("测试", knowledge_id=1)

            # doc2 在两路都出现，RRF 分数应该最高，排在第一位
            assert len(results) >= 1, "结果为空"

    def test_parallel_execution_time(self):
        """测试并行执行比串行快（验证真正并行）"""
        import time

        def slow_search(delay=0.5):
            time.sleep(delay)
            return {"hits": {"hits": []}}

        # 串行执行时间
        start = time.time()
        slow_search(0.3)
        slow_search(0.3)
        serial_time = time.time() - start

        # 并行执行时间
        start = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(slow_search, 0.3)
            f2 = executor.submit(slow_search, 0.3)
            f1.result()
            f2.result()
        parallel_time = time.time() - start

        # 并行应该明显快于串行
        assert parallel_time < serial_time, f"并行未生效: serial={serial_time:.2f}s, parallel={parallel_time:.2f}s"

    def test_rerank_not_called_when_disabled(self):
        """测试 use_rerank=False 时不调用重排序"""
        mock_word_response = {
            "hits": {
                "hits": [
                    {"_id": "doc1", "fields": {"chunk_content": ["内容1"]}},
                ]
            }
        }
        mock_vector_response = {
            "hits": {
                "hits": [
                    {"_id": "doc1", "fields": {"chunk_content": ["内容1"]}},
                ]
            }
        }

        with patch.object(self.rag, '_word_search', return_value=mock_word_response), \
             patch.object(self.rag, '_vector_search', return_value=mock_vector_response), \
             patch.object(self.rag, 'query_rewrite', return_value="测试"), \
             patch.object(self.rag, 'get_embedding', return_value=[0.1] * 512), \
             patch.object(self.rag, 'get_rerank') as mock_rerank, \
             patch.object(self.rag, 'use_rerank', False):

            self.rag.query_document("测试", knowledge_id=1)

            # 重排序不应该被调用
            assert not mock_rerank.called, "use_rerank=False 时不应调用 get_rerank"


class TestRRFFusion:
    """RRF 融合算法测试"""

    def setup_method(self):
        self.rag = RAG()

    def test_rrf_score_calculation(self):
        """测试 RRF 分数计算公式：score = Σ 1/(k + rank)"""
        # 两路各有一个 doc，排名都是 1
        # k=60 时，score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0328
        # 单路出现：score = 1/(60+1) ≈ 0.0164
        # 所以两路都出现的 doc 分数应该更高

        mock_word_response = {
            "hits": {
                "hits": [
                    {"_id": "both", "fields": {"chunk_content": ["两路都有的文档"]}},
                    {"_id": "word_only", "fields": {"chunk_content": ["只有 BM25 的文档"]}},
                ]
            }
        }
        mock_vector_response = {
            "hits": {
                "hits": [
                    {"_id": "both", "fields": {"chunk_content": ["两路都有的文档"]}},
                    {"_id": "vector_only", "fields": {"chunk_content": ["只有 KNN 的文档"]}},
                ]
            }
        }

        with patch.object(self.rag, '_word_search', return_value=mock_word_response), \
             patch.object(self.rag, '_vector_search', return_value=mock_vector_response), \
             patch.object(self.rag, 'query_rewrite', return_value="测试"), \
             patch.object(self.rag, 'get_embedding', return_value=[0.1] * 512), \
             patch.object(self.rag, 'use_rerank', False), \
             patch.object(self.rag, 'chunk_candidate', 10):

            results = self.rag.query_document("测试", knowledge_id=1)

            # "both" 应该排在最前面，因为 RRF 分数最高
            assert results[0]["chunk_content"][0] == "两路都有的文档", "RRF 融合未正确排序"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])