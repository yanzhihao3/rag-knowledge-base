import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router_schemas import RerankRequest


def test_rerank_request_has_text_pair():
    req = RerankRequest(
        model="bge-reranker-base",
        text_pair=[["什么是RAG？", "RAG是检索增强生成技术"]],
    )
    assert req.text_pair == [["什么是RAG？", "RAG是检索增强生成技术"]]
