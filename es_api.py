import yaml
from elasticsearch import Elasticsearch
import os
import time
import logging

from utils import with_retry

logger = logging.getLogger(__name__)

#project_root = os.path.dirname(os.path.abspath(__file__))
#config_path = os.path.join(project_root, 'config.yaml')
with open("config.yaml", 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

es_host = os.environ.get("ES_HOST", config["elasticsearch"]["host"])
es_port = int(os.environ.get("ES_PORT", config["elasticsearch"]["port"]))
es_scheme = config["elasticsearch"]["scheme"]
es_username = config["elasticsearch"]["username"]
es_password = config["elasticsearch"]["password"]

if es_username != "" and es_password != "":
    es = Elasticsearch([{'host': es_host, 'port': es_port, 'scheme': es_scheme}],
                       basic_auth=(es_username, es_password)
                       )
else:
    es = Elasticsearch([{'host': es_host, 'port': es_port, 'scheme': es_scheme}],)

embedding_dims = config["models"]["embedding_model"][
    config["rag"]["embedding_model"]
]["dims"]

def init_es():
    """
    检查es环境配置
    :return: 环境是否配置成功
    """
    # Docker 环境下等待 ES 就绪（最多重试 30 次，每次 2 秒）
    for i in range(30):
        if es.ping():
            break
        logger.info("等待 Elasticsearch... (%d/30)", i + 1)
        time.sleep(2)
    else:
        logger.error("无法连接 Elasticsearch")
        return False

    document_meta_mapping = {
        "mappings": {
            'properties': {
                'document_id': {'type': 'integer'},
                'knowledge_id': {'type': 'integer'},
                'owner_id': {'type': 'integer'},
                'department_id': {'type': 'integer'},
                'document_name': {
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                },
                'file_path': {'type': 'keyword'},
                'abstract': {
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word',
                }
            }
        }
    }
    try:
        if not es.indices.exists(index="document_meta"):
            es.indices.create(index="document_meta", body=document_meta_mapping)
    except Exception:
        logger.exception("创建 document_meta 索引失败")
        return False

    chunk_info_mapping = {
        'mappings': {
            'properties': {
                'document_id': {'type': 'integer'},
                'knowledge_id': {'type': 'integer'},
                'owner_id': {'type': 'integer'},
                'department_id': {'type': 'integer'},
                'page_number': {'type': 'integer'},
                'chunk_id': {'type': 'integer'},
                'chunk_content': {
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word',
                },
                'chunk_images': {'type': 'keyword'},
                'chunk_tables': {'type': 'keyword'},
                'embedding_vector': {
                    'type': 'dense_vector',
                    'element_type': "float",
                    'dims': embedding_dims,
                    'index': True,
                    'index_options': {
                        'type': 'int8_hnsw'
                    }
                }
            }
        }
    }

    try:
        if not es.indices.exists(index="chunk_info"):
            es.indices.create(index="chunk_info", body=chunk_info_mapping)
    except Exception:
        logger.exception("创建 chunk_info 索引失败")
        return False
    logger.info("成功连接 Elasticsearch")
    return True
init_es()

# resp: ES的delete_by_query操作返回的响应字典
# "version_conflicts": 3,  // 版本冲突的文档数   "failures":  // 失败详情
def _assert_no_delete_failures(index_name: str, resp: dict) -> None:
    """delete_by_query 部分失败（版本冲突/失败项）时抛异常，避免静默残留幽灵分块。"""
    if resp.get("version_conflicts") or resp.get("failures"):
        raise RuntimeError(
            f"{index_name} 删除部分失败: version_conflicts={resp.get('version_conflicts')}, "
            f"failures={resp.get('failures')}"
        )


@with_retry(max_retries=3, base_delay=1)
def delete_document_chunks(document_id: int) -> int:
    """删除一个文档在 ES 的全部数据（分块 + 摘要）。失败抛异常，调用方阻断。"""
    resp1 = es.delete_by_query(
        index="chunk_info",
        query={"term": {"document_id": document_id}},
        refresh=True,
    )
    _assert_no_delete_failures("chunk_info", resp1)
    resp2 = es.delete_by_query(
        index="document_meta",
        query={"term": {"document_id": document_id}},
        refresh=True,
    )
    _assert_no_delete_failures("document_meta", resp2) # 检查是否完全删除
    deleted = int(resp1["deleted"]) + int(resp2["deleted"])
    logger.info("已从 ES 删除 document_id=%d 共 %d 条", document_id, deleted)
    return deleted


@with_retry(max_retries=3, base_delay=1)
def delete_knowledge_chunks(knowledge_id: int) -> int:
    """删除一个知识库在 ES 的全部数据（按 knowledge_id 一把清）。失败抛异常。"""
    resp1 = es.delete_by_query(
        index="chunk_info",
        query={"term": {"knowledge_id": knowledge_id}},
        refresh=True,
    )
    _assert_no_delete_failures("chunk_info", resp1)
    resp2 = es.delete_by_query(
        index="document_meta",
        query={"term": {"knowledge_id": knowledge_id}},
        refresh=True,
    )
    _assert_no_delete_failures("document_meta", resp2)
    deleted = int(resp1["deleted"]) + int(resp2["deleted"])
    logger.info("已从 ES 删除 knowledge_id=%d 共 %d 条", knowledge_id, deleted)
    return deleted


# 重新启动
#cd D:\elasticsearch-9.3.1-windows-x86_64\elasticsearch-9.3.1
#bin\elasticsearch.bat

