import yaml
from elasticsearch import Elasticsearch
import traceback
import os

#project_root = os.path.dirname(os.path.abspath(__file__))
#config_path = os.path.join(project_root, 'config.yaml')
with open("config.yaml", 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

es_host = config["elasticsearch"]["host"]
es_port = config["elasticsearch"]["port"]
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
    if not es.ping():
        print("Could not connect to Elasticsearch")
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
    except:
        print(traceback.format_exc())
        print("Could not create index")
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
    except:
        print(traceback.format_exc())
        print("Could not create index of chunk_info")
        return False
    print("Successfully connected to Elasticsearch")
    return True
init_es()


# 重新启动
#cd D:\elasticsearch-9.3.1-windows-x86_64\elasticsearch-9.3.1
#bin\elasticsearch.bat

