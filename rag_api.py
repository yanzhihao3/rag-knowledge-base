import yaml
from typing import Union, List, Any, Dict
import numpy as np
import datetime
import pdfplumber
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from es_api import es
import os
from concurrent.futures import ThreadPoolExecutor
from utils import with_retry, TaskStateMachine, task_state_machine

#project_root = os.path.dirname(os.path.abspath(__file__))
#config_path = os.path.join(project_root, 'config.yaml')

with open("config.yaml", 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

device = config['device']
EMBEDDING_MODEL_PARAMS: Dict[Any, Any] = {}




BASIC_QA_TEMPLATE = '''现在的时间{#TIME#}. 你是一个专家，你擅长回答用户提问 ，帮我结合给定的资料，回答下面的问题
如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
如果问题可以从资料中获得，则请逐步回答

资料：
{#RELATED_DOCUMENT#}

问题: {#QUESTION#}
'''

def load_embedding_model(model_name: str, model_path: str) -> None:
    """
    加载编码模型
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return: None
    """
    global EMBEDDING_MODEL_PARAMS
    if model_name in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
        EMBEDDING_MODEL_PARAMS["embedding_model"] = SentenceTransformer(model_path)

def load_rerank_model(model_name: str, model_path: str) -> None:
    """
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return: None
    """
    global EMBEDDING_MODEL_PARAMS
    if model_name in ["bge-reranker-base"]:
        EMBEDDING_MODEL_PARAMS["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_model"].eval()
        EMBEDDING_MODEL_PARAMS["rerank_model"].to(device)
if config["rag"]["use_embedding"]:
    model_name = config["rag"]["embedding_model"]
    model_path = config["models"]["embedding_model"][model_name]["local_url"]
    print(f"Loading embedding model {model_name} from model_path...")
    load_embedding_model(model_name, model_path)
if config["rag"]["use_rerank"]:
    model_name = config["rag"]["rerank_model"]
    model_path = config["models"]["rerank_model"][model_name]["local_url"]
    print(f"Loading rerank model {model_name} from model_path...")
    load_rerank_model(model_name, model_path)

def split_text_with_overlap(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + chunk_size - chunk_overlap
    return chunks

class RAG:
    def __init__(self):
        self.embedding_model = config["rag"]["embedding_model"]
        self.rerank_model = config["rag"]["rerank_model"]
        self.use_rerank = config["rag"]["use_rerank"]
        self.embedding_dims = config["models"]["embedding_model"][
            config["rag"]["embedding_model"]
        ]["dims"]
        self.chunk_size = config["rag"]["chunk_size"]
        self.chunk_overlap = config["rag"]["chunk_overlap"]
        self.chunk_candidate = config["rag"]["chunk_candidate"]

        self.client = OpenAI(
            api_key=config["rag"]["llm_api_key"],
            base_url=config["rag"]["llm_base"],
        )
        self.llm_model = config["rag"]["llm_model"]

    def _extract_pdf_content(self, knowledge_id, document_id, title, file_path) -> bool:
        try:
            pdf = pdfplumber.open(file_path)
        except:
            print("打开失败！")
            return False
        print(f"{file_path} pages:", len(pdf.pages))

        abstract = ""
        prev_page_tail = ""  # 跨页断句：保存上页末尾的不完整句子

        for page_number in range(len(pdf.pages)):
            page = pdf.pages[page_number]

            # === 1. 提取表格 ===
            tables = page.extract_tables()
            table_content = ""
            if tables:
                for table_idx, table in enumerate(tables):
                    if table and any(table):
                        # 将表格转换为字符串格式存储
                        table_rows = []
                        for row in table:
                            if row:
                                cleaned_row = [str(cell) if cell else "" for cell in row]
                                table_rows.append(" | ".join(cleaned_row))
                        if table_rows:
                            table_content += "\n[表格]\n" + "\n".join(table_rows) + "\n[/表格]\n"

            # === 2. 提取文本 ===
            current_page_text = page.extract_text()
            if page_number <= 3:
                abstract = abstract + '\n' + current_page_text

            # === 3. 跨页断句处理 ===
            if prev_page_tail:
                # 把上页末尾的不完整句子拼到这页开头
                current_page_text = prev_page_tail + current_page_text
                prev_page_tail = ""

            # 检测本页末尾是否是不完整句子
            stripped = current_page_text.strip()
            if stripped and stripped[-1] not in ('。', '！', '？', '.', '!', '?'):
                # 句子被切断，找到最后一个完整句子之前的位置
                last_punct = max(
                    stripped.rfind('。') if stripped.rfind('。') > 0 else -1,
                    stripped.rfind('！') if stripped.rfind('！') > 0 else -1,
                    stripped.rfind('？') if stripped.rfind('？') > 0 else -1,
                )
                if last_punct > 0:
                    prev_page_tail = stripped[last_punct + 1:]
                    current_page_text = stripped[:last_punct + 1]

            # 整页向量（包含表格内容）
            page_text_with_table = current_page_text + table_content
            embedding_vector = self.get_embedding(page_text_with_table)
            page_data = {
                "document_id": document_id,
                "knowledge_id": knowledge_id,
                "owner_id": 0,
                "department_id": 0,
                "page_number": page_number,
                "chunk_id": 0,
                "chunk_content": current_page_text,
                "chunk_images": [],
                "chunk_tables": [table_content] if table_content else [],
                "embedding_vector": [float(x) for x in list(embedding_vector)]
            }
            response = es.index(index="chunk_info", document=page_data)

            # 分块向量
            page_chunks = split_text_with_overlap(current_page_text, self.chunk_size, self.chunk_overlap)
            embedding_vector = self.get_embedding(page_chunks)  # 批量生成向量
            for chunk_idx in range(1, len(page_chunks)+1):
                page_data = {
                    "document_id": document_id,
                    "knowledge_id": knowledge_id,
                    "owner_id": 0,
                    "department_id": 0,
                    "page_number": page_number,
                    "chunk_id": chunk_idx,
                    "chunk_content": page_chunks[chunk_idx - 1],
                    "chunk_images": [],
                    "chunk_tables": [],
                    "embedding_vector": [float(x) for x in list(embedding_vector[chunk_idx - 1])]

                }
                response = es.index(index="chunk_info", document=page_data)

        document_data = {
            "document_id": document_id,
            "knowledge_id": knowledge_id,
            "owner_id": 0,
            "department_id": 0,
            "document_name": title,
            "file_path": file_path,
            "abstract": abstract,
        }
        response = es.index(index="document_meta", document=document_data)

    def _extract_word_content(self):
        pass

    @with_retry(max_retries=3, base_delay=2)
    def extract_content(self, knowledge_id, document_id, title, file_type, file_path):
        doc_id_str = str(document_id)
        task_state_machine.set_state(doc_id_str, TaskStateMachine.STATE_PROCESSING)

        try:
            if "pdf" in file_type:
                self._extract_pdf_content(knowledge_id, document_id, title, file_path)
            elif "word" in file_type:
                pass
            task_state_machine.set_state(doc_id_str, TaskStateMachine.STATE_COMPLETED)
            print("提取完成", document_id, file_type, file_path)
        except Exception as e:
            task_state_machine.set_state(doc_id_str, TaskStateMachine.STATE_FAILED)
            raise e

# {
    #   "text": "提示词工程",
    #   "token": "1",
    #   "model": "bge-small-zh-v1.5"
    # }
    def get_embedding(self, text) -> np.ndarray:
        """
        对文本进行编码
        :param text: 待便秘文本
        :return: 编码结果
        """
        if self.embedding_model in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
            return EMBEDDING_MODEL_PARAMS["embedding_model"].encode(text, normalize_embeddings=True)
        raise NotImplemented
# {
    #     "text_pair": [
    #         ["什么是RAG？", "RAG是检索增强生成技术"],
    #         ["什么是RAG？", "今天天气很好"],
    #         ["什么是RAG？", "RAG结合了向量搜索和LLM生成"]
    #     ],
    #     "token": "123",
    #     "model": "bge-reranker-base"
    # }  这是测试格式
    def get_rerank(self, text_pair):
        """
        对文本进行重排序
        :param text_pair: 待排序文本
        :return: 匹配打分结果
        """
        if self.rerank_model in ["bge-reranker-base"]:
            with torch.no_grad():
                inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                    text_pair, padding=True, return_tensors="pt",
                    truncation=True, max_length=512,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                scores = EMBEDDING_MODEL_PARAMS["rerank_model"](**inputs, return_dict=True).logits.view(-1, ).float()
                scores = scores.data.cpu().numpy()
                return scores
        raise NotImplemented

    def _word_search(self, rewritten_query, knowledge_id):
        """第一路：关键词检索"""
        return es.search(index="chunk_info",
                         body={
                             "query": {
                                 "bool": {
                                     "must": [{"match": {"chunk_content": rewritten_query}}],
                                     "filter": [
                                         {"term": {"knowledge_id": knowledge_id}},
                                     ]
                                 }
                             },
                             "size": 50
                         },
                         fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content"],
                         source=False)

    def _vector_search(self, embedding_vector, knowledge_id):
        """第二路：向量检索"""
        knn_query = {
            "field": "embedding_vector",
            "query_vector": [float(x) for x in list(embedding_vector)],
            "k": 50,
            "num_candidates": 100,
            "filter": {
                "bool": {
                    "must": [
                        {"term": {"knowledge_id": knowledge_id}},
                    ]
                }
            }
        }
        return es.search(index="chunk_info", knn=knn_query,
                         fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content"],
                         source=False)

    def query_document(self, query: str, knowledge_id: int, history: List[Dict] = None) -> List[str]:
        # Query改写：消除指代词，完整表达（传入历史帮助理解指代）
        rewritten_query = self.query_rewrite(query, history)

        # 向量化（串行在并行之前，因为向量检索依赖结果）
        embedding_vector = self.get_embedding(rewritten_query)

        # 双路并行召回
        with ThreadPoolExecutor(max_workers=2) as executor:
            word_future = executor.submit(self._word_search, rewritten_query, knowledge_id)
            vector_future = executor.submit(self._vector_search, embedding_vector, knowledge_id)
            word_search_response = word_future.result()
            vector_search_response = vector_future.result()
        # ===== RRF融合 =====
        k = 60
        fusion_score = {}
        search_id2record = {}
        # 两路各取前50条，按排名给分
        for idx, record in enumerate(word_search_response["hits"]["hits"]):
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k) # 排名越靠前分数越高
            else:
                fusion_score[_id] += 1 / (idx + k)
            if _id not in search_id2record:
                search_id2record[_id] = record["fields"]

        for idx, record in enumerate(vector_search_response["hits"]["hits"]):
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)  # 两路分数叠加
            else:
                fusion_score[_id] += 1 / (idx + k)
            if _id not in search_id2record:
                search_id2record[_id] = record["fields"] # 把这个文档块的所有字段存下来，以它的 _id 为 key。

        # 按总分排序，取前chunk_candidate条（config里是10条）
        sorted_dict = sorted(fusion_score.items(), key=lambda kv: kv[1], reverse=True)
        sorted_records = [search_id2record[x[0]] for x in sorted_dict][:self.chunk_candidate]
        sorted_content = [x["chunk_content"] for x in sorted_records]

        # 构造 debug 信息
        debug_chunks = []
        for _id, score in sorted_dict[:self.chunk_candidate]:
            record = search_id2record[_id]
            content = (record.get("chunk_content") or [""])[0]
            page = record.get("page_number")
            page = page[0] if isinstance(page, list) else (page or 0)
            doc_id = record.get("document_id")
            doc_id = doc_id[0] if isinstance(doc_id, list) else (doc_id or 0)
            debug_chunks.append({
                "content": content[:200],
                "page_number": page,
                "document_id": doc_id,
                "rrf_score": round(float(score), 4),
                "rerank_score": None,
            })

        if self.use_rerank:
            test_pair = []
            for chunk_content in sorted_content:
                text = chunk_content[0] if isinstance(chunk_content, list) else chunk_content
                test_pair.append([rewritten_query, text])
            if not sorted_content:
                self._last_debug_info = {"rewritten_query": rewritten_query, "chunks": debug_chunks}
                return sorted_records

            rerank_score = self.get_rerank(test_pair)
            rerank_idx = np.argsort(rerank_score)[::-1]

            sorted_records = [sorted_records[i] for i in rerank_idx]
            sorted_content = [sorted_content[i] for i in rerank_idx]
            debug_chunks = [debug_chunks[i] for i in rerank_idx]
            for i, idx in enumerate(rerank_idx):
                debug_chunks[i]["rerank_score"] = round(float(rerank_score[idx]), 4)

        self._last_debug_info = {"rewritten_query": rewritten_query, "chunks": debug_chunks}
        return sorted_records

    def chat_with_rag(self, knowledge_id: int, message:List[Dict]):
        print(f"[DEBUG chat_with_rag] message length: {len(message)}")
        print(f"[DEBUG chat_with_rag] message: {message}")

        if len(message) == 1:
            query = message[0]["content"]
            related_records = self.query_document(query, knowledge_id, None)
            debug_info = getattr(self, '_last_debug_info', None)
            try:
                print(related_records)
            except UnicodeEncodeError:
                print(f"[DEBUG] {len(related_records)} records retrieved")
            related_document = '\n'.join([x["chunk_content"][0] for x in related_records])

            rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
                          .replace("{#QUESTION#}", query) \
                          .replace("{#RELATED_DOCUMENT#}", related_document)
            rag_response = self.chat(
                [{"role": "system", "content": rag_query}],
                0.7, 0.9
            ).content
            message.append({"role": "system", "content": rag_response})
        else:
            # 多轮对话：从历史消息中提取最新用户问题，做 RAG 检索
            query = message[-1]["content"]
            history = message[:-1]  # 传入历史帮助 query_rewrite 理解指代
            print(f"[DEBUG chat_with_rag] history length: {len(history)}")
            print(f"[DEBUG chat_with_rag] history: {history}")
            related_records = self.query_document(query, knowledge_id, history)
            debug_info = getattr(self, '_last_debug_info', None)
            related_document = '\n'.join([x["chunk_content"][0] for x in related_records])

            rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
                          .replace("{#QUESTION#}", query) \
                          .replace("{#RELATED_DOCUMENT#}", related_document)

            # 将 RAG 上下文插入到历史消息中，位于最后一条用户消息之前
            rag_context_message = {"role": "system", "content": rag_query}
            messages_with_context = message[:-1] + [rag_context_message, message[-1]]

            normal_response = self.chat(
                messages_with_context,
                0.7, 0.9
            ).content
            message.append({"role": "system", "content": normal_response})

        return message, debug_info or {}




    def chat(self, message: List[Dict], top_p: float, temperature: float) -> Any:
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=message,
            top_p=top_p,
            temperature=temperature,
        )
        return completion.choices[0].message

    def query_parse(self, query: str) -> str:
        return ""

    def query_rewrite(self, query: str, history: List[Dict] = None) -> str:
        print(f"[DEBUG] query_rewrite called with query={query}, history={history}")

        # 没有历史对话时，问题没有指代需要消除，直接用原问题
        if not history or len(history) < 2:
            print(f"[Query改写] 无历史对话，使用原问题: {query}")
            return query

        # 提取上一轮对话内容作为上下文
        last_user_msg = history[-2].get("content", "") if len(history) >= 2 else ""
        last_ai_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "system":
                last_ai_msg = msg.get("content", "")
                break

        prompt = f"""你只做问题改写，不要回答问题。

把用户的问题改写成完整表达，消除指代词（它、这、那、它们等）。
如果有省略部分，补充完整。如果问题已经完整，原样输出。

只输出改写后的句子，不要任何解释。

示例：
问：它是什么？
答：RAG是什么？

问：它怎么实现的？
答：RAG怎么实现的？

对话历史：
用户：{last_user_msg}
助手：{last_ai_msg[:200]}

当前问题：{query}
改写结果："""
        print(f"[DEBUG] prompt sent to LLM:\n{prompt}")
        try:
            response = self.chat(
                [{"role": "user", "content": prompt}],
                0.3, 0.5
            )
            rewritten = response.content.strip()
            print(f"[Query改写] 原始问题: {query}")
            print(f"[Query改写] 改写后: {rewritten}")
            return rewritten if rewritten else query
        except Exception as e:
            print(f"Query改写失败，使用原问题: {e}")
            return query





