import yaml
import re
from typing import Union, List, Any, Dict, Optional
import numpy as np
import datetime
import time
import pdfplumber
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from es_api import es, delete_document_chunks
import os
from concurrent.futures import ThreadPoolExecutor
from utils import with_retry
import logging

logger = logging.getLogger(__name__)

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
    logger.info("加载 embedding 模型: %s", model_name)
    load_embedding_model(model_name, model_path)
if config["rag"]["use_rerank"]:
    model_name = config["rag"]["rerank_model"]
    model_path = config["models"]["rerank_model"][model_name]["local_url"]
    logger.info("加载 rerank 模型: %s", model_name)
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


def _format_related_document(records) -> str:
    """把检索结果拼成喂给 LLM 的资料。

    只拼 chunk_content 会让表格只参与召回、不参与生成；这里把 chunk_tables 一并拼入，
    保证表格问题能拿到表格原文。
    """
    parts = []
    for x in records:
        content = (x.get("chunk_content") or [""])[0]
        parts.append(content)
        for table in (x.get("chunk_tables") or []):
            parts.append(table)
    return "\n".join(parts)


_SENTENCE_END_PUNCT = set('。！？.!?')
_SENTENCE_END_RE = re.compile('[。！？.!?]')


def split_incomplete_sentence(text):
    """检测文本末尾句子是否被切断，返回 (完整部分, 尾部不完整部分)。

    原实现切开时只查中文标点，英文句号被忽略；这里统一按中英文结束标点切。
    """
    stripped = (text or "").strip()
    if not stripped or stripped[-1] in _SENTENCE_END_PUNCT:
        return stripped, ""
    matches = [m for m in _SENTENCE_END_RE.finditer(stripped) if m.start() > 0]
    if not matches:
        return stripped, ""
    last = matches[-1]
    return stripped[:last.end()], stripped[last.end():]


# 触发改写的标记：指代词 / 口语词 / 口语语气词
# （“其它”里的“它”不算指代，用负向断言排除）
_NEEDS_REWRITE_RE = re.compile(
    r"(?<!其)它|这|那|这些|那些|它们|这个|那个|"
    r"咋|啥|搞|弄|呗|啦|来着|啥子|咋样|咋整|玩意儿|再说|再讲"
)
# 极短问题往往是省略句（如“性能呢？”），也触发改写
_SHORT_QUERY_TRIGGER_LEN = 4

# 改写输出里常见的前缀/引号，需要清洗
_REWRITE_PREFIX_RE = re.compile(r"^(改写结果|改写|结果|答)[：:\s]*")


def _needs_rewrite(query: str) -> bool:
    """判断问题是否需要改写：指代 / 口语化 / 极短省略句。"""
    q = (query or "").strip()
    if not q:
        return False
    if _NEEDS_REWRITE_RE.search(q):
        return True
    return len(q) <= _SHORT_QUERY_TRIGGER_LEN


def _extract_recent_turns(history: List[Dict], max_turns: int = 2):
    """从历史里取最近 max_turns 轮一问一答，按时间正序返回 [(user, assistant), ...]。

    约定：本系统把助手回答以 role=system 存进历史，user 为用户消息。
    按消息类型收集，而不是依赖固定下标，对消息顺序更稳健。
    """
    turns = []
    assistant_msg = ""
    for msg in reversed(history):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            assistant_msg = content
        elif role == "user":
            turns.append((content, assistant_msg))
            assistant_msg = ""
            if len(turns) >= max_turns:
                break
    turns.reverse()
    return turns


def _clean_rewritten(text, original: str) -> str:
    """清洗改写输出：去前缀/引号；为空或异常长则回退原问题。"""
    out = (text or "").strip()
    out = _REWRITE_PREFIX_RE.sub("", out).strip()
    out = out.strip('"\'“”‘’「」')
    if not out or len(out) > 300:
        return original
    return out


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
        # 最终进 Prompt 的条数：召回/重排阶段保留 chunk_candidate 条保证召回率，
        # 最后一步截断到 rerank_top_k 条，控制噪音与上下文长度。
        self.rerank_top_k = config["rag"].get("rerank_top_k", 5)

        self.client = OpenAI(
            api_key=config["rag"]["llm_api_key"],
            base_url=config["rag"]["llm_base"],
        )
        self.llm_model = config["rag"]["llm_model"]

    def _extract_pdf_content(self, knowledge_id, document_id, title, file_path,
                             owner_id: int = 0, department_id: int = 0) -> bool:
        try:
            pdf = pdfplumber.open(file_path)
        except Exception:
            logger.warning("打开 PDF 失败: %s", file_path)
            raise
        logger.info("PDF %s 共 %d 页", file_path, len(pdf.pages))

        try:
            # 幂等写入第一步：解析前先清掉该文档残留的分块/摘要。
            # 配合下方“确定性 _id 覆盖写”，即使解析中途失败被 @with_retry 重试，
            # 每次尝试都从干净状态开始，不会产生重复分块。
            delete_document_chunks(document_id)

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

                # 检测本页末尾是否是不完整句子（中英文标点统一处理）
                current_page_text, tail = split_incomplete_sentence(current_page_text)
                if tail:
                    prev_page_tail = tail

                # 整页向量（包含表格内容）
                page_text_with_table = current_page_text + table_content
                embedding_vector = self.get_embedding(page_text_with_table)
                page_data = {
                    "document_id": document_id,
                    "knowledge_id": knowledge_id,
                    "owner_id": owner_id,
                    "department_id": department_id,
                    "page_number": page_number,
                    "chunk_id": 0,
                    "chunk_content": current_page_text,
                    "chunk_images": [],
                    "chunk_tables": [table_content] if table_content else [],
                    "embedding_vector": [float(x) for x in list(embedding_vector)]
                }
                es.index(
                    index="chunk_info",
                    id=f"doc_{document_id}_page_{page_number}",
                    document=page_data,
                )

                # 分块向量
                page_chunks = split_text_with_overlap(current_page_text, self.chunk_size, self.chunk_overlap)
                embedding_vector = self.get_embedding(page_chunks)  # 批量生成向量
                for chunk_idx in range(1, len(page_chunks)+1):
                    page_data = {
                        "document_id": document_id,
                        "knowledge_id": knowledge_id,
                        "owner_id": owner_id,
                        "department_id": department_id,
                        "page_number": page_number,
                        "chunk_id": chunk_idx,
                        "chunk_content": page_chunks[chunk_idx - 1],
                        "chunk_images": [],
                        "chunk_tables": [],
                        "embedding_vector": [float(x) for x in list(embedding_vector[chunk_idx - 1])]

                    }
                    es.index(
                        index="chunk_info",
                        id=f"doc_{document_id}_page_{page_number}_chunk_{chunk_idx}",
                        document=page_data,
                    )

            document_data = {
                "document_id": document_id,
                "knowledge_id": knowledge_id,
                "owner_id": owner_id,
                "department_id": department_id,
                "document_name": title,
                "file_path": file_path,
                "abstract": abstract,
            }
            es.index(
                index="document_meta",
                id=f"docmeta_{document_id}",
                document=document_data,
            )
            return True
        finally:
            pdf.close()

    def _extract_word_content(self):
        pass

    @with_retry(max_retries=3, base_delay=2)
    def extract_content(self, knowledge_id, document_id, title, file_type, file_path,
                        owner_id: int = 0, department_id: int = 0):
        # 类型白名单：只处理明确支持的 PDF；Word/未知类型直接置为失败，
        # 绝不静默“成功”（以前 Word 走到 pass 后照样上报 completed）。
        # 判断依据优先看服务端可控的文件扩展名，不轻信客户端自报的 content_type。
        is_pdf = ("pdf" in (file_type or "")) or str(file_path or "").lower().endswith(".pdf")
        if not is_pdf:
            logger.warning("不支持的文件类型，文档解析失败: document_id=%s file_type=%s", document_id, file_type)
            return

        try:
            self._extract_pdf_content(knowledge_id, document_id, title, file_path,
                                      owner_id=owner_id, department_id=department_id)
            logger.info("文档提取完成 document_id=%s file_type=%s path=%s", document_id, file_type, file_path)
        except Exception as e:
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

    def _word_search(self, rewritten_query, knowledge_id, department_id: Optional[int] = None):
        """第一路：关键词检索"""
        filters = [{"term": {"knowledge_id": knowledge_id}}]
        if department_id is not None:
            # 租户隔离：只召回本部门的分块（admin/system 传 None 表示不限制）
            filters.append({"term": {"department_id": department_id}})
        return es.search(index="chunk_info",
                         body={
                             "query": {
                                 "bool": {
                                     "must": [{"match": {"chunk_content": rewritten_query}}],
                                     "filter": filters,
                                 }
                             },
                             "size": 50
                         },
                         fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content", "chunk_tables"],
                         source=False)

    def _vector_search(self, embedding_vector, knowledge_id, department_id: Optional[int] = None):
        """第二路：向量检索"""
        filters = [{"term": {"knowledge_id": knowledge_id}}]
        if department_id is not None:
            filters.append({"term": {"department_id": department_id}})
        knn_query = {
            "field": "embedding_vector",
            "query_vector": [float(x) for x in list(embedding_vector)],
            "k": 50,
            "num_candidates": 100,
            "filter": {
                "bool": {
                    "must": filters
                }
            }
        }
        return es.search(index="chunk_info", knn=knn_query,
                         fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content", "chunk_tables"],
                         source=False)

    def query_document(self, query: str, knowledge_id: int, history: List[Dict] = None,
                       department_id: Optional[int] = None) -> List[str]:
        # Query改写：消除指代词，完整表达（传入历史帮助理解指代）
        t0 = time.monotonic()
        rewritten_query = self.query_rewrite(query, history)
        t1 = time.monotonic()

        # 向量化（串行在并行之前，因为向量检索依赖结果）
        embedding_vector = self.get_embedding(rewritten_query)
        t2 = time.monotonic()

        # 双路并行召回
        with ThreadPoolExecutor(max_workers=2) as executor:
            word_future = executor.submit(self._word_search, rewritten_query, knowledge_id, department_id)
            vector_future = executor.submit(self._vector_search, embedding_vector, knowledge_id, department_id)
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
                search_id2record[_id] = {**record["fields"], "_id": _id}

        for idx, record in enumerate(vector_search_response["hits"]["hits"]):
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)  # 两路分数叠加
            else:
                fusion_score[_id] += 1 / (idx + k)
            if _id not in search_id2record:
                search_id2record[_id] = {**record["fields"], "_id": _id}  # 附带 _id，便于评估时定位答案块

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
        t3 = time.monotonic()

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
            # sorted_content = [sorted_content[i] for i in rerank_idx]
            debug_chunks = [debug_chunks[i] for i in rerank_idx]
            for i, idx in enumerate(rerank_idx):
                debug_chunks[i]["rerank_score"] = round(float(rerank_score[idx]), 4)

        # 最终只保留前 rerank_top_k 条：重排（或 RRF）后的排序已确定，多余的候选
        # 大概率是噪音，塞进 Prompt 只会干扰 LLM，还拉长生成时间。
        sorted_records = sorted_records[:self.rerank_top_k]
        # sorted_content = sorted_content[:self.rerank_top_k]
        debug_chunks = debug_chunks[:self.rerank_top_k]

        t4 = time.monotonic()
        logger.info(
            "[RAG] 改写=%.3fs 向量=%.3fs 召回=%.3fs 重排=%.3fs",
            t1 - t0, t2 - t1, t3 - t2, t4 - t3,
        )

        self._last_debug_info = {"rewritten_query": rewritten_query, "chunks": debug_chunks}
        return sorted_records

    def _retrieve_context(self, knowledge_id: int, message: List[Dict],
                          department_id: Optional[int] = None):
        """检索 + 组装 LLM 消息，返回 (llm_messages, debug_info)。

        单轮: llm_messages = [system(资料+问题)]
        多轮: llm_messages = 历史 + system(资料) + 最后一问
        """
        logger.info("[chat] 收到 %d 条消息", len(message))
        if len(message) == 1:
            query = message[0]["content"]
            history = None
        else:
            query = message[-1]["content"]
            history = message[:-1]  # 传入历史帮助 query_rewrite 理解指代
            logger.info("[chat] 历史 %d 条", len(history))

        related_records = self.query_document(query, knowledge_id, history, department_id=department_id)
        debug_info = getattr(self, '_last_debug_info', None) or {}
        logger.info("[RAG] 检索到 %d 条记录", len(related_records))
        related_document = _format_related_document(related_records)

        rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
                      .replace("{#QUESTION#}", query) \
                      .replace("{#RELATED_DOCUMENT#}", related_document)

        if len(message) == 1:
            llm_messages = [{"role": "system", "content": rag_query}]
        else:
            # 将 RAG 上下文插入到历史消息中，位于最后一条用户消息之前
            rag_context_message = {"role": "system", "content": rag_query}
            llm_messages = message[:-1] + [rag_context_message, message[-1]]
        return llm_messages, debug_info

    def chat_with_rag(self, knowledge_id: int, message: List[Dict], department_id: Optional[int] = None):
        """非流式对话（保留，供 benchmark 直接调用）。"""
        llm_messages, debug_info = self._retrieve_context(knowledge_id, message, department_id=department_id)
        rag_response = self.chat(llm_messages, 0.7, 0.9).content
        message.append({"role": "system", "content": rag_response})
        return message, debug_info

    def _stream_tokens(self, llm_messages: List[Dict], message: List[Dict]):
        """LLM 流式生成，yield (event, data)：token / error / done。"""
        full_response = ""
        t_gen = time.monotonic()
        try:
            stream = self.client.chat.completions.create(
                model=self.llm_model,
                messages=llm_messages,
                top_p=0.7,
                temperature=0.9,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = (delta and delta.content) or ""
                if content:
                    full_response += content
                    yield ("token", content)
        except Exception as e:
            logger.exception("[chat] 流式生成失败")
            yield ("error", {"message": str(e)})
            return
        logger.info("[chat] 流式生成完成: %d tokens, 生成=%.3fs", len(full_response), time.monotonic() - t_gen)
        message.append({"role": "system", "content": full_response})
        yield ("done", {"message": message})

    def chat_stream(self, knowledge_id: int, message: List[Dict], department_id: Optional[int] = None):
        """流式对话入口，yield (event, data)：debug / token / error / done。"""
        llm_messages, debug_info = self._retrieve_context(knowledge_id, message, department_id=department_id)
        yield ("debug", debug_info)
        yield from self._stream_tokens(llm_messages, message)




    def chat(self, message: List[Dict], top_p: float, temperature: float, timeout: float = None) -> Any:
        kwargs = dict(
            model=self.llm_model,
            messages=message,
            top_p=top_p,
            temperature=temperature,
        )
        if timeout is not None:
            kwargs["timeout"] = timeout
        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message

    def query_parse(self, query: str) -> str:
        return ""

    def query_rewrite(self, query: str, history: List[Dict] = None) -> str:
        logger.debug("query_rewrite query=%s history=%d条", query, len(history) if history else 0)

        # 没有历史对话时，问题没有指代需要消除，直接用原问题
        if not history or len(history) < 2:
            logger.debug("无历史对话，使用原问题: %s", query)
            return query

        # 规则先筛：没有指代/口语标记、且长度足够的完整问题，不值得花一次 LLM 调用
        if not _needs_rewrite(query):
            logger.debug("问题完整，跳过改写: %s", query)
            return query

        # 取最近 2 轮一问一答作为上下文（按消息类型收集，而非依赖固定下标）
        turns = _extract_recent_turns(history, max_turns=2)
        history_text = "\n".join(
            f"用户：{user_msg}\n助手：{assistant_msg[:200]}"
            for user_msg, assistant_msg in turns
        )

        prompt = f"""你只做问题改写，不要回答问题。

把用户的问题改写成适合检索的完整书面语，要求：
1. 消除指代词（它、这、那、它们等），把省略的部分补充完整；
2. 口语化表达改写成正式书面语（例如“咋搞的”→“是如何实现的”）；
3. 如果问题已经完整且书面，原样输出。

只输出改写后的句子，不要任何解释。

示例：
问：它是什么？
答：RAG是什么？

问：它咋搞的？
答：RAG是如何实现的？

对话历史：
{history_text}

当前问题：{query}
改写结果："""
        logger.debug("发给 LLM 的 prompt 长度: %d 字符", len(prompt))
        try:
            response = self.chat(
                [{"role": "user", "content": prompt}],
                0.3, 0.5,
                timeout=10,
            )
            rewritten = _clean_rewritten(response.content, query)
            logger.info("[Query改写] 原始=%s -> 改写=%s", query, rewritten)
            return rewritten
        except Exception as e:
            logger.warning("Query改写失败，使用原问题: %s", e)
            return query





