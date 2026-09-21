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


class DocumentParseError(Exception):
    """文档本身无法解析（不是临时故障），重试没有意义。

    典型场景：上传的是扫描件 / 纯图片 PDF，没有文本层。
    这种情况必须明确失败，绝不能静默上报"处理成功" —— 用户看到成功却一个字
    都搜不到，比直接报错糟糕得多。
    """


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


_SENT_END_CHARS = "。！？!?；;"
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
# 结构行：章节号（3.2 / 7.3.1）、括号序号（（4）/ 3））、图表题注（图3-2 / 表4-1）、第N章。
# 这类行是独立单元：既不能被当成"上一行的折行"粘到正文里，也不能让正文粘到它后面。
_STRUCT_LINE_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.\s"                      # "2. RAG" / "7.3. 召回优化"
    r"|\d+(?:\.\d+)+\s"                       # "7.3 召回环节优化"
    r"|第[一二三四五六七八九十百\d]+[章节]"     # "第6章"
    r"|[图表]\s*\d+"                          # "图3-2" / "表4-1"
    r"|（\d+）|\d+）"                          # "（4）输出解析器" / "3）硬件加速"
    r")"
)


def _ends_sentence(text: str) -> bool:
    return bool(text) and text[-1] in _SENT_END_CHARS


def _is_structural(line: str) -> bool:
    return bool(_STRUCT_LINE_RE.match(line or ""))


def join_visual_lines(text):
    """把 pdfplumber 的"视觉行"拼回逻辑行。

    为什么需要这一步：这本 PDF 用 pdfplumber 抽出来是**每个视觉行一个 \\n**，
    平均行长只有 26~35 字，而且会在词中间断开（`一定数\\n量`、`Co-\\noccurrence`）。
    直接拿这种文本去切块，等于在一个"已经被切碎"的流上再切一次。

    拼接规则：
      - 上一行没有以句末标点结尾，且两行都不是结构行 → 行内折行，直接拼接
        （中文不补空格；英文连字符正好接回 `Co-occurrence`）；
      - 上一行以句末标点结尾 → 这是真正的语义边界，保留；
      - 任一行是章节号 / 图表题注 / 括号序号 → 强制断段，避免"7.3 标题正文..."
        这种标题粘正文的块。
    """
    lines = []
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        prev = lines[-1] if lines else ""
        if (prev and not _ends_sentence(prev)
                and not _is_structural(prev) and not _is_structural(line)):
            lines[-1] += line
        else:
            lines.append(line)
    return lines


def split_sentences(text):
    """按中英文句末标点切句，标点保留在前一句末尾；末尾没标点的残句也保留。"""
    return [p.strip() for p in _SENT_SPLIT_RE.split(text or "") if p.strip()]


def split_text_semantic(text, target_size=400, overlap_sentences=1, max_size=None):
    """语义分块：视觉行拼回 → 按句切 → 按目标长度聚合，块间用"整句"重叠。

    和 split_text_with_overlap（纯字符窗口）的区别：
      - 不在句子中间切（唯一例外：单句本身就超过 max_size，只能硬切）；
      - 重叠不是 20 个字符而是最后一整句，不会有半句话横跨两个块；
      - 块长按句子边界就近取，不追求精确等于 target_size。

    :param target_size: 目标块大小（字）。累计超过它就换块。
    :param overlap_sentences: 块间重叠的整句数，0 表示不重叠。
    :param max_size: 单块硬上限，超过它的单句会被硬切；默认 target_size * 1.5。
    """
    if max_size is None:
        max_size = int(target_size * 1.5)

    sentences = []
    for line in join_visual_lines(text):
        sentences.extend(split_sentences(line))

    chunks, buf, size = [], [], 0
    for sent in sentences:
        if len(sent) > max_size:
            # 单句超长（表格被串成一行、代码块等）：冲掉缓冲后硬切
            if buf:
                chunks.append("".join(buf))
                buf, size = [], 0
            for i in range(0, len(sent), target_size):
                chunks.append(sent[i:i + target_size])
            continue
        if buf and size + len(sent) > target_size:
            chunks.append("".join(buf))
            buf = buf[-overlap_sentences:] if overlap_sentences > 0 else []
            size = sum(len(x) for x in buf)
        buf.append(sent)
        size += len(sent)
    if buf:
        chunks.append("".join(buf))
    return chunks


def _as_text(value) -> str:
    """统一取值：ES 的 fields 返回列表（一个字段可能多值），_source 返回标量。

    这里必须两种都吃：flat 模式的记录来自 fields（chunk_content 是 ["文本"]），
    parent_child 模式的记录是我们自己拼的（chunk_content 是 "文本"）。
    历史上这里写死了 [0] —— 在 parent_child 下就变成"取第一个字符"，
    结果喂给 LLM 的资料只剩 5 个字，模型只能回"无法回答"。召回率指标完全看不出来，
    是回答准确率评测才暴露的。
    """
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def _format_related_document(records) -> str:
    """把检索结果拼成喂给 LLM 的资料。

    只拼 chunk_content 会让表格只参与召回、不参与生成；这里把 chunk_tables 一并拼入，
    保证表格问题能拿到表格原文。
    """
    parts = []
    for x in records:
        parts.append(_as_text(x.get("chunk_content")))
        for table in (x.get("chunk_tables") or []):
            parts.append(_as_text(table))
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
        # 分块策略：
        #   semantic（默认）= 视觉行拼回 + 按句边界聚合，重叠按"整句"算；
        #   fixed           = 旧的定长字符窗口，保留它只为复现旧基线（256/20）。
        self.chunk_strategy = config["rag"].get("chunk_strategy", "semantic")
        self.chunk_overlap_sentences = config["rag"].get("chunk_overlap_sentences", 1)
        # 检索模式：
        #   flat         = 页级块和子块在同一个池子里竞争（改前的行为）；
        #   parent_child = 只在子块上召回 + RRF，再按页聚合、把整页正文送去生成。
        # 允许用环境变量覆盖，方便在同一份索引上做 A/B：
        #   $env:RAG_RETRIEVAL_MODE='parent_child'
        self.retrieval_mode = (os.environ.get("RAG_RETRIEVAL_MODE")
                               or config["rag"].get("retrieval_mode", "flat"))
        self.parent_max_chars = config["rag"].get("parent_max_chars", 1200)
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
            pages_parsed = 0     # 真正写出内容的页数
            pages_skipped = 0    # 无文字且无表格、被跳过的页数

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
                # 纯图页上 extract_text() 会返回 None，必须兜底成空串，
                # 否则下面的字符串拼接会抛 TypeError（这是另一种"静默"的来源：
                # 报错信息里看不出是"没有文本层"）。
                current_page_text = page.extract_text() or ""

                # === 3. 跨页断句处理 ===
                if prev_page_tail:
                    # 把上页末尾的不完整句子拼到这页开头
                    current_page_text = prev_page_tail + current_page_text
                    prev_page_tail = ""

                # 检测本页末尾是否是不完整句子（中英文标点统一处理）
                current_page_text, tail = split_incomplete_sentence(current_page_text)
                if tail:
                    prev_page_tail = tail

                # 页级（父）块也存归一化后的文本：把 pdfplumber 的视觉行拼回逻辑行。
                # 之前只归一了子块，页级块留的是原始输出（"一定数\n量的"），带来两个问题：
                #   1. 跨行的词在页级文本里匹配不上 → 评测的"答案页"口径会低估；
                #   2. parent_child 把整页送进 Prompt 时，LLM 看到的是断行文本。
                page_text_clean = "\n".join(join_visual_lines(current_page_text))

                # === 4. 空页守卫 ===
                # 这一页既没有文本层、也没抽到表格 → 直接跳过，不写空向量。
                # 空 chunk_content 的记录照样会进索引、被 kNN 召回，纯属噪音
                # （修复前这本书里有 14 条这样的记录）。
                if not page_text_clean.strip() and not table_content.strip():
                    pages_skipped += 1
                    logger.debug("[parse] 跳过无内容页 document_id=%s page=%s",
                                 document_id, page_number)
                    continue

                pages_parsed += 1
                if page_number <= 3:
                    abstract = abstract + '\n' + page_text_clean

                # 整页向量（包含表格内容）
                page_text_with_table = page_text_clean + table_content
                embedding_vector = self.get_embedding(page_text_with_table)
                page_data = {
                    "document_id": document_id,
                    "knowledge_id": knowledge_id,
                    "owner_id": owner_id,
                    "department_id": department_id,
                    "page_number": page_number,
                    "chunk_id": 0,
                    "chunk_content": page_text_clean,
                    "chunk_images": [],
                    "chunk_tables": [table_content] if table_content else [],
                    "embedding_vector": [float(x) for x in list(embedding_vector)]
                }
                es.index(
                    index="chunk_info",
                    id=f"doc_{document_id}_page_{page_number}",
                    document=page_data,
                )

                # 分块向量（策略见 config.rag.chunk_strategy）
                if self.chunk_strategy == "semantic":
                    page_chunks = split_text_semantic(
                        page_text_clean, self.chunk_size, self.chunk_overlap_sentences)
                else:
                    page_chunks = split_text_with_overlap(
                        page_text_clean, self.chunk_size, self.chunk_overlap)
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

            # === 全篇守卫 ===
            # 一页有效内容都没有 → 这份 PDF 没有文本层（多半是扫描件）。
            # 必须明确失败：静默"成功"会让用户以为处理好了，实际一个字都搜不到。
            if pages_parsed == 0:
                raise DocumentParseError(
                    f"PDF 无可提取的文本层（共 {len(pdf.pages)} 页，"
                    f"{pages_skipped} 页既无文字也无表格）。"
                    f"可能是扫描件或纯图片 PDF：请先做 OCR，或换一份带文本层的文件。"
                )
            logger.info("[parse] document_id=%s 共 %d 页：解析 %d 页，跳过空页 %d 页",
                        document_id, len(pdf.pages), pages_parsed, pages_skipped)

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

    @with_retry(max_retries=3, base_delay=2, no_retry=(DocumentParseError,))
    def extract_content(self, knowledge_id, document_id, title, file_type, file_path,
                        owner_id: int = 0, department_id: int = 0):
        # 类型白名单：只处理明确支持的 PDF；Word/未知类型直接置为失败，
        # 绝不静默“成功”（以前 Word 走到 pass 后照样上报 completed）。
        # 判断依据优先看服务端可控的文件扩展名，不轻信客户端自报的 content_type。
        is_pdf = ("pdf" in (file_type or "")) or str(file_path or "").lower().endswith(".pdf")
        if not is_pdf:
            # 原来是 return（等于静默成功：任务照样被标成 success），
            # 现在抛出来，让 task 表如实记成 failure。
            raise DocumentParseError(
                f"不支持的文件类型: {file_type or '未知'}（当前只支持 PDF）"
            )

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
        if self.retrieval_mode == "parent_child":
            # 父子块：召回阶段只打子块，父块（页级，chunk_id=0）不参与竞争
            filters.append({"range": {"chunk_id": {"gte": 1}}})
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
        if self.retrieval_mode == "parent_child":
            filters.append({"range": {"chunk_id": {"gte": 1}}})
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

    @staticmethod
    def _first(value):
        """ES 的 fields 返回的是列表（一个字段可能有多个值），这里取第一个；标量原样返回。"""
        if isinstance(value, list):
            return value[0] if value else None
        return value

    # 结构化内容的类型标记。这类内容在 BM25 里天然吃亏（正文块一页 600-800 字、
    # 词频高；代码块大半是符号、中文匹配词少），所以给它们单独开一路召回。
    _TYPED_CHUNK_TYPES = ("code", "table")

    def _typed_search(self, rewritten_query, embedding_vector, knowledge_id,
                      department_id: Optional[int] = None):
        """第三路：只在代码/表格这类结构化内容里召回，返回合并去重后的有序命中列表。

        为什么需要单独一路：正文块和代码块放同一个池子里比 BM25，代码永远被压制；
        而用户问"代码怎么写"时，召回正文块等于没给代码（正文只会说
        "代码如下所示"）。分开召回再 RRF 融合，保证每种内容都有机会进上下文 ——
        这也是 RAGFlow / EnsembleRetriever 这类多路召回的标准做法。

        注意：这一路不改动现有两路的任何行为，只是多一路，所以不影响原来的排序。
        """
        filters = [{"term": {"knowledge_id": knowledge_id}},
                   {"terms": {"chunk_type": list(self._TYPED_CHUNK_TYPES)}}]
        if department_id is not None:
            filters.append({"term": {"department_id": department_id}})
        fields = ["chunk_id", "document_id", "knowledge_id", "page_number",
                  "chunk_content", "chunk_tables", "chunk_type"]

        word_resp = es.search(
            index="chunk_info",
            body={"query": {"bool": {"must": [{"match": {"chunk_content": rewritten_query}}],
                                     "filter": filters}},
                  "size": 50},
            fields=fields, source=False,
        )
        knn_query = {
            "field": "embedding_vector",
            "query_vector": [float(x) for x in list(embedding_vector)],
            "k": 50,
            "num_candidates": 100,
            "filter": {"bool": {"must": filters}},
        }
        vec_resp = es.search(index="chunk_info", knn=knn_query, fields=fields, source=False)

        merged, seen = [], set()
        for h in list(word_resp["hits"]["hits"]) + list(vec_resp["hits"]["hits"]):
            if h["_id"] in seen:
                continue
            seen.add(h["_id"])
            merged.append(h)
        return merged[:50]

    def _expand_to_parents(self, sorted_dict, search_id2record):
        """父子块检索的第 2、3 步：按父块（页）聚合子块分数，再把父块正文捞回来。

        为什么按页聚合，而不是把命中的子块逐个展开：
          - 同一页命中多个子块时，逐个展开会在 Prompt 里塞进好几份同样的整页文本，
            token 直接翻倍，还会把 LLM 的注意力稀释掉；
          - 分数累加本身是有意义的信号：一页里有 3 个子块命中，说明"整页都相关"，
            比单个子块偶然命中更可信。

        返回的记录里：
          _id           —— 保留命中的**子块** id（评测定位和溯源都用它，不能换成父块 id）
          parent_id     —— 父块（页级）id
          chunk_content —— 父块全文（截断到 parent_max_chars），送 LLM 用
          child_content —— 命中的子块原文，重排和 debug 用
        """
        groups, order = {}, []
        for _id, score in sorted_dict:
            rec = search_id2record[_id]
            did = self._first(rec.get("document_id"))
            page_no = self._first(rec.get("page_number"))
            if did is None or page_no is None:
                continue
            pid = f"doc_{did}_page_{page_no}"
            g = groups.get(pid)
            if g is None:
                groups[pid] = {"rrf": score, "best_id": _id, "best_score": score}
                order.append(pid)
            else:
                g["rrf"] += score
                if score > g["best_score"]:
                    g["best_score"], g["best_id"] = score, _id

        ranked = sorted(order, key=lambda p: groups[p]["rrf"], reverse=True)[:self.chunk_candidate]
        if not ranked:
            return []

        resp = es.mget(index="chunk_info", ids=ranked)
        parents = {d["_id"]: (d.get("_source") or {})
                   for d in resp.get("docs", []) if d.get("found")}

        out = []
        for pid in ranked:
            g = groups[pid]
            child = search_id2record[g["best_id"]]
            child_text = self._first(child.get("chunk_content")) or ""
            src = parents.get(pid) or {}
            parent_text = src.get("chunk_content") or child_text
            out.append({
                "_id": g["best_id"],
                "parent_id": pid,
                "chunk_content": parent_text[:self.parent_max_chars],
                "child_content": child_text,
                "rerank_text": child_text,
                "chunk_tables": src.get("chunk_tables"),
                "page_number": self._first(child.get("page_number")),
                "document_id": self._first(child.get("document_id")),
                "knowledge_id": self._first(child.get("knowledge_id")),
                "rrf_score": g["rrf"],
            })
        return out

    def query_document(self, query: str, knowledge_id: int, history: List[Dict] = None,
                       department_id: Optional[int] = None) -> List[str]:
        # Query改写：消除指代词，完整表达（传入历史帮助理解指代）
        t0 = time.monotonic()
        rewritten_query = self.query_rewrite(query, history)
        t1 = time.monotonic()

        # 向量化（串行在并行之前，因为向量检索依赖结果）
        embedding_vector = self.get_embedding(rewritten_query)
        t2 = time.monotonic()

        # 三路并行召回：BM25 / 向量 / 只查代码表格的"结构化内容"路
        with ThreadPoolExecutor(max_workers=3) as executor:
            word_future = executor.submit(self._word_search, rewritten_query, knowledge_id, department_id)
            vector_future = executor.submit(self._vector_search, embedding_vector, knowledge_id, department_id)
            typed_future = executor.submit(self._typed_search, rewritten_query, embedding_vector,
                                           knowledge_id, department_id)
            word_search_response = word_future.result()
            vector_search_response = vector_future.result()
            typed_hits = typed_future.result()
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

        # 第三路：结构化内容（代码/表格）。保证这两类内容一定有机会进候选，
        # 不会被正文块用词频压死。
        for idx, record in enumerate(typed_hits):
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)
            else:
                fusion_score[_id] += 1 / (idx + k)
            if _id not in search_id2record:
                search_id2record[_id] = {**record["fields"], "_id": _id}

        # 按总分排序，取前 chunk_candidate 条（config 里是 10 条）
        sorted_dict = sorted(fusion_score.items(), key=lambda kv: kv[1], reverse=True)
        if self.retrieval_mode == "parent_child":
            # 父子块：先按父块（页）聚合子块分数，再把父块正文捞回来
            sorted_records = self._expand_to_parents(sorted_dict, search_id2record)
        else:
            sorted_records = [search_id2record[x[0]] for x in sorted_dict][:self.chunk_candidate]

        # 构造 debug 信息
        debug_chunks = []
        if self.retrieval_mode == "parent_child":
            for rec in sorted_records:
                debug_chunks.append({
                    "content": (rec.get("child_content") or "")[:200],
                    "page_number": rec.get("page_number"),
                    "document_id": rec.get("document_id"),
                    "parent_id": rec.get("parent_id"),
                    "rrf_score": round(float(rec.get("rrf_score") or 0.0), 4),
                    "rerank_score": None,
                })
        else:
            for _id, score in sorted_dict[:self.chunk_candidate]:
                record = search_id2record[_id]
                content = _as_text(record.get("chunk_content"))
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
            for rec in sorted_records:
                # 父子块模式用命中的子块文本重排（语义更聚焦），送 LLM 时才换成父块全文
                text = rec.get("rerank_text")
                if text is None:
                    chunk_content = rec.get("chunk_content")
                    text = chunk_content[0] if isinstance(chunk_content, list) else chunk_content
                test_pair.append([rewritten_query, text])
            if not sorted_records:
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





