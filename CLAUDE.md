# RAG 智能问答系统

## 项目概述

基于 RAG（检索增强生成）的知识库智能问答系统，支持 PDF 文档上传、自然语言提问、多轮对话。

**技术栈**：FastAPI / Elasticsearch / SQLite / Ollama / SBert / BM25 / pdfplumber

**运行**：Ollama 本地运行 LLM（默认 `qwen2.5:1.5b`），服务端口 6010。

## 项目结构

```
.
├── main.py              # FastAPI 入口，REST API 端点
├── rag_api.py           # RAG 核心逻辑：检索、召回、重排、多轮对话
├── db_api.py            # SQLite 操作：知识库、文档元数据
├── es_api.py            # Elasticsearch 操作：向量索引、全文检索
├── router_schemas.py    # API 请求/响应数据结构
├── config.yaml          # 配置：ES、数据库、模型路径、LLM
├── upload_files/        # 上传的 PDF 文件
├── rag.db               # SQLite 数据库
└── test/                # 单元测试
```

## 核心模块说明

### rag_api.py — RAG 类

核心类 `RAG` 包含：
- `get_embedding()` — 调用 BGE 模型生成向量
- `get_rerank()` — 调用 rerank 模型重排序
- `extract_content()` — 解析 PDF、分块、写入 ES
- `query_document()` — 混合检索（BM25 + 向量 + RRF）
- `chat_with_rag()` — 多轮对话，组装 Prompt 调用 LLM

### es_api.py — Elasticsearch 操作

- `init_es()` — 初始化 ES 连接和索引
- `chunk_info` 索引 — 存储文档块、向量、全文
- `document_meta` 索引 — 存储文档元数据

### db_api.py — 数据库操作

- `KnowledgeDatabase` — 知识库表
- `KnowledgeDocument` — 文档元数据表

## 配置说明 (config.yaml)

```yaml
rag:
  llm_base: "http://localhost:11434/v1"   # Ollama 地址
  llm_model: "qwen2.5:1.5b"               # LLM 模型
  embedding_model: "bge-small-zh-v1.5"    # Embedding 模型
  rerank_model: "bge-reranker-base"       # 重排模型
  chunk_size: 256                         # 分块大小
  chunk_overlap: 20                       # 重叠窗口
  use_rrf: true                           # 启用 RRF 融合
  use_rerank: true                        # 启用重排序
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/knowledge_base` | 查询知识库（SQLite） |
| POST | `/v1/knowledge_base` | 创建知识库（SQLite） |
| DELETE | `/v1/knowledge_base` | 删除知识库（SQLite） |
| GET | `/v1/document` | 查询文档（SQLite） |
| POST | `/v1/document` | 上传 PDF（异步解析） |
| DELETE | `/v1/document` | 删除文档（SQLite） |
| POST | `/v1/embedding` | 文本向量化 |
| POST | `/v1/rerank` | 重排序 |
| POST | `/chat` | RAG 多轮对话 |

## 检索流程

1. **PDF 上传** → `BackgroundTasks` 异步调用 `extract_content()` 解析页面文本
2. **分块** → 按 256 token 分块，20 overlap
3. **向量化** → BGE 生成 512 维向量，写入 ES
4. **检索时** → Query 改写（结合历史理解指代）→ BM25 + KNN **双路并行召回** → RRF 融合 → Cross-Encoder 重排
5. **生成** → 检索结果注入 Prompt → 调用 Ollama LLM

## 数据存储架构

| 存储 | 用途 |
|------|------|
| SQLite | 知识库、文档元数据的 CRUD |
| ES chunk_info | 文档块文本、全文索引、向量索引 |
| ES document_meta | 文档摘要、文件名等元数据 |

## 多轮对话说明

服务端无状态，对话历史由调用方维护。每轮对话传入完整 `message` 列表（含历史消息），服务端基于最新问题检索，结合历史生成回答。

## 测试

```bash
pytest test/ -v
```

主要测试文件：
- `test_recall.py` — 召回率测试
- `test_query_rewrite.py` — Query 改写测试
- `test_permission.py` — 权限测试
- `test_retry.py` — 重试逻辑测试
- `test_state_machine.py` — 状态机测试
- `test_pdf_processing.py` — PDF 解析测试

## 常见问题

1. **Ollama 未运行** — 确保 `ollama serve` 启动，且模型已拉取
2. **ES 索引不存在** — 首次上传文档时自动创建
3. **向量维度不匹配** — 检查 config.yaml 中 `dims: 512` 与模型一致