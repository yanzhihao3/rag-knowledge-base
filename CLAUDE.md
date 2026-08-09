# RAG 智能问答系统

## 项目概述

基于 RAG（检索增强生成）的知识库智能问答系统，支持 PDF 文档上传、自然语言提问、多轮对话。

**技术栈**：FastAPI / Elasticsearch / SQLite / Ollama / SBert / BM25 / pdfplumber

**观测**：Python 标准库 `logging` + `request_id` 关联 + 检索链路耗时拆分 + 敏感内容脱敏

**前端**：Vue 3 / Element Plus / Axios / Vite

**运行**：Ollama 本地运行 LLM（默认 `qwen2.5:1.5b`），服务端口 6010。

## 项目结构

```
.
├── main.py              # FastAPI 入口，REST API 端点
├── rag_api.py           # RAG 核心逻辑：检索、召回、重排、多轮对话
├── db_api.py            # SQLite 操作：知识库、文档元数据
├── es_api.py            # Elasticsearch 操作：向量索引、全文检索、级联删除
├── logging_config.py    # 日志底座：统一格式 + request_id 注入 + 轮转文件
├── router_schemas.py    # API 请求/响应数据结构
├── config.yaml          # 配置：ES、数据库、模型路径、LLM
├── upload_files/        # 上传的 PDF 文件
├── rag.db               # SQLite 数据库
├── frontend/            # Vue 3 前端
│   ├── src/
│   │   ├── App.vue          # 根组件（布局）
│   │   ├── api.js           # Axios API 封装
│   │   ├── components/
│   │   │   ├── Sidebar.vue  # 侧边栏（知识库+文档管理）
│   │   │   └── ChatView.vue # 聊天界面
│   │   └── main.js          # 入口
│   ├── vite.config.js
│   └── package.json
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
- `delete_document_chunks()` — 按 `document_id` 级联删除分块 + 摘要
- `delete_knowledge_chunks()` — 按 `knowledge_id` 一把清（删知识库用）
- `_assert_no_delete_failures()` — 检查 `delete_by_query` 的 `version_conflicts`/`failures`，部分删除失败即抛异常

### logging_config.py — 日志底座

- `setup_logging()` — 幂等初始化：INFO 级别 + 控制台 + 轮转文件（5MB×3），统一格式 `时间 | 级别 | 模块 | request_id | 内容`
- `request_id_var` — `contextvars.ContextVar`，每个请求一个唯一 ID，线程/协程安全
- `RequestIdFilter` — 把当前请求的 request_id 注入每条日志记录

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
| GET | `/v1/knowledge_base/list` | 知识库列表 |
| POST | `/v1/knowledge_base` | 创建知识库（SQLite） |
| DELETE | `/v1/knowledge_base` | 删除知识库（级联清空库下 ES 分块/摘要 + PDF + 子文档） |
| GET | `/v1/document` | 查询文档（SQLite） |
| GET | `/v1/document/list` | 文档列表（按知识库） |
| POST | `/v1/document` | 上传 PDF（异步解析） |
| DELETE | `/v1/document` | 删除文档（级联清理 ES 分块/摘要 + PDF） |
| POST | `/v1/embedding` | 文本向量化 |
| POST | `/v1/rerank` | 重排序 |
| POST | `/chat` | RAG 多轮对话（含 debug_info 检索详情） |

## 检索流程

1. **PDF 上传** → `BackgroundTasks` 异步调用 `extract_content()` 解析页面文本
2. **分块** → 按 256 token 分块，20 overlap
3. **向量化** → BGE 生成 512 维向量，写入 ES
4. **检索时** → Query 改写（结合历史理解指代）→ BM25 + KNN **双路并行召回** → RRF 融合 → Cross-Encoder 重排；四段耗时（改写/向量/召回/重排）单独打点记录日志
5. **生成** → 检索结果注入 Prompt → 调用 Ollama LLM
6. **结果可视化** → `/chat` 返回 `debug_info`（改写后 query、召回chunks、RRF/重排分数），前端折叠面板展示

## 数据存储架构

| 存储 | 用途 |
|------|------|
| SQLite | 知识库、文档元数据的 CRUD |
| ES chunk_info | 文档块文本、全文索引、向量索引 |
| ES document_meta | 文档摘要、文件名等元数据 |

## 数据一致性（级联删除）

三存储（SQLite 元数据 / ES 分块与摘要 / 磁盘 PDF）**没有事务边界**，删除在代码里做级联清理，顺序钉死为 **ES → 物理文件 → SQLite**：

- **ES 先删，失败阻断**：删了元数据却留分块 = 还能搜到已删内容（幽灵分块），危害最大。ES 清理失败抛异常 → 接口 500 → SQLite 不删。
- **物理文件次删，失败只告警**：孤儿文件低危、可手动清理，不该被文件锁阻塞整个删除。
- **SQLite 最后删，同一事务**：删文档删一行；删知识库先删子文档行、再删知识库行。
- 删文档按 `document_id` 精确删；删知识库按 `knowledge_id` 一把清，不误删其他库。
- `delete_by_query` 显式检查 `version_conflicts`/`failures`，部分删除失败会抛 `RuntimeError`。
- 响应字段在 `commit` 前快照，避免 `ObjectDeletedError`；pdfplumber 解析用 try/finally 关闭，防 Windows 文件锁删不掉 PDF。

## 监控日志

- **request_id 关联**：HTTP 中间件为每个请求生成唯一 ID，经 `contextvars` 贯穿整条调用链（含线程池），业务日志自动携带。
- **请求中间件**：记录 `方法 路径 -> 状态码 | 耗时`（`time.monotonic()`），异常路径用 `logger.exception()` 保留堆栈。
- **检索链路耗时拆分**：改写/向量/召回/重排四段独立计时，输出 `[RAG] 改写=..s 向量=..s 召回=..s 重排=..s`。
- **脱敏**：对话全文、历史全文、prompt 全文不落日志只记长度；Query 改写结果保留 INFO（RAG 特有可观测点）。
- **轮转**：`rag.log` 5MB 轮转、保留 3 份，已被 `.gitignore` 忽略。

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
- `test_logging.py` — 日志底座测试（CI 可跑）
- `test_utils.py` — 工具函数测试（`safe_remove_file` 等）

## 常见问题

1. **Ollama 未运行** — 确保 `ollama serve` 启动，且模型已拉取
2. **ES 索引不存在** — 首次上传文档时自动创建
3. **向量维度不匹配** — 检查 config.yaml 中 `dims: 512` 与模型一致