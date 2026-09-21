# RAG 智能问答系统

## 项目概述

基于 RAG（检索增强生成）的知识库智能问答系统，支持 PDF 文档上传、自然语言提问、多轮对话。

**技术栈**：FastAPI / Elasticsearch / MySQL（默认，可切 SQLite）/ Ollama / SBert / BM25 / pdfplumber
／ **PyMuPDF + RapidOCR**（图片型表格 / 代码的提取，本地 CPU、0 成本）

**认证与权限**：JWT（access + refresh）+ RBAC 角色权限 + 多租户数据隔离（`security.py` / `auth_routes.py`）

**异步任务**：Celery + Redis（`celery_app.py` / `tasks.py` / `task_store.py`）

**观测**：Python 标准库 `logging` + `request_id` 关联 + 检索链路耗时拆分 + 敏感内容脱敏

**运行**：Ollama 本地运行 LLM（默认 `qwen2.5:1.5b`），服务端口 6010。

## 项目结构

```
.
├── main.py              # FastAPI 入口，REST API 端点
├── rag_api.py           # RAG 核心逻辑：检索、召回、重排、多轮对话
├── image_ocr.py         # 图片内容提取：定位→OCR→分类→拼块（表格/代码）
├── db_api.py            # 数据库层：MySQL（默认）/ SQLite 双引擎
├── es_api.py            # Elasticsearch 操作：向量索引、全文检索、级联删除
├── logging_config.py    # 日志底座：统一格式 + request_id 注入 + 轮转文件
├── router_schemas.py    # API 请求/响应数据结构
├── security.py          # 认证与授权：bcrypt 密码、JWT、权限依赖、租户过滤
├── auth_routes.py       # 认证接口：登录 / 刷新 / 当前用户 / 创建用户
├── celery_app.py        # Celery 应用（Redis broker / backend，acks_late 不丢任务）
├── tasks.py             # 文档解析任务（幂等投递 + 分层重试 + 状态回写）
├── task_store.py        # 任务状态存储（task 表）
├── env_loader.py        # .env 加载（环境变量 > .env > config.yaml）
├── alembic/             # 数据库迁移（表结构版本管理）
├── scripts/             # 运维 / 调试脚本（管理员创建、数据搬迁、表格定位、代码分类、ES 迁移等）
├── config.yaml          # 配置：ES、数据库、模型路径、LLM
├── upload_files/        # 上传的 PDF 文件
├── test/                # 单元测试
│   ├── benchmark_report.py         # RAG 评测脚本（召回率 / 准确率）
│   ├── benchmark_cases_rag_book.py # 评测用例（70 条：概念 50 / 表格 10 / 代码 10）
│   └── baselines/                  # 评测基线（每次改动的对比留档）
```

## 核心模块说明

### rag_api.py — RAG 类

核心类 `RAG` 包含：
- `get_embedding()` — 调用 BGE 模型生成向量
- `get_rerank()` — 调用 rerank 模型重排序
- `extract_content()` — 解析 PDF、分块、写入 ES（幂等：确定性 `_id` + 解析前先清旧分块）；
  内部还会调 `_index_image_blocks()` 把**图片型的表格/代码** OCR 出来一并入库
- `join_visual_lines()` / `split_text_semantic()` — 语义分块（视觉行拼回 → 按句切 → 整句重叠）
- `query_document()` — 三路并行召回（BM25 + 向量 + 结构化内容）→ RRF → 重排
- `_typed_search()` — 第三路：只查 `chunk_type ∈ {code, table}`，避免代码/表格被正文块压死
- `_identifier_clauses()` — 标识符分词：把英文类名拆开查，让用户搜"半截类名"也能命中
- `_expand_to_parents()` — 父子块（`retrieval_mode=parent_child`，默认关闭）
- `query_rewrite()` — 指代/口语规则触发 + 最近 2 轮历史 + 口语转书面 + 输出校验（10s 超时回退）
- `chat_with_rag()` — 多轮对话，组装 Prompt 调用 LLM

### image_ocr.py — 图片内容提取

- `extract_image_blocks(doc)` — 一站式入口：**定位 → OCR → 分类 → 拼块**
- 定位：表格用正文里的"表X-Y"引用反查；代码用"代码如下所示"这类引用句（靠书自己说的话定位）
- 分类：`classify()` 判表格/代码/公式/架构图，核心判据是 `cjk_outside_comment()`（注释外的中文数）
- OCR：本地 RapidOCR，纯 CPU、0 成本、平均 3.5 秒/张

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
  chunk_candidate: 10                     # RRF 融合后召回候选数
  rerank_top_k: 5                         # 最终进 Prompt 的条数
  use_rrf: true                           # 启用 RRF 融合
  use_rerank: true                        # 启用重排序
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/knowledge_base` | 查询知识库（数据库） |
| GET | `/v1/knowledge_base/list` | 知识库列表 |
| POST | `/v1/knowledge_base` | 创建知识库（数据库） |
| DELETE | `/v1/knowledge_base` | 删除知识库（级联清空库下 ES 分块/摘要 + PDF + 子文档） |
| GET | `/v1/document` | 查询文档（数据库） |
| GET | `/v1/document/list` | 文档列表（按知识库） |
| POST | `/v1/document` | 上传 PDF（异步解析） |
| DELETE | `/v1/document` | 删除文档（级联清理 ES 分块/摘要 + PDF） |
| POST | `/v1/embedding` | 文本向量化 |
| POST | `/v1/rerank` | 重排序 |
| POST | `/chat` | RAG 多轮对话（含 debug_info 检索详情） |

## 检索流程

1. **PDF 上传** → 投递 Celery 任务（Redis 队列），worker 调用 `extract_content()` 解析页面文本；状态写入 `task` 表，可用 `/v1/task` 查询
2. **图片内容提取** → 正文解析完后，`image_ocr.py` 把**图片型的表格/代码** OCR 出来。
   技术文档的代码示例和表格常是图片（实测这本书 251 张图占 20.9% 版面、
   `extract_tables()` 全书抽到 0 张），只靠文本层一个字都拿不到。
   开关：`config.yaml` 的 `image_ocr`（默认开；关掉可省约 10 分钟解析时间）
3. **分块** → 语义分块：视觉行拼回逻辑行 → 按句边界切 → 聚合到目标 400 字 + 整句重叠 1 句
   （`chunk_strategy: semantic`；`fixed` 是旧的 256 字符定长切分，用于回退对比）
4. **向量化** → BGE 生成 512 维向量，写入 ES
5. **检索时** → Query 改写（规则触发，结合最近 2 轮历史理解指代）→ **三路并行召回**
   （BM25 / 向量 / 只查 code+table 的结构化内容路）→ RRF 融合取前 10 条 → Cross-Encoder 重排
   → **只保留前 5 条（rerank_top_k）进 Prompt**；四段耗时（改写/向量/召回/重排）单独打点记录日志
6. **生成** → 检索结果注入 Prompt → 调用 Ollama LLM
7. **结果可视化** → `/chat` 返回 `debug_info`（改写后 query、召回chunks、RRF/重排分数），可在 Swagger `/docs` 或任意客户端查看

## 数据存储架构

| 存储 | 用途 |
|------|------|
| MySQL（默认）/ SQLite | 知识库、文档元数据的 CRUD |
| ES chunk_info | 文档块文本、全文索引、向量索引 |
| ES document_meta | 文档摘要、文件名等元数据 |

## 数据一致性（级联删除）

三存储（MySQL 元数据 / ES 分块与摘要 / 磁盘 PDF）**没有事务边界**，删除在代码里做级联清理，顺序钉死为 **ES → 物理文件 → 数据库**：

- **ES 先删，失败阻断**：删了元数据却留分块 = 还能搜到已删内容（幽灵分块），危害最大。ES 清理失败抛异常 → 接口 500 → 数据库不删。
- **物理文件次删，失败只告警**：孤儿文件低危、可手动清理，不该被文件锁阻塞整个删除。
- **数据库最后删，同一事务**：删文档删一行；删知识库先删子文档行、再删知识库行。
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
- `benchmark_report.py` — RAG 评测：70 条用例 × 四口径（关键词/答案块/仅子块/答案页）+ 回答准确率，
  并**按类别拆分统计**（概念题 50 / 表格题 10 / 代码题 10）
- `benchmark_cases_rag_book.py` — 评测用例数据（基于《大模型RAG实战》，含答案特征词；
  末尾 20 条是图片类的表格题和代码题，`case_category()` 负责分类）
- `test_chunking.py` — 分块（视觉行拼回 / 按句聚合 / 整句重叠）、文档拼装、要点打分
- `test_parse_guard.py` — 解析"静默失败"守门（空页跳过 / 全篇无内容报错），用假 ES 测试不污染知识库
- `test_query_rewrite.py` — Query 改写测试
- `test_permission.py` — 权限测试
- `test_retry.py` — 重试逻辑测试
- `test_task_store.py` — 任务状态存储测试（task 表状态流转）
- `test_pdf_processing.py` — PDF 解析测试
- `test_logging.py` — 日志底座测试（CI 可跑）
- `test_utils.py` — 工具函数测试（`safe_remove_file` 等）

评测运行（需 ES + 模型；准确率评测还需 Ollama）：

```bash
pytest test/benchmark_report.py -v -s -k recall     # 召回率：关键词 / 答案块 / MRR
pytest test/benchmark_report.py -v -s -k accuracy   # 回答准确率（有/无重排对比）
```

## 常见问题

1. **Ollama 未运行** — 确保 `ollama serve` 启动，且模型已拉取
2. **ES 索引不存在** — 首次上传文档时自动创建
3. **向量维度不匹配** — 检查 config.yaml 中 `dims: 512` 与模型一致
4. **评测全 0%** — 检查 ES 是否启动、`benchmark_report.py` 的 `KB_ID` 是否与知识库一致、文档是否已解析完成
