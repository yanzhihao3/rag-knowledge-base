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

---

## 待办与已知短板（按面试风险排序）

> 最后更新：2026-11-10。这里记录的是**代码里真实存在、但还没修**的问题。
> 面试时能讲清"我知道它、为什么先不修、准备怎么修"即可，不要假装不存在。

### 高 —— 面试官一追问就会露的

1. **`query_rewrite` 零自动化测试**
   - 位置：`rag_api.py:1035-1088`
   - 原因：评测全是单轮 —— `test/benchmark_report.py:200` 传 `history=None`，`:379` 只传一条 user 消息，
     所以 `rag_api.py:1039` 的 `if not history` 每次都短路。`rag.log` 里只有 5 条 `[Query改写]`，
     全部集中在 2026-08-16 / 08-21 的手工测试。
   - 修法：`_needs_rewrite` 补参数化单测（指代 / 口语 / 极短 / 完整问句四种），再 mock `chat`
     验证"超长输出回退原句"与"调用失败回退原句"；另外补一组多轮评测题。

2. **`temperature=0.9` 硬编码，配置里的 `0.1` 从未生效**
   - 位置：`rag_api.py:993-994` 写死 `top_p=0.7, temperature=0.9`；全仓库搜不到对
     `config["models"]["llm"]["config"]` 的读取，`temperature` / `top_p` / `max_token_to_sample` 都是死配置。
   - 影响：RAG 问答用 0.9 偏高，直接放大"脑补原文没有的对比"。
   - 修法：从配置读取 + 加一条断言测试（`RAG().temperature == config[...]`）+ 重跑
     `pytest test/benchmark_report.py -k accuracy` 做前后对比。

3. **embedding 静默截断**
   - 位置：`rag_api.py:622` 的 `encode()` 不传任何长度参数，模型默认 `max_seq_length = 512`。
   - 实测：中文约 1 字 = 1 token（400 字 → 402 tokens；600 字 → 602 tokens），
     超过 512 的部分被**静默丢弃**，不报错、不打日志。
   - 影响：页级块（600–800 字正文 + 表格文本）的后半页没有进入向量，只能靠 BM25 兜底。
     对比：同一文件的 `get_rerank`（`rag_api.py:643`）显式写了 `truncation=True, max_length=512`。
   - 修法：分块按 token 而不是按字符计；页级块送 embedding 前截断；日志记录截断发生的次数。

4. **上传没有内容校验和大小上限**
   - 位置：`main.py:599` 只校验扩展名白名单，`main.py:601` 的 `file.file.read()` 把整个文件读进内存。
   - 影响：任意文件改名为 `.pdf` 即可通过；超大文件直接打爆 API 进程内存。
   - 修法：加 `content-length` / `UploadFile.size` 上限（超限返回 413）+ 校验 PDF 魔数 `%PDF`；
     `content_type` 只当参考，不作为落库依据。

### 中 —— 能讲清"我知道但没做"就行

5. **删除与解析的并发竞态**
   - 位置：`tasks.py:49-55` 只在任务开始时查一次文档是否存在，之后一路往 ES 写；
     `main.py:322` / `:525` 删除时直接清 ES；`main.py:332-335` 只删 task 行，不会取消正在跑的任务。
   - 窗口：加 OCR 后单文档解析实测 564 秒（约 9.4 分钟），撞车概率被放大。
   - 修法：写 ES 前二次校验 `document_id` 是否还在 DB；或删除时先置 `deleting` 状态。

6. **`classify` 的经验常数没有标注集 / 回归测试**
   - 位置：`image_ocr.py:114-175`，包括 `cjk < 3`、`cjk_out <= 2`、`max(widths) > 8`、
     `threshold = 0.4 / 0.5`、`avg <= 8` 等阈值，全部来自这一本书的试错。
   - 修法：抽 20–30 张图人工标注（表格 / 代码 / 架构图 / 公式），写分类准确率与混淆矩阵测试；
     否则以后调任何阈值都无法判断是改进还是退化。

7. **`department_id=None` 是 fail-open 默认值**
   - 位置：`rag_api.py:654-656`（词路）、`:680-681`（向量路）、`:725-726`（结构化路）。
   - 影响：`None` 表示"不限制"，是默认值。任何调用方忘了传，系统不报错、只静默做全部门检索。
   - 修法：用显式哨兵值表达"不限制"，或在入口断言"要么是 admin/system，要么 `department_id` 非空"。

8. **`es_api` 在 import 时 ping ES**
   - 位置：`es_api.py:137` 在模块顶层调用 `init_es()`；`es_api.py:39-43` 会重试 30 次 × 2 秒。
   - 影响：`import rag_api` 在 ES 未启动时会卡最多 60 秒；`test_parse_guard.py` 因此被放进
     CI 的 `collect_ignore` 名单，尽管它本身用假 ES。
   - 修法：顶层只构造 client（惰性、不发请求），把 `init_es()` 移到显式启动点
     （FastAPI 启动事件 / worker 启动 / 脚本入口）。

### 低 —— 属于"下个迭代再说"

9. **页级块与子块在索引里重复**
   - 位置：`rag_api.py:431-475` 同一页既写页级块（`page_text_clean`）又写子块。
   - 影响：一份文字存两份、都参与召回；是"父子块收益≈0"与"答案块口径虚高 6 个点"的共同根因。
   - 方向：页级块只作为 parent 存在、不参与向量召回。

10. **第三路"解压"与"重复计票"的贡献未分离**
    - 位置：`rag_api.py:710-754`。第三路本质上是对结构化内容的一次系统性加权。
    - 现象：只要词路命中满 50 条，`merged[:50]` 会把向量独有命中全部丢弃，向量分支形同虚设；
      候选池只有 83 条（10 表格 + 73 代码），取走 50 条等于六成内容都进候选，排名区分度很弱。
    - 验证：让第三路只补充前两路没捞到的块（纯解压），再跑 70 题；差值即重复计票的贡献。
      合并方式也可改成按名次交错（词第 1、向量第 1、词第 2、向量第 2…）。

11. **代码题答案块 70% 的成因未查**
    - 现象：代码题关键词命中 100%，答案块命中只有 70%。
    - 假设：`answer_terms` 是人工按原书排版标注的，而代码块文本来自 OCR，术语可能对不上 →
      gold 定位失败。
    - 验证：挑出失败的 3 道题，把 `answer_terms` 直接拿去代码块文本里搜；搜得到 = 检索问题，
      搜不到 = gold 构造有缺陷。

12. **三套测试跳过机制并存**
    - `test/conftest.py:41-59`：`collect_ignore` + `RAG_SKIP_INTEGRATION`（收集阶段跳过）
    - `test/test_recall.py:33`、`test/test_benchmark.py:90`：`skipif` + `RUN_INTEGRATION_TESTS`（导入后跳过）
    - `test/test_auth_api.py:18`：`pytestmark = pytest.mark.integration`，但 CI 从未用 `-m` 过滤，形同虚设
    - 修法：统一成一种机制，并把环境变量名对齐。

### 建议的处理顺序

1. **先提交工作区改动**（4 个修改 + 3 个新增文档），别在上面继续叠加。
2. **再做第 2 条（temperature）**：改一行 + 重跑 accuracy，性价比最高，也是"模型幻觉"问题的直接答案。
3. **然后第 3 条（embedding 截断）**：影响面最大，且不需要重跑全部评测就能验证（单测 token 数即可）。
4. **面试前不要动第 9、10 条**：它们会改变检索候选与评测基线，会让简历上的数字对不上。
