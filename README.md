# 基于 RAG 的知识库智能问答系统

一个本地部署的 RAG 智能问答系统，支持 PDF 文档上传、自然语言提问、多轮对话。

## 技术栈

FastAPI / Elasticsearch / SQLite / Ollama / SBert / BM25 / pdfplumber

## 项目结构

```
.
├── main.py              # FastAPI 入口，REST API 端点
├── rag_api.py           # RAG 核心逻辑：检索、召回、重排、多轮对话
├── db_api.py            # SQLite 操作：知识库、文档元数据
├── es_api.py            # Elasticsearch 操作：向量索引、全文检索
├── router_schemas.py    # API 请求/响应数据结构
├── config.yaml          # 配置文件
├── upload_files/        # 上传的 PDF 文件
├── rag.db               # SQLite 数据库
└── test/                # 单元测试
```

## 快速开始

### 环境要求

- Python 3.10+
- Elasticsearch 8.x
- Ollama（本地运行 LLM）

### 启动服务

```bash
# 1. 启动 Elasticsearch
# 2. 启动 Ollama 并确保 qwen2.5:1.5b 模型已拉取
ollama pull qwen2.5:1.5b
ollama serve

# 3. 运行服务
python main.py
```

服务运行在 `http://localhost:6010`

## 核心功能

### 1. 文档上传与解析

上传 PDF 文档，系统自动：
- 提取页面文本和表格
- 处理跨页断句
- 按 256 tokens 分块，20 overlap
- 批量生成 BGE 向量
- 写入 ES 索引

### 2. 双路并行召回

检索时同时执行两路搜索：

```
用户问题 → Query 改写（结合历史理解指代）→ Embedding
                        ↓
              ┌─────────┴─────────┐
              ↓                   ↓
         BM25 检索            KNN 向量检索
              ↓                   ↓
              └─────────┬─────────┘
                        ↓
                  RRF 融合（k=60）
                        ↓
                  重排序（可选）
                        ↓
                     返回结果
```

- **BM25**：关键词精确匹配
- **KNN**：向量相似度检索
- **RRF**：Reciprocal Rank Fusion，两路结果按排名融合
- **重排序**：bge-reranker-base Cross-Encoder 精排

### 3. 多轮对话

每轮对话基于最新问题重新检索，结合对话历史生成回答。

### 4. 异步处理

PDF 上传后立即返回，解析任务后台执行，不阻塞 API。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/knowledge_base` | 查询知识库 |
| POST | `/v1/knowledge_base` | 创建知识库 |
| DELETE | `/v1/knowledge_base` | 删除知识库 |
| GET | `/v1/document` | 查询文档 |
| POST | `/v1/document` | 上传 PDF（异步解析） |
| DELETE | `/v1/document` | 删除文档 |
| POST | `/v1/embedding` | 文本向量化 |
| POST | `/v1/rerank` | 重排序 |
| POST | `/chat` | RAG 多轮对话 |

## 数据存储

| 存储 | 用途 |
|------|------|
| SQLite | 知识库、文档元数据的 CRUD |
| ES chunk_info | 文档块文本、全文索引、向量索引 |
| ES document_meta | 文档摘要、文件名等元数据 |

## 配置说明

关键配置在 `config.yaml`：

```yaml
rag:
  llm_base: "http://localhost:11434/v1"   # Ollama 地址
  llm_model: "qwen2.5:1.5b"              # LLM 模型
  embedding_model: "bge-small-zh-v1.5"   # Embedding 模型
  rerank_model: "bge-reranker-base"      # 重排模型
  chunk_size: 256                        # 分块大小
  chunk_overlap: 20                      # 重叠窗口
```

## 测试

```bash
pytest test/ -v
```

## 项目亮点

1. **双路并行召回**：BM25 + KNN 通过 ThreadPoolExecutor 并行执行，降低检索延迟
2. **RRF 融合**：无需调参，用排名而非分数融合结果，避免量纲不一致问题
3. **Query 改写**：结合多轮对话历史，基于 LLM 消除指代消解与话题漂移，提升多轮对话检索准确性
4. **跨页断句处理**：保留页面边界语义完整性
5. **表格提取**：识别并存储 PDF 中的表格结构
6. **任务状态机**：跟踪文档解析进度，指数退避重试保障可靠性