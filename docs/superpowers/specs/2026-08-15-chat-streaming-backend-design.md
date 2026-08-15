# /chat 流式输出（后端 SSE）设计

## 目标

将 `POST /chat` 从一次性返回 JSON 改为 SSE（Server-Sent Events）流式响应：先推检索详情（`debug`），再逐 token 推 LLM 生成内容（`token`），最后推完整消息列表与耗时（`done`）。

只改后端。前端不在本次范围。

## 非目标

- 不改前端
- 不引入 WebSocket
- 不改 `query_rewrite`、`chat()` 非流式调用、`query_document` 检索逻辑

## 现状

- `RAG.chat()`（rag_api.py:464）用 `openai` SDK 非流式调用 Ollama，返回 `completion.choices[0].message`。
- `RAG.chat_with_rag()`（rag_api.py:418）做检索 + 组装 prompt + 非流式生成，末尾把回答 append 进 `message`。
- `/chat`（main.py:518）同步返回 `RAGResponse`（含 `message`、`debug_info`、`processing_time`）。
- `test/benchmark_report.py:308` 直接调用 `chat_with_rag` 测回答准确率（有/无重排对比）。

## 架构

```
rag_api.py  RAG._retrieve_context(knowledge_id, message)
                单轮:  query = message[0]["content"], history = None
                多轮:  query = message[-1]["content"], history = message[:-1]
                检索(query_document) → 格式化资料 → 组装 llm_messages → 返回 (llm_messages, debug_info)

            RAG.chat_with_rag()   # 保留，非流式薄封装，供 benchmark 直调
                = _retrieve_context() + chat()(非流式) + append 回答

            RAG.chat_stream(knowledge_id, message)   # 新增，返回事件生成器
                = _retrieve_context() + stream=True 逐 token + append 回答
                yield (event, data) 元组序列

main.py   /chat
                检索在 StreamingResponse 之前执行（失败 → HTTPException → 统一错误信封）
                event_stream(): 把 (event, data) 格式化为 SSE 文本
                返回 StreamingResponse(media_type="text/event-stream")
```

## SSE 事件协议

| 事件 | data 载荷 | 时机 |
|------|-----------|------|
| `debug` | `debug_info`（rewritten_query、chunks、分数） | 检索完成、出字前 |
| `token` | 单段 token 文本（`delta.content`） | LLM 逐 token 生成时 |
| `done` | `{"message": [...], "processing_time": ...}` | 流正常结束 |
| `error` | `{"message": "..."}` | 生成中途失败（检索失败不走此事件） |

SSE 文本格式（每个事件两行 + 空行）：

```
event: debug
data: {json}

event: token
data: 文本

```

注意：`token` 的 data 是不带 JSON 包裹的原始文本，前端按行读取时直接拼接。

## 错误处理

- **检索/组装失败**：发生在 SSE 头发出之前 → 抛 `HTTPException(500)` → 走 main.py:70 统一错误信封，HTTP 非 200。
- **生成中途失败**：已发出 `debug`，无法改 HTTP 状态 → `yield ("error", {"message": str(e)})` 后终止流，客户端靠 `error` 事件识别。

## 多轮历史延续

沿用现状契约：服务端无状态，调用方维护历史。`chat_stream` 结束后把回答 append 进 `message`，由 `done` 事件返回完整列表，调用方下一轮继续传入。

回答角色沿用现状 `{"role": "system"}`，不做变更（避免扩大行为面）。

## 受影响文件

| 文件 | 改动 |
|------|------|
| `rag_api.py` | 新增 `_retrieve_context()`、`chat_stream()`；`chat_with_rag()` 改调 `_retrieve_context()` |
| `main.py` | `/chat` 改为 `StreamingResponse`，检索前置 + SSE 事件流 |
| `test/test_chat.py` | CLI 脚本从 `resp.json()` 改为按 SSE 行解析、拼回完整回答 |
| `test/benchmark_report.py` | 不动（继续用 `chat_with_rag`） |

## 测试策略

新增 `test/test_chat_stream.py`，用 TestClient 消费 SSE，`patch` LLM 的流式 create 返回模拟 chunk 迭代器：

- 单轮：断言事件序列 `debug → token×N → done`，done 的 message 含 append 后的回答。
- 多轮：断言 `_retrieve_context` 把历史传给 query_document（指代改写），llm_messages 顺序为「历史 + system 上下文 + 最后一问」。
- 生成中途异常：断言发出 `error` 事件。
- 检索失败：断言非 200 + 错误信封。

mock 方式：`patch.object(rag.client.chat.completions, "create", return_value=iter([...]))`，每个 chunk 含 `choices[0].delta.content`。
