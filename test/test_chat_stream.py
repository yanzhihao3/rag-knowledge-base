"""test_chat_stream.py — /chat SSE 流式输出测试。

重依赖（真实 ES 检索 / 模型向量化 / Ollama 生成）全部通过 patch 隔离：
- RAG._retrieve_context / RAG.query_document  → 返回固定内容，避免真实检索
- rag_api.OpenAI                              → 喂模拟 chunk 流，避免真实 LLM

注意：端点内部是 `rag = RAG()` 现建实例，所以不能 patch 某个 rag 实例的
client，而要 patch 模块级 `rag_api.OpenAI`，让新建的 RAG 也拿到模拟客户端。

运行：pytest test/test_chat_stream.py -v
"""
import os
import sys
import json
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
import rag_api
from rag_api import RAG
from main import _format_sse, app
from fastapi.testclient import TestClient


# ---------- 模拟 OpenAI 流式客户端 ----------

_CTRL = {"chunks": [], "raise_at": None}


class _FakeCompletions:
    def create(self, **kwargs):
        return _iter_chunks(_CTRL["chunks"], _CTRL["raise_at"])


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = _FakeChat()


def _chunk(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=content))]
    )


def _iter_chunks(contents, raise_at=None):
    """逐 chunk 生成；i == raise_at 时抛异常，模拟生成中途失败。"""
    i = 0
    while True:
        if raise_at is not None and i == raise_at:
            raise RuntimeError("llm stream failed")
        if i >= len(contents):
            return
        yield _chunk(contents[i])
        i += 1


def _parse_sse(text):
    events, cur = [], None
    for line in text.split("\n"):
        if line == "":
            if cur is not None:
                events.append(cur)
                cur = None
            continue
        if cur is None:
            cur = {}
        if line.startswith("event:"):
            cur["event"] = line[6:].strip()
        elif line.startswith("data:"):
            cur.setdefault("data_lines", []).append(line[5:].lstrip())
    if cur:
        events.append(cur)
    return events


def _chat_payload(message):
    return {"knowledge_id": 1, "message": message}


# ---------- _format_sse ----------

def test_format_sse_json_and_token():
    sse_debug = _format_sse("debug", {"rewritten_query": "什么是RAG？", "chunks": []})
    assert sse_debug == 'event: debug\ndata: {"rewritten_query": "什么是RAG？", "chunks": []}\n\n'

    sse_token = _format_sse("token", "你好\n世界")
    assert sse_token == "event: token\ndata: 你好\ndata: 世界\n\n"


# ---------- _retrieve_context ----------

def test_retrieve_context_single_turn(monkeypatch):
    rag = RAG()
    seen = {}
    monkeypatch.setattr(
        RAG, "query_document",
        lambda self, query, knowledge_id, history: (seen.update(query=query, history=history), [])[1],
    )
    llm_messages, _ = rag._retrieve_context(1, [{"role": "user", "content": "什么是RAG？"}])

    assert seen["query"] == "什么是RAG？"
    assert seen["history"] is None
    assert len(llm_messages) == 1
    assert llm_messages[0]["role"] == "system"
    assert "什么是RAG？" in llm_messages[0]["content"]


def test_retrieve_context_multi_turn_passes_history(monkeypatch):
    rag = RAG()
    seen = {}
    monkeypatch.setattr(
        RAG, "query_document",
        lambda self, query, knowledge_id, history: (seen.update(query=query, history=history), [])[1],
    )
    msgs = [
        {"role": "user", "content": "RAG是什么"},
        {"role": "system", "content": "RAG是检索增强生成"},
        {"role": "user", "content": "它有什么优点"},
    ]
    llm_messages, _ = rag._retrieve_context(1, msgs)

    # 历史传给 query_document，供指代改写
    assert seen["query"] == "它有什么优点"
    assert seen["history"] == msgs[:-1]
    # llm_messages = 历史 + system(资料) + 最后一问
    assert len(llm_messages) == 4
    assert llm_messages[:2] == msgs[:2]
    assert llm_messages[2]["role"] == "system"
    assert llm_messages[3] == msgs[2]


# ---------- /chat SSE 端点 ----------

def test_chat_stream_events_sequence(monkeypatch):
    monkeypatch.setattr(RAG, "_retrieve_context", lambda self, kid, msg: (
        [{"role": "system", "content": "资料+问题"}],
        {"rewritten_query": "什么是RAG？", "chunks": []},
    ))
    monkeypatch.setattr(rag_api, "OpenAI", _FakeOpenAI)
    _CTRL["chunks"], _CTRL["raise_at"] = ["你", "好", "！"], None

    resp = TestClient(app).post(
        "/chat",
        json=_chat_payload([{"role": "user", "content": "什么是RAG？"}]),
        headers={"X-API-Key": main.API_KEY},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    names = [e["event"] for e in events]
    assert names == ["debug", "token", "token", "token", "done"], names

    debug = json.loads("".join(events[0]["data_lines"]))
    assert debug["rewritten_query"] == "什么是RAG？"

    token_text = "".join("".join(e["data_lines"]) for e in events[1:-1])
    assert token_text == "你好！"

    done = json.loads("".join(events[-1]["data_lines"]))
    assert done["message"][-1]["content"] == "你好！"
    assert "processing_time" in done


def test_chat_stream_error_event(monkeypatch):
    monkeypatch.setattr(RAG, "_retrieve_context", lambda self, kid, msg: (
        [{"role": "system", "content": "资料+问题"}],
        {"rewritten_query": "x", "chunks": []},
    ))
    monkeypatch.setattr(rag_api, "OpenAI", _FakeOpenAI)
    _CTRL["chunks"], _CTRL["raise_at"] = ["你"], 1  # 第 2 个 chunk 抛异常

    resp = TestClient(app).post(
        "/chat",
        json=_chat_payload([{"role": "user", "content": "hi"}]),
        headers={"X-API-Key": main.API_KEY},
    )
    assert resp.status_code == 200  # 已发出 debug/token，只能靠 error 事件识别
    events = _parse_sse(resp.text)
    names = [e["event"] for e in events]
    assert names == ["debug", "token", "error"], names
    err = json.loads("".join(events[-1]["data_lines"]))
    assert "message" in err


def test_chat_retrieval_failure_envelope(monkeypatch):
    def _boom(self, kid, msg):
        raise RuntimeError("ES down")

    monkeypatch.setattr(RAG, "_retrieve_context", _boom)

    resp = TestClient(app).post(
        "/chat",
        json=_chat_payload([{"role": "user", "content": "hi"}]),
        headers={"X-API-Key": main.API_KEY},
    )
    assert resp.status_code == 500
    body = resp.json()
    assert body["process_status"] == "failed"
    assert body["response_code"] == 500
    assert body["response_msg"] == "ES down"
