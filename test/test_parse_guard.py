# -*- coding: utf-8 -*-
"""解析阶段的"静默失败"守门（2026-09 修复）。

修复前有两个问题：

  1. **纯图页写入空记录**：`chunk_content` 为空的块照样进索引，kNN 会召回它，
     纯属噪音（实测那本书 240 页里有 14 条这样的记录）。
  2. **没有文本层却报成功**：整份 PDF 抽不出一个字时，任务照样被标成 success——
     用户看到"处理完成"，实际一个字都搜不到。这比直接失败糟糕得多。

这里用**假的 ES** 替换真索引，测解析逻辑本身：不污染知识库，也不需要 ES 在跑。
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag_api  # noqa: E402
from rag_api import DocumentParseError, RAG  # noqa: E402


class _FakeES:
    """只收下 index() 的调用，不真的写 ES。"""

    def __init__(self):
        self.docs = {}

    def index(self, index, id, document):
        self.docs[id] = document
        return {"result": "created"}


def _write_pdf(path, pages):
    """生成 PDF：pages 里 None 表示空白页，字符串表示写一行文字。"""
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=14)
    doc.save(path)
    doc.close()


@pytest.fixture
def parse_env(monkeypatch):
    """返回 (rag, fake_es, tmpdir)，ES 和删块函数都已替换成假的。"""
    fake = _FakeES()
    monkeypatch.setattr(rag_api, "es", fake)
    monkeypatch.setattr(rag_api, "delete_document_chunks", lambda doc_id: 0)
    tmp = tempfile.mkdtemp(prefix="rag_parse_guard_")
    return RAG(), fake, tmp


def _page_records(fake, document_id):
    """取该文档的页级块 id（排除子块和 document_meta）。"""
    prefix = f"doc_{document_id}_page_"
    return [k for k in fake.docs
            if k.startswith(prefix) and "_chunk_" not in k]


class TestParseGuard:
    def test_纯图片PDF_明确报错(self, parse_env):
        """一页有效内容都没有 → 必须抛错，不能静默成功。"""
        rag, _, tmp = parse_env
        path = os.path.join(tmp, "blank.pdf")
        _write_pdf(path, [None, None])

        with pytest.raises(DocumentParseError) as exc:
            rag._extract_pdf_content(1, 1, "blank", path)

        msg = str(exc.value)
        assert "文本层" in msg, f"错误信息要说清原因，实际是: {msg}"

    def test_混合PDF_空页被跳过(self, parse_env):
        """第 0 页有文字、第 1 页是空白 → 只写入第 0 页，且不写空内容。"""
        rag, fake, tmp = parse_env
        path = os.path.join(tmp, "mixed.pdf")
        _write_pdf(path, ["This page has a real text layer.", None])

        rag._extract_pdf_content(1, 2, "mixed", path)

        assert len(_page_records(fake, 2)) == 1, "空白页不该被写入索引"
        for key, doc in fake.docs.items():
            if key.startswith("docmeta_"):
                continue
            assert (doc.get("chunk_content") or "").strip(), \
                f"不允许写入空的 chunk_content: {key}"

    def test_正常PDF_每页都写入(self, parse_env):
        rag, fake, tmp = parse_env
        path = os.path.join(tmp, "normal.pdf")
        _write_pdf(path, ["First page text.", "Second page text."])

        rag._extract_pdf_content(1, 3, "normal", path)

        assert len(_page_records(fake, 3)) == 2
        assert "docmeta_3" in fake.docs, "document_meta 应该照常写入"

    def test_不支持的类型_抛错而不是静默返回(self, parse_env):
        """原来这里 return（等于静默成功），现在必须抛错。"""
        rag, _, tmp = parse_env
        path = os.path.join(tmp, "fake.docx")
        with open(path, "w", encoding="utf-8") as f:
            f.write("not a pdf")

        with pytest.raises(DocumentParseError):
            rag.extract_content(1, 4, "docx", "application/msword", path)
