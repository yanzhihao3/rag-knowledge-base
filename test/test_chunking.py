# -*- coding: utf-8 -*-
"""语义分块单元测试（需要 rag_api，会加载模型，属于重型测试）。

对照物：旧实现 split_text_with_overlap 是纯字符窗口，会在句子中间和英文单词
中间切开（`Co-\\noccurrence` 这种）。新实现 split_text_semantic 的硬约束是：
除"单句本身就超长"这一种情况外，**任何一块都必须以完整句子结尾**。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_api import (  # noqa: E402
    _format_related_document,
    join_visual_lines,
    split_sentences,
    split_text_semantic,
    split_text_with_overlap,
)


def _make_text(n_sent=30, sent_len=20):
    """造一段"每句都是完整句子、且互不相同"的中文文本，每句长度固定。

    句子必须互不相同：否则"块间是否重叠"的断言会因为相邻句子内容一样而误判。
    """
    out = []
    for i in range(n_sent):
        prefix = f"句{i:03d}"
        out.append(prefix + "甲" * (sent_len - len(prefix) - 1) + "。")
    return "".join(out)


class TestJoinVisualLines:
    """pdfplumber 的视觉行 → 逻辑行。用例取自真实的《大模型RAG实战》PDF。"""

    def test_中文折行拼接不补空格(self):
        text = ("相比词袋模型，跳跃模型增加了计算的复杂性，因为它必须根据一定数\n"
                "量的相邻单词来预测邻近单词。")
        assert join_visual_lines(text) == [
            "相比词袋模型，跳跃模型增加了计算的复杂性，因为它必须根据一定数量的相邻单词来预测邻近单词。"
        ]

    def test_英文连字符折行接回单词(self):
        text = "GloVe通过引入共现矩阵（Co-\noccurrence Probabilities Matrix）来表征。"
        assert join_visual_lines(text) == [
            "GloVe通过引入共现矩阵（Co-occurrence Probabilities Matrix）来表征。"
        ]

    def test_句末标点后是新段(self):
        assert join_visual_lines("第一句结束了。\n第二句开始。") == [
            "第一句结束了。", "第二句开始。"
        ]

    def test_章节标题既不粘前也不粘后(self):
        text = "上一段没有句号结尾\n7.3 基于Streamlit搭建应用\n正文从这里开始。"
        lines = join_visual_lines(text)
        assert "7.3 基于Streamlit搭建应用" in lines, "标题必须独立成段"
        assert "正文从这里开始。" in lines, "正文不能被粘到标题后面"

    def test_图表题注独立成段(self):
        text = "图3-2 稠密向量检索模型输入输出模式\n1）将[cls]对应的向量作为句向量。"
        lines = join_visual_lines(text)
        assert "图3-2 稠密向量检索模型输入输出模式" in lines
        assert "1）将[cls]对应的向量作为句向量。" in lines

    def test_数字点标题独立成段(self):
        """这本 PDF 里的 "2. RAG" 这种小节标题，不能被粘到下一行正文上。"""
        text = "这是上一段的结尾。\n2. RAG\n2020年，Facebook公司首先提出了RAG框架。"
        lines = join_visual_lines(text)
        assert "2. RAG" in lines, "「2. RAG」这种标题必须独立成段"
        assert "2020年，Facebook公司首先提出了RAG框架。" in lines

    def test_普通数字开头的正文不被误判(self):
        """正文行以数字开头但没有点号时，不该被当成结构行强行断开。"""
        text = "上一行没有标点结尾\n2020年，Facebook公司首先提出了RAG框架。"
        assert join_visual_lines(text) == [
            "上一行没有标点结尾2020年，Facebook公司首先提出了RAG框架。"
        ]

    def test_空输入(self):
        assert join_visual_lines("") == []
        assert join_visual_lines(None) == []


class TestSplitSentences:
    def test_中英文标点(self):
        assert split_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"]

    def test_末尾残句也保留(self):
        assert split_sentences("完整一句。半句没有标点") == ["完整一句。", "半句没有标点"]

    def test_分号算边界(self):
        assert split_sentences("A；B。") == ["A；", "B。"]

    def test_空(self):
        assert split_sentences("") == []


class TestSplitTextSemantic:
    def test_任何一块都以完整句子结尾(self):
        chunks = split_text_semantic(_make_text(), target_size=200, overlap_sentences=1)
        assert len(chunks) > 1, "这段文本应该被切成多块"
        for c in chunks:
            assert c.endswith("。"), f"切在句子中间了: ...{c[-30:]}"

    def test_块长不超过硬上限(self):
        chunks = split_text_semantic(_make_text(), target_size=200, overlap_sentences=1)
        assert all(len(c) <= 300 for c in chunks), "单块不应超过 target_size * 1.5"

    def test_重叠是整句而不是半句(self):
        chunks = split_text_semantic(_make_text(), target_size=120, overlap_sentences=1)
        for prev, nxt in zip(chunks, chunks[1:]):
            last_sent = split_sentences(prev)[-1]
            assert nxt.startswith(last_sent), "块间重叠必须是一整句"

    def test_不重叠时不共享内容(self):
        chunks = split_text_semantic(_make_text(), target_size=120, overlap_sentences=0)
        for prev, nxt in zip(chunks, chunks[1:]):
            assert not nxt.startswith(split_sentences(prev)[-1])

    def test_超长单句只能硬切(self):
        text = "甲" * 1000
        chunks = split_text_semantic(text, target_size=200, overlap_sentences=1)
        assert all(len(c) <= 200 for c in chunks)
        assert "".join(chunks) == text, "硬切不应丢字"

    def test_空输入返回空(self):
        assert split_text_semantic("") == []
        assert split_text_semantic(None) == []

    def test_对比旧实现会切在词中间(self):
        """这条用例的作用是把"为什么要换分块"钉死在测试里。"""
        text = "前面有一句铺垫。Co-occurrence 这个词必须完整保留。"
        fixed = split_text_with_overlap(text, 20, 0)
        assert not fixed[0].endswith("。"), "旧实现会在句子中间切"
        assert "Co-occurrenc" in fixed[0], "旧实现会把英文单词切断"

        semantic = split_text_semantic(text, target_size=20, overlap_sentences=0)
        assert all(c.endswith("。") for c in semantic), "新实现每块都是完整句子"


class TestFormatRelatedDocument:
    """喂给 LLM 的资料拼装。

    这段曾经有一个只在 parent_child 模式下才出现的致命 bug：
    chunk_content 是**字符串**时，`(x.get("chunk_content") or [""])[0]` 取到的是
    **第一个字符**，于是 5 条记录只给 LLM 送了 5 个字，模型只能回"无法回答"。
    召回率指标完全看不到这个问题，是回答准确率评测暴露的。
    """

    def test_fields形状是列表(self):
        records = [{"chunk_content": ["正文A"], "chunk_tables": ["表格B"]}]
        assert _format_related_document(records) == "正文A\n表格B"

    def test_source形状是字符串(self):
        records = [{"chunk_content": "正文A", "chunk_tables": ["表格B"]}]
        assert _format_related_document(records) == "正文A\n表格B"

    def test_字符串不能被截成一个字(self):
        records = [{"_id": "doc_3_page_7_chunk_1", "parent_id": "doc_3_page_7",
                    "chunk_content": "这是整页正文，长度远超一个字。",
                    "chunk_tables": None, "child_content": "子块"}]
        out = _format_related_document(records)
        assert out.startswith("这是整页正文")
        assert len(out) > 5

    def test_缺字段和空列表不炸(self):
        assert _format_related_document([{}]) == ""
        assert _format_related_document([]) == ""


class TestAnswerScoring:
    """回答准确率的要点覆盖判定（来自真实案例的回归测试）。

    模型的回答语义完全正确，但旧的逐字匹配给它 0 分 —— 这条用例把修法钉死。
    """

    def test_语义相同但措辞不同要算命中(self):
        from test.benchmark_report import _point_covered

        answer = ("RAG通过提供每个答案的来源信息，让用户可以看到来源文档，"
                  "增强了回答的可信度。")
        assert _point_covered(answer, "给出答案来源"), "「来源」对上了，不能判 0"
        assert _point_covered(answer, "增强可信度")

    def test_真正没答到的不算命中(self):
        from test.benchmark_report import _point_covered

        answer = "RAG是一种检索增强生成技术，它结合了检索和生成。"
        assert not _point_covered(answer, "可追溯文档")
        assert not _point_covered(answer, "增强可信度")

    def test_逐字一致当然算命中(self):
        from test.benchmark_report import _point_covered

        assert _point_covered("答案更个性化", "答案更个性化")

    def test_空回答和空要点(self):
        from test.benchmark_report import _point_covered

        assert not _point_covered("", "给出答案来源")
        assert _point_covered("随便什么", "")
