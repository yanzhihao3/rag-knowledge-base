import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_api import split_text_with_overlap, _format_related_document, split_incomplete_sentence


class TestPDFProcessing:
    """PDF处理功能测试：表格提取 + 跨页断句"""

    def test_split_text_with_overlap(self):
        """测试分块函数基本功能"""
        text = "这是第一句话。这是第二句话。这是第三句话。"
        chunks = split_text_with_overlap(text, chunk_size=5, chunk_overlap=2)
        assert len(chunks) > 0
        assert isinstance(chunks, list)

    def test_split_text_preserves_content(self):
        """测试分块后内容被保留"""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = split_text_with_overlap(text, chunk_size=10, chunk_overlap=3)
        # 所有chunk拼接应该包含原始文本
        combined = "".join(chunks)
        assert "A" in combined
        assert "Z" in combined

    def test_cross_page_sentence_detection(self):
        """测试跨页断句检测逻辑"""
        # 如果文本末尾没有句号，说明句子被切断了
        incomplete_sentence = "这是被切断的句子没有句号"
        complete_sentence = "这是完整的句子。"

        def is_complete(text):
            if not text:
                return True
            return text.strip()[-1] in ('。', '！', '？', '.', '!', '?')

        assert is_complete(complete_sentence) is True
        assert is_complete(incomplete_sentence) is False
        assert is_complete("") is True  # 空文本算完整

    def test_cross_page_join(self):
        """测试跨页句子拼接逻辑"""
        prev_tail = "没有句号的尾巴"
        current = "这是下一页的内容。"
        joined = prev_tail + current
        assert "没有句号的尾巴这是下一页的内容。" == joined

    def test_table_content_format(self):
        """测试表格内容格式化"""
        mock_table = [
            ["姓名", "年龄", "城市"],
            ["张三", "25", "北京"],
            ["李四", "30", "上海"],
        ]

        table_rows = []
        for row in mock_table:
            if row:
                cleaned_row = [str(cell) if cell else "" for cell in row]
                table_rows.append(" | ".join(cleaned_row))

        table_content = "\n".join(table_rows)
        assert "姓名 | 年龄 | 城市" in table_content
        assert "张三 | 25 | 北京" in table_content
        assert "李四 | 30 | 上海" in table_content


class TestRelatedDocument:
    """检索结果拼接喂给 LLM 的资料（表格必须拼进去）"""

    def test_content_only_when_no_tables(self):
        """没有表格时只拼文本"""
        records = [
            {"chunk_content": ["这是页面文本。"], "chunk_tables": []},
        ]
        assert _format_related_document(records) == "这是页面文本。"

    def test_table_appended_after_content(self):
        """有表格时表格拼在文本后面，喂给 LLM"""
        records = [
            {
                "chunk_content": ["页面文本。"],
                "chunk_tables": ["[表格]\n姓名 | 年龄 | 城市\n张三 | 25 | 北京\n[/表格]"],
            },
        ]
        result = _format_related_document(records)
        assert "页面文本。" in result
        assert "张三 | 25 | 北京" in result

    def test_multiple_records_tables_in_order(self):
        """多条记录、多个表格都拼进去，顺序为 文本->表格->文本->表格"""
        records = [
            {"chunk_content": ["文本A"], "chunk_tables": ["表1"]},
            {"chunk_content": ["文本B"], "chunk_tables": ["表2", "表3"]},
        ]
        result = _format_related_document(records)
        assert result.index("文本A") < result.index("表1") < result.index("文本B") < result.index("表2") < result.index("表3")


class TestCrossPageSentence:
    """跨页断句中英文标点处理"""

    def test_chinese_complete_sentence(self):
        """末尾是中文句号 → 完整，无尾部"""
        assert split_incomplete_sentence("这是完整的句子。") == ("这是完整的句子。", "")

    def test_chinese_split_at_last_punctuation(self):
        """中文句子被切断 → 从最后一个句号处切开"""
        complete, tail = split_incomplete_sentence("前句完整。后句被切断没有句号")
        assert complete == "前句完整。"
        assert tail == "后句被切断没有句号"

    def test_english_split_at_last_period(self):
        """英文句子被切断 → 从最后一个英文句号处切开（原代码只查中文标点会漏）"""
        complete, tail = split_incomplete_sentence("First sentence. Second sentence cut off")
        assert complete == "First sentence."
        assert tail == " Second sentence cut off"

    def test_mixed_chinese_english(self):
        """中英混合 → 从最后一个句子结束标点切开"""
        complete, tail = split_incomplete_sentence("中文句。English here. more cut")
        assert complete == "中文句。English here."
        assert tail == " more cut"

    def test_empty_text(self):
        """空文本 → 完整，无尾部"""
        assert split_incomplete_sentence("") == ("", "")
        assert split_incomplete_sentence("   ") == ("", "")

    def test_no_punctuation_returns_unchanged(self):
        """没有任何句子结束标点 → 原样返回，无尾部"""
        assert split_incomplete_sentence("没有句号的文本") == ("没有句号的文本", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])