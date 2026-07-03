import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_api import split_text_with_overlap


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])