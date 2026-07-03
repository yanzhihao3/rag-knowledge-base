import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_api import with_retry


class TestRetryMechanism:
    """失败重试机制测试"""

    def test_retry_success_first_try(self):
        """测试一次成功：不应该重试"""
        call_count = 0

        @with_retry(max_retries=3)
        def succeed_once():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed_once()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """测试失败后重试成功"""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.1)
        def fail_twice_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"第{call_count}次失败")
            return "success"

        result = fail_twice_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        """测试重试耗尽：应该抛出最后一次异常"""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.1)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("总是失败")

        with pytest.raises(ValueError) as exc_info:
            always_fail()

        assert "总是失败" in str(exc_info.value)
        assert call_count == 3

    def test_retry_decorator_preserves_function_name(self):
        """测试装饰器保留原函数名"""
        @with_retry(max_retries=2)
        def my_function():
            return "done"

        assert my_function.__name__ == "my_function"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])