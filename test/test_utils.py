import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import safe_remove_file, with_retry


def test_safe_remove_file_deletes_existing(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hi")
    assert p.exists()
    safe_remove_file(str(p))
    assert not p.exists()


def test_safe_remove_file_missing_is_silent(tmp_path):
    safe_remove_file(str(tmp_path / "nope.txt"))  # 不存在不抛异常


def test_safe_remove_file_empty_path_is_silent():
    safe_remove_file(None)  # 空路径不抛异常
    safe_remove_file("")


class _DeterministicError(Exception):
    """模拟"重试也没用"的确定性失败（比如 PDF 没有文本层）。"""


def test_with_retry_默认会重试():
    calls = []

    @with_retry(max_retries=3, base_delay=0.01)
    def always_fail():
        calls.append(1)
        raise _DeterministicError("临时抖动")

    with pytest.raises(_DeterministicError):
        always_fail()
    assert len(calls) == 3, "默认行为应该重试满 max_retries 次"


def test_with_retry_no_retry_只调用一次():
    """确定性失败不该重试：重试只是白等几秒 + 刷误导性的重试日志。"""
    calls = []

    @with_retry(max_retries=3, base_delay=0.01, no_retry=(_DeterministicError,))
    def always_fail():
        calls.append(1)
        raise _DeterministicError("这份 PDF 没有文本层")

    with pytest.raises(_DeterministicError):
        always_fail()
    assert len(calls) == 1, f"no_retry 的异常应该只调用一次，实际 {len(calls)} 次"


def test_with_retry_no_retry_不影响其他异常():
    """no_retry 只拦住指定类型，别的异常照旧重试。"""
    calls = []

    @with_retry(max_retries=3, base_delay=0.01, no_retry=(_DeterministicError,))
    def fail_other():
        calls.append(1)
        raise ValueError("临时故障")

    with pytest.raises(ValueError):
        fail_other()
    assert len(calls) == 3
