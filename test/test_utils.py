import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import safe_remove_file


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
