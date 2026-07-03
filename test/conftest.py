"""
运行所有测试：
    pytest test/ -v

只运行权限测试：
    pytest test/test_permission.py -v

只运行Query改写测试：
    pytest test/test_query_rewrite.py -v
"""
import pytest
import sys
import os

# 确保从项目根目录运行
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)
