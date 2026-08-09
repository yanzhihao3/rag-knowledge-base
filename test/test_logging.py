import os
import sys
import logging
from logging.handlers import RotatingFileHandler

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_config import setup_logging, request_id_var, RequestIdFilter


class TestLoggingConfig:
    # 为什么先写测试：logging_config.py 会被整个项目 import，
    # 写错会污染全局、所有模块的日志一起坏且难查。
    # 所以用测试把"正确行为"钉死，实现完跑绿 = 地基就位。
    # 后续 main.py/rag_api.py 只是在上层盖楼，不再改动这里。
    # 这套测试真实抓过 bug：第一次跑 assert 30 == 20 失败（级别是 WARNING 不是 INFO），
    # 根因是 pytest 环境里 root logger 已有其他 handler，最初的守卫在设级别前就 return 了。

    def test_setup_logging_creates_rotating_handler(self):
        # 锁住：setup_logging() 后，全局 = INFO 级别 + 文件handler + 控制台handler。
        # 任何一个缺失，日志系统就是残缺的。
        setup_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert any(isinstance(h, RotatingFileHandler) for h in root.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    def test_setup_logging_is_idempotent(self):
        # 锁住：多次调用 setup_logging()，handler 数量不变（幂等）。
        # 否则每次 import 都重复加 handler，日志会打 N 遍。
        setup_logging()
        before = len(logging.getLogger().handlers)
        setup_logging()
        after = len(logging.getLogger().handlers)
        assert before == after

    def test_request_id_filter_injects_request_id(self):
        # 锁住：RequestIdFilter 能把当前上下文里的 request_id 挂到日志记录上。
        # set/reset 成对出现是 ContextVar 的规范用法，用完还原，不影响其他测试。
        record = logging.LogRecord("test", logging.INFO, "file.py", 1, "msg", None, None)
        token = request_id_var.set("req-123")
        try:
            assert RequestIdFilter().filter(record) is True
            assert record.request_id == "req-123"
        finally:
            request_id_var.reset(token)

    def test_log_writes_to_file_with_request_id(self, tmp_path):
        # 锁住：request_id 真的写进文件，不是只在内存里。
        # 这是全链路最关键的一条——证明"带id的日志落到文件"的通路是通的，
        # main.py 中间件只是给这个 id 赋值而已。
        # 用 tmp_path + 独立 logger，避免污染全局 handler，测试互相独立。
        logger = logging.getLogger("test_rag_log")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(tmp_path / "test.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(request_id)s | %(message)s"))
        handler.addFilter(RequestIdFilter())
        logger.addHandler(handler)
        token = request_id_var.set("req-abc")
        try:
            logger.info("hello world")
        finally:
            request_id_var.reset(token)
        handler.flush()
        content = (tmp_path / "test.log").read_text(encoding="utf-8")
        assert "req-abc" in content
        assert "hello world" in content
