import os
import sys
import logging
from logging.handlers import RotatingFileHandler

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_config import setup_logging, request_id_var, RequestIdFilter


class TestLoggingConfig:
    def test_setup_logging_creates_rotating_handler(self):
        setup_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert any(isinstance(h, RotatingFileHandler) for h in root.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    def test_setup_logging_is_idempotent(self):
        setup_logging()
        before = len(logging.getLogger().handlers)
        setup_logging()
        after = len(logging.getLogger().handlers)
        assert before == after

    def test_request_id_filter_injects_request_id(self):
        record = logging.LogRecord("test", logging.INFO, "file.py", 1, "msg", None, None)
        token = request_id_var.set("req-123")
        try:
            assert RequestIdFilter().filter(record) is True
            assert record.request_id == "req-123"
        finally:
            request_id_var.reset(token)

    def test_log_writes_to_file_with_request_id(self, tmp_path):
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
