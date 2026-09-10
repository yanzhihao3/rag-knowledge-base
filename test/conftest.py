"""
pytest 全局配置，负责两件事：

1. 测试隔离：强制所有测试使用一次性 SQLite 临时库，
   绝不触碰开发库（MySQL 的 rag 库 / 本地 rag.db）。
   原理：真实环境变量优先级高于 .env，所以即使 .env 指向 MySQL，
   测试进程也会被强制切到临时 SQLite。

2. 测试分层：CI 里设 RAG_SKIP_INTEGRATION=1，
   跳过需要 Elasticsearch / 本地模型 / Ollama 的重型测试
   （这些测试 import rag_api 会加载 torch，CI 跑不动也没必要跑）。

常用命令：
    pytest test/ -v                      # 本地全量（需要 ES/模型）
    $env:RAG_SKIP_INTEGRATION='1'; pytest # 只跑轻量单元测试（CI 用）
"""
import atexit
import os
import shutil
import sys
import tempfile

# 确保从项目根目录运行
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

# ===== 1. 测试隔离：必须早于任何测试模块 import db_api =====
_TEST_DB_DIR = tempfile.mkdtemp(prefix="rag_test_db_")
os.environ["RAG_DB_ENGINE"] = "sqlite"
os.environ["RAG_DB_PATH"] = os.path.join(_TEST_DB_DIR, "test_rag.db")


def _cleanup_test_db() -> None:
    shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)


atexit.register(_cleanup_test_db)

# ===== 2. 测试分层：CI 跳过的重型测试（需要 ES / 模型 / Ollama）=====
_INTEGRATION_TESTS = [
    "test_recall.py",
    "test_benchmark.py",
    "benchmark_report.py",
    "test_chat_stream.py",
    "test_chat.py",
    "test_parallel_recall.py",
    "test_pdf_processing.py",
    "test_query_rewrite.py",
]

collect_ignore = []
if os.environ.get("RAG_SKIP_INTEGRATION") == "1":
    collect_ignore += [
        os.path.join(os.path.dirname(__file__), name) for name in _INTEGRATION_TESTS
    ]
