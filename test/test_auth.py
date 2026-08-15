import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import is_public_path, resolve_api_key


def test_public_paths_are_public():
    for p in ["/health", "/docs", "/openapi.json", "/redoc"]:
        assert is_public_path(p) is True


def test_api_paths_require_auth():
    for p in ["/chat", "/v1/knowledge_base", "/v1/document", "/v1/embedding"]:
        assert is_public_path(p) is False


def test_resolve_api_key_env_overrides(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "env-key")
    assert resolve_api_key({"rag": {"api_key": "cfg-key"}}) == "env-key"


def test_resolve_api_key_config_fallback(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    assert resolve_api_key({"rag": {"api_key": "cfg-key"}}) == "cfg-key"


def test_resolve_api_key_default(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    assert resolve_api_key({}) == "rag-dev-key"
