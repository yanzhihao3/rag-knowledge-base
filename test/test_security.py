"""认证与授权核心逻辑测试（轻量，不需要 ES / 模型 / Ollama，CI 可跑）。"""
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# JWT 密钥走环境变量；CI 里没有 .env，这里给测试值即可
os.environ.setdefault("RAG_JWT_SECRET", "test-only-secret-do-not-use-in-prod")

import pytest
from fastapi import HTTPException

from db_api import KnowledgeDatabase, User
from security import (
    ROLE_PERMISSIONS,
    Principal,
    _create_token,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    tenant_conditions,
    verify_password,
)


def _fake_user() -> User:
    """构造未落库的 User 对象，只用于签发 token。"""
    user = User(username="alice", role="editor", department_id=2)
    user.user_id = 7
    return user


class TestPassword:
    def test_hash_is_not_plaintext_and_verifies(self):
        hashed = hash_password("S3cret-pass")
        assert hashed != "S3cret-pass"
        assert hashed.startswith("$2")            # bcrypt 哈希格式
        assert verify_password("S3cret-pass", hashed) is True
        assert verify_password("wrong-pass", hashed) is False

    def test_overlong_password_rejected(self):
        with pytest.raises(ValueError):
            hash_password("x" * 73)               # bcrypt 上限 72 字节

    def test_invalid_hash_returns_false(self):
        assert verify_password("whatever", "not-a-valid-hash") is False


class TestToken:
    def test_access_token_roundtrip(self):
        claims = decode_token(create_access_token(_fake_user()), "access")
        assert claims["sub"] == "7"
        assert claims["role"] == "editor"
        assert claims["department_id"] == 2
        assert claims["type"] == "access"

    def test_refresh_token_cannot_be_used_as_access(self):
        refresh = create_refresh_token(_fake_user())
        with pytest.raises(HTTPException) as exc:
            decode_token(refresh, "access")
        assert exc.value.status_code == 401

    def test_expired_token_rejected(self):
        expired = _create_token(_fake_user(), "access", timedelta(seconds=-1))
        with pytest.raises(HTTPException) as exc:
            decode_token(expired, "access")
        assert exc.value.status_code == 401

    def test_tampered_token_rejected(self):
        token = create_access_token(_fake_user())
        tampered = token[:-3] + ("aaa" if not token.endswith("aaa") else "bbb")
        with pytest.raises(HTTPException) as exc:
            decode_token(tampered, "access")
        assert exc.value.status_code == 401


class TestAuthorization:
    def test_role_permission_matrix(self):
        assert "user:manage" in ROLE_PERMISSIONS["admin"]
        assert "user:manage" not in ROLE_PERMISSIONS["editor"]
        assert "kb:write" not in ROLE_PERMISSIONS["viewer"]
        assert "doc:delete" in ROLE_PERMISSIONS["editor"]

    def test_tenant_scope_by_role(self):
        viewer = Principal(user_id=1, username="v", role="viewer", department_id=2)
        admin = Principal(user_id=2, username="a", role="admin", department_id=1)
        system = Principal(user_id=None, username="system", role="system", department_id=None)

        assert viewer.department_scope() == 2
        assert admin.department_scope() is None          # admin 可跨部门
        assert system.department_scope() is None         # 系统通道可跨部门

    def test_tenant_conditions_generate_department_filter(self):
        viewer = Principal(user_id=1, username="v", role="viewer", department_id=2)
        admin = Principal(user_id=2, username="a", role="admin", department_id=1)

        conditions = tenant_conditions(KnowledgeDatabase, viewer)
        assert len(conditions) == 1
        assert "department_id" in str(conditions[0])
        assert tenant_conditions(KnowledgeDatabase, admin) == []
