"""API 级认证 / 授权 / 租户隔离测试。

需要导入 main（会加载 torch 与 ES 连接），属重型测试，CI 里由 conftest 跳过；
本地运行：pytest test/test_auth_api.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from db_api import Base, Session, User, engine
from security import hash_password
import main

pytestmark = pytest.mark.integration

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def clean_users():
    Base.metadata.create_all(engine)
    with Session() as session:
        session.query(User).delete()
        session.commit()
    yield
    with Session() as session:
        session.query(User).delete()
        session.commit()


def _add_user(username: str, password: str, role: str, department_id: int) -> None:
    with Session() as session:
        session.add(User(
            username=username,
            password_hash=hash_password(password),
            role=role,
            department_id=department_id,
            is_active=True,
        ))
        session.commit()


def _login(username: str, password: str) -> str:
    resp = client.post("/v1/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_login_success_and_me():
    _add_user("alice", "pass1234", "viewer", 2)
    token = _login("alice", "pass1234")
    resp = client.get("/v1/auth/me", headers=_auth(token))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"user_id": 1, "username": "alice", "role": "viewer", "department_id": 2}


def test_login_wrong_password_returns_401():
    _add_user("bob", "pass1234", "viewer", 2)
    resp = client.post("/v1/auth/login", data={"username": "bob", "password": "wrong-pass"})
    assert resp.status_code == 401


def test_business_endpoint_without_token_returns_401():
    assert client.get("/v1/knowledge_base/list").status_code == 401


def test_viewer_cannot_create_knowledge_base():
    _add_user("carol", "pass1234", "viewer", 2)
    token = _login("carol", "pass1234")
    resp = client.post("/v1/knowledge_base",
                       json={"title": "t", "category": "c"}, headers=_auth(token))
    assert resp.status_code == 403


def test_tenant_isolation_ignores_client_department():
    _add_user("dave", "pass1234", "editor", 2)
    _add_user("eve", "pass1234", "editor", 3)
    dave_token = _login("dave", "pass1234")
    eve_token = _login("eve", "pass1234")

    # 请求体伪造 department_id/owner_id 应被忽略，归属以 token（数据库）为准
    resp = client.post("/v1/knowledge_base",
                       json={"title": "dept2-kb", "category": "c",
                             "department_id": 99, "owner_id": 99},
                       headers=_auth(dave_token))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    knowledge_id = body["knowledge_id"]
    assert body["department_id"] == 2

    # 同部门可见
    same = client.get(f"/v1/knowledge_base?knowledge_id={knowledge_id}", headers=_auth(dave_token))
    assert same.json()["response_code"] == 200

    # 跨部门不可见：与"不存在"同样返回 404，不泄露资源是否存在
    cross = client.get(f"/v1/knowledge_base?knowledge_id={knowledge_id}", headers=_auth(eve_token))
    assert cross.json()["response_code"] == 404

    # 跨部门问答同样被拦（在检索之前就返回 404）
    chat = client.post("/chat",
                       json={"knowledge_id": knowledge_id,
                             "message": [{"role": "user", "content": "hi"}]},
                       headers=_auth(eve_token))
    assert chat.status_code == 404


def test_system_api_key_can_cross_departments():
    """X-API-Key 作为系统通道，仍可用于脚本 / CI（跨部门）。"""
    resp = client.get("/v1/knowledge_base/list", headers={"X-API-Key": "rag-dev-key"})
    assert resp.status_code == 200, resp.text
