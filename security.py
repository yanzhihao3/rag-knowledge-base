"""认证与授权：密码哈希、JWT 签发/校验、权限校验、租户过滤条件。

设计要点：
- 密码只存 bcrypt 哈希，永不存明文；
- access token 短有效期（30 分钟），refresh token 长有效期（7 天），用 type 字段防混用；
- 每次请求都从数据库读用户最新状态：用户被禁用/调岗后立即生效，不长期信任 token 里的旧信息；
- 兼容 X-API-Key：映射为 system 角色（跨部门），供脚本 / CI 使用。
"""
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import bcrypt
import jwt
import yaml
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer

from auth import resolve_api_key
from db_api import Session as SessionLocal, User

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
BCRYPT_MAX_PASSWORD_BYTES = 72   # bcrypt 只取前 72 字节，超长必须拦截

# 角色 → 权限集合：集中维护，便于测试与扩展
ROLE_PERMISSIONS = {
    "admin": {
        "ai:use", "kb:read", "kb:write", "kb:delete",
        "doc:read", "doc:write", "doc:delete", "user:manage",
    },
    "editor": {
        "ai:use", "kb:read", "kb:write", "kb:delete",
        "doc:read", "doc:write", "doc:delete",
    },
    "viewer": {"ai:use", "kb:read", "doc:read"},
    # 系统通道（X-API-Key）：跨部门管理员，用于脚本 / CI
    "system": {
        "ai:use", "kb:read", "kb:write", "kb:delete",
        "doc:read", "doc:write", "doc:delete", "user:manage",
    },
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _jwt_secret() -> str:
    secret = os.environ.get("RAG_JWT_SECRET") or _load_config().get("rag", {}).get("jwt_secret")
    if not secret:
        raise RuntimeError("未配置 JWT 密钥：请设置环境变量 RAG_JWT_SECRET（参考 .env.example）")
    return secret


# ===== 密码哈希 =====
def hash_password(raw_password: str) -> str:
    """生成 bcrypt 哈希（自带随机盐）。超长密码直接拒绝，避免静默截断。"""
    if len(raw_password.encode("utf-8")) > BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError(f"密码过长：bcrypt 最多支持 {BCRYPT_MAX_PASSWORD_BYTES} 字节")
    return bcrypt.hashpw(raw_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(raw_password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(raw_password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ===== JWT =====
def _create_token(user: User, token_type: str, expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.user_id),
        "username": user.username,
        "role": user.role,
        "department_id": user.department_id,
        "type": token_type,
        "jti": uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)


def create_access_token(user: User) -> str:
    return _create_token(user, "access", timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))


def create_refresh_token(user: User) -> str:
    return _create_token(user, "refresh", timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))


def decode_token(token: str, expected_type: str) -> dict:
    """校验签名与有效期，并确认 token 类型（access 不能当 refresh 用，反之亦然）。"""
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token 已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="token 无效")
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=401, detail="token 类型不匹配")
    return payload


# ===== 当前身份与权限 =====
@dataclass
class Principal:
    """一次请求的调用者身份。user_id 为 None 表示系统通道（X-API-Key）。"""
    user_id: Optional[int]
    username: str
    role: str
    department_id: Optional[int]

    def department_scope(self) -> Optional[int]:
        """返回数据过滤用的部门 ID；None 表示不限制（admin / system 可跨部门）。"""
        if self.role in ("admin", "system"):
            return None
        return self.department_id


def tenant_conditions(model, principal: Principal) -> list:
    """生成租户过滤条件，调用处用 *tenant_conditions(...) 展开到 filter() 里。"""
    scope = principal.department_scope()
    if scope is None:
        return []
    return [model.department_id == scope]


def _principal_from_token(token: str) -> Principal:
    payload = decode_token(token, "access")
    user_id = int(payload["sub"])
    with SessionLocal() as session:
        user = session.query(User).filter(User.user_id == user_id).first()
        if user is None or not user.is_active:
            raise HTTPException(status_code=401, detail="用户不存在或已被禁用")
        # 以数据库为准，不信任 token 里可能过期的 role / department_id
        return Principal(
            user_id=user.user_id,
            username=user.username,
            role=user.role,
            department_id=user.department_id,
        )


def get_current_principal(token: str = Depends(oauth2_scheme),
                          api_key: str = Security(api_key_header)) -> Principal:
    """认证入口：优先 Bearer JWT，其次 X-API-Key（系统通道）。"""
    if token:
        return _principal_from_token(token)
    if api_key:
        expected = resolve_api_key(_load_config())
        if secrets.compare_digest(api_key, expected):
            return Principal(user_id=None, username="system", role="system", department_id=None)
        raise HTTPException(status_code=401, detail="API Key 无效")
    raise HTTPException(status_code=401, detail="缺少认证凭据")


def require_permission(permission: str):
    """依赖工厂：require_permission("kb:write") → 校验当前用户是否具备该权限。"""
    def _checker(principal: Principal = Depends(get_current_principal)) -> Principal:
        if permission not in ROLE_PERMISSIONS.get(principal.role, set()):
            raise HTTPException(status_code=403, detail=f"缺少权限: {permission}")
        return principal

    return _checker
