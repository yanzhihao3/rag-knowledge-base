"""认证接口：登录、刷新、当前用户、创建用户（仅管理员）。"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from db_api import Session as SessionLocal, User
from router_schemas import RefreshRequest, RegisterRequest, TokenResponse, UserResponse
from security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    ROLE_PERMISSIONS,
    Principal,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_principal,
    hash_password,
    require_permission,
    verify_password,
)

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user),
        refresh_token=create_refresh_token(user),
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """表单登录（OAuth2 password 流程，Swagger 的 Authorize 按钮直接可用）。"""
    with SessionLocal() as session:
        user = session.query(User).filter(User.username == form.username).first()
        # 统一错误信息：不区分“用户不存在/密码错误”，防止用户名枚举
        if user is None or not user.is_active or not verify_password(form.password, user.password_hash):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        user.last_login_dt = datetime.now()
        session.commit()
        session.refresh(user)
        return _token_response(user)


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest):
    """用 refresh token 换一组新 token（refresh 轮换，降低泄露影响）。"""
    claims = decode_token(payload.refresh_token, "refresh")
    with SessionLocal() as session:
        user = session.query(User).filter(User.user_id == int(claims["sub"])).first()
        if user is None or not user.is_active:
            raise HTTPException(status_code=401, detail="用户不存在或已被禁用")
        return _token_response(user)


@router.get("/me", response_model=UserResponse)
def me(principal: Principal = Depends(get_current_principal)):
    return UserResponse(
        user_id=principal.user_id or 0,
        username=principal.username,
        role=principal.role,
        department_id=principal.department_id or 0,
    )


@router.post("/register", response_model=UserResponse, status_code=201)
def register(payload: RegisterRequest,
             principal: Principal = Depends(require_permission("user:manage"))):
    """创建用户：仅 admin / system（X-API-Key）可调用。"""
    if payload.role not in ROLE_PERMISSIONS:
        raise HTTPException(status_code=400, detail=f"非法角色: {payload.role}")
    try:
        password_hash = hash_password(payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    with SessionLocal() as session:
        if session.query(User).filter(User.username == payload.username).first():
            raise HTTPException(status_code=409, detail="用户名已存在")
        user = User(
            username=payload.username,
            password_hash=password_hash,
            role=payload.role,
            department_id=payload.department_id,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            role=user.role,
            department_id=user.department_id,
        )
