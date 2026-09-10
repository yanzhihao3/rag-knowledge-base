import yaml
import os
import re
import sqlite3
import time
import json
import numpy as np
import uuid
import datetime
import logging
import uvicorn
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError
from typing_extensions import Annotated
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Request, Depends, HTTPException
from pydantic import BaseModel
from router_schemas import (
     EmbeddingRequest, EmbeddingResponse,
     RAGRequest,
     RerankRequest, RerankResponse,
     KnowledgeRequest, KnowledgeResponse,
     DocumentRequest, DocumentResponse,
)
from logging_config import setup_logging, request_id_var
from fastapi.responses import JSONResponse, StreamingResponse
from auth_routes import router as auth_router
from security import Principal, require_permission, tenant_conditions

setup_logging()

from rag_api import RAG
from db_api import (
     KnowledgeDocument, KnowledgeDatabase,
     Session, db_type
)
from es_api import delete_document_chunks, delete_knowledge_chunks
from utils import safe_remove_file, task_state_machine

logger = logging.getLogger(__name__)

#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#config_path = os.path.join(project_root, 'config.yaml')

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.2
# MySQL 值得重试的错误码：1213 死锁、1205 锁等待超时
_MYSQL_RETRYABLE_ERRCODES = {1205, 1213}


def _is_retryable_db_error(exc: BaseException) -> bool:
    """按数据库引擎判断错误是否值得重试。

    - SQLite：写锁 "database is locked" 等暂时性错误值得重试；
    - MySQL：只重试死锁(1213)/锁等待超时(1205)，
      业务/参数/逻辑错误重试没有意义，直接抛出。
    """
    if isinstance(exc, sqlite3.OperationalError):
        return db_type == 'sqlite'
    if not isinstance(exc, SQLAlchemyOperationalError):
        return False
    if db_type in ('mysql', 'mysql+pymysql'):
        orig = getattr(exc, 'orig', None)
        code = getattr(orig, 'args', [None])[0] if orig is not None else None
        return code in _MYSQL_RETRYABLE_ERRCODES
    return True

ALLOWED_UPLOAD_EXTENSIONS = {".pdf"}
_UPLOAD_DIR = "upload_files"


def _db_operation_with_retry(fn, *args, **kwargs):
    """执行数据库操作，遇到锁等暂时性错误时指数退避重试。"""
    last_exc = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable_db_error(exc):
                raise
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "[DB重试] %s 第%d次失败，%.1fs后重试: %s",
                    getattr(fn, "__name__", str(fn)), attempt + 1, delay, exc,
                )
                time.sleep(delay)
    logger.error("[DB重试耗尽] %s 失败: %s", getattr(fn, "__name__", str(fn)), last_exc)
    raise last_exc


def _sanitize_filename(filename: str) -> str:
    """把客户端文件名清洗成安全的纯文件名（去掉路径、非法字符）。"""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    # 统一分隔符后只保留最后一段，消除 ../、..\ 这类路径穿越
    base = os.path.basename(filename.replace("\\", "/"))
    # 去掉 Windows/Linux 下的非法字符和控制字符
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip().strip(".")
    if not base or base in {".", ".."}:
        raise HTTPException(status_code=400, detail="非法文件名")
    return base


def _validate_upload_filename(filename: str) -> None:
    """上传入口校验：文件名必须安全，且扩展名在白名单内。"""
    base = _sanitize_filename(filename)
    ext = os.path.splitext(base)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext or '无扩展名'}，仅支持 {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
        )


def _safe_upload_path(document_id: int, filename: str) -> str:
    """生成服务端控制的上传落盘路径，并兜底校验路径不越界。"""
    base = _sanitize_filename(filename)
    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(_UPLOAD_DIR, f"document_{document_id}_{base}")
    uploads_abs = os.path.abspath(_UPLOAD_DIR)
    if os.path.commonpath([uploads_abs, os.path.abspath(file_path)]) != uploads_abs:
        raise HTTPException(status_code=400, detail="非法文件名")
    return file_path


def _rollback_partial_document(document_id: int, file_path: str) -> None:
    """上传中途失败时回收残留：删物理文件 + 删刚插入的 DB 行。"""
    safe_remove_file(file_path)
    try:
        with Session() as session:
            session.query(KnowledgeDocument).filter(
                KnowledgeDocument.document_id == document_id
            ).delete()
            session.commit()
    except Exception:
        logger.exception("回滚残留文档记录失败: document_id=%d", document_id)


with open("config.yaml", 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

app = FastAPI(
    swagger_ui_parameters={
        # 替换成国内较快的 unpkg 镜像源
        "swagger_js_url": "https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        "swagger_css_url": "https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
        # favicon 地址可改可不改
        "swagger_favicon_url": "https://fastapi.tiangolo.com/img/favicon.png",
    }
)

# 认证路由（/v1/auth/login 等）无需全局鉴权，各端点自行声明所需权限
app.include_router(auth_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # 统一错误信封：request_id / response_code / response_msg / process_status
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": request_id_var.get(),
            "response_code": exc.status_code,
            "response_msg": exc.detail,
            "process_status": "failed",
        },
    )

# 日志链路：log_requests 中间件为每个请求发 request_id、记录总耗时与成败；
# 业务层出错时用 logger.exception 保留堆栈，统一错误信封由 HTTPException 处理器输出。
# 鉴权链路：各业务端点通过 Depends(require_permission(...)) 做认证 + 授权，
# OAuth2PasswordBearer 会让 Swagger UI 出现 Authorize 按钮（支持账密登录换 token，也兼容 X-API-Key）。

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    token = request_id_var.set(request_id)
    start = time.monotonic()
    status = 500
    try:
        response = await call_next(request)  #  一句话总结:call_next(request) = "把这个请求放行给真正的业务处理,我等它办完,再接着记录"。
        status = response.status_code
        return response
    except Exception:
        logger.exception("请求处理异常: %s %s", request.method, request.url.path)
        raise
    finally:
        duration = time.monotonic() - start
        logger.info("%s %s -> %d | %.3fs", request.method, request.url.path, status, duration)
        request_id_var.reset(token)


@app.get("/v1/knowledge_base")
def get_knowledge_base(knowledge_id: int,
                       principal: Principal = Depends(require_permission("kb:read"))) -> KnowledgeResponse:
    start_time = time.time()
    try:
        def _query():
            with Session() as session:
                return session.query(KnowledgeDatabase).filter(
                    KnowledgeDatabase.knowledge_id == knowledge_id,
                    # 租户隔离：非 admin 只能查本部门（越权访问与"不存在"同样返回 404）
                    *tenant_conditions(KnowledgeDatabase, principal),
                ).first()

        record = _db_operation_with_retry(_query)
        if record is not None:
            return KnowledgeResponse(
                request_id=str(uuid.uuid4()),
                knowledge_id=knowledge_id,
                title=str(record.title),
                category=str(record.category),
                owner_id=record.owner_id,
                department_id=record.department_id,
                response_code=200,
                response_msg="查询知识库成功",
                process_status="completed",
                process_time=time.time() - start_time,
            )
        return KnowledgeResponse(
            request_id=str(uuid.uuid4()),
            knowledge_id=knowledge_id,
            title="",
            category="",
            owner_id=0,
            department_id=0,
            response_code=404,
            response_msg="知识库不存在",
            process_status="failed",
            process_time=time.time() - start_time,
        )
    except Exception:
        logger.exception("查询知识库失败: knowledge_id=%d", knowledge_id)
    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        title="",
        category="",
        owner_id=0,
        department_id=0,
        response_code=500,
        response_msg="查询知识库失败",
        process_status="failed",
        process_time=time.time() - start_time,
    )

@app.get("/v1/knowledge_base/list")
def list_knowledge_base(principal: Principal = Depends(require_permission("kb:read"))):
    start_time = time.time()
    try:
        with Session() as session:
            records = session.query(KnowledgeDatabase).filter(
                *tenant_conditions(KnowledgeDatabase, principal)
            ).all()
            return {
                "request_id": str(uuid.uuid4()),
                "knowledge_list": [
                    {
                        "knowledge_id": r.knowledge_id,
                        "title": r.title,
                        "category": r.category,
                    }
                    for r in records
                ],
                "response_code": 200,
                "response_msg": "ok",
                "process_status": "completed",
                "process_time": time.time() - start_time,
            }
    except Exception as e:
        logger.exception("查询知识库列表失败")
        return {
            "request_id": str(uuid.uuid4()),
            "knowledge_list": [],
            "response_code": 500,
            "response_msg": str(e),
            "process_status": "failed",
            "process_time": time.time() - start_time,
        }

@app.delete("/v1/knowledge_base")
def delete_knowledge_base(knowledge_id: int,
                          principal: Principal = Depends(require_permission("kb:delete"))) -> KnowledgeResponse:
    start_time = time.time()
    try:
        with Session() as session:
            record = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.knowledge_id == knowledge_id,
                *tenant_conditions(KnowledgeDatabase, principal),
            ).first()
            if record is None:
                return KnowledgeResponse(
                    request_id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id, title="", category="",
                    owner_id=0, department_id=0,
                    response_code=404, response_msg="知识库不存在",
                    process_status="failed",
                    process_time=time.time() - start_time,
                )
            # 响应字段先快照（commit 后实例过期，读取会抛 ObjectDeletedError）快照的作用：在删除记录之前，先把需要返回的字段保存到变量 即使后续记录被删除，这些变量仍然有效
            title, category, owner_id, department_id = (
                record.title, record.category,
                record.owner_id, record.department_id,
            )
            # 该库下所有文档（文件路径 + 子记录都要用）
            documents = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.knowledge_id == knowledge_id
            ).all()
            # id 先快照：commit 后实例过期，读取 document_id 会抛 ObjectDeletedError
            document_ids = [doc.document_id for doc in documents]
            # 1) ES：按 knowledge_id 一把清（失败抛异常 → 阻断，绝不留幽灵分块）
            delete_knowledge_chunks(knowledge_id)
            # 2) 物理文件：失败只告警
            for doc in documents:
                safe_remove_file(doc.file_path)
            # 3) SQLite：先删子文档、再删知识库，同一事务提交
            # 核心理解：doc 不只是一个普通对象，它是带着"数据库上下文"的对象，知道自己从哪里来，也知道如何删除自己。
            for doc in documents:
                session.delete(doc)
            session.delete(record)
            session.commit()
            # 4) 清理任务状态：库下所有文档已删，状态行过期，删掉防误报（低危，失败只告警）
            for document_id in document_ids:
                try:
                    task_state_machine.reset(str(document_id))
                except Exception:
                    logger.warning("清理任务状态失败，需手动清理: document_id=%s", document_id)
            return KnowledgeResponse(
                request_id=str(uuid.uuid4()),
                knowledge_id=knowledge_id,
                title=str(title), category=str(category),
                owner_id=owner_id, department_id=department_id,
                response_code=200, response_msg="知识库删除成功",
                process_status="completed",
                process_time=time.time() - start_time,
            )
    except Exception as e:
        logger.exception("删除知识库失败: knowledge_id=%d", knowledge_id)
    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id, title="", category="",
        owner_id=0, department_id=0,
        response_code=500, response_msg=str(e),
        process_status="failed",
        process_time=time.time() - start_time,
    )

@app.post("/v1/knowledge_base")
def add_knowledge_base(req: KnowledgeRequest,
                       principal: Principal = Depends(require_permission("kb:write"))) -> KnowledgeResponse:
    start_time = time.time()
    # 归属由服务端决定：请求体里传的 owner/department 一律忽略，防止越权写入
    owner_id = principal.user_id or 0
    department_id = principal.department_id or 0
    try:
        def _insert():
            with Session() as session:
                record = KnowledgeDatabase(
                    title=req.title,
                    category=req.category,
                    owner_id=owner_id,
                    department_id=department_id,
                    create_dt=datetime.datetime.now(),
                    update_dt=datetime.datetime.now(),
                )
                session.add(record)
                session.flush()
                knowledge_id = record.knowledge_id
                session.commit()
                return knowledge_id

        knowledge_id = _db_operation_with_retry(_insert)
        return KnowledgeResponse(
            request_id=str(uuid.uuid4()),
            knowledge_id=knowledge_id,
            title=req.title,
            category=req.category,
            owner_id=owner_id,
            department_id=department_id,
            response_code=200,
            response_msg="知识库插入成功",
            process_status="completed",
            process_time=time.time() - start_time,
        )
    except Exception:
        logger.exception("新增知识库失败: title=%s", req.title)
    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=0,
        title="",
        category="",
        owner_id=0,
        department_id=0,
        response_code=500,
        response_msg="知识库插入失败",
        process_status="failed",
        process_time=time.time() - start_time,

    )

@app.get("/v1/document")
def get_document(document_id: int,
                 principal: Principal = Depends(require_permission("doc:read"))) -> DocumentResponse:
    start_time = time.time()
    try:
        def _query():
            with Session() as session:
                return session.query(KnowledgeDocument).filter(
                    KnowledgeDocument.document_id == document_id,
                    *tenant_conditions(KnowledgeDocument, principal),
                ).first()

        record = _db_operation_with_retry(_query)
        if record is not None:
            return DocumentResponse(
                request_id=str(uuid.uuid4()),
                document_id=document_id,
                title=record.title,
                category=record.category,
                knowledge_id=record.knowledge_id,
                file_type=record.file_type,
                response_code=200,
                response_msg="查询文档成功",
                process_status="completed",
                process_time=time.time() - start_time,
            )
        return DocumentResponse(
            request_id=str(uuid.uuid4()),
            document_id=document_id,
            title="",
            category="",
            knowledge_id=0,
            file_type="",
            response_code=404,
            response_msg="文档不存在",
            process_status="failed",
            process_time=time.time() - start_time,
        )
    except Exception:
        logger.exception("查询文档失败: document_id=%d", document_id)
    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id,
        title="",
        category="",
        knowledge_id=0,
        file_type="",
        response_code=500,
        response_msg="查询文档失败",
        process_status="failed",
        process_time=time.time() - start_time,

    )

@app.get("/v1/document/list")
def list_document(knowledge_id: int,
                  principal: Principal = Depends(require_permission("doc:read"))):
    start_time = time.time()
    try:
        with Session() as session:
            records = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.knowledge_id == knowledge_id,
                *tenant_conditions(KnowledgeDocument, principal),
            ).all()
            return {
                "request_id": str(uuid.uuid4()),
                "document_list": [
                    {
                        "document_id": d.document_id,
                        "title": d.title,
                        "category": d.category,
                        "file_type": d.file_type,
                        "create_dt": str(d.create_dt),
                    }
                    for d in records
                ],
                "response_code": 200,
                "response_msg": "ok",
                "process_status": "completed",
                "process_time": time.time() - start_time,
            }
    except Exception as e:
        logger.exception("查询文档列表失败: knowledge_id=%d", knowledge_id)
        return {
            "request_id": str(uuid.uuid4()),
            "document_list": [],
            "response_code": 500,
            "response_msg": str(e),
            "process_status": "failed",
            "process_time": time.time() - start_time,
        }

@app.delete("/v1/document")
def delete_document(document_id: int,
                    principal: Principal = Depends(require_permission("doc:delete"))) -> DocumentResponse:
    start_time = time.time()
    try:
        with Session() as session:
            record = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.document_id == document_id,
                *tenant_conditions(KnowledgeDocument, principal),
            ).first()
            if record is None:
                return DocumentResponse(
                    request_id=str(uuid.uuid4()),
                    document_id=document_id, title="", category="",
                    knowledge_id=0, file_type="",
                    response_code=404, response_msg="文档不存在",
                    process_status="failed",
                    process_time=time.time() - start_time,
                )
            file_path, knowledge_id, title, category, file_type = (
                record.file_path, record.knowledge_id,
                record.title, record.category, record.file_type,
            )
            # 1) ES：按 document_id 删分块+摘要（失败抛异常 → 阻断）
            delete_document_chunks(document_id)
            # 2) 物理文件：失败只告警
            safe_remove_file(file_path)
            # 3) SQLite：删元数据行
            session.delete(record)
            session.commit()
            # 4) 清理任务状态：文档已删，状态行是过期垃圾，删掉防误报（低危，失败只告警）
            try:
                task_state_machine.reset(str(document_id))
            except Exception:
                logger.warning("清理任务状态失败，需手动清理: document_id=%d", document_id)
            return DocumentResponse(
                request_id=str(uuid.uuid4()),
                document_id=document_id, knowledge_id=knowledge_id,
                title=title, category=category, file_type=file_type,
                response_code=200, response_msg="文档删除成功！",
                process_status="completed",
                process_time=time.time() - start_time,
            )
    except Exception as e:
        logger.exception("删除文档失败: document_id=%d", document_id)
    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id, title="", category="",
        knowledge_id=0, file_type="",
        response_code=500, response_msg=str(e),
        process_status="failed",
        process_time=time.time() - start_time,
    )

@app.post("/v1/document")
def add_document(
        knowledge_id: int = Form(),
        title: str = Form(),
        category: str = Form(),
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None,
        principal: Principal = Depends(require_permission("doc:write")),
) -> DocumentResponse:
    start_time = time.time()
    response_msg = "新增文档失败"
    # 文档归属跟随当前用户；部门字段只在服务端写入
    owner_id = principal.user_id or 0
    department_id = principal.department_id or 0
    try:
        # 上传文件名先校验：拒绝路径穿越、非法字符和不支持的扩展名（尽早拦截，不留脏数据）
        _validate_upload_filename(file.filename)
        # 内容一次性读入内存：重试写文件时文件流不会因已消费而变空
        content = file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="文件内容为空")

        def _create_document():
            with Session() as session:
                kb = session.query(KnowledgeDatabase).filter(
                    KnowledgeDatabase.knowledge_id == knowledge_id,
                    # 只能往自己部门的知识库里传文档
                    *tenant_conditions(KnowledgeDatabase, principal),
                ).first()
                if kb is None:
                    return None
                record = KnowledgeDocument(
                    title=title,
                    category=category,
                    knowledge_id=knowledge_id,
                    owner_id=owner_id,
                    department_id=department_id,
                    file_path="",
                    file_type=file.content_type,
                    create_dt=datetime.datetime.now(),
                    update_dt=datetime.datetime.now(),

                )
                session.add(record)
                session.flush()
                document_id = record.document_id
                session.commit()

            new_path = _safe_upload_path(document_id, file.filename)
            try:
                with open(new_path, "wb") as buffer:
                    buffer.write(content)
                with Session() as session:
                    record = session.query(KnowledgeDocument).filter(
                        KnowledgeDocument.document_id == document_id
                    ).first()
                    record.file_path = new_path
                    session.commit()
            except Exception:
                # 回滚部分成果（文件 + 刚插入的 DB 行），重试才能从干净状态重新开始
                _rollback_partial_document(document_id, new_path)
                raise
            return document_id, new_path

        result = _db_operation_with_retry(_create_document)
        if result is None:
            response_msg = "知识库不存在，请提前创建"
            return DocumentResponse(
                request_id=str(uuid.uuid4()),
                document_id=0,
                title="",
                category="",
                knowledge_id=0,
                file_type="",
                response_code=404,
                response_msg=response_msg,
                process_status="failed",
                process_time=time.time() - start_time,
            )
        document_id, file_path = result
        background_tasks.add_task(
            RAG().extract_content,
            knowledge_id=knowledge_id,
            document_id=document_id,
            title=title,
            file_type=file.content_type,
            file_path=file_path,
            owner_id=owner_id,
            department_id=department_id,
        )
        return DocumentResponse(
            request_id=str(uuid.uuid4()),
            document_id=document_id,
            title=title,
            category=category,
            file_type=file.content_type,
            knowledge_id=knowledge_id,
            response_code=200,
            response_msg="文档添加成功",
            process_status="completed",
            process_time=time.time() - start_time,
        )
    except HTTPException:
        raise  # 参数校验类错误走统一错误信封，不被吞掉
    except Exception:
        logger.exception("新增文档失败: title=%s", title)
    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=0,
        title="",
        category="",
        knowledge_id=0,
        file_type="",
        response_code=500,
        response_msg=response_msg,
        process_status="failed",
        process_time=time.time() - start_time,
    )

@app.post("/v1/embedding")
async def semantic_embedding(req: EmbeddingRequest,
                             _principal: Principal = Depends(require_permission("ai:use"))) -> EmbeddingResponse:
    start_time = time.time()
    if not isinstance(req.text, list):
        text = [req.text]
    else:
        text = req.text
    vector: np.ndarray = RAG().get_embedding(text)
    return EmbeddingResponse(
        request_id=str(uuid.uuid4()),
        embedding=vector.astype(float).tolist(),
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time,
    )

@app.post("/v1/rerank")
async def semantic_rerank(req: RerankRequest,
                          _principal: Principal = Depends(require_permission("ai:use"))) -> RerankResponse:
    start_time = time.time()
    vector: np.ndarray = RAG().get_rerank(req.text_pair)
    return RerankResponse(
        request_id=str(uuid.uuid4()),
        rank=vector.astype(float).tolist(),
        response_code=200,
        response_msg="ok",
        process_status="completed",
        process_time=time.time() - start_time,
    )

def _format_sse(event: str, data) -> str:
    """把 (event, data) 格式化为 SSE 文本（事件两行 + 空行）。

    token 的 data 是原始文本，可能含换行，拆成多行 data 保持语义；
    其余事件 data 为 JSON 字符串。
    """
    if event == "token":
        payload = "".join(f"data: {line}\n" for line in str(data).split("\n"))
    else:
        payload = f"data: {json.dumps(data, ensure_ascii=False)}\n"
    return f"event: {event}\n{payload}\n"


@app.post("/chat")
def chat(req: RAGRequest, principal: Principal = Depends(require_permission("ai:use"))):
    start_time = time.time()
    rag = RAG()
    request_id = request_id_var.get()
    # 只能对自己部门的知识库提问；跨部门或不存在的库统一返回 404
    with Session() as session:
        kb = session.query(KnowledgeDatabase).filter(
            KnowledgeDatabase.knowledge_id == req.knowledge_id,
            *tenant_conditions(KnowledgeDatabase, principal),
        ).first()
    if kb is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    # 检索/组装在 SSE 头发出之前完成：失败走 HTTPException → 统一错误信封（非200）
    try:
        llm_messages, debug_info = rag._retrieve_context(
            req.knowledge_id, req.message, department_id=principal.department_scope())
    except Exception as e:
        logger.exception("[chat] 检索失败: knowledge_id=%d", req.knowledge_id)
        raise HTTPException(status_code=500, detail=str(e))

    def event_stream():
        # 流式 body 在中间件 reset 之后才发送，这里重新注入 request_id，流阶段日志才能关联到请求
        request_id_var.set(request_id)
        yield _format_sse("debug", debug_info)
        for event, data in rag._stream_tokens(llm_messages, req.message):
            if event == "done":
                total_cost = time.time() - start_time
                data = {**data, "processing_time": total_cost}
                logger.info("[chat] 完成: 总耗时=%.3fs", total_cost)
            yield _format_sse(event, data)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config["rag"]["port"], workers=1)
