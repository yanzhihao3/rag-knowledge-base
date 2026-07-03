import yaml
import os
import time
import numpy as np
import uuid
import datetime
import traceback
import uvicorn
from typing_extensions import Annotated
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from router_schemas import (
     EmbeddingRequest, EmbeddingResponse,
     RAGRequest, RAGResponse,
     RerankRequest, RerankResponse,
     KnowledgeRequest, KnowledgeResponse,
     DocumentRequest, DocumentResponse,
)
from rag_api import RAG
from db_api import (
     KnowledgeDocument, KnowledgeDatabase,
     Session
)

#project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#config_path = os.path.join(project_root, 'config.yaml')

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

@app.get("/v1/knowledge_base")
def get_knowledge_base(knowledge_id: int, token: str) -> KnowledgeResponse:
    start_time = time.time()
    try:
        for retry_time in range(10):
            with Session() as session:
                record = session.query(KnowledgeDatabase).filter(KnowledgeDatabase.knowledge_id == knowledge_id).first()
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
    except Exception as e:
        traceback.print_exc()
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

@app.delete("/v1/knowledge_base")
def delete_knowledge_base(knowledge_id: int, token: str) -> KnowledgeResponse:
    start_time = time.time()
    try:
        for retry_time in range(10):
            with Session() as session:
                record = session.query(KnowledgeDatabase).filter(KnowledgeDatabase.knowledge_id == knowledge_id).first()
                if record is None:
                    break
                session.delete(record)
                session.commit()
                return KnowledgeResponse(
                    request_id=str(uuid.uuid4()),
                    knowledge_id=knowledge_id,
                    title=str(record.title),
                    category=str(record.category),
                    owner_id=record.owner_id,
                    department_id=record.department_id,
                    response_code=200,
                    response_msg="知识库删除成功",
                    process_status="completed",
                    process_time=time.time() - start_time,
                )
    except Exception as e:
        traceback.print_exc()
    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        title= "",
        category="",
        owner_id=0,
        department_id=0,
        response_code=404,
        response_msg="知识库不存在",
        process_status="failed",
        process_time=time.time() - start_time,
    )

@app.post("/v1/knowledge_base")
def add_knowledge_base(req: KnowledgeRequest) -> KnowledgeResponse:
    start_time = time.time()
    try:
        for retry_time in range(10):
            with Session() as session:
                record = KnowledgeDatabase(
                    title=req.title,
                    category=req.category,
                    owner_id=req.owner_id,
                    department_id=req.department_id,
                    create_dt=datetime.datetime.now(),
                    update_dt=datetime.datetime.now(),
                )
                session.add(record)
                session.flush()
                knowledge_id = record.knowledge_id
                session.commit()
            return KnowledgeResponse(
                request_id=str(uuid.uuid4()),
                knowledge_id=knowledge_id,
                title=req.title,
                category=req.category,
                owner_id=req.owner_id,
                department_id=req.department_id,
                response_code=200,
                response_msg="知识库插入成功",
                process_status="completed",
                process_time=time.time() - start_time,
            )
    except Exception as e:
        traceback.print_exc()
        pass
    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=0,
        title="",
        category="",
        owner_id=0,
        department_id=0,
        response_code=404,
        response_msg="知识库插入失败",
        process_status="failed",
        process_time=time.time() - start_time,

    )

@app.get("/v1/document")
def get_document(document_id: int, token: str) -> DocumentResponse:
    start_time = time.time()
    try:
        for retry_time in range(10):
            with Session() as session:
                record = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).first()
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
                break
    except Exception as e:
        traceback.print_exc()
        pass
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

@app.delete("/v1/document")
def delete_document(document_id: int, token: str) -> DocumentResponse:
    start_time = time.time()
    try:
        for retry_time in range(10):
            with Session() as session:
                record = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).first()
                if record is None:
                    break
                session.delete(record)
                session.commit()
                return DocumentResponse(
                    request_id=str(uuid.uuid4()),
                    document_id=document_id,
                    knowledge_id=record.knowledge_id,
                    title=record.title,
                    category=record.category,
                    file_type=record.file_type,
                    response_code=200,
                    response_msg="文档删除成功！",
                    process_status="completed",
                    process_time=time.time() - start_time,
                )
    except Exception as e:
        traceback.print_exc()
        pass
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

@app.post("/v1/document")
def add_document(
        knowledge_id: int = Form(),
        title: str = Form(),
        category: str = Form(),
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None,
) -> DocumentResponse:
    start_time = time.time()
    response_msg = "新增文档失败"
    try:
        for retry_time in range(10):
            with Session() as session:
                record = session.query(KnowledgeDatabase).filter(KnowledgeDatabase.knowledge_id == knowledge_id).first()
                if record is None:
                    response_msg = "知识库不存在， 请提前创建"
                    break
                record = KnowledgeDocument(
                    title=title,
                    category=category,
                    knowledge_id=knowledge_id,
                    file_path="",
                    file_type=file.content_type,
                    create_dt=datetime.datetime.now(),
                    update_dt=datetime.datetime.now(),

                )
                session.add(record)
                session.flush()
                document_id = record.document_id
                session.commit()
                file_path = f"upload_files/document_{document_id}_" + file.filename
                with open(file_path, "wb") as buffer:
                    buffer.write(file.file.read())

                record = session.query(KnowledgeDocument).filter(KnowledgeDocument.document_id == document_id).first()
                record.file_path = file_path
                session.commit()
            background_tasks.add_task(
                RAG().extract_content,
                knowledge_id=knowledge_id,
                document_id=document_id,
                title=title,
                file_type=file.content_type,
                file_path=file_path,
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
    except Exception as e:
        traceback.print_exc()
        pass
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

@app.post("/v1/embedding")
async def semantic_embedding(req: EmbeddingRequest) -> EmbeddingResponse:
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
async def semantic_rerank(req: RerankRequest) -> RerankResponse:
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

@app.post("/chat")
def chat(req: RAGRequest) -> RAGResponse:
    start_time = time.time()
    message = RAG().chat_with_rag(req.knowledge_id, req.message)
    return RAGResponse(
        request_id=str(uuid.uuid4()),
        message=message,
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time,

    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config["rag"]["port"], workers=1)




