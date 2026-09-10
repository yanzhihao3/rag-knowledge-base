from pydantic import BaseModel, Field
from typing import List, Union, Any, Tuple, Dict, Optional
from fastapi import File, UploadFile, Form
from typing_extensions import Annotated

class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]
    model: str

class EmbeddingResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    embedding: List[List[float]]=Field(description="文本对应的向量表示")
    response_code: int = Field(description="响应代码， 用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息， 详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态， 例如 'completed','pending','failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")

class RerankRequest(BaseModel):
    model: str
    text_pair: List[List[str]]

class RerankResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    rank: List[float]
    response_code: int =Field(description="响应代码，用于成功信息或错误状态")
    response_msg: str = Field(description="响应信息， 详细描述响应状态或错误信息")
    process_status: str= Field(description="处理状态， 例如 'completed','pending','failed'")
    process_time: float = Field(description="处理请求的耗时（秒）")

class KnowledgeRequest(BaseModel):
    category: str
    title: str
    # owner_id / department_id 由服务端从当前登录用户写入，不接受客户端指定（防越权）


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=72, description="最长 72 字节（bcrypt 限制）")
    role: str = Field(default="viewer", description="admin / editor / viewer")
    department_id: int = Field(default=0, ge=0)


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="access token 有效期（秒）")


class UserResponse(BaseModel):
    user_id: int
    username: str
    role: str
    department_id: int

class KnowledgeResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    knowledge_id: int
    category: str
    title: str
    owner_id: int = 0
    department_id: int = 0
    response_code: int = Field(description="响应代码， 用于表示成功或错误信息")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态， 例如 'completed','pending','failed'")
    process_time: float = Field(description="处理请求的耗时（秒）")

class DocumentRequest(BaseModel):
    knowledge_id: int = Field(description="请求ID")
    title: str = Annotated[str, Form()]
    category: str = Annotated[str, Form()]
    file: UploadFile = Annotated[str, Form()]


class DocumentResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    document_id: int
    category: str
    title: str
    knowledge_id: int
    file_type: str
    response_code: int = Field(description="响应代码，用于成功或错误信息")
    response_msg: str = Field(description="响应信息， 详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed','pending','failed'")
    process_time: float = Field(description="处理请求的耗时（秒）")

class RAGRequest(BaseModel):
    knowledge_id: int
    message: List[Dict] = Field(..., min_length=1, description="对话消息列表，至少包含一条消息")

class RAGResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    message: List[Dict]
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")
    debug_info: Optional[dict] = Field(None, description="检索中间结果（改写后query、召回chunks等）")

