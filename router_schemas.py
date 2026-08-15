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
    owner_id: int = 0
    department_id: int = 0

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
    message: List[Dict]

class RAGResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    message: List[Dict]
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")
    debug_info: Optional[dict] = Field(None, description="检索中间结果（改写后query、召回chunks等）")

