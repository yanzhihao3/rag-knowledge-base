import os

# 无需鉴权的路径 使用集合的好处是查找时间复杂度为O(1)，高效判断路径是否公开。
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


def resolve_api_key(config: dict) -> str:
    """优先级：环境变量 RAG_API_KEY > config.rag.api_key > 默认 dev key"""
    return (
        os.environ.get("RAG_API_KEY")
        or config.get("rag", {}).get("api_key", "rag-dev-key")
    )


def is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS
