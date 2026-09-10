import os

# 无需鉴权的路径 使用集合的好处是查找时间复杂度为O(1)，高效判断路径是否公开。
# 无需认证的路径：探活、文档、登录（登录本身不能要求先登录）
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/v1/auth/login"}


def resolve_api_key(config: dict) -> str:
    """解析 API Key：环境变量 RAG_API_KEY 优先，其次 config.rag.api_key。

    两者都没有时直接抛错拒绝启动，避免带着默认密钥上线。
    """
    api_key = os.environ.get("RAG_API_KEY") or config.get("rag", {}).get("api_key")
    if not api_key:
        raise RuntimeError("未配置 API Key：请设置环境变量 RAG_API_KEY 或 config.rag.api_key")
    return api_key


def is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS
