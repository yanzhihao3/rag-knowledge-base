FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（torch 编译需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 6010

# 健康检查：探活 /health 端点（该路径无需鉴权，可直接探）
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:6010/health')" || exit 1

CMD ["python", "main.py"]
