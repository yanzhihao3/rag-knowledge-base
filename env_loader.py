"""加载项目根目录的 .env 文件，供 db_api / celery_app 等模块复用。

优先级约定：真实环境变量 > .env > config.yaml > 代码默认值。
.env 已被 .gitignore 忽略，密码与密钥不会进版本库。
"""
import os


def load_dotenv(path: str = ".env") -> None:
    """把 .env 的键值写入环境变量（不覆盖已存在的真实环境变量）。"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
