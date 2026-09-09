from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
import yaml
import os

#project_root = os.path.dirname(os.path.abspath(__file__))
#config_path = os.path.join(project_root, 'config.yaml')
with open("config.yaml", 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

def _cfg(db_config, key, env_name, default=None, cast=None):
    """配置取值：环境变量优先（容器/云端/上线时用），其次才是 config.yaml。"""
    value = os.environ.get(env_name)
    if value is None:
        value = db_config.get(key, default)
    if value is not None and cast is not None:
        value = cast(value)
    return value


db_config = config['database']
# 引擎选择：默认 sqlite（本地开发、rag.db 原样保留），
# 可通过环境变量 RAG_DB_ENGINE=mysql 或 config.yaml 切换
db_type = _cfg(db_config, 'engine', 'RAG_DB_ENGINE', 'sqlite')
# 默认不打印 SQL，避免文档标题/路径等敏感数据进日志；调试时可设 SQL_ECHO=1
SQL_ECHO = os.environ.get("SQL_ECHO", "0") == "1"

if db_type == 'sqlite':
    # SQLite 分支保持原样：本地开发默认，rag.db 文件不动
    db_path = _cfg(db_config, 'path', 'RAG_DB_PATH', 'rag.db')
    engine = create_engine(f'sqlite:///{db_path}', echo=SQL_ECHO)
elif db_type in ('mysql', 'mysql+pymysql'):
    # MySQL 分支：URL.create 负责安全编码（密码里含 @ / # 也不会拼坏连接串）
    engine = create_engine(
        URL.create(
            drivername="mysql+pymysql",                  # 驱动：pymysql
            username=_cfg(db_config, 'username', 'RAG_DB_USER', 'rag'),
            password=_cfg(db_config, 'password', 'RAG_DB_PASSWORD', ''),
            host=_cfg(db_config, 'host', 'RAG_DB_HOST', 'localhost'),
            port=_cfg(db_config, 'port', 'RAG_DB_PORT', 3306, int),
            database=_cfg(db_config, 'database', 'RAG_DB_NAME', 'rag'),
            query={'charset': 'utf8mb4'},                # 完整中文支持
        ),
        echo=SQL_ECHO,
        pool_pre_ping=True,                              # 取连接前探活，避免拿到死连接
        pool_recycle=3600,                               # 1小时回收，避开 MySQL wait_timeout
        pool_size=int(os.environ.get('RAG_DB_POOL_SIZE', '5')),
        max_overflow=int(os.environ.get('RAG_DB_MAX_OVERFLOW', '10')),
    )
else:
    raise ValueError(f"不支持的数据库引擎: {db_type}，可选 sqlite / mysql")
Base = declarative_base()

class KnowledgeDatabase(Base):
    __tablename__ = 'knowledge_database'
    knowledge_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255))
    category = Column(String(100))
    owner_id = Column(Integer, default=0)
    department_id = Column(Integer, default=0)
    create_dt = Column(DateTime, default=datetime.now)
    update_dt = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    documents = relationship("KnowledgeDocument", back_populates="knowledge")

    def __str__(self):
        return (f"KnowledgeDatabase(knowledge_id={self.knowledge_id})"
                f"(title={self.title}, category={self.category})"
                f"(create_dt={self.create_dt}, update_dt={self.update_dt})")

class KnowledgeDocument(Base):
    __tablename__ = 'knowledge_document'

    document_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255))
    category = Column(String(100))
    knowledge_id = Column(Integer, ForeignKey('knowledge_database.knowledge_id'), index=True)
    owner_id = Column(Integer, default=0)
    department_id = Column(Integer, default=0)
    file_path = Column(String(500))
    file_type = Column(String(50))
    create_dt = Column(DateTime, default=datetime.now)
    update_dt = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    knowledge = relationship("KnowledgeDatabase", back_populates="documents")

# 表结构交给 Alembic 管理：新环境启动前先执行 `alembic upgrade head`。
# 这里不再 import 时自动建表，避免“代码改了模型、库却没跟着变”的问题。
Session = sessionmaker(bind=engine)
