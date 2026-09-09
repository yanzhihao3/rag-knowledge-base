"""Alembic 迁移环境：让迁移复用 db_api 的模型与引擎。

- 引擎：db_api.engine（由 config.yaml + RAG_DB_* 环境变量决定），
  所以同一套迁移文件既能迁 SQLite，也能迁 MySQL；
- 模型：db_api.Base.metadata，autogenerate 以此为准；
- SQLite 开启 batch 模式：SQLite 改列需要重建表，Alembic 会自动处理。
"""
import os
import sys
from logging.config import fileConfig

from alembic import context

# 保证能 import 到项目根目录的 db_api
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_api import Base, engine  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _is_sqlite() -> bool:
    return engine.dialect.name == "sqlite"


def run_migrations_offline() -> None:
    """Offline 模式：只生成 SQL 文本，不连接数据库。"""
    context.configure(
        url=engine.url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=_is_sqlite(),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Online 模式：连接 db_api 当前配置指向的数据库执行迁移。"""
    connectable = engine

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite 的 ALTER 能力有限：batch 模式会自动“建新表→搬数据→删旧表”
            render_as_batch=_is_sqlite(),
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
