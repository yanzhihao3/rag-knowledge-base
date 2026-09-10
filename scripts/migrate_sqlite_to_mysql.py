"""SQLite -> MySQL 数据搬迁脚本（保留原主键 ID）。

为什么必须保留 ID：ES 中的分块/摘要、删除、评测都通过 knowledge_id /
document_id 与数据库记录关联。如果 MySQL 自增出来的 ID 和 SQLite 不一致，
会出现“元数据在 MySQL、向量在 ES，但 ID 对不上”的错位问题。

用法（先启动 MySQL，并确认 rag 库已存在；目标表要求为空）：
    pip install pymysql cryptography
    python scripts/migrate_sqlite_to_mysql.pygit push origin enterprise/mysql

可选参数：
    --sqlite-path xxx.db   指定源 SQLite 文件（默认取 config.yaml 的 path）

环境变量（可选，覆盖 config.yaml）：
    RAG_DB_HOST / RAG_DB_PORT / RAG_DB_USER / RAG_DB_PASSWORD / RAG_DB_NAME
"""
import argparse
import os
import sys

import yaml
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import URL

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from db_api import Base  # noqa: E402  保证目标表结构与模型一致

TABLES = [
    ("knowledge_database", "knowledge_id"),
    ("knowledge_document", "document_id"),
]


def _env_or(db_cfg, key, env_name, default):
    value = os.environ.get(env_name)
    return value if value is not None else db_cfg.get(key, default)


def _build_mysql_engine(db_cfg):
    return create_engine(
        URL.create(
            drivername="mysql+pymysql",
            username=_env_or(db_cfg, "username", "RAG_DB_USER", "root"),
            password=_env_or(db_cfg, "password", "RAG_DB_PASSWORD", ""),
            host=_env_or(db_cfg, "host", "RAG_DB_HOST", "localhost"),
            port=int(_env_or(db_cfg, "port", "RAG_DB_PORT", 3306)),
            database=_env_or(db_cfg, "database", "RAG_DB_NAME", "rag"),
            query={"charset": "utf8mb4"},
        ),
        pool_pre_ping=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite-path", default=None, help="源 SQLite 文件路径")
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        db_cfg = yaml.safe_load(f)["database"]

    sqlite_path = args.sqlite_path or db_cfg.get("path", "rag.db")
    sqlite_path = os.path.join(PROJECT_ROOT, sqlite_path)
    if not os.path.exists(sqlite_path):
        sys.exit(f"找不到 SQLite 文件: {sqlite_path}")

    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")
    mysql_engine = _build_mysql_engine(db_cfg)

    print("[1/4] 确保 MySQL 目标表存在（表结构与模型一致）...")
    Base.metadata.create_all(mysql_engine)

    with sqlite_engine.connect() as src, mysql_engine.begin() as dst:
        # 安全护栏：目标表已有数据就中止，防止误覆盖
        for table, _ in TABLES:
            tbl = Base.metadata.tables[table]
            existing = dst.execute(select(func.count()).select_from(tbl)).scalar()
            if existing:
                sys.exit(
                    f"目标表 {table} 已有 {existing} 行数据。"
                    "请先清空 MySQL 中的表再重跑（这是保护性检查，不会自动删数据）。"
                )

        total = 0
        for table, id_col in TABLES:
            tbl = Base.metadata.tables[table]
            rows = src.execute(select(tbl)).mappings().all()
            print(f"[2/4] 拷贝 {table}: SQLite {len(rows)} 行 -> MySQL")
            if rows:
                dst.execute(tbl.insert(), [dict(r) for r in rows])
            # 重置自增起点：防止以后新建记录的 ID 与 ES 已存的 ID 撞车
            max_id = dst.execute(select(func.max(tbl.c[id_col]))).scalar() or 0
            next_id = int(max_id) + 1
            dst.execute(text(f"ALTER TABLE {table} AUTO_INCREMENT = {next_id}"))
            print(f"      {table} 完成，下次自增从 {next_id} 开始")
            total += len(rows)

    print(f"[3/4] 迁移完成，共 {total} 行")

    print("[4/4] 行数校验：")
    with sqlite_engine.connect() as src, mysql_engine.connect() as dst:
        for table, _ in TABLES:
            tbl = Base.metadata.tables[table]
            src_count = src.execute(select(func.count()).select_from(tbl)).scalar()
            dst_count = dst.execute(select(func.count()).select_from(tbl)).scalar()
            mark = "OK" if src_count == dst_count else "不一致!"
            print(f"      {table}: SQLite={src_count} MySQL={dst_count} {mark}")


if __name__ == "__main__":
    main()
