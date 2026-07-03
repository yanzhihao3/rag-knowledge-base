from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
import yaml
import os

#project_root = os.path.dirname(os.path.abspath(__file__))
#config_path = os.path.join(project_root, 'config.yaml')
with open("config.yaml", 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

db_config = config['database']
db_type = db_config['engine']
if db_type == 'sqlite':
    db_path = db_config.get('path', 'rag.db')
    engine = create_engine(f'sqlite:///{db_path}', echo=True)
else:
    host = db_config.get('host', 'localhost')
    port = db_config.get('port', 3306)
    username = db_config.get('username', 'user')
    password = db_config.get('password', 'password')
    database = db_config.get('database', 'mydb')

    engine = create_engine(
        f"{db_type}://{username}:{password}@{host}:{port}/{database}",
        echo=True
    )
Base = declarative_base()

class KnowledgeDatabase(Base):
    __tablename__ = 'knowledge_database'
    knowledge_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    category = Column(String)
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
    title = Column(String)
    category = Column(String)
    knowledge_id = Column(Integer, ForeignKey('knowledge_database.knowledge_id'))
    owner_id = Column(Integer, default=0)
    department_id = Column(Integer, default=0)
    file_path = Column(String)
    file_type = Column(String)
    create_dt = Column(DateTime, default=datetime.now)
    update_dt = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    knowledge = relationship("KnowledgeDatabase", back_populates="documents")

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)