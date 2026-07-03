import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_api import KnowledgeDatabase, KnowledgeDocument, Session, Base, engine


class TestPermissionFields:
    """权限字段功能测试"""

    def setup_method(self):
        """每个测试前创建表"""
        Base.metadata.create_all(engine)

    def teardown_method(self):
        """每个测试后清理"""
        with Session() as session:
            session.query(KnowledgeDatabase).delete()
            session.query(KnowledgeDocument).delete()
            session.commit()

    def test_knowledge_database_has_owner_id(self):
        """测试KnowledgeDatabase表有权owner_id字段"""
        kb = KnowledgeDatabase(
            title="测试知识库",
            category="技术",
            owner_id=1,
            department_id=10
        )
        assert kb.owner_id == 1
        assert kb.department_id == 10

    def test_knowledge_database_owner_id_default(self):
        """测试owner_id默认值：创建时为None，INSERT时会用默认值0"""
        kb = KnowledgeDatabase(
            title="测试知识库",
            category="技术"
        )
        # SQLAlchemy的default只在INSERT时生效，创建对象时是None
        assert kb.owner_id is None  # 创建时为None（不是0）
        with Session() as session:
            session.add(kb)
            session.commit()
            session.refresh(kb)
            assert kb.owner_id == 0  # INSERT后数据库用默认值0
        assert kb.department_id == 0  # 默认值

    def test_knowledge_document_has_permission_fields(self):
        """测试KnowledgeDocument表有权限字段"""
        with Session() as session:
            # 先创建知识库
            kb = KnowledgeDatabase(
                title="测试知识库",
                category="技术",
                owner_id=1,
                department_id=10
            )
            session.add(kb)
            session.flush()

            # 再创建文档
            doc = KnowledgeDocument(
                title="测试文档",
                category="技术",
                knowledge_id=kb.knowledge_id,
                owner_id=2,
                department_id=20
            )
            session.add(doc)
            session.commit()

            # 验证
            assert doc.owner_id == 2
            assert doc.department_id == 20

    def test_permission_isolation_concept(self):
        """测试权限隔离概念：不同owner_id的文档应该被区分"""
        with Session() as session:
            # 创建两个不同租户的知识库
            kb1 = KnowledgeDatabase(
                title="租户A知识库",
                category="技术",
                owner_id=1,
                department_id=10
            )
            kb2 = KnowledgeDatabase(
                title="租户B知识库",
                category="技术",
                owner_id=2,
                department_id=20
            )
            session.add_all([kb1, kb2])
            session.commit()

            # 查询时按owner_id过滤
            result = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.owner_id == 1
            ).first()

            assert result is not None
            assert result.title == "租户A知识库"
            assert result.owner_id == 1
            assert result.owner_id != result.department_id  # owner_id和department_id是不同概念

    def test_cross_tenant_isolation(self):
        """测试跨租户隔离：owner_id不同则查不到对方数据"""
        with Session() as session:
            # 创建两个租户的数据
            kb1 = KnowledgeDatabase(
                title="租户A知识库",
                category="技术",
                owner_id=100,
                department_id=1000
            )
            kb2 = KnowledgeDatabase(
                title="租户B知识库",
                category="技术",
                owner_id=200,
                department_id=2000
            )
            session.add_all([kb1, kb2])
            session.commit()

            # 租户A只能查到自己的
            tenant_a_data = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.owner_id == 100
            ).all()

            assert len(tenant_a_data) == 1
            assert tenant_a_data[0].title == "租户A知识库"

            # 租户B也只能查到自己的
            tenant_b_data = session.query(KnowledgeDatabase).filter(
                KnowledgeDatabase.owner_id == 200
            ).all()

            assert len(tenant_b_data) == 1
            assert tenant_b_data[0].title == "租户B知识库"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
