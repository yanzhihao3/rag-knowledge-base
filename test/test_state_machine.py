import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import TaskStateMachine, task_state_machine, SqliteStateStore


class TestStateMachine:
    """任务状态机测试"""

    def setup_method(self):
        """每个测试前清空状态机"""
        self.sm = TaskStateMachine()

    def test_initial_state_is_pending(self):
        """测试初始状态是pending"""
        assert self.sm.get_state("doc_1") == TaskStateMachine.STATE_PENDING

    def test_set_and_get_state(self):
        """测试设置和获取状态"""
        self.sm.set_state("doc_1", TaskStateMachine.STATE_PROCESSING)
        assert self.sm.get_state("doc_1") == TaskStateMachine.STATE_PROCESSING

    def test_state_transitions(self):
        """测试状态流转"""
        doc_id = "doc_1"

        # pending → processing
        self.sm.set_state(doc_id, TaskStateMachine.STATE_PROCESSING)
        assert self.sm.get_state(doc_id) == TaskStateMachine.STATE_PROCESSING

        # processing → completed
        self.sm.set_state(doc_id, TaskStateMachine.STATE_COMPLETED)
        assert self.sm.is_completed(doc_id)

    def test_is_completed(self):
        """测试is_completed判断"""
        self.sm.set_state("doc_1", TaskStateMachine.STATE_COMPLETED)
        assert self.sm.is_completed("doc_1") is True

        self.sm.set_state("doc_2", TaskStateMachine.STATE_PROCESSING)
        assert self.sm.is_completed("doc_2") is False

    def test_is_failed(self):
        """测试is_failed判断"""
        self.sm.set_state("doc_1", TaskStateMachine.STATE_FAILED)
        assert self.sm.is_failed("doc_1") is True

        self.sm.set_state("doc_2", TaskStateMachine.STATE_COMPLETED)
        assert self.sm.is_failed("doc_2") is False

    def test_reset(self):
        """测试重置状态"""
        self.sm.set_state("doc_1", TaskStateMachine.STATE_COMPLETED)
        self.sm.reset("doc_1")
        assert self.sm.get_state("doc_1") == TaskStateMachine.STATE_PENDING

    def test_multiple_documents(self):
        """测试多文档独立状态"""
        self.sm.set_state("doc_1", TaskStateMachine.STATE_COMPLETED)
        self.sm.set_state("doc_2", TaskStateMachine.STATE_FAILED)
        self.sm.set_state("doc_3", TaskStateMachine.STATE_PROCESSING)

        assert self.sm.is_completed("doc_1")
        assert self.sm.is_failed("doc_2")
        assert self.sm.get_state("doc_3") == TaskStateMachine.STATE_PROCESSING
        assert self.sm.get_state("doc_4") == TaskStateMachine.STATE_PENDING  # 未设置过的默认pending


class TestStateMachinePersistence:
    """状态机持久化：挂 SQLite 后重启不丢状态"""

    def test_state_restored_after_restart(self, tmp_path):
        """模拟重启：新实例从 SQLite 恢复之前的状态"""
        db = str(tmp_path / "state.db")
        TaskStateMachine(store=SqliteStateStore(db)).set_state("doc_1", TaskStateMachine.STATE_COMPLETED)

        sm2 = TaskStateMachine(store=SqliteStateStore(db))
        assert sm2.get_state("doc_1") == TaskStateMachine.STATE_COMPLETED

    def test_failed_state_persists(self, tmp_path):
        """failed 状态也要持久化"""
        db = str(tmp_path / "state.db")
        TaskStateMachine(store=SqliteStateStore(db)).set_state("doc_2", TaskStateMachine.STATE_FAILED)

        sm2 = TaskStateMachine(store=SqliteStateStore(db))
        assert sm2.is_failed("doc_2")

    def test_reset_clears_persisted_state(self, tmp_path):
        """reset 要同时清掉持久层的状态"""
        db = str(tmp_path / "state.db")
        sm1 = TaskStateMachine(store=SqliteStateStore(db))
        sm1.set_state("doc_3", TaskStateMachine.STATE_COMPLETED)
        sm1.reset("doc_3")

        sm2 = TaskStateMachine(store=SqliteStateStore(db))
        assert sm2.get_state("doc_3") == TaskStateMachine.STATE_PENDING

    def test_without_store_state_not_shared(self):
        """不挂持久层时，新实例不共享状态（默认 pending）"""
        TaskStateMachine().set_state("doc_1", TaskStateMachine.STATE_COMPLETED)
        sm2 = TaskStateMachine()
        assert sm2.get_state("doc_1") == TaskStateMachine.STATE_PENDING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])