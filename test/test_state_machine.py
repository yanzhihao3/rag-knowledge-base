import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_api import TaskStateMachine, task_state_machine


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])