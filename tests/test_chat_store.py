"""
tests/test_chat_store.py
========================

Unit tests for the chat archive. No API calls are made; a small fake
chat model is used for the agent test.
"""

import tempfile
import unittest
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from utils.chat_store import (
    create_conversation, list_conversations, load_messages, save_turn,
)
from utils.chat_agent import build_agent


class ChatStoreTests(unittest.TestCase):
    """Persistence and memory behaviour."""

    def test_two_conversations_are_isolated(self):
        with tempfile.TemporaryDirectory() as folder:
            database_path = Path(folder) / "chat.db"
            first = create_conversation(database_path)
            second = create_conversation(database_path)

            save_turn(first, "Hi Ali", "Hi Ali", "Hello Ali", database_path)
            save_turn(second, "Hi Sara", "Hi Sara", "Hello Sara", database_path)

            self.assertEqual(len(load_messages(first, database_path)), 2)
            self.assertEqual(load_messages(first, database_path)[0]["content"], "Hi Ali")
            self.assertEqual(load_messages(second, database_path)[0]["content"], "Hi Sara")
            self.assertEqual(len(list_conversations(database_path)), 2)

    def test_recent_turns_keeps_short_history(self):
        agent = build_agent(
            chat_model=FakeListChatModel(responses=["ok"]),
            system_prompt="Be helpful.",
            memory_mode="Recent turns",
            recent_turns=1,
        )
        config = {"configurable": {"thread_id": "test"}}
        for message in ["one", "two", "three"]:
            agent.invoke({"messages": [{"role": "user", "content": message}]}, config)

        messages = agent.get_state(config).values["messages"]
        self.assertEqual(len(messages), 4)


if __name__ == "__main__":
    unittest.main()
