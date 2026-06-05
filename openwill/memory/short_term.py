"""Short-term memory - working memory, stores current context"""

import time
from typing import Optional

from ..llm.interface import Message


class ShortTermMemory:
    """Short-term memory: stores current conversation context and recent events"""

    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.messages: list[Message] = []
        self.current_context: dict = {}

    def add(self, role: str, content: str, metadata: Optional[dict] = None):
        """Add a message"""
        msg = Message(role=role, content=content, metadata=metadata or {})
        msg.metadata["timestamp"] = time.time()
        self.messages.append(msg)

        # Remove oldest messages when exceeding limit (preserve system messages)
        while len(self.messages) > self.max_messages:
            if self.messages[0].role == "system":
                # Skip system messages, remove the next one
                if len(self.messages) > 1:
                    self.messages.pop(1)
                else:
                    break
            else:
                self.messages.pop(0)

    def add_thought(self, content: str):
        """Add a thought record"""
        self.add("assistant", content, {"type": "thought"})

    def add_observation(self, content: str):
        """Add an observation record"""
        self.add("user", content, {"type": "observation"})

    def add_action(self, action: str, result: str):
        """Add an action record"""
        self.add("assistant", f"[Action] {action}", {"type": "action"})
        self.add("user", f"[Result] {result}", {"type": "action_result"})

    def get_messages(self) -> list[Message]:
        """Get all messages"""
        return self.messages.copy()

    def get_recent(self, n: int = 10) -> list[Message]:
        """Get the most recent N messages"""
        return self.messages[-n:]

    def set_context(self, key: str, value):
        """Set current context"""
        self.current_context[key] = value

    def get_context(self, key: str, default=None):
        """Get current context"""
        return self.current_context.get(key, default)

    def clear(self):
        """Clear short-term memory"""
        self.messages.clear()
        self.current_context.clear()

    def get_summary(self) -> str:
        """Get short-term memory summary"""
        if not self.messages:
            return "(empty)"
        recent = self.get_recent(10)
        lines = []
        for msg in recent:
            prefix = {"system": "System", "user": "Observation", "assistant": "Thought"}.get(msg.role, msg.role)
            lines.append(f"[{prefix}] {msg.content[:100]}")
        return "\n".join(lines)
