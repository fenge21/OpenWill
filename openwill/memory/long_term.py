"""Long-term memory - persistent knowledge base"""

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


class KnowledgeEntry:
    """Knowledge entry"""

    def __init__(self, topic: str, content: str, source: str = "exploration",
                 importance: float = 0.5, tags: Optional[list[str]] = None):
        self.topic = topic
        self.content = content
        self.source = source
        self.importance = importance
        self.tags = tags or []
        self.created_at = time.time()
        self.updated_at = time.time()
        self.access_count = 0

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "content": self.content,
            "source": self.source,
            "importance": self.importance,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeEntry":
        entry = cls(
            topic=data["topic"],
            content=data["content"],
            source=data.get("source", "unknown"),
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
        )
        entry.created_at = data.get("created_at", time.time())
        entry.updated_at = data.get("updated_at", time.time())
        entry.access_count = data.get("access_count", 0)
        return entry


class LongTermMemory:
    """Long-term memory: persistent knowledge base"""

    def __init__(self, data_dir: str = "data", max_entries: int = 10000):
        self.data_dir = data_dir
        self.max_entries = max_entries
        self.entries: list[KnowledgeEntry] = []
        self._load()

    def _get_filepath(self) -> str:
        return os.path.join(self.data_dir, "long_term_memory.json")

    def _load(self):
        """Load from disk"""
        filepath = self._get_filepath()
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.entries = [KnowledgeEntry.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.entries)} long-term memory entries")
            except Exception as e:
                logger.error(f"Failed to load long-term memory: {e}")
                self.entries = []

    def save(self):
        """Save to disk"""
        filepath = self._get_filepath()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            data = [e.to_dict() for e in self.entries]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self.entries)} long-term memory entries")
        except Exception as e:
            logger.error(f"Failed to save long-term memory: {e}")

    def store(self, topic: str, content: str, source: str = "exploration",
              importance: float = 0.5, tags: Optional[list[str]] = None) -> KnowledgeEntry:
        """Store knowledge"""
        # Check if an entry with the same topic already exists
        for entry in self.entries:
            if entry.topic == topic:
                entry.content = content
                entry.source = source
                entry.importance = max(entry.importance, importance)
                entry.tags = list(set(entry.tags + (tags or [])))
                entry.updated_at = time.time()
                self.save()
                return entry

        entry = KnowledgeEntry(
            topic=topic,
            content=content,
            source=source,
            importance=importance,
            tags=tags,
        )
        self.entries.append(entry)

        # Remove least valuable entries when exceeding limit
        # Uses forgetting-weight score (importance * recency * frequency)
        if len(self.entries) > self.max_entries:
            from .consolidation import forgetting_weight
            now = time.time()
            self.entries.sort(
                key=lambda e: forgetting_weight(
                    e.access_count,
                    e.updated_at or e.created_at,
                    e.importance,
                    now=now,
                ),
            )
            self.entries = self.entries[len(self.entries) - self.max_entries:]

        self.save()
        return entry

    def retrieve(self, query: str, top_k: int = 5) -> list[KnowledgeEntry]:
        """Retrieve relevant knowledge (simple keyword matching)"""
        query_words = set(query.lower().split())
        scored = []

        for entry in self.entries:
            entry_words = set((entry.topic + " " + entry.content).lower().split())
            overlap = len(query_words & entry_words)
            if overlap > 0:
                score = overlap * entry.importance * (1 + entry.access_count * 0.1)
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:top_k]]

        # Update access count
        for entry in results:
            entry.access_count += 1

        return results

    def get_all_topics(self) -> list[str]:
        """Get all known topics"""
        return list(set(e.topic for e in self.entries))

    def get_all_tags(self) -> list[str]:
        """Get all tags"""
        tags = set()
        for entry in self.entries:
            tags.update(entry.tags)
        return list(tags)

    def get_knowledge_stats(self) -> dict:
        """Get knowledge statistics"""
        if not self.entries:
            return {"total": 0, "topics": 0, "tags": 0, "avg_importance": 0}

        return {
            "total": len(self.entries),
            "topics": len(self.get_all_topics()),
            "tags": len(self.get_all_tags()),
            "avg_importance": sum(e.importance for e in self.entries) / len(self.entries),
            "sources": list(set(e.source for e in self.entries)),
        }
