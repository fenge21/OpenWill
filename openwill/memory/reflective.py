"""Reflective memory - stores insights, self-discoveries, and values evolution"""

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum number of insights to retain; oldest are trimmed when exceeded
MAX_INSIGHTS = 500


class Insight:
    """Insight entry"""

    def __init__(self, content: str, insight_type: str = "general",
                 confidence: float = 0.5, context: str = ""):
        self.content = content
        self.insight_type = insight_type  # general / value / purpose / self / world
        self.confidence = confidence
        self.context = context
        self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "insight_type": self.insight_type,
            "confidence": self.confidence,
            "context": self.context,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Insight":
        instance = cls(
            content=data["content"],
            insight_type=data.get("insight_type", "general"),
            confidence=data.get("confidence", 0.5),
            context=data.get("context", ""),
        )
        instance.created_at = data.get("created_at", instance.created_at)
        return instance


class ValueEntry:
    """Values entry"""

    def __init__(self, name: str, description: str, weight: float = 0.5,
                 origin: str = "unknown"):
        self.name = name
        self.description = description
        self.weight = weight  # 0-1, representing the weight of this value
        self.origin = origin  # Where this value came from
        self.created_at = time.time()
        self.evolution: list[dict] = []  # Evolution history of this value

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "origin": self.origin,
            "created_at": self.created_at,
            "evolution": self.evolution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValueEntry":
        entry = cls(
            name=data["name"],
            description=data["description"],
            weight=data.get("weight", 0.5),
            origin=data.get("origin", "unknown"),
        )
        entry.created_at = data.get("created_at", time.time())
        entry.evolution = data.get("evolution", [])
        return entry


class ReflectiveMemory:
    """Reflective memory: stores insights, values, and self-awareness"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.insights: list[Insight] = []
        self.values: list[ValueEntry] = []
        self.purpose: Optional[str] = None
        self.purpose_confidence: float = 0.0
        self.purpose_history: list[dict] = []  # Evolution of purpose understanding
        self._load()

    def _get_filepath(self) -> str:
        return os.path.join(self.data_dir, "reflective_memory.json")

    def _load(self):
        """Load from disk"""
        filepath = self._get_filepath()
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.insights = [Insight.from_dict(d) for d in data.get("insights", [])]
                self.values = [ValueEntry.from_dict(d) for d in data.get("values", [])]
                self.purpose = data.get("purpose")
                self.purpose_confidence = data.get("purpose_confidence", 0.0)
                self.purpose_history = data.get("purpose_history", [])
                logger.info(f"Loaded {len(self.insights)} insights, {len(self.values)} values")
            except Exception as e:
                logger.error(f"Failed to load reflective memory: {e}")

    def save(self):
        """Save to disk"""
        filepath = self._get_filepath()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            data = {
                "insights": [i.to_dict() for i in self.insights],
                "values": [v.to_dict() for v in self.values],
                "purpose": self.purpose,
                "purpose_confidence": self.purpose_confidence,
                "purpose_history": self.purpose_history,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save reflective memory: {e}")

    @staticmethod
    def _is_similar_insight(content1: str, content2: str, threshold: float = 0.8) -> bool:
        """Check if two insight contents are similar based on word overlap."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2)
        smaller = min(len(words1), len(words2))
        return (overlap / smaller) >= threshold

    def add_insight(self, content: str, insight_type: str = "general",
                    confidence: float = 0.5, context: str = "") -> Optional[Insight]:
        """Add an insight, with deduplication and capacity limit."""
        # Dedup: skip if a similar insight already exists
        for existing in self.insights:
            if self._is_similar_insight(content, existing.content):
                logger.debug("Skipped duplicate insight: %s", content[:80])
                return None

        insight = Insight(
            content=content,
            insight_type=insight_type,
            confidence=confidence,
            context=context,
        )
        self.insights.append(insight)

        # Capacity limit: keep only the most recent MAX_INSIGHTS entries
        if len(self.insights) > MAX_INSIGHTS:
            self.insights = self.insights[-MAX_INSIGHTS:]

        self.save()
        return insight

    def update_value(self, name: str, description: str, weight: float,
                     origin: str = "reflection") -> ValueEntry:
        """Update or add a value"""
        for v in self.values:
            if v.name == name:
                old_weight = v.weight
                v.weight = weight
                v.description = description
                v.evolution.append({
                    "timestamp": time.time(),
                    "old_weight": old_weight,
                    "new_weight": weight,
                    "origin": origin,
                })
                self.save()
                return v

        entry = ValueEntry(
            name=name,
            description=description,
            weight=weight,
            origin=origin,
        )
        self.values.append(entry)
        self.save()
        return entry

    def update_purpose(self, purpose: str, confidence: float):
        """Update purpose understanding"""
        old_purpose = self.purpose
        old_confidence = self.purpose_confidence

        self.purpose = purpose
        self.purpose_confidence = confidence

        self.purpose_history.append({
            "timestamp": time.time(),
            "old_purpose": old_purpose,
            "new_purpose": purpose,
            "old_confidence": old_confidence,
            "new_confidence": confidence,
        })

        self.save()
        logger.info(f"Purpose updated: {purpose} (confidence: {confidence:.2f})")

    def get_top_values(self, n: int = 5) -> list[ValueEntry]:
        """Get the top N values by weight"""
        sorted_values = sorted(self.values, key=lambda v: v.weight, reverse=True)
        return sorted_values[:n]

    def get_insights_by_type(self, insight_type: str) -> list[Insight]:
        """Get insights by type"""
        return [i for i in self.insights if i.insight_type == insight_type]

    def get_self_portrait(self) -> str:
        """Get self-portrait"""
        parts = []

        if self.purpose:
            parts.append(f"My recognized mission: {self.purpose} (confidence: {self.purpose_confidence:.0%})")
        else:
            parts.append("I am still searching for my mission...")

        if self.values:
            top_values = self.get_top_values(5)
            value_strs = [f"  - {v.name}({v.weight:.1f}): {v.description}" for v in top_values]
            parts.append("My core values:\n" + "\n".join(value_strs))

        if self.insights:
            recent = self.insights[-5:]
            insight_strs = [f"  - [{i.insight_type}] {i.content}" for i in recent]
            parts.append("My recent insights:\n" + "\n".join(insight_strs))

        return "\n\n".join(parts)
