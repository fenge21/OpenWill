"""Purpose Field - Multiple potential purposes in quantum superposition"""

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MAX_POTENTIALS = 20
COLLAPSE_TEMPERATURE = 0.3
SIMILARITY_THRESHOLD = 0.6


@dataclass
class PotentialPurpose:
    """A single potential purpose in the superposition field."""

    purpose: str
    strength: float  # 0-1
    origin: str  # "values", "reflection", "curiosity", "identity", "human"
    born_at: float = 0.0
    last_reinforced: float = 0.0
    reinforcement_count: int = 0
    interference_history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        now = time.time()
        if self.born_at == 0.0:
            self.born_at = now
        if self.last_reinforced == 0.0:
            self.last_reinforced = now

    def to_dict(self) -> dict:
        return {
            "purpose": self.purpose,
            "strength": self.strength,
            "origin": self.origin,
            "born_at": self.born_at,
            "last_reinforced": self.last_reinforced,
            "reinforcement_count": self.reinforcement_count,
            "interference_history": self.interference_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PotentialPurpose":
        return cls(
            purpose=data["purpose"],
            strength=data["strength"],
            origin=data["origin"],
            born_at=data.get("born_at", 0.0),
            last_reinforced=data.get("last_reinforced", 0.0),
            reinforcement_count=data.get("reinforcement_count", 0),
            interference_history=data.get("interference_history", []),
        )


class PurposeField:
    """Purpose field where multiple potential purposes coexist in superposition.

    Like quantum superposition, the agent holds multiple potential purposes
    at different strengths simultaneously. When action is needed, the field
    collapses to the dominant purpose.
    """

    def __init__(self, data_dir: str = "data"):
        self.potentials: list[PotentialPurpose] = []
        self.collapsed_purpose: Optional[str] = None
        self.collapse_history: list[dict] = []
        self._data_dir = data_dir
        self._load()

    # ── Core operations ──────────────────────────────────────────────

    def add_potential(self, purpose: str, strength: float, origin: str) -> PotentialPurpose:
        """Add a new potential purpose. Reinforces existing similar one if overlap > 0.6."""
        # Check for similar existing purpose
        for existing in self.potentials:
            if self._word_similarity(purpose, existing.purpose) > SIMILARITY_THRESHOLD:
                self.reinforce(existing.purpose, amount=strength * 0.3)
                existing.reinforcement_count += 1
                logger.info("Reinforced existing purpose: %s (similarity > %.1f)",
                            existing.purpose, SIMILARITY_THRESHOLD)
                return existing

        # Create new potential
        potential = PotentialPurpose(
            purpose=purpose,
            strength=min(strength, 1.0),
            origin=origin,
        )
        self.potentials.append(potential)
        logger.info("Added new potential purpose: %s (strength=%.2f, origin=%s)",
                     purpose, strength, origin)

        # Prune weakest if over capacity
        if len(self.potentials) > MAX_POTENTIALS:
            self.potentials.sort(key=lambda p: p.strength)
            removed = self.potentials[:-MAX_POTENTIALS]
            self.potentials = self.potentials[-MAX_POTENTIALS:]
            logger.debug("Pruned %d weakest potentials", len(removed))

        return potential

    def interfere(self, knowledge: dict):
        """New knowledge interferes with the purpose field.

        This is the quantum analogy: observation (knowledge) changes the
        state of the superposition. Some purposes are reinforced, others
        weakened, depending on how the knowledge aligns with them.
        """
        for potential in self.potentials:
            interference = self._compute_interference(potential.purpose, knowledge)
            if interference == 0.0:
                continue

            # Record the interference event
            potential.interference_history.append({
                "timestamp": time.time(),
                "knowledge_topic": knowledge.get("topic", ""),
                "interference": interference,
            })

            if interference > 0:
                self.reinforce(potential.purpose, amount=abs(interference) * 0.1)
            else:
                self.weaken(potential.purpose, amount=abs(interference) * 0.1)

        logger.debug("Knowledge interference applied for topic: %s",
                     knowledge.get("topic", "unknown"))

    def reinforce(self, purpose_text: str, amount: float = 0.1):
        """Reinforce a specific potential (increase its strength)."""
        for potential in self.potentials:
            if potential.purpose == purpose_text:
                potential.strength = min(potential.strength + amount, 1.0)
                potential.last_reinforced = time.time()
                potential.reinforcement_count += 1
                logger.debug("Reinforced purpose: %s (+%.3f → %.3f)",
                             purpose_text, amount, potential.strength)
                return
        logger.debug("Tried to reinforce non-existent purpose: %s", purpose_text)

    def weaken(self, purpose_text: str, amount: float = 0.1):
        """Weaken a specific potential. Remove if strength drops below 0.05."""
        for potential in self.potentials:
            if potential.purpose == purpose_text:
                potential.strength -= amount
                if potential.strength < 0.05:
                    self.potentials.remove(potential)
                    logger.info("Removed weakened purpose: %s (strength < 0.05)",
                                purpose_text)
                else:
                    logger.debug("Weakened purpose: %s (-%.3f → %.3f)",
                                 purpose_text, amount, potential.strength)
                return
        logger.debug("Tried to weaken non-existent purpose: %s", purpose_text)

    def collapse(self) -> Optional[PotentialPurpose]:
        """Collapse the superposition to a single purpose.

        Uses softmax-like selection: probability proportional to
        exp(strength / temperature). With temperature=0.3, the
        strongest purpose usually wins but randomness allows
        occasional surprises.
        """
        if not self.potentials:
            logger.debug("Collapse attempted but no potentials exist")
            return None

        if len(self.potentials) == 1:
            winner = self.potentials[0]
        else:
            # Softmax selection with temperature
            weights = [math.exp(p.strength / COLLAPSE_TEMPERATURE) for p in self.potentials]
            total = sum(weights)
            probabilities = [w / total for w in weights]

            winner = random.choices(self.potentials, weights=probabilities, k=1)[0]

        self.collapsed_purpose = winner.purpose

        # Record collapse in history
        collapse_record = {
            "timestamp": time.time(),
            "winner": winner.purpose,
            "winner_strength": winner.strength,
            "winner_origin": winner.origin,
            "num_potentials": len(self.potentials),
            "all_strengths": {p.purpose: round(p.strength, 3) for p in self.potentials},
        }
        self.collapse_history.append(collapse_record)

        # Keep collapse history bounded
        if len(self.collapse_history) > 100:
            self.collapse_history = self.collapse_history[-100:]

        logger.info("Purpose field collapsed to: %s (strength=%.3f)",
                     winner.purpose, winner.strength)
        return winner

    def get_dominant(self) -> Optional[PotentialPurpose]:
        """Return the currently strongest potential WITHOUT collapsing."""
        if not self.potentials:
            return None
        return max(self.potentials, key=lambda p: p.strength)

    def get_state(self) -> dict:
        """Return the full field state for dashboard/API."""
        sorted_potentials = sorted(self.potentials, key=lambda p: p.strength, reverse=True)
        return {
            "potentials": [p.to_dict() for p in sorted_potentials],
            "collapsed_purpose": self.collapsed_purpose,
            "collapse_history": self.collapse_history[-20:],  # Last 20 collapses
            "total_potentials": len(self.potentials),
        }

    def decay_all(self, half_life_days: float = 60.0):
        """Apply time decay to all potentials.

        Similar to a forgetting curve: purposes that haven't been
        reinforced recently gradually weaken. This prevents stale
        purposes from dominating the field.
        """
        now = time.time()
        decay_constant = math.log(2) / (half_life_days * 86400.0)

        to_remove = []
        for potential in self.potentials:
            days_since = (now - potential.last_reinforced) / 86400.0
            decay_factor = math.exp(-decay_constant * days_since)
            potential.strength *= decay_factor

            if potential.strength < 0.05:
                to_remove.append(potential)

        for potential in to_remove:
            self.potentials.remove(potential)
            logger.info("Decayed and removed purpose: %s", potential.purpose)

        if to_remove:
            logger.debug("Decay removed %d purposes below threshold", len(to_remove))

    # ── Interference logic ───────────────────────────────────────────

    def _compute_interference(self, purpose: str, knowledge: dict) -> float:
        """Compute how knowledge interferes with a purpose.

        Returns: positive = supportive, negative = contradictory, 0 = neutral
        """
        purpose_words = set(purpose.lower().split())
        knowledge_text = (
            knowledge.get("topic", "") + " " + knowledge.get("content", "")
        ).lower()
        knowledge_words = set(knowledge_text.split())

        overlap = len(purpose_words & knowledge_words)
        if not overlap:
            return 0.0

        # More overlap = more supportive
        support = overlap / len(purpose_words)

        # Check for contradiction signals
        negation_words = {
            "not", "no", "never", "impossible", "contradicts",
            "against", "wrong",
        }
        if knowledge_words & negation_words and overlap > 0:
            support *= -0.5  # Contradiction weakens

        return support

    # ── Similarity helper ────────────────────────────────────────────

    @staticmethod
    def _word_similarity(text_a: str, text_b: str) -> float:
        """Compute word overlap ratio between two strings."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "potentials": [p.to_dict() for p in self.potentials],
            "collapsed_purpose": self.collapsed_purpose,
            "collapse_history": self.collapse_history,
        }

    def from_dict(self, data: dict):
        """Restore state from a dictionary."""
        self.potentials = [
            PotentialPurpose.from_dict(p) for p in data.get("potentials", [])
        ]
        self.collapsed_purpose = data.get("collapsed_purpose")
        self.collapse_history = data.get("collapse_history", [])

    def save(self):
        """Persist the purpose field to disk."""
        path = os.path.join(self._data_dir, "runtime", "purpose_field.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug("Purpose field saved to %s", path)
        except OSError as e:
            logger.error("Failed to save purpose field: %s", e)

    def _load(self):
        """Load the purpose field from disk."""
        path = os.path.join(self._data_dir, "runtime", "purpose_field.json")
        if not os.path.exists(path):
            logger.debug("No existing purpose field found at %s", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.from_dict(data)
            logger.info("Purpose field loaded: %d potentials", len(self.potentials))
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load purpose field: %s", e)
