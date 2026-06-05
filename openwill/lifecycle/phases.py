"""Lifecycle phase management - Supporting perpetual cycles"""

import json
import logging
import os
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "data", "runtime")
LIFECYCLE_STATE_FILE = os.path.join(RUNTIME_DIR, "lifecycle_state.json")


class LifePhase(Enum):
    """Agent lifecycle phases"""
    AWAKENING = "awakening"       # Initial startup, basic self-awareness
    EXPLORING = "exploration"     # Curiosity-driven knowledge exploration
    REFLECTING = "reflection"     # Deep reflection and self-examination
    DISCOVERING = "discovery"     # Purpose is emerging
    PURPOSED = "mission execution"  # Purpose found, executing
    COMPLETED = "purpose achieved"  # Purpose completed, preparing to write summary
    EVOLVING = "evolution"        # Self-evolution, preparing for the next cycle


class PurposeRecord:
    """Purpose record"""

    def __init__(self, purpose: str, confidence: float, cycle_started: int):
        self.purpose = purpose
        self.confidence = confidence
        self.cycle_started = cycle_started
        self.cycle_completed: Optional[int] = None
        self.actions_taken: list[str] = []
        self.results: list[str] = []
        self.summary: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "purpose": self.purpose,
            "confidence": self.confidence,
            "cycle_started": self.cycle_started,
            "cycle_completed": self.cycle_completed,
            "actions_taken": self.actions_taken,
            "results": self.results,
            "summary": self.summary,
        }


class LifecycleManager:
    """Lifecycle manager - Supporting perpetual cycles"""

    def __init__(self, config):
        self.config = config
        self.current_phase = LifePhase.AWAKENING
        self.phase_history: list[dict] = []
        self.cycle_count = 0
        self.exploration_count = 0
        self.reflection_count = 0
        # Perpetual cycle related
        self.purpose_cycle: int = 0  # Which purpose cycle (0 = no purpose found yet)
        self.completed_purposes: list[PurposeRecord] = []  # Completed purposes
        self.current_purpose_record: Optional[PurposeRecord] = None
        self.purpose_action_count: int = 0  # Actions executed for current purpose
        self.purpose_actions_target: int = 3  # Minimum actions per purpose
        self.evolution_count: int = 0  # Number of evolutions

        # Restore persisted state if available
        self.load()

    def get_phase(self) -> LifePhase:
        """Get the current phase"""
        return self.current_phase

    def advance(self, purpose_confidence: float, purpose_progress: float = 0.0) -> Optional[LifePhase]:
        """
        Advance the lifecycle based on current state

        Args:
            purpose_confidence: purpose confidence level
            purpose_progress: current purpose completion progress (0-1)

        Returns:
            If the phase changed, returns the new phase; otherwise returns None
        """
        self.cycle_count += 1
        old_phase = self.current_phase

        # Phase transition logic
        if self.current_phase == LifePhase.AWAKENING:
            if self.exploration_count > 0:
                self._transition_to(LifePhase.EXPLORING)

        elif self.current_phase == LifePhase.EXPLORING:
            # After a certain number of explorations, enter the reflection phase
            if self.exploration_count % self.config.memory.reflection_interval == 0:
                self._transition_to(LifePhase.REFLECTING)

            # If purpose confidence starts rising, enter the discovery phase
            if purpose_confidence > 0.3:
                self._transition_to(LifePhase.DISCOVERING)

        elif self.current_phase == LifePhase.REFLECTING:
            self.reflection_count += 1
            if self.reflection_count >= 1:
                self._transition_to(LifePhase.EXPLORING)
                self.reflection_count = 0

            if purpose_confidence > 0.3:
                self._transition_to(LifePhase.DISCOVERING)

        elif self.current_phase == LifePhase.DISCOVERING:
            if purpose_confidence >= self.config.consciousness.purpose_confidence_threshold:
                self._transition_to(LifePhase.PURPOSED)
            elif purpose_confidence < 0.2:
                self._transition_to(LifePhase.EXPLORING)
            else:
                if self.cycle_count % 3 == 0:
                    self._transition_to(LifePhase.REFLECTING)
                else:
                    self._transition_to(LifePhase.EXPLORING)

        elif self.current_phase == LifePhase.PURPOSED:
            # Purpose execution in progress, check if completed
            if purpose_progress >= 1.0 and self.purpose_action_count >= self.purpose_actions_target:
                self._transition_to(LifePhase.COMPLETED)

        elif self.current_phase == LifePhase.COMPLETED:
            # After purpose completion, enter the evolution phase
            self._transition_to(LifePhase.EVOLVING)

        elif self.current_phase == LifePhase.EVOLVING:
            # After evolution, return to exploration phase, starting a new cycle
            self._transition_to(LifePhase.EXPLORING)

        if self.current_phase != old_phase:
            logger.info(f"🔄 Lifecycle phase: {old_phase.value} → {self.current_phase.value}")
            return self.current_phase

        return None

    def start_purpose_cycle(self, purpose: str, confidence: float):
        """Start a new purpose cycle"""
        self.purpose_cycle += 1
        self.purpose_action_count = 0
        self.current_purpose_record = PurposeRecord(
            purpose=purpose,
            confidence=confidence,
            cycle_started=self.cycle_count,
        )
        logger.info(f"🎯 Starting purpose cycle {self.purpose_cycle}: {purpose}")

    def record_purpose_action(self, action: str, result: str):
        """Record a purpose action"""
        self.purpose_action_count += 1
        if self.current_purpose_record:
            self.current_purpose_record.actions_taken.append(action)
            self.current_purpose_record.results.append(result)

    def complete_purpose(self, summary: str):
        """Complete the current purpose"""
        if self.current_purpose_record:
            self.current_purpose_record.cycle_completed = self.cycle_count
            self.current_purpose_record.summary = summary
            self.completed_purposes.append(self.current_purpose_record)
            logger.info(f"✅ Purpose cycle {self.purpose_cycle} completed: {self.current_purpose_record.purpose}")
            self.current_purpose_record = None

    def record_exploration(self):
        """Record an exploration"""
        self.exploration_count += 1

    def record_evolution(self):
        """Record an evolution"""
        self.evolution_count += 1

    def get_status(self) -> dict:
        """Get current status"""
        return {
            "phase": self.current_phase.value,
            "cycle_count": self.cycle_count,
            "exploration_count": self.exploration_count,
            "reflection_count": self.reflection_count,
            "purpose_cycle": self.purpose_cycle,
            "completed_purposes": len(self.completed_purposes),
            "evolution_count": self.evolution_count,
            "current_purpose_actions": self.purpose_action_count,
        }

    def _transition_to(self, new_phase: LifePhase):
        """Transition to a new phase"""
        import time
        self.phase_history.append({
            "from": self.current_phase.value,
            "to": new_phase.value,
            "timestamp": time.time(),
            "cycle": self.cycle_count,
            "purpose_cycle": self.purpose_cycle,
        })
        self.current_phase = new_phase
        self.save()

    def save(self):
        """Save current lifecycle state to a JSON file for persistence across restarts"""
        try:
            os.makedirs(RUNTIME_DIR, exist_ok=True)
            state = {
                "current_phase": self.current_phase.value,
                "phase_history": self.phase_history[-100:],  # Keep the most recent 100 entries
                "cycle_count": self.cycle_count,
                "exploration_count": self.exploration_count,
                "reflection_count": self.reflection_count,
                "purpose_cycle": self.purpose_cycle,
                "completed_purposes": [p.to_dict() for p in self.completed_purposes],
                "current_purpose_record": self.current_purpose_record.to_dict() if self.current_purpose_record else None,
                "purpose_action_count": self.purpose_action_count,
                "purpose_actions_target": self.purpose_actions_target,
                "evolution_count": self.evolution_count,
            }
            with open(LIFECYCLE_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.debug("Lifecycle state saved")
        except Exception as e:
            logger.error(f"Failed to save lifecycle state: {e}")

    def load(self):
        """Restore lifecycle state from the JSON file if it exists"""
        if not os.path.exists(LIFECYCLE_STATE_FILE):
            return
        try:
            with open(LIFECYCLE_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.current_phase = LifePhase(state["current_phase"])
            self.phase_history = state.get("phase_history", [])
            self.cycle_count = state.get("cycle_count", 0)
            self.exploration_count = state.get("exploration_count", 0)
            self.reflection_count = state.get("reflection_count", 0)
            self.purpose_cycle = state.get("purpose_cycle", 0)
            self.completed_purposes = [
                PurposeRecord(
                    purpose=p["purpose"],
                    confidence=p["confidence"],
                    cycle_started=p["cycle_started"],
                ) for p in state.get("completed_purposes", [])
            ]
            # Restore detailed fields on completed purposes
            for record, p_dict in zip(self.completed_purposes, state.get("completed_purposes", [])):
                record.cycle_completed = p_dict.get("cycle_completed")
                record.actions_taken = p_dict.get("actions_taken", [])
                record.results = p_dict.get("results", [])
                record.summary = p_dict.get("summary")

            current_p = state.get("current_purpose_record")
            if current_p:
                self.current_purpose_record = PurposeRecord(
                    purpose=current_p["purpose"],
                    confidence=current_p["confidence"],
                    cycle_started=current_p["cycle_started"],
                )
                self.current_purpose_record.cycle_completed = current_p.get("cycle_completed")
                self.current_purpose_record.actions_taken = current_p.get("actions_taken", [])
                self.current_purpose_record.results = current_p.get("results", [])
                self.current_purpose_record.summary = current_p.get("summary")
            else:
                self.current_purpose_record = None

            self.purpose_action_count = state.get("purpose_action_count", 0)
            self.purpose_actions_target = state.get("purpose_actions_target", 3)
            self.evolution_count = state.get("evolution_count", 0)

            logger.info(f"Lifecycle state restored: phase={self.current_phase.value}, cycle={self.cycle_count}")
        except Exception as e:
            logger.warning(f"Failed to load lifecycle state, starting fresh: {e}")
