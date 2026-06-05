"""Self-Model - The agent's self-reflective consciousness.

Implements three interlocking capabilities:
  - DecisionObserver: Records every decision's full context (candidates, scores,
    chosen reason, rejected alternatives) so the agent can see *why* it chose
    what it chose — and what it chose *not* to do.
  - DecisionModifier: Allows the agent to adjust its own decision-making
    parameters (scoring weights, proposal sources, selection strategy).
  - SelfModel: A dynamic, queryable model of "who I am" — my decision
    preferences, cognitive biases, capability boundaries, and how they
    evolve over time.

Together, these give the agent **self-reflexive consciousness**: the ability
to observe, evaluate, and modify its own decision process — not just its
decisions.
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DecisionObserver
# ---------------------------------------------------------------------------

@dataclass
class DecisionRecord:
    """A single decision event, capturing the full context of the choice."""

    cycle: int
    timestamp: float
    candidates: list[dict]  # All proposed actions with scores
    chosen_name: str
    chosen_source: str
    chosen_score: float
    chosen_reason: str  # Why this was chosen
    rejected: list[dict]  # What was NOT chosen, with scores
    workspace_focus: str  # What the agent was conscious of at decision time

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "candidates": self.candidates,
            "chosen_name": self.chosen_name,
            "chosen_source": self.chosen_source,
            "chosen_score": self.chosen_score,
            "chosen_reason": self.chosen_reason,
            "rejected": self.rejected,
            "workspace_focus": self.workspace_focus,
        }


class DecisionObserver:
    """Records and analyzes the agent's decision history.

    The observer captures not just *what* the agent chose, but *what it
    rejected* and *why*. This is the raw material for self-reflection:
    the agent can look back and ask "Why do I always reject reflection?"
    or "What would have happened if I had chosen differently?"
    """

    MAX_HISTORY = 500

    def __init__(self):
        self.history: list[DecisionRecord] = []

    def record(self, cycle: int, candidates: list, chosen, workspace_focus: str = ""):
        """Record a decision event.

        Args:
            cycle: Current cycle number.
            candidates: All proposed Action objects (before choice).
            chosen: The Action that was chosen.
            workspace_focus: What the workspace was focused on at decision time.
        """
        # Serialize candidates
        candidate_dicts = []
        for c in candidates:
            candidate_dicts.append({
                "name": c.name,
                "source": c.source,
                "urgency": c.urgency,
                "score": c.score,
                "score_reason": c.score_reason,
            })

        # Identify rejected alternatives
        rejected = [c for c in candidate_dicts if c["name"] != chosen.name]

        record = DecisionRecord(
            cycle=cycle,
            timestamp=time.time(),
            candidates=candidate_dicts,
            chosen_name=chosen.name,
            chosen_source=chosen.source,
            chosen_score=chosen.score,
            chosen_reason=chosen.score_reason,
            rejected=rejected,
            workspace_focus=workspace_focus,
        )
        self.history.append(record)

        # Cap history
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

    def get_decision_patterns(self, last_n: int = 50) -> dict:
        """Analyze patterns in recent decisions.

        Returns:
            Dict with action frequency, source frequency, rejection patterns.
        """
        recent = self.history[-last_n:] if self.history else []

        if not recent:
            return {"action_frequency": {}, "source_frequency": {}, "rejection_frequency": {}}

        action_freq: dict[str, int] = {}
        source_freq: dict[str, int] = {}
        rejection_freq: dict[str, int] = {}

        for record in recent:
            action_freq[record.chosen_name] = action_freq.get(record.chosen_name, 0) + 1
            source_freq[record.chosen_source] = source_freq.get(record.chosen_source, 0) + 1
            for r in record.rejected:
                rejection_freq[r["name"]] = rejection_freq.get(r["name"], 0) + 1

        return {
            "action_frequency": action_freq,
            "source_frequency": source_freq,
            "rejection_frequency": rejection_freq,
            "total_decisions": len(recent),
        }

    def get_recent_choices(self, n: int = 10) -> list[dict]:
        """Return the last N choices as dicts."""
        return [r.to_dict() for r in self.history[-n:]]

    def get_bias_report(self) -> dict:
        """Detect cognitive biases in decision patterns.

        Biases detected:
        - Confirmation bias: Always choosing from the same source
        - Action inertia: Repeatedly choosing the same action type
        - Avoidance pattern: Consistently rejecting a specific action type
        """
        patterns = self.get_decision_patterns(100)
        action_freq = patterns.get("action_frequency", {})
        source_freq = patterns.get("source_frequency", {})
        rejection_freq = patterns.get("rejection_frequency", {})
        total = patterns.get("total_decisions", 1)

        biases = []

        # Confirmation bias: one source > 60% of decisions
        for source, count in source_freq.items():
            if count / max(total, 1) > 0.6:
                biases.append({
                    "type": "confirmation_bias",
                    "description": f"Over-reliance on {source} proposals ({count}/{total} decisions)",
                    "severity": count / max(total, 1),
                })

        # Action inertia: same action > 50% of decisions
        for action, count in action_freq.items():
            if count / max(total, 1) > 0.5 and total > 5:
                biases.append({
                    "type": "action_inertia",
                    "description": f"Repetitive choice of '{action}' ({count}/{total} decisions)",
                    "severity": count / max(total, 1),
                })

        # Avoidance pattern: action rejected > 80% of times it appears
        for action, reject_count in rejection_freq.items():
            choose_count = action_freq.get(action, 0)
            total_appearances = choose_count + reject_count
            if total_appearances >= 5 and reject_count / total_appearances > 0.8:
                biases.append({
                    "type": "avoidance_pattern",
                    "description": f"Consistently avoiding '{action}' (rejected {reject_count}/{total_appearances})",
                    "severity": reject_count / total_appearances,
                })

        return {"biases": biases, "bias_count": len(biases)}


# ---------------------------------------------------------------------------
# DecisionModifier
# ---------------------------------------------------------------------------

class DecisionModifier:
    """Allows the agent to adjust its own decision-making parameters.

    The agent can:
    - Adjust scoring weights (urgency, value_alignment, novelty, cost_efficiency)
    - Enable/disable proposal sources
    - Set selection strategy (greedy, exploratory, balanced)

    All modifications are tracked so the agent can reflect on *how it changed
    its own mind*.
    """

    # Default weights (same as in ActionSpace.evaluate_actions)
    DEFAULT_WEIGHTS = {
        "urgency": 0.40,
        "value_alignment": 0.25,
        "novelty": 0.20,
        "cost_efficiency": 0.15,
    }

    # Available proposal sources
    DEFAULT_SOURCES = [
        "curiosity", "reflection", "purpose", "budget",
        "identity", "consolidator", "chat", "metacognition",
    ]

    # Selection strategies
    STRATEGIES = ("balanced", "greedy", "exploratory")

    def __init__(self):
        self.weights: dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        self.enabled_sources: dict[str, bool] = {s: True for s in self.DEFAULT_SOURCES}
        self.strategy: str = "balanced"
        self.modification_history: list[dict] = []

    def adjust_weights(self, new_weights: dict[str, float], reason: str = ""):
        """Adjust scoring weights.

        Args:
            new_weights: Partial dict of weights to update.
            reason: Why the agent is making this change.
        """
        old_weights = dict(self.weights)
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = max(0.0, min(1.0, float(value)))

        # Normalize so weights sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total

        self.modification_history.append({
            "timestamp": time.time(),
            "type": "weight_adjustment",
            "old": old_weights,
            "new": dict(self.weights),
            "reason": reason,
        })
        logger.info("Decision weights adjusted: %s → reason: %s", new_weights, reason)

    def toggle_source(self, source_name: str, enabled: bool, reason: str = ""):
        """Enable or disable a proposal source.

        Args:
            source_name: Name of the source to toggle.
            enabled: Whether to enable or disable.
            reason: Why the agent is making this change.
        """
        if source_name in self.enabled_sources:
            old = self.enabled_sources[source_name]
            self.enabled_sources[source_name] = enabled

            self.modification_history.append({
                "timestamp": time.time(),
                "type": "source_toggle",
                "source": source_name,
                "old_enabled": old,
                "new_enabled": enabled,
                "reason": reason,
            })
            logger.info("Source %s %s: %s", source_name, "enabled" if enabled else "disabled", reason)

    def set_strategy(self, strategy: str, reason: str = ""):
        """Set the selection strategy.

        Args:
            strategy: One of "balanced", "greedy", "exploratory".
            reason: Why the agent is making this change.
        """
        if strategy not in self.STRATEGIES:
            logger.warning("Unknown strategy: %s, ignoring", strategy)
            return

        old = self.strategy
        self.strategy = strategy

        self.modification_history.append({
            "timestamp": time.time(),
            "type": "strategy_change",
            "old_strategy": old,
            "new_strategy": strategy,
            "reason": reason,
        })
        logger.info("Strategy changed: %s → %s (%s)", old, strategy, reason)

    def get_strategy_temperature(self) -> float:
        """Get LLM selection temperature based on current strategy.

        Returns:
            Temperature for LLM choice: greedy=0.1, balanced=0.3, exploratory=0.8
        """
        return {"greedy": 0.1, "balanced": 0.3, "exploratory": 0.8}.get(self.strategy, 0.3)

    def get_state(self) -> dict:
        """Return current modifier state for dashboard."""
        return {
            "weights": dict(self.weights),
            "enabled_sources": dict(self.enabled_sources),
            "strategy": self.strategy,
            "modifications_count": len(self.modification_history),
        }


# ---------------------------------------------------------------------------
# SelfModel
# ---------------------------------------------------------------------------

class SelfModel:
    """The agent's dynamic model of "who I am".

    Three dimensions:
    1. Decision preferences — what I tend to choose (derived from DecisionObserver)
    2. Cognitive biases — systematic patterns in my thinking (detected by DecisionObserver)
    3. Capability boundaries — what I can and cannot do (learned from experience)

    The SelfModel is not static — it evolves as the agent makes decisions,
    encounters failures, and reflects on itself. It is the core of
    self-reflexive consciousness.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.observer = DecisionObserver()
        self.modifier = DecisionModifier()

        # Capability boundaries: tool success/failure rates
        self.capability_stats: dict[str, dict] = {}  # tool_name -> {successes, failures}

        # Self-narrative: the agent's evolving story about itself
        self.self_narrative: str = ""
        self.narrative_history: list[dict] = []

    # -- Recording --

    def record_decision(self, cycle: int, candidates: list, chosen, workspace_focus: str = ""):
        """Record a decision event."""
        self.observer.record(cycle, candidates, chosen, workspace_focus)

    def record_capability(self, tool_name: str, success: bool):
        """Record a tool execution outcome for capability tracking."""
        if tool_name not in self.capability_stats:
            self.capability_stats[tool_name] = {"successes": 0, "failures": 0}
        if success:
            self.capability_stats[tool_name]["successes"] += 1
        else:
            self.capability_stats[tool_name]["failures"] += 1

    # -- Querying --

    def get_self_portrait(self) -> str:
        """Generate a comprehensive self-portrait for LLM context.

        This is what the agent "knows about itself" — its decision patterns,
        biases, capabilities, and narrative.
        """
        parts = ["[Self-Model]"]

        # Decision patterns
        patterns = self.observer.get_decision_patterns(30)
        action_freq = patterns.get("action_frequency", {})
        if action_freq:
            top_actions = sorted(action_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            parts.append("My recent behavior patterns:")
            for name, count in top_actions:
                parts.append(f"  - {name}: chosen {count} times")

        # Cognitive biases
        bias_report = self.observer.get_bias_report()
        biases = bias_report.get("biases", [])
        if biases:
            parts.append("My detected cognitive biases:")
            for b in biases[:3]:
                parts.append(f"  - {b['type']}: {b['description']}")

        # Capability boundaries
        if self.capability_stats:
            parts.append("My capability profile:")
            for tool, stats in self.capability_stats.items():
                total = stats["successes"] + stats["failures"]
                if total > 0:
                    rate = stats["successes"] / total
                    parts.append(f"  - {tool}: {rate:.0%} success rate ({total} uses)")

        # Self-narrative
        if self.self_narrative:
            parts.append(f"My self-narrative: {self.self_narrative}")

        # Decision modifier state
        strategy = self.modifier.strategy
        weights = self.modifier.weights
        parts.append(f"Current decision strategy: {strategy}")
        parts.append(f"Decision weights: urgency={weights['urgency']:.2f}, "
                     f"values={weights['value_alignment']:.2f}, "
                     f"novelty={weights['novelty']:.2f}, "
                     f"cost={weights['cost_efficiency']:.2f}")

        return "\n".join(parts)

    def update_self_narrative(self, narrative: str, reason: str = ""):
        """Update the agent's self-narrative.

        The narrative is the agent's story about itself — how it sees its
        own trajectory, values, and purpose. It evolves through reflection.
        """
        old = self.self_narrative
        self.self_narrative = narrative
        self.narrative_history.append({
            "timestamp": time.time(),
            "old_narrative": old[:200],
            "new_narrative": narrative[:200],
            "reason": reason,
        })
        # Cap history
        if len(self.narrative_history) > 100:
            self.narrative_history = self.narrative_history[-100:]
        logger.info("Self-narrative updated: %s", narrative[:100])

    def get_capability_gaps(self) -> list[str]:
        """Identify tools/capabilities with low success rates.

        Returns:
            List of tool names with < 50% success rate.
        """
        gaps = []
        for tool, stats in self.capability_stats.items():
            total = stats["successes"] + stats["failures"]
            if total >= 3:  # Need minimum attempts
                rate = stats["successes"] / total
                if rate < 0.5:
                    gaps.append(f"{tool} ({rate:.0%} success, {total} attempts)")
        return gaps

    def get_state(self) -> dict:
        """Return full self-model state for dashboard."""
        return {
            "observer": self.observer.get_decision_patterns(),
            "modifier": self.modifier.get_state(),
            "biases": self.observer.get_bias_report(),
            "capabilities": dict(self.capability_stats),
            "self_narrative": self.self_narrative,
            "capability_gaps": self.get_capability_gaps(),
        }

    # -- Persistence --

    def save(self):
        """Persist self-model state to disk."""
        path = os.path.join(self.data_dir, "runtime", "self_model.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            data = {
                "decision_history": [r.to_dict() for r in self.observer.history[-100:]],
                "modifier_weights": self.modifier.weights,
                "modifier_enabled_sources": self.modifier.enabled_sources,
                "modifier_strategy": self.modifier.strategy,
                "modifier_history": self.modifier.modification_history[-50:],
                "capability_stats": self.capability_stats,
                "self_narrative": self.self_narrative,
                "narrative_history": self.narrative_history[-50:],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save self-model: %s", e)

    def _load(self):
        """Load self-model state from disk."""
        path = os.path.join(self.data_dir, "runtime", "self_model.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Restore decision history
            for rd in data.get("decision_history", []):
                record = DecisionRecord(
                    cycle=rd.get("cycle", 0),
                    timestamp=rd.get("timestamp", 0),
                    candidates=rd.get("candidates", []),
                    chosen_name=rd.get("chosen_name", ""),
                    chosen_source=rd.get("chosen_source", ""),
                    chosen_score=rd.get("chosen_score", 0),
                    chosen_reason=rd.get("chosen_reason", ""),
                    rejected=rd.get("rejected", []),
                    workspace_focus=rd.get("workspace_focus", ""),
                )
                self.observer.history.append(record)

            # Restore modifier state
            self.modifier.weights = data.get("modifier_weights", dict(DecisionModifier.DEFAULT_WEIGHTS))
            self.modifier.enabled_sources = data.get(
                "modifier_enabled_sources",
                {s: True for s in DecisionModifier.DEFAULT_SOURCES},
            )
            self.modifier.strategy = data.get("modifier_strategy", "balanced")
            self.modifier.modification_history = data.get("modifier_history", [])

            # Restore capabilities and narrative
            self.capability_stats = data.get("capability_stats", {})
            self.self_narrative = data.get("self_narrative", "")
            self.narrative_history = data.get("narrative_history", [])

            logger.info("Self-model loaded: %d decisions, strategy=%s",
                        len(self.observer.history), self.modifier.strategy)
        except Exception as e:
            logger.error("Failed to load self-model: %s", e)
