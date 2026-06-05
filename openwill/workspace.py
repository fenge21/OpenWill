"""Global Workspace for consciousness integration.

Implements Baars' Global Workspace Theory: multiple parallel processes compete
for access to a shared "spotlight", and the winner is broadcast to all modules.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BroadcastMessage:
    """A message competing for access to the global workspace spotlight."""

    source: str  # Which module sent this (e.g. "curiosity", "reflection", "identity")
    content: str  # The message content
    urgency: float  # 0-1, how important this is
    timestamp: float  # When the message was submitted
    metadata: dict = field(default_factory=dict)  # Optional extra info


class GlobalWorkspace:
    """Global Workspace implementing Baars' Global Workspace Theory.

    Multiple parallel processes compete for access to a shared "spotlight".
    The winner is broadcast to all modules, creating a unified conscious experience.
    """

    MAX_HISTORY = 100
    FATIGUE_WINDOW = 3  # Number of consecutive wins to trigger fatigue
    FATIGUE_PENALTY = 0.7  # Multiplier applied when a source is fatigued

    def __init__(self):
        self.current_focus: Optional[BroadcastMessage] = None
        self.broadcast_history: list[BroadcastMessage] = []
        self.competition_buffer: list[BroadcastMessage] = []
        self.module_states: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, source: str, content: str, urgency: float,
               metadata: dict = None) -> None:
        """A module submits content to compete for attention.

        Args:
            source: Which module is submitting (e.g. "curiosity").
            content: The message content.
            urgency: Importance from 0 to 1.
            metadata: Optional extra information.
        """
        if not 0.0 <= urgency <= 1.0:
            urgency = max(0.0, min(1.0, urgency))
        msg = BroadcastMessage(
            source=source,
            content=content,
            urgency=urgency,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        with self._lock:
            self.competition_buffer.append(msg)

    def _fatigue_penalty(self, source: str) -> float:
        """Compute fatigue multiplier for a source.

        If the same source won the last FATIGUE_WINDOW broadcasts, its effective
        urgency is reduced by 30%. This prevents one module from monopolizing
        consciousness.

        Returns:
            0.7 if fatigued, 1.0 otherwise.
        """
        if len(self.broadcast_history) < self.FATIGUE_WINDOW:
            return 1.0
        recent_sources = [
            msg.source for msg in self.broadcast_history[-self.FATIGUE_WINDOW:]
        ]
        if all(s == source for s in recent_sources):
            return self.FATIGUE_PENALTY
        return 1.0

    def resolve(self) -> Optional[BroadcastMessage]:
        """Resolve competition: highest effective urgency wins.

        Urgency is adjusted by fatigue penalty. If tied, the most recent
        submission wins. The winner becomes current_focus, is appended to
        broadcast_history (capped at MAX_HISTORY), and the competition buffer
        is cleared.

        Returns:
            The winning BroadcastMessage, or None if no submissions.
        """
        with self._lock:
            if not self.competition_buffer:
                return None

            # Score each message with fatigue-adjusted urgency
            def _effective_urgency(msg: BroadcastMessage) -> float:
                return msg.urgency * self._fatigue_penalty(msg.source)

            winner = max(
                self.competition_buffer,
                key=lambda m: (_effective_urgency(m), m.timestamp),
            )

            self.current_focus = winner
            self.broadcast_history.append(winner)
            if len(self.broadcast_history) > self.MAX_HISTORY:
                self.broadcast_history = self.broadcast_history[-self.MAX_HISTORY:]
            self.competition_buffer.clear()
            return winner

    def broadcast(self) -> Optional[BroadcastMessage]:
        """Return the current focus (the last resolved broadcast).

        All modules can read this to know what the agent is currently
        "thinking about".
        """
        with self._lock:
            return self.current_focus

    def get_context_for_llm(self) -> str:
        """Generate a text summary of the workspace state for LLM injection.

        Includes current focus, recent broadcast history (last 5), and a
        module states summary.

        Returns:
            Formatted string describing the consciousness state.
        """
        with self._lock:
            lines = ["[Consciousness State]"]

            # Current focus
            if self.current_focus:
                lines.append(
                    f"Current focus: {self.current_focus.source}: "
                    f"{self.current_focus.content}"
                )
            else:
                lines.append("Current focus: none")

            # Recent thoughts (last 5)
            recent = self.broadcast_history[-5:] if self.broadcast_history else []
            if recent:
                lines.append("Recent thoughts:")
                for msg in recent:
                    lines.append(
                        f"- {msg.source}: {msg.content} "
                        f"(urgency: {msg.urgency:.2f})"
                    )

            # Module states summary
            if self.module_states:
                lines.append("Module states:")
                for name, state in self.module_states.items():
                    state_summary = ", ".join(
                        f"{k}: {v}" for k, v in state.items()
                    )
                    lines.append(f"- {name}: {state_summary}")

            return "\n".join(lines)

    def update_module_state(self, module_name: str, state: dict) -> None:
        """Register a module's current state.

        Args:
            module_name: Name of the module (e.g. "curiosity").
            state: Current state dict for the module.
        """
        with self._lock:
            self.module_states[module_name] = state

    def get_module_state(self, module_name: str) -> dict:
        """Get a module's last registered state.

        Args:
            module_name: Name of the module.

        Returns:
            The module's state dict, or empty dict if not registered.
        """
        with self._lock:
            return self.module_states.get(module_name, {})

    def get_full_state(self) -> dict:
        """Return the entire workspace state for dashboard/API.

        Returns:
            Dict containing current focus, recent history, and module states.
        """
        with self._lock:
            return {
                "current_focus": (
                    {
                        "source": self.current_focus.source,
                        "content": self.current_focus.content,
                        "urgency": self.current_focus.urgency,
                        "timestamp": self.current_focus.timestamp,
                        "metadata": self.current_focus.metadata,
                    }
                    if self.current_focus
                    else None
                ),
                "broadcast_history": [
                    {
                        "source": msg.source,
                        "content": msg.content,
                        "urgency": msg.urgency,
                        "timestamp": msg.timestamp,
                        "metadata": msg.metadata,
                    }
                    for msg in self.broadcast_history
                ],
                "module_states": dict(self.module_states),
                "competition_buffer_size": len(self.competition_buffer),
            }

    def clear_focus(self) -> None:
        """Reset current_focus to None (e.g. at start of new cycle)."""
        with self._lock:
            self.current_focus = None


def inject_into_prompt(prompt: str, workspace: GlobalWorkspace) -> str:
    """Append workspace context to any LLM prompt.

    If the workspace has a current focus, the consciousness state is added
    to the prompt. This ensures ALL LLM calls are aware of the agent's
    current conscious state.

    Args:
        prompt: The original LLM prompt.
        workspace: The GlobalWorkspace instance to read state from.

    Returns:
        The prompt with workspace context appended.
    """
    context = workspace.get_context_for_llm()
    if context.strip() == "[Consciousness State]":
        # No meaningful state to inject
        return prompt
    return f"{prompt}\n\n{context}"
