"""
Conversation state machine for the OpenWill chat server.

Tracks per-session state transitions, task progress, and context
preservation so that multi-step agent interactions can be paused,
resumed, and reflected upon reliably.
"""

import json
import logging
import os
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default persistence path (relative to project root)
_DEFAULT_PERSIST_PATH = os.path.join("data", "runtime", "conversation_sessions.json")


# ----------------------------------------------------------------------
# ConversationState enum
# ----------------------------------------------------------------------


class ConversationState(Enum):
    """States a conversation session can be in."""

    IDLE = "IDLE"                      # No active task, waiting for user input
    TASK_PLANNING = "TASK_PLANNING"    # Agent is planning how to approach a task
    TASK_EXECUTING = "TASK_EXECUTING"  # Agent is executing a multi-step task
    TASK_WAITING = "TASK_WAITING"      # Agent needs user confirmation or input
    TASK_PAUSED = "TASK_PAUSED"        # Task was interrupted, can be resumed
    REFLECTING = "REFLECTING"          # Agent is in a reflection/thinking phase


# ----------------------------------------------------------------------
# Valid state transitions
# ----------------------------------------------------------------------

# Each key maps to the set of states that may follow it.
_VALID_TRANSITIONS: dict[ConversationState, set[ConversationState]] = {
    ConversationState.IDLE: {
        ConversationState.TASK_PLANNING,
        ConversationState.REFLECTING,
        ConversationState.IDLE,           # reset
        ConversationState.TASK_PAUSED,    # interrupt
    },
    ConversationState.TASK_PLANNING: {
        ConversationState.TASK_EXECUTING,
        ConversationState.REFLECTING,
        ConversationState.TASK_PAUSED,    # interrupt
        ConversationState.IDLE,           # reset
    },
    ConversationState.TASK_EXECUTING: {
        ConversationState.TASK_EXECUTING,  # another tool call
        ConversationState.TASK_WAITING,
        ConversationState.IDLE,            # task complete
        ConversationState.REFLECTING,
        ConversationState.TASK_PAUSED,     # interrupt
    },
    ConversationState.TASK_WAITING: {
        ConversationState.TASK_EXECUTING,  # user provided input
        ConversationState.TASK_PAUSED,     # user said "wait" / "pause"
        ConversationState.REFLECTING,
        ConversationState.IDLE,            # reset
    },
    ConversationState.TASK_PAUSED: {
        ConversationState.TASK_EXECUTING,  # resume
        ConversationState.IDLE,            # cancel
        ConversationState.REFLECTING,
        ConversationState.TASK_PAUSED,     # interrupt (idempotent)
    },
    ConversationState.REFLECTING: {
        # REFLECTING can return to any non-IDLE previous state or IDLE.
        # The *actual* return target is stored in context_snapshot["_prev_state"]
        # and validated at transition time.
        ConversationState.IDLE,
        ConversationState.TASK_PLANNING,
        ConversationState.TASK_EXECUTING,
        ConversationState.TASK_WAITING,
        ConversationState.TASK_PAUSED,
        ConversationState.REFLECTING,      # interrupt while reflecting
    },
}


# ----------------------------------------------------------------------
# ConversationSession
# ----------------------------------------------------------------------


class ConversationSession:
    """Tracks state for a single conversation session."""

    __slots__ = (
        "session_id",
        "state",
        "current_task",
        "task_steps",
        "pending_tool_calls",
        "context_snapshot",
        "created_at",
        "updated_at",
        "metadata",
    )

    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.state: ConversationState = ConversationState.IDLE
        self.current_task: Optional[str] = None
        self.task_steps: list[dict] = []
        self.pending_tool_calls: list[dict] = []
        self.context_snapshot: dict = {}
        self.created_at: float = time.time()
        self.updated_at: float = self.created_at
        self.metadata: dict = {}

    # -- Serialisation helpers ------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the session."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "current_task": self.current_task,
            "task_steps": self.task_steps,
            "pending_tool_calls": self.pending_tool_calls,
            "context_snapshot": self.context_snapshot,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationSession":
        """Reconstruct a session from a persisted dict."""
        session = cls(session_id=data["session_id"])
        session.state = ConversationState(data["state"])
        session.current_task = data.get("current_task")
        session.task_steps = data.get("task_steps", [])
        session.pending_tool_calls = data.get("pending_tool_calls", [])
        session.context_snapshot = data.get("context_snapshot", {})
        session.created_at = data.get("created_at", time.time())
        session.updated_at = data.get("updated_at", session.created_at)
        session.metadata = data.get("metadata", {})
        return session


# ----------------------------------------------------------------------
# ConversationManager
# ----------------------------------------------------------------------


class ConversationManager:
    """
    Manages all conversation sessions, enforces valid state transitions,
    persists state to disk, and provides integration helpers for the
    chat server.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = threading.Lock()
        self._persist_path = persist_path or _DEFAULT_PERSIST_PATH
        self._load_sessions()

    # -- Public API -----------------------------------------------------

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Return an existing session or create a new one."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationSession(session_id)
                logger.debug("Created new conversation session: %s", session_id)
            return self._sessions[session_id]

    def transition(
        self,
        session_id: str,
        new_state: ConversationState,
        context: Optional[dict] = None,
    ) -> None:
        """
        Validate and execute a state transition.

        Raises ``ValueError`` if the transition is not allowed.
        On success the context snapshot is updated and the session is
        auto-saved.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            old_state = session.state

            # Special handling for REFLECTING: remember where to return to.
            if new_state == ConversationState.REFLECTING and old_state != ConversationState.REFLECTING:
                context = context or {}
                context["_prev_state"] = old_state.value

            # Special handling for leaving REFLECTING: return to previous state
            # if the caller just says "go back to previous".
            if old_state == ConversationState.REFLECTING and new_state != ConversationState.REFLECTING:
                prev_state_value = session.context_snapshot.get("_prev_state")
                if prev_state_value:
                    prev_state = ConversationState(prev_state_value)
                    # Validate that the requested target is compatible
                    allowed = _VALID_TRANSITIONS.get(ConversationState.REFLECTING, set())
                    if new_state not in allowed:
                        raise ValueError(
                            f"Invalid transition from REFLECTING to {new_state.value}"
                        )
                    # If the caller passes the previous state explicitly that's fine;
                    # otherwise we auto-route to the stored previous state.
                    if new_state == prev_state:
                        pass  # explicit and matches
                    # Both are valid; proceed with new_state as given.

            # General validation
            allowed = _VALID_TRANSITIONS.get(old_state, set())
            if new_state not in allowed:
                raise ValueError(
                    f"Invalid state transition: {old_state.value} -> {new_state.value}"
                )

            # Save context snapshot before transitioning
            self._save_context_snapshot(session, context)

            # Apply transition
            session.state = new_state
            session.updated_at = time.time()

            # Clear task info when going back to IDLE
            if new_state == ConversationState.IDLE:
                session.current_task = None
                session.task_steps = []
                session.pending_tool_calls = []

            logger.info(
                "Session %s: %s -> %s",
                session_id,
                old_state.value,
                new_state.value,
            )

            self._save_sessions()

    def get_state(self, session_id: str) -> ConversationState:
        """Return the current state of a session (IDLE if unknown)."""
        with self._lock:
            session = self._sessions.get(session_id)
            return session.state if session else ConversationState.IDLE

    def add_step(self, session_id: str, step: str, result: str) -> None:
        """Record a task step in the session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            session.task_steps.append({
                "step": step,
                "result": result,
                "timestamp": time.time(),
            })
            session.updated_at = time.time()
            self._save_sessions()

    def interrupt(self, session_id: str) -> None:
        """Pause the current task and save context for later resume."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.state == ConversationState.TASK_PAUSED:
                return  # Already paused
            if session.state == ConversationState.IDLE:
                return  # Nothing to interrupt
            # Save a rich snapshot before pausing
            self._save_context_snapshot(session, {"interrupted_from": session.state.value})
            session.state = ConversationState.TASK_PAUSED
            session.updated_at = time.time()
            logger.info("Session %s interrupted -> TASK_PAUSED", session_id)
            self._save_sessions()

    def resume(self, session_id: str) -> None:
        """Resume a paused session back to TASK_EXECUTING."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.state != ConversationState.TASK_PAUSED:
                raise ValueError(
                    f"Cannot resume session in state {session.state.value}; "
                    f"expected TASK_PAUSED"
                )
            session.state = ConversationState.TASK_EXECUTING
            session.updated_at = time.time()
            logger.info("Session %s resumed -> TASK_EXECUTING", session_id)
            self._save_sessions()

    def reset(self, session_id: str) -> None:
        """Clear a session back to IDLE."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            session.state = ConversationState.IDLE
            session.current_task = None
            session.task_steps = []
            session.pending_tool_calls = []
            session.context_snapshot = {}
            session.updated_at = time.time()
            logger.info("Session %s reset -> IDLE", session_id)
            self._save_sessions()

    # -- Integration helpers --------------------------------------------

    def build_state_context(self, session_id: str) -> str:
        """
        Generate a text summary of the current session state suitable
        for injection into an LLM system prompt.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return "[Session State: IDLE]"

        lines: list[str] = []
        lines.append(f"[Session State: {session.state.value}]")

        if session.current_task:
            lines.append(f"Current task: {session.current_task}")

        if session.task_steps:
            lines.append(f"Steps completed: {len(session.task_steps)}/?")
            for i, s in enumerate(session.task_steps, 1):
                result_preview = s["result"][:120]
                lines.append(f"  - Step {i}: {s['step']} -> {result_preview}")

        if session.pending_tool_calls:
            lines.append(f"Pending tool calls: {len(session.pending_tool_calls)}")
            for tc in session.pending_tool_calls:
                lines.append(f"  - {tc.get('tool', '?')}({tc.get('args', {})})")

        if session.state == ConversationState.TASK_PAUSED:
            prev = session.context_snapshot.get("interrupted_from", "unknown")
            lines.append(f"Paused (was: {prev}). Awaiting resume or cancel.")

        if session.state == ConversationState.TASK_WAITING:
            lines.append("Next: Awaiting user input to continue.")

        return "\n".join(lines)

    def should_persist_state(self, session_id: str) -> bool:
        """Return True if the session has meaningful state worth preserving."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            # IDLE sessions with no history are not worth persisting
            if session.state == ConversationState.IDLE and not session.task_steps:
                return False
            return True

    # -- Context preservation -------------------------------------------

    @staticmethod
    def _save_context_snapshot(
        session: ConversationSession,
        context: Optional[dict] = None,
    ) -> None:
        """
        Save a snapshot of key context at the time of a state change.

        Captures: current task, steps so far, last tool call result,
        and any caller-supplied context dict.
        """
        snapshot: dict = {
            "current_task": session.current_task,
            "steps_count": len(session.task_steps),
            "last_step": session.task_steps[-1] if session.task_steps else None,
            "last_tool_result": (
                session.task_steps[-1]["result"] if session.task_steps else None
            ),
        }
        if context:
            snapshot.update(context)
        # Merge into existing snapshot so we don't lose previously stored keys
        # (e.g. _prev_state for REFLECTING) unless the new context overwrites.
        session.context_snapshot.update(snapshot)

    # -- Persistence ----------------------------------------------------

    def _save_sessions(self) -> None:
        """Persist all sessions to disk."""
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            data = {
                sid: s.to_dict()
                for sid, s in self._sessions.items()
            }
            tmp_path = self._persist_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Atomic-ish rename on Windows (may fail if dest exists on some
            # Python versions, but write+rename is still safer than direct).
            try:
                os.replace(tmp_path, self._persist_path)
            except OSError:
                # Fallback: just overwrite directly
                with open(self._persist_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        except Exception as exc:
            logger.error("Failed to persist conversation sessions: %s", exc)

    def _load_sessions(self) -> None:
        """Load sessions from disk (best-effort)."""
        if not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, sdata in data.items():
                try:
                    self._sessions[sid] = ConversationSession.from_dict(sdata)
                except Exception as exc:
                    logger.warning("Skipping corrupt session %s: %s", sid, exc)
            logger.info("Loaded %d conversation sessions", len(self._sessions))
        except Exception as exc:
            logger.error("Failed to load conversation sessions: %s", exc)
