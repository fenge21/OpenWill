"""Cron-based task scheduler for the OpenWill agent.

Runs as a background daemon thread inside the agent process.  When a
scheduled task becomes due, it calls ``agent.llm.chat()`` with the
task description so the agent can use its full capabilities (tools,
memory, etc.) to carry it out.
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

_DEFAULT_TASKS_PATH = os.path.join("data", "runtime", "scheduled_tasks.json")


def _tasks_path(data_dir: str) -> str:
    """Return the path used for persisting scheduled tasks."""
    return os.path.join(data_dir, "runtime", "scheduled_tasks.json")


# ---------------------------------------------------------------------------
# ScheduledTask dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScheduledTask:
    """A single cron-based scheduled task."""

    id: str
    name: str
    description: str
    cron_expression: str
    created_at: float
    last_run: Optional[float] = None
    last_result: Optional[str] = None
    enabled: bool = True


# ---------------------------------------------------------------------------
# Natural-language cron parser
# ---------------------------------------------------------------------------

# Ordered list of (compiled regex, replacement template)
_NL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "every day at 9am" / "every day at 9:30am"
    (
        re.compile(
            r"every\s+day\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            re.IGNORECASE,
        ),
        lambda m: _replace_daily(m),
    ),
    # "daily at midnight"
    (
        re.compile(r"daily\s+at\s+midnight", re.IGNORECASE),
        "0 0 * * *",
    ),
    # "daily at noon"
    (
        re.compile(r"daily\s+at\s+noon", re.IGNORECASE),
        "0 12 * * *",
    ),
    # "every monday at 10am" / "every weekday at 9am"
    (
        re.compile(
            r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            re.IGNORECASE,
        ),
        lambda m: _replace_weekly(m),
    ),
    # "every hour"
    (
        re.compile(r"every\s+hour", re.IGNORECASE),
        "0 * * * *",
    ),
    # "every 30 minutes" / "every 15 minutes"
    (
        re.compile(r"every\s+(\d+)\s+minutes?", re.IGNORECASE),
        lambda m: f"*/{m.group(1)} * * * *",
    ),
    # "every 2 hours"
    (
        re.compile(r"every\s+(\d+)\s+hours?", re.IGNORECASE),
        lambda m: f"0 */{m.group(1)} * * *",
    ),
]

_WEEKDAY_MAP = {
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6",
    "sunday": "0",
}


def _to_24h(hour: int, minute: int, ampm: Optional[str]) -> tuple[int, int]:
    """Convert 12-hour time to 24-hour."""
    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
    return hour, minute


def _replace_daily(m: re.Match) -> str:
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    ampm = m.group(3)
    hour, minute = _to_24h(hour, minute, ampm)
    return f"{minute} {hour} * * *"


def _replace_weekly(m: re.Match) -> str:
    day_name = m.group(1).lower()
    hour = int(m.group(2))
    minute = int(m.group(3)) if m.group(3) else 0
    ampm = m.group(4)
    hour, minute = _to_24h(hour, minute, ampm)
    dow = _WEEKDAY_MAP.get(day_name, "*")
    return f"{minute} {hour} * * {dow}"


def parse_natural_time(text: str) -> str:
    """Convert common natural-language time patterns to a 5-field cron expression.

    If no known pattern matches, the original text is returned unchanged so
    that a raw cron expression can pass through, or the caller can handle the
    failure gracefully.
    """
    for pattern, replacement in _NL_PATTERNS:
        m = pattern.search(text)
        if m:
            if callable(replacement):
                return replacement(m)
            return replacement
    return text


# ---------------------------------------------------------------------------
# Cron matching
# ---------------------------------------------------------------------------

def _cron_match(expr: str, dt: datetime) -> bool:
    """Return True if *dt* satisfies the 5-field cron *expr*.

    Supported field syntax:
      *   — match any
      N   — exact value
      */N — every Nth step
      N,M — list of values
      N-M — inclusive range
    """
    fields = expr.strip().split()
    if len(fields) != 5:
        logger.warning(f"Invalid cron expression (expected 5 fields): {expr}")
        return False

    minute, hour, day_month, month, day_week = fields
    checks = [
        (minute, dt.minute, 0, 59),
        (hour, dt.hour, 0, 23),
        (day_month, dt.day, 1, 31),
        (month, dt.month, 1, 12),
        (day_week, dt.isoweekday() % 7, 0, 6),  # Sunday=0
    ]

    for field, value, lo, hi in checks:
        if not _field_match(field, value, lo, hi):
            return False
    return True


def _field_match(field: str, value: int, lo: int, hi: int) -> bool:
    """Check a single cron field against a value."""
    # Asterisk — always matches
    if field == "*":
        return True

    # Comma-separated list (e.g. "1,15")
    if "," in field:
        return any(_field_match(part, value, lo, hi) for part in field.split(","))

    # Range (e.g. "1-5")
    if "-" in field and "/" not in field:
        parts = field.split("-")
        if len(parts) == 2:
            try:
                return int(parts[0]) <= value <= int(parts[1])
            except ValueError:
                return False

    # Step (e.g. "*/5" or "0-30/5")
    if "/" in field:
        base, step = field.split("/", 1)
        try:
            step_val = int(step)
        except ValueError:
            return False
        if step_val <= 0:
            return False
        if base == "*":
            return value % step_val == 0
        # e.g. "10-50/5"
        if "-" in base:
            parts = base.split("-")
            if len(parts) == 2:
                try:
                    start, end = int(parts[0]), int(parts[1])
                    return start <= value <= end and (value - start) % step_val == 0
                except ValueError:
                    return False
        try:
            start = int(base)
            return value >= start and (value - start) % step_val == 0
        except ValueError:
            return False

    # Exact value
    try:
        return int(field) == value
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# CronScheduler
# ---------------------------------------------------------------------------

class CronScheduler:
    """Cron-based task scheduler that runs as a background daemon thread.

    Receives the agent object so it can call ``agent.llm.chat()`` with the
    task description when a scheduled task is due.
    """

    TICK_INTERVAL = 60  # seconds between checks

    def __init__(self, agent):
        self.agent = agent
        self._tasks: list[ScheduledTask] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._load_tasks()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_path(self) -> str:
        data_dir = getattr(self.agent.config, "memory", None)
        data_dir = getattr(data_dir, "data_dir", "data") if data_dir else "data"
        return _tasks_path(data_dir)

    def _load_tasks(self):
        """Load tasks from the JSON file on disk."""
        path = self._persist_path()
        if not os.path.exists(path):
            self._tasks = []
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._tasks = [ScheduledTask(**item) for item in raw]
            logger.info(f"Loaded {len(self._tasks)} scheduled tasks from {path}")
        except Exception as e:
            logger.error(f"Failed to load scheduled tasks: {e}")
            self._tasks = []

    def _save_tasks(self):
        """Persist the current task list to disk."""
        path = self._persist_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([asdict(t) for t in self._tasks], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save scheduled tasks: {e}")

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    def add_task(self, name: str, description: str, cron_expression: str) -> ScheduledTask:
        """Add a new scheduled task and persist."""
        cron_expression = parse_natural_time(cron_expression)
        task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            cron_expression=cron_expression,
            created_at=time.time(),
        )
        with self._lock:
            self._tasks.append(task)
            self._save_tasks()
        logger.info(f"Scheduled task added: {name} ({cron_expression})")
        return task

    def remove_task(self, task_id: str) -> bool:
        """Remove a task by ID. Returns True if found and removed."""
        with self._lock:
            for i, t in enumerate(self._tasks):
                if t.id == task_id:
                    self._tasks.pop(i)
                    self._save_tasks()
                    logger.info(f"Scheduled task removed: {t.name} ({task_id})")
                    return True
        return False

    def list_tasks(self) -> list[ScheduledTask]:
        """Return a snapshot of all tasks."""
        with self._lock:
            return list(self._tasks)

    # ------------------------------------------------------------------
    # Background thread lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background scheduler thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("CronScheduler started (background thread)")

    def stop(self):
        """Stop the background scheduler thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("CronScheduler stopped")

    def _run_loop(self):
        """Main loop: tick every 60 seconds."""
        while self._running:
            try:
                self.tick()
            except Exception as e:
                logger.error(f"CronScheduler tick error: {e}", exc_info=True)
            # Sleep in small increments so stop() is responsive
            for _ in range(self.TICK_INTERVAL):
                if not self._running:
                    break
                time.sleep(1)

    # ------------------------------------------------------------------
    # Tick — check and execute due tasks
    # ------------------------------------------------------------------

    def tick(self):
        """Check if any task is due and execute it.

        Called automatically by the background thread every 60 seconds.
        """
        now = datetime.now()
        with self._lock:
            due_tasks = [t for t in self._tasks if t.enabled and _cron_match(t.cron_expression, now)]

        if not due_tasks:
            return

        logger.info(f"CronScheduler: {len(due_tasks)} task(s) due at {now.isoformat()}")

        for task in due_tasks:
            self._execute_task(task)

    def _execute_task(self, task: ScheduledTask):
        """Execute a single scheduled task via the agent's LLM."""
        logger.info(f"Executing scheduled task: {task.name} ({task.id})")

        from .llm.interface import Message

        system_prompt = (
            "You are OpenWill, an autonomous agent with free will. "
            "A scheduled task is now due for execution. "
            "You have full access to your tools, memory, and capabilities. "
            "Carry out the task described below as best you can. "
            "After completing the task, briefly summarize what you did."
        )

        try:
            response = self.agent.llm.chat(
                messages=[Message(role="user", content=task.description)],
                system_prompt=system_prompt,
                temperature=0.7,
            )
            result_text = response.content
        except Exception as e:
            logger.error(f"Scheduled task execution failed ({task.name}): {e}")
            result_text = f"ERROR: {e}"

        with self._lock:
            # Re-find the task in case the list changed
            for t in self._tasks:
                if t.id == task.id:
                    t.last_run = time.time()
                    t.last_result = result_text[:2000] if result_text else None
                    break
            self._save_tasks()

        logger.info(f"Scheduled task completed: {task.name}")

    # ------------------------------------------------------------------
    # API handlers (called by chat_server)
    # ------------------------------------------------------------------

    def handle_api_list(self) -> list[dict]:
        """Return all tasks as a list of dicts."""
        with self._lock:
            return [asdict(t) for t in self._tasks]

    def handle_api_add(self, data: dict) -> dict:
        """Add a task from an API request dict.

        Expected keys: name, description, cron_expression.
        """
        name = data.get("name", "Untitled task")
        description = data.get("description", "")
        cron_expression = data.get("cron_expression", "* * * * *")
        task = self.add_task(name, description, cron_expression)
        return asdict(task)

    def handle_api_remove(self, task_id: str) -> dict:
        """Remove a task by ID."""
        removed = self.remove_task(task_id)
        if removed:
            return {"status": "ok", "removed": task_id}
        return {"status": "not_found", "task_id": task_id}

    def handle_api_toggle(self, task_id: str) -> dict:
        """Toggle the enabled/disabled state of a task."""
        with self._lock:
            for t in self._tasks:
                if t.id == task_id:
                    t.enabled = not t.enabled
                    self._save_tasks()
                    return {"status": "ok", "task_id": task_id, "enabled": t.enabled}
        return {"status": "not_found", "task_id": task_id}
