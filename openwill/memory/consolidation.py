"""Memory consolidation - summarization, skill extraction, and forgetting curve.

Inspired by:
  - OpenClaw: Memory Flush + Dreaming (periodic log → essence promotion)
  - Hermes:   Skills engine (auto-extract reusable execution paths)
  - Academic: Ebbinghaus forgetting curve + Knowledge-Memory-Wisdom layers
"""

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forgetting curve helpers
# ---------------------------------------------------------------------------

def forgetting_weight(
    access_count: int,
    last_accessed: float,
    importance: float,
    now: Optional[float] = None,
    half_life_days: float = 30.0,
) -> float:
    """Compute a composite score combining importance, recency, and access.

    Uses an exponential decay (Ebbinghaus-inspired) for recency, boosted by
    access frequency and base importance.

    Formula:
        recency = exp(-lambda * days_since_last_access)
        score   = importance * recency * (1 + log(1 + access_count))

    A higher score means the entry is more worth keeping.
    """
    now = now or time.time()
    days_since = max(0.0, (now - last_accessed) / 86400.0)
    lam = 0.693 / half_life_days  # ln(2) / half_life
    recency = pow(2.71828, -lam * days_since)
    freq_boost = 1.0 + _log1p(access_count)
    return importance * recency * freq_boost


def _log1p(n: int) -> float:
    """Natural log of (1 + n), safe for n=0."""
    import math
    return math.log(1 + n)


# ---------------------------------------------------------------------------
# Skill (episodic memory)
# ---------------------------------------------------------------------------

class Skill:
    """A reusable execution template extracted from experience."""

    def __init__(
        self,
        name: str,
        description: str,
        steps: list[str],
        tools_used: list[str],
        source_context: str = "",
        success_count: int = 0,
        fail_count: int = 0,
    ):
        self.name = name
        self.description = description
        self.steps = steps
        self.tools_used = tools_used
        self.source_context = source_context
        self.success_count = success_count
        self.fail_count = fail_count
        self.created_at = time.time()
        self.updated_at = time.time()

    @property
    def reliability(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5
        return self.success_count / total

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "tools_used": self.tools_used,
            "source_context": self.source_context,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        skill = cls(
            name=data["name"],
            description=data["description"],
            steps=data.get("steps", []),
            tools_used=data.get("tools_used", []),
            source_context=data.get("source_context", ""),
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
        )
        skill.created_at = data.get("created_at", time.time())
        skill.updated_at = data.get("updated_at", time.time())
        return skill


# ---------------------------------------------------------------------------
# MemoryConsolidator
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """Periodic memory consolidation engine.

    Three responsibilities:
      1. Summarize short-term memory → store key points in long-term memory
      2. Extract reusable Skills from multi-step action sequences
      3. Apply forgetting curve to long-term memory (decay + prune)
    """

    MAX_SKILLS = 200

    def __init__(self, llm, short_term, long_term, reflective, data_dir: str = "data"):
        self.llm = llm
        self.short_term = short_term
        self.long_term = long_term
        self.reflective = reflective
        self.data_dir = data_dir
        self.skills: list[Skill] = []
        self._last_consolidation = 0.0
        self._load()

    # -- persistence --

    def _get_filepath(self) -> str:
        return os.path.join(self.data_dir, "skills.json")

    def _load(self):
        filepath = self._get_filepath()
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.skills = [Skill.from_dict(d) for d in data.get("skills", [])]
                self._last_consolidation = data.get("last_consolidation", 0.0)
                logger.info(f"Loaded {len(self.skills)} skills")
            except Exception as e:
                logger.error(f"Failed to load skills: {e}")
                self.skills = []

    def save(self):
        filepath = self._get_filepath()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            data = {
                "skills": [s.to_dict() for s in self.skills],
                "last_consolidation": self._last_consolidation,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save skills: {e}")

    # -- public API --

    def consolidate(self, cycle_count: int, interval: int = 5):
        """Run full consolidation if enough cycles have passed.

        Args:
            cycle_count: Current cycle number.
            interval: Minimum cycles between consolidations.
        """
        if cycle_count - self._last_consolidation < interval:
            return

        logger.info("🧠 Starting memory consolidation...")
        self._summarize_short_term()
        self._extract_skills()
        self._apply_forgetting_curve()
        self._last_consolidation = cycle_count
        self.save()
        logger.info("🧠 Memory consolidation complete")

    def get_skills_description(self) -> str:
        """Get a compact description of all skills for LLM context (progressive disclosure level-0)."""
        if not self.skills:
            return "(no skills yet)"
        lines = []
        for s in sorted(self.skills, key=lambda x: x.reliability, reverse=True)[:20]:
            lines.append(f"- {s.name}: {s.description} (reliability: {s.reliability:.0%})")
        return "\n".join(lines)

    def get_skill_detail(self, name: str) -> Optional[Skill]:
        """Get full skill detail by name (progressive disclosure level-1)."""
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def record_skill_outcome(self, name: str, success: bool):
        """Record whether a skill execution succeeded or failed."""
        for s in self.skills:
            if s.name == name:
                if success:
                    s.success_count += 1
                else:
                    s.fail_count += 1
                s.updated_at = time.time()
                self.save()
                return

    def match_skill(self, task_description: str) -> Optional[Skill]:
        """Use LLM to determine if a task matches any existing skill.

        Returns the best-matching Skill or None.
        """
        if not self.skills:
            return None

        # Quick keyword pre-filter to avoid LLM call when no skill is remotely relevant
        task_words = set(task_description.lower().split())
        candidates = []
        for s in self.skills:
            skill_words = set((s.name + " " + s.description).lower().split())
            if task_words & skill_words:
                candidates.append(s)
        if not candidates:
            # Also try all skills if none matched by keyword (LLM may see semantic similarity)
            candidates = self.skills[:10]

        skill_list = "\n".join(
            f"- {s.name}: {s.description}" for s in candidates
        )

        prompt = f"""Given this task: "{task_description}"

Which of these skills (if any) is the best match?

{skill_list}

Respond with ONLY the skill name, or "none" if no skill matches."""

        try:
            response = self.llm.chat(
                messages=[],
                system_prompt=prompt,
                temperature=0.1,
            )
            name = response.content.strip().lower()
            if name == "none" or not name:
                return None
            for s in candidates:
                if s.name.lower() == name:
                    return s
            return None
        except Exception as e:
            logger.error(f"Skill matching failed: {e}")
            return None

    def execute_skill(self, skill: Skill, task_description: str, agent) -> dict:
        """Execute a skill's steps using the agent's tools.

        Returns a dict with:
          - success: bool
          - steps_completed: int
          - results: list of step results
          - error: str (if any)
        """
        results = []
        success = True
        error = ""

        for i, step in enumerate(skill.steps):
            logger.info(f"  Skill '{skill.name}' step {i+1}/{len(skill.steps)}: {step[:80]}")

            # Build a prompt for this step
            step_prompt = f"""You are executing step {i+1}/{len(skill.steps)} of the skill "{skill.name}".

Skill description: {skill.description}
Current task: {task_description}

Step instruction: {step}

Previous results:
{chr(10).join(f'- Step {j+1}: {r[:200]}' for j, r in enumerate(results)) if results else '(first step)'}

Execute this step. If you need to use a tool, include a tool call in the standard format:
{{"tool": "tool_name", "args": {{"param": "value"}}}}

If the step is complete, respond with the result as plain text."""

            try:
                from ..llm.interface import Message
                response = agent.llm.chat(
                    messages=[Message(role="user", content=step_prompt)],
                    system_prompt="You are OpenWill executing a learned skill step. Follow the instruction precisely.",
                    temperature=0.3,
                )

                result_text = response.content.strip()
                results.append(result_text)

                # Check if the step contains a tool call
                from ..chat_server import _TOOL_CALL_STANDARD, _TOOL_CALL_SHORTHAND, _TOOL_CALL_PARENS
                import re
                tool_match = _TOOL_CALL_STANDARD.search(result_text)
                if tool_match:
                    tool_name = tool_match.group(1)
                    try:
                        import json as _json
                        tool_args = _json.loads(tool_match.group(2))
                    except Exception:
                        tool_args = {}
                    tool_result = agent.use_tool(tool_name, **tool_args)
                    results.append(f"[Tool {tool_name}: {'OK' if tool_result.success else 'FAIL'}] {tool_result.output[:200]}")

                    if not tool_result.success:
                        success = False
                        error = f"Tool {tool_name} failed: {tool_result.error}"
                        break

            except Exception as e:
                success = False
                error = str(e)
                results.append(f"[Error] {e}")
                break

        # Record outcome
        self.record_skill_outcome(skill.name, success)

        return {
            "success": success,
            "steps_completed": len([r for r in results if not r.startswith("[Error]")]),
            "results": results,
            "error": error,
        }

    # -- internal steps --

    def _summarize_short_term(self):
        """Use LLM to summarize short-term memory and store key points in long-term."""
        messages = self.short_term.get_messages()
        if len(messages) < 5:
            return  # Not enough to summarize

        # Build a compact representation of recent messages
        lines = []
        for msg in messages:
            prefix = {"system": "System", "user": "Obs", "assistant": "Thought"}.get(msg.role, msg.role)
            lines.append(f"[{prefix}] {msg.content[:200]}")
        context_text = "\n".join(lines[-30:])  # Last 30 messages max

        prompt = f"""Analyze the following recent activity log and extract key knowledge worth remembering long-term.

For each piece of knowledge, provide:
- topic: A short topic name
- content: The key fact or insight (concise)
- importance: 0.0-1.0 how important this is to remember

Activity log:
{context_text}

Respond in JSON format:
{{"knowledge": [{{"topic": "...", "content": "...", "importance": 0.5}}]}}

Only include genuinely new and useful information. Skip trivial observations."""

        try:
            response = self.llm.chat(
                messages=[],
                system_prompt=prompt,
                temperature=0.3,
            )
            raw = response.content.strip()

            # Extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                for item in data.get("knowledge", []):
                    self.long_term.store(
                        topic=item.get("topic", "unknown"),
                        content=item.get("content", ""),
                        source="consolidation",
                        importance=item.get("importance", 0.5),
                    )
                logger.info(f"  Consolidated {len(data.get('knowledge', []))} knowledge entries")

        except Exception as e:
            logger.error(f"Short-term summarization failed: {e}")

        # Clear old messages, keep only the most recent few
        while len(self.short_term.messages) > 10:
            self.short_term.messages.pop(0)

    def _extract_skills(self):
        """Use LLM to extract reusable skills from recent action sequences."""
        messages = self.short_term.get_messages()
        # Look for action sequences (3+ actions)
        action_msgs = [
            m for m in messages
            if m.metadata.get("type") in ("action", "action_result")
        ]
        if len(action_msgs) < 6:  # Need at least 3 action+result pairs
            return

        # Build action sequence text
        seq_lines = []
        for m in action_msgs[-20:]:  # Last 20 action messages
            seq_lines.append(f"  {m.content[:200]}")
        seq_text = "\n".join(seq_lines)

        # Existing skill names to avoid duplicates
        existing_names = [s.name for s in self.skills]

        prompt = f"""Analyze the following action sequence and determine if it contains a reusable workflow pattern.

Action sequence:
{seq_text}

Existing skills: {', '.join(existing_names) if existing_names else 'none'}

If this sequence demonstrates a clear, repeatable workflow (not just random actions), extract it as a skill.
If no clear pattern exists, respond with: {{"skill": null}}

If a pattern exists, respond in JSON:
{{"skill": {{"name": "short-kebab-name", "description": "one-line description", "steps": ["step 1", "step 2", ...], "tools_used": ["tool1", "tool2"]}}}}

Be selective - only extract genuinely reusable patterns."""

        try:
            response = self.llm.chat(
                messages=[],
                system_prompt=prompt,
                temperature=0.3,
            )
            raw = response.content.strip()

            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start < 0 or end <= start:
                return

            data = json.loads(raw[start:end])
            skill_data = data.get("skill")
            if not skill_data:
                return

            name = skill_data.get("name", "")
            if not name:
                return

            # Check for duplicate or similar skill
            for existing in self.skills:
                if existing.name == name:
                    # Update existing skill steps if improved
                    existing.steps = skill_data.get("steps", existing.steps)
                    existing.tools_used = skill_data.get("tools_used", existing.tools_used)
                    existing.updated_at = time.time()
                    logger.info(f"  Updated skill: {name}")
                    return

            skill = Skill(
                name=name,
                description=skill_data.get("description", ""),
                steps=skill_data.get("steps", []),
                tools_used=skill_data.get("tools_used", []),
                source_context=seq_text[:500],
            )
            self.skills.append(skill)

            # Cap skills
            if len(self.skills) > self.MAX_SKILLS:
                self.skills.sort(key=lambda s: s.reliability)
                self.skills = self.skills[-self.MAX_SKILLS:]

            logger.info(f"  Extracted new skill: {name}")

        except Exception as e:
            logger.error(f"Skill extraction failed: {e}")

    def _apply_forgetting_curve(self):
        """Apply Ebbinghaus-inspired forgetting curve to long-term memory.

        Entries with low composite scores (importance * recency * frequency)
        are decayed in importance. Entries below a minimum threshold are pruned.
        """
        entries = self.long_term.entries
        if len(entries) < 100:
            return  # Don't bother with small memory

        now = time.time()
        min_score = 0.01  # Below this, entry is effectively forgotten
        pruned = 0
        decayed = 0

        to_keep = []
        for entry in entries:
            # Update last_accessed time based on access_count and created_at
            # If never accessed after creation, use created_at as last_accessed
            last_accessed = entry.updated_at or entry.created_at

            score = forgetting_weight(
                access_count=entry.access_count,
                last_accessed=last_accessed,
                importance=entry.importance,
                now=now,
            )

            if score < min_score:
                pruned += 1
                continue

            # Decay importance slightly based on score
            # This makes importance converge toward the composite score over time
            if entry.importance > 0:
                decay_factor = min(1.0, score / entry.importance) if entry.importance > 0 else 1.0
                if decay_factor < 0.95:
                    entry.importance *= (0.95 + 0.05 * decay_factor)  # Gentle decay
                    decayed += 1

            to_keep.append(entry)

        if pruned > 0 or decayed > 0:
            self.long_term.entries = to_keep
            self.long_term.save()
            logger.info(f"  Forgetting curve: pruned {pruned}, decayed {decayed}, kept {len(to_keep)}")
