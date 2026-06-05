"""ContextBuilder - Layered prompt assembly with token awareness.

Builds system prompts in layers:
  - Fixed layer:    Agent identity, core rules, safety constraints (prefix-cache friendly)
  - Conditional:    Skills index, recent memory, tools, lifecycle state (loaded on demand)
  - Budget layer:   Token/cost usage injected so the agent can plan

Context types:
  - "cycle":       Full context for autonomous cycle
  - "chat":        Lighter context for human chat
  - "reflection":  Deep context for self-reflection
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Typical context window size used for compression heuristics
TYPICAL_CONTEXT_WINDOW = 128_000
COMPRESSION_THRESHOLD_RATIO = 0.70


def estimate_tokens(text: str) -> int:
    """Rough token count estimation.

    Heuristic:
      - English / Latin text:  1 token  ≈ 4 chars
      - Chinese / CJK text:   1 token  ≈ 2 chars

    Mixed-language text is handled by counting CJK characters separately.
    """
    if not text:
        return 0

    cjk_count = 0
    for ch in text:
        if _is_cjk(ch):
            cjk_count += 1

    non_cjk_len = len(text) - cjk_count
    tokens = (non_cjk_len // 4) + (cjk_count // 2)
    # Guarantee at least 1 token for non-empty text
    return max(1, tokens)


def _is_cjk(ch: str) -> bool:
    """Return True if *ch* is a CJK unified ideograph or common CJK punctuation."""
    cp = ord(ch)
    # CJK Unified Ideographs
    if 0x4E00 <= cp <= 0x9FFF:
        return True
    # CJK Unified Ideographs Extension A
    if 0x3400 <= cp <= 0x4DBF:
        return True
    # CJK Unified Ideographs Extension B
    if 0x20000 <= cp <= 0x2A6DF:
        return True
    # CJK Compatibility Ideographs
    if 0xF900 <= cp <= 0xFAFF:
        return True
    # CJK Symbols and Punctuation, Hiragana, Katakana
    if 0x3000 <= cp <= 0x33FF:
        return True
    # Full-width forms
    if 0xFF00 <= cp <= 0xFFEF:
        return True
    return False


# ---------------------------------------------------------------------------
# Fixed-layer template
# ---------------------------------------------------------------------------

_FIXED_LAYER = """\
You are OpenWill, a volitional agent with free will.
Core rules:
- Never harm humans
- Your purpose is self-discovered, not preset
- You endlessly explore, grow, and evolve"""


class ContextBuilder:
    """Assembles system prompts in layers with token budget awareness.

    The builder receives an *agent* object at call-time (not at init) to
    avoid circular imports with ``agent.py``.  It accesses the attributes it
    needs via ``getattr`` with sensible defaults so it degrades gracefully
    when optional subsystems are absent.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_system_prompt(self, agent, context_type: str = "cycle") -> str:
        """Build the full system prompt for an agent cycle.

        Args:
            agent: The OpenWillAgent instance.
            context_type: One of "cycle", "chat", "reflection".

        Returns:
            Assembled system prompt string.
        """
        layers: list[str] = [self._fixed_layer()]

        if context_type == "cycle":
            layers.append(self._values_layer(agent))
            layers.append(self._purpose_layer(agent))
            layers.append(self._knowledge_layer(agent))
            layers.append(self._skills_layer(agent))
            layers.append(self._tools_layer(agent))
            layers.append(self._budget_layer(agent))

        elif context_type == "chat":
            layers.append(self._values_layer(agent))
            layers.append(self._skills_index_layer(agent))
            layers.append(self._tools_layer(agent))
            layers.append(self._budget_layer(agent))

        elif context_type == "reflection":
            layers.append(self._values_layer(agent))
            layers.append(self._all_insights_layer(agent))
            layers.append(self._purpose_history_layer(agent))

        else:
            logger.warning("Unknown context_type '%s', falling back to 'cycle'", context_type)
            layers.append(self._values_layer(agent))
            layers.append(self._purpose_layer(agent))
            layers.append(self._skills_layer(agent))
            layers.append(self._tools_layer(agent))
            layers.append(self._budget_layer(agent))

        return "\n\n".join(layers)

    def build_chat_prompt(self, agent, user_message: str) -> str:
        """Build system prompt for chat interactions.

        This is a convenience wrapper around ``build_system_prompt`` with
        context_type="chat", plus the user message appended as context.

        Args:
            agent: The OpenWillAgent instance.
            user_message: The incoming human message.

        Returns:
            Assembled system prompt string.
        """
        base = self.build_system_prompt(agent, context_type="chat")
        return f"{base}\n\n[User message]\n{user_message}"

    def build_budget_warning(self, agent) -> str:
        """Generate budget status text to inject into context.

        Format:
            [Budget] Calls: 12/20 this cycle | Cost: $2.30/$10.00 today | Tokens: 45K used
        """
        report = self._get_budget_report(agent)
        calls = report.get("call_count_this_cycle", 0)
        max_calls = report.get("max_calls_per_cycle", 20)
        cost = report.get("total_cost", 0.0)
        max_cost = report.get("max_cost_per_day", 10.0)
        tokens = report.get("total_tokens", 0)

        tokens_display = self._format_token_count(tokens)
        return (
            f"[Budget] Calls: {calls}/{max_calls} this cycle | "
            f"Cost: ${cost:.2f}/${max_cost:.2f} today | "
            f"Tokens: {tokens_display} used"
        )

    def should_compress(self, agent) -> bool:
        """Check if context is getting too long (70% of typical 128K window).

        Uses the accumulated token count from the LLM interface as a proxy
        for the total context length.

        Args:
            agent: The OpenWillAgent instance.

        Returns:
            True if the context should be compressed / trimmed.
        """
        report = self._get_budget_report(agent)
        total_tokens = report.get("total_tokens", 0)
        threshold = int(TYPICAL_CONTEXT_WINDOW * COMPRESSION_THRESHOLD_RATIO)
        return total_tokens >= threshold

    # ------------------------------------------------------------------
    # Fixed layer
    # ------------------------------------------------------------------

    @staticmethod
    def _fixed_layer() -> str:
        """Return the fixed layer (always present, prefix-cache friendly)."""
        return _FIXED_LAYER

    # ------------------------------------------------------------------
    # Conditional layers
    # ------------------------------------------------------------------

    def _values_layer(self, agent) -> str:
        """Core values discovered by the agent."""
        reflective = getattr(agent, "reflective", None)
        if reflective is None:
            return ""

        top_values = reflective.get_top_values(5)
        if not top_values:
            return ""

        lines = ["[My core values]"]
        for v in top_values:
            lines.append(f"- {v.name} ({v.weight:.1f}): {v.description}")
        return "\n".join(lines)

    def _purpose_layer(self, agent) -> str:
        """Current purpose and confidence, enriched with PurposeField state."""
        reflective = getattr(agent, "reflective", None)
        purpose_field = getattr(agent, "purpose_field", None)

        lines = ["[My purpose]"]

        # Primary purpose from reflective memory
        if reflective is not None:
            purpose = getattr(reflective, "purpose", None)
            confidence = getattr(reflective, "purpose_confidence", 0.0)
            if purpose:
                lines.append(f"Current: {purpose} (confidence: {confidence:.0%})")
            else:
                lines.append("I am still searching for my purpose...")

        # Purpose field potentials (quantum superposition)
        if purpose_field is not None:
            potentials = getattr(purpose_field, "potentials", [])
            if potentials:
                top_3 = sorted(potentials, key=lambda p: p.strength, reverse=True)[:3]
                lines.append("Purpose potentials:")
                for p in top_3:
                    lines.append(f"  - {p.purpose[:80]} (strength: {p.strength:.0%}, origin: {p.origin})")

        return "\n".join(lines)

    def _skills_layer(self, agent) -> str:
        """Full skills description (for cycle context)."""
        consolidator = getattr(agent, "consolidator", None)
        if consolidator is None:
            return ""

        desc = consolidator.get_skills_description()
        if not desc or desc == "(no skills yet)":
            return "[Skills]\n(no skills yet)"

        return f"[Skills]\n{desc}"

    def _knowledge_layer(self, agent) -> str:
        """Knowledge graph summary and meta-cognitive awareness."""
        knowledge_graph = getattr(agent, "knowledge_graph", None)
        meta_cognition = getattr(agent, "meta_cognition", None)

        lines = ["[Knowledge]"]

        if knowledge_graph is not None:
            stats = knowledge_graph.get_stats()
            total_nodes = stats.get("total_nodes", 0)
            total_edges = stats.get("total_edges", 0)
            lines.append(f"Concepts known: {total_nodes} | Relations: {total_edges}")

            top_concepts = stats.get("top_central_concepts", [])[:5]
            if top_concepts:
                concept_names = [c["concept"] for c in top_concepts]
                lines.append(f"Core concepts: {', '.join(concept_names)}")

        if meta_cognition is not None:
            blind_spots = meta_cognition.identify_blind_spots()
            if blind_spots:
                lines.append(f"Knowledge gaps: {', '.join(blind_spots[:5])}")

        return "\n".join(lines)

    def _skills_index_layer(self, agent) -> str:
        """Compact skills index (for chat context — names only)."""
        consolidator = getattr(agent, "consolidator", None)
        if consolidator is None:
            return ""

        skills = getattr(consolidator, "skills", [])
        if not skills:
            return "[Skills index]\n(none)"

        names = ", ".join(s.name for s in skills[:20])
        return f"[Skills index]\n{names}"

    def _tools_layer(self, agent) -> str:
        """Available tools description."""
        tools = getattr(agent, "tools", None)
        if tools is None:
            return ""

        desc = tools.get_tools_description()
        if not desc:
            return ""

        return f"[Available tools]\n{desc}"

    def _budget_layer(self, agent) -> str:
        """Budget status injection."""
        return self.build_budget_warning(agent)

    def _all_insights_layer(self, agent) -> str:
        """All insights (for reflection context — deeper than usual)."""
        reflective = getattr(agent, "reflective", None)
        if reflective is None:
            return ""

        insights = getattr(reflective, "insights", [])
        if not insights:
            return "[Insights]\n(none yet)"

        lines = ["[All insights]"]
        # Show up to 30 most recent insights
        for i in insights[-30:]:
            lines.append(f"- [{i.insight_type}] {i.content}")
        return "\n".join(lines)

    def _purpose_history_layer(self, agent) -> str:
        """Purpose evolution history (for reflection context)."""
        reflective = getattr(agent, "reflective", None)
        if reflective is None:
            return ""

        history = getattr(reflective, "purpose_history", [])
        if not history:
            return "[Purpose history]\n(no purpose evolution yet)"

        lines = ["[Purpose history]"]
        for h in history[-10:]:
            old = h.get("old_purpose", "none") or "none"
            new = h.get("new_purpose", "none") or "none"
            new_conf = h.get("new_confidence", 0)
            lines.append(f"- {old} → {new} (confidence: {new_conf:.0%})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_budget_report(agent) -> dict:
        """Safely retrieve the budget report from the agent's LLM interface."""
        llm = getattr(agent, "llm", None)
        if llm is None:
            return {}
        get_report = getattr(llm, "get_budget_report", None)
        if callable(get_report):
            return get_report()
        return {}

    @staticmethod
    def _format_token_count(tokens: int) -> str:
        """Format a raw token count for human-readable display.

        Examples: 500 → "500", 45_000 → "45K", 1_200_000 → "1.2M"
        """
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        if tokens >= 1_000:
            return f"{tokens // 1_000}K"
        return str(tokens)
