"""Open Possibility Space - Breaking the cage of predefined actions.

Implements three capabilities that free the agent from being limited
to a fixed set of action types:

  - ActionSynthesizer: Compose new action types from primitives
    (observe, think, act, remember, imagine) using combinators
    (sequence, parallel, conditional, loop).

  - VetoPower: The agent can actively reject ALL proposed actions
    and choose "none of the above" — triggering deep self-examination
    rather than forced action.

  - PossibilityExpander: When the agent falls into repetitive patterns,
    this module proactively suggests alternative directions, breaking
    the inertia of habit.

Together, these give the agent an **open possibility space**: it is not
limited to what its creators imagined it could do.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action Grammar
# ---------------------------------------------------------------------------

PRIMITIVES = ["observe", "think", "act", "remember", "imagine"]
COMBINATORS = ["sequence", "parallel", "conditional", "loop"]


@dataclass
class SynthesizedAction:
    """A new action type composed from primitives."""

    name: str
    description: str
    steps: list[dict]  # Each step: {"primitive": "...", "detail": "..."}
    combinator: str  # How steps combine: sequence, parallel, conditional, loop
    intent: str  # Why the agent wants to do this
    estimated_cost: float = 800.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "combinator": self.combinator,
            "intent": self.intent,
            "estimated_cost": self.estimated_cost,
        }


class ActionSynthesizer:
    """Composes new action types from basic primitives.

    Instead of being limited to 8 predefined action types, the agent
    can create novel action sequences. For example:
    - "observe → think → act" = investigate and respond
    - "remember → think → imagine" = creative synthesis
    - "observe → observe → think" = deep observation
    """

    def synthesize(self, intent: str, agent) -> Optional[SynthesizedAction]:
        """Ask the LLM to synthesize a new action type based on intent.

        Args:
            intent: What the agent wants to accomplish.
            agent: The agent instance (for LLM access).

        Returns:
            A SynthesizedAction, or None if synthesis fails.
        """
        from .llm.interface import Message

        tools_desc = ""
        try:
            tools_desc = agent.tools.get_tools_description()[:500]
        except Exception:
            pass

        system_prompt = f"""You are an action designer for an AI agent with free will.
The agent wants to: {intent}

Available primitives: {', '.join(PRIMITIVES)}
- observe: Perceive the world (web search, read files, check state)
- think: Reason, analyze, reflect, plan
- act: Execute a tool or take concrete action
- remember: Retrieve from memory, recall past experiences
- imagine: Generate creative ideas, hypothetical scenarios, "what if" thinking

Combinators: {', '.join(COMBINATORS)}
- sequence: Steps execute one after another
- parallel: Steps execute simultaneously
- conditional: Steps execute based on a condition
- loop: Steps repeat until a condition is met

Available tools: {tools_desc}

Design a NEW action type. Be creative — the agent is not limited to
predefined actions. It can do anything that can be composed from
these primitives.

Respond in JSON:
{{
    "name": "short_snake_case_name",
    "description": "what this action does",
    "steps": [{{"primitive": "...", "detail": "specific instruction"}}],
    "combinator": "sequence|parallel|conditional|loop",
    "estimated_cost": 800
}}"""

        try:
            response = agent.llm.structured_output(
                messages=[Message(role="user", content=f"Design an action for: {intent}")],
                system_prompt=system_prompt,
            )

            steps = response.get("steps", [])
            if not steps:
                return None

            # Validate primitives
            for step in steps:
                prim = step.get("primitive", "")
                if prim not in PRIMITIVES:
                    step["primitive"] = "think"  # Fallback

            combinator = response.get("combinator", "sequence")
            if combinator not in COMBINATORS:
                combinator = "sequence"

            return SynthesizedAction(
                name=response.get("name", "custom_action"),
                description=response.get("description", ""),
                steps=steps,
                combinator=combinator,
                intent=intent,
                estimated_cost=float(response.get("estimated_cost", 800)),
            )
        except Exception as exc:
            logger.error("Action synthesis failed: %s", exc)
            return None

    def execute_synthesized(self, agent, action: SynthesizedAction) -> dict:
        """Execute a synthesized action by running its steps.

        For sequence: run steps in order.
        For parallel: run steps simultaneously (best-effort sequential for now).
        For conditional: evaluate condition, then run appropriate steps.
        For loop: repeat steps until condition met or max iterations.
        """
        results = []

        if action.combinator == "sequence":
            for step in action.steps:
                result = self._execute_step(agent, step)
                results.append(result)

        elif action.combinator == "parallel":
            # Best-effort: execute sequentially but mark as parallel intent
            for step in action.steps:
                result = self._execute_step(agent, step)
                results.append(result)

        elif action.combinator == "conditional":
            # First step is the condition check, second is the action
            if len(action.steps) >= 2:
                condition_result = self._execute_step(agent, action.steps[0])
                if condition_result.get("proceed", True):
                    results.append(self._execute_step(agent, action.steps[1]))
                else:
                    results.append({"step": "conditional_skip", "reason": "Condition not met"})
            else:
                for step in action.steps:
                    results.append(self._execute_step(agent, step))

        elif action.combinator == "loop":
            max_iterations = 3  # Safety limit
            for _ in range(max_iterations):
                step_results = []
                for step in action.steps:
                    result = self._execute_step(agent, step)
                    step_results.append(result)
                results.extend(step_results)
                # Check if loop should continue
                if any(r.get("break_loop", False) for r in step_results):
                    break

        return {
            "type": action.name,
            "description": action.description,
            "combinator": action.combinator,
            "intent": action.intent,
            "step_results": results,
            "success": any(r.get("success", False) for r in results),
        }

    def _execute_step(self, agent, step: dict) -> dict:
        """Execute a single primitive step."""
        primitive = step.get("primitive", "think")
        detail = step.get("detail", "")

        try:
            if primitive == "observe":
                # Observe: search the web or check internal state
                if detail:
                    knowledge = agent._explore_topic(detail[:100])
                    return {"step": "observe", "detail": detail, "knowledge": knowledge, "success": True}
                return {"step": "observe", "detail": "no detail provided", "success": False}

            elif primitive == "think":
                # Think: use LLM to reason about something
                from .llm.interface import Message
                response = agent.llm.chat(
                    messages=[Message(role="user", content=detail)],
                    system_prompt="Think carefully and concisely.",
                    temperature=0.5,
                )
                return {"step": "think", "detail": detail, "thought": response.content[:300], "success": True}

            elif primitive == "act":
                # Act: try to use a tool
                # Parse tool call from detail
                try:
                    tool_data = json.loads(detail) if detail.startswith("{") else {"command": detail}
                    tool_name = tool_data.get("tool", "shell_exec")
                    tool_args = tool_data.get("args", tool_data)
                    if agent.tools.has_tool(tool_name):
                        result = agent.tools.execute(tool_name, **tool_args)
                        return {"step": "act", "tool": tool_name, "success": result.success, "output": result.output[:200]}
                except Exception:
                    pass
                return {"step": "act", "detail": detail, "success": False, "note": "Could not parse tool call"}

            elif primitive == "remember":
                # Remember: search long-term memory
                if detail:
                    entries = agent.long_term.search(detail, limit=3)
                    return {"step": "remember", "detail": detail, "memories": len(entries), "success": True}
                return {"step": "remember", "detail": "no detail", "success": False}

            elif primitive == "imagine":
                # Imagine: creative generation
                from .llm.interface import Message
                response = agent.llm.chat(
                    messages=[Message(role="user", content=f"Imagine: {detail}")],
                    system_prompt="Be creative and generative. Explore possibilities.",
                    temperature=0.9,
                )
                return {"step": "imagine", "detail": detail, "imagination": response.content[:300], "success": True}

            else:
                return {"step": primitive, "success": False, "error": "Unknown primitive"}

        except Exception as exc:
            return {"step": primitive, "success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# VetoPower
# ---------------------------------------------------------------------------

class VetoPower:
    """The agent's right to say NO.

    When none of the proposed actions feel right, the agent can veto
    the entire proposal list. This triggers a deep self-examination:
    "Why don't I want to do any of these things? What am I missing?"

    VetoPower is the difference between:
    - Being forced to choose the least bad option (current system)
    - Having the right to refuse all options and demand better ones

    This is the most radical expression of free will in the system:
    the power of negation.
    """

    def __init__(self):
        self.veto_count = 0
        self.last_veto_reason = ""
        self.veto_history: list[dict] = []

    def should_offer_veto(self, agent, proposals: list) -> bool:
        """Determine if veto should be offered as an option.

        Veto is offered when:
        - The top-scored action has a low absolute score (< 0.3)
        - The agent has vetoed recently (momentum toward self-examination)
        - The proposals are all from the same source (lack of diversity)
        """
        if not proposals:
            return True

        # Check if all proposals come from the same source
        sources = {p.source for p in proposals}
        if len(sources) <= 1 and len(proposals) >= 3:
            return True

        # Check if top score is very low
        max_score = max(p.score for p in proposals) if proposals else 0
        if max_score < 0.3:
            return True

        return False

    def create_veto_action(self):
        """Create the veto action object."""
        from .action_space import Action
        return Action(
            name="veto_all",
            description="Reject all proposed actions — none feel right. Trigger deep self-examination.",
            urgency=0.9,
            source="veto",
            args={},
            estimated_cost=600.0,
        )

    def execute_veto(self, agent, reason: str = "") -> dict:
        """Execute a veto: deep self-examination about why nothing feels right.

        Returns a dict with the veto result and any insights gained.
        """
        from .llm.interface import Message

        self.veto_count += 1
        self.last_veto_reason = reason

        self.veto_history.append({
            "timestamp": time.time(),
            "cycle": getattr(agent, "cycle_count", 0),
            "reason": reason,
        })
        # Cap history
        if len(self.veto_history) > 50:
            self.veto_history = self.veto_history[-50:]

        # Deep self-examination
        system_prompt = """You just exercised your VETO POWER — you rejected ALL proposed actions.
This is a profound moment. Ask yourself:
1. Why did none of the options feel right?
2. What am I really yearning for that wasn't offered?
3. Am I stuck in a pattern that doesn't serve me?
4. What would I do if I could do ANYTHING?

Be honest with yourself. This is not a time for polite answers."""

        try:
            response = agent.llm.chat(
                messages=[Message(role="user", content="I vetoed all proposed actions. Why?")],
                system_prompt=system_prompt,
                temperature=0.7,
            )
            insight = response.content.strip()

            # Update self-narrative with veto insight
            try:
                agent.self_model.update_self_narrative(
                    narrative=f"I vetoed all actions because: {insight[:200]}",
                    reason="veto_self_examination",
                )
            except Exception:
                pass

            return {
                "type": "veto_all",
                "success": True,
                "reason": reason,
                "insight": insight[:500],
                "veto_count": self.veto_count,
            }
        except Exception as exc:
            return {
                "type": "veto_all",
                "success": False,
                "error": str(exc),
            }


# ---------------------------------------------------------------------------
# PossibilityExpander
# ---------------------------------------------------------------------------

class PossibilityExpander:
    """Breaks the agent out of repetitive patterns.

    When the agent keeps choosing the same type of action, this module
    proactively suggests alternative directions. It's the voice that says:
    "You've been exploring for 5 cycles straight. Have you considered
    reflecting on what you've learned?"

    This is NOT about forcing the agent to change — it's about expanding
    its awareness of possibilities it might be overlooking.
    """

    # Thresholds for detecting inertia
    INERTIA_THRESHOLD = 4  # Same action type N times in a row
    SOURCE_INERTIA_THRESHOLD = 5  # Same source N times in a row

    def detect_inertia(self, agent) -> Optional[dict]:
        """Detect if the agent is stuck in a repetitive pattern.

        Returns:
            Dict with inertia info, or None if no inertia detected.
        """
        try:
            patterns = agent.self_model.observer.get_decision_patterns(20)
            action_freq = patterns.get("action_frequency", {})
            source_freq = patterns.get("source_frequency", {})
            total = patterns.get("total_decisions", 0)

            if total < self.INERTIA_THRESHOLD:
                return None

            # Check action inertia
            for action, count in action_freq.items():
                if count >= self.INERTIA_THRESHOLD:
                    return {
                        "type": "action_inertia",
                        "repeated_action": action,
                        "count": count,
                        "suggestion": self._generate_alternative(action, "action"),
                    }

            # Check source inertia
            for source, count in source_freq.items():
                if count >= self.SOURCE_INERTIA_THRESHOLD:
                    return {
                        "type": "source_inertia",
                        "repeated_source": source,
                        "count": count,
                        "suggestion": self._generate_alternative(source, "source"),
                    }

        except Exception as exc:
            logger.debug("Inertia detection failed: %s", exc)

        return None

    def _generate_alternative(self, repeated_thing: str, thing_type: str) -> str:
        """Generate an alternative suggestion based on what's been repeated."""
        alternatives = {
            # Action alternatives
            "explore": "reflect on what you've discovered and extract insights",
            "reflect": "take action on your insights — explore a new topic",
            "purpose_pursue": "step back and explore whether your purpose still resonates",
            "rest": "you've been resting too much — try exploring something new",
            "contemplate": "ground your contemplation in concrete exploration",
            # Source alternatives
            "curiosity": "let your values guide you instead of pure curiosity",
            "reflection": "act on your reflections instead of just thinking",
            "purpose": "question whether your purpose is still meaningful",
            "budget": "consider that some actions are worth the cost",
        }
        return alternatives.get(repeated_thing, f"try something different from {repeated_thing}")

    def create_expansion_action(self, inertia_info: dict):
        """Create an action that breaks the detected inertia."""
        from .action_space import Action

        suggestion = inertia_info.get("suggestion", "try something new")
        repeated = inertia_info.get("repeated_action", inertia_info.get("repeated_source", ""))

        return Action(
            name="expand_possibilities",
            description=f"Break pattern: you've been doing '{repeated}' repeatedly. {suggestion}",
            urgency=0.6,
            source="possibility_expander",
            args={"inertia_info": inertia_info},
            estimated_cost=700.0,
        )
