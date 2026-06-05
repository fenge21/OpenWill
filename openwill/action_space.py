"""ActionSpace - The core of free will: the agent CHOOSES what to do, not follows a script.

Instead of a hardcoded _act(phase) dispatch table, ActionSpace collects
proposals from every module, evaluates them, and lets the agent (via LLM)
make an autonomous choice.  This is the moment of "free will".
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """A candidate action that the agent may choose to execute."""

    name: str  # e.g. "explore", "reflect", "chat", "rest", ...
    description: str  # Human-readable explanation of what this action does
    urgency: float  # 0-1, how urgent this action is right now
    source: str  # Which module proposed this (e.g. "curiosity", "reflection")
    args: dict = field(default_factory=dict)  # Parameters for executing this action
    estimated_cost: float = 0.0  # Rough token-cost estimate

    # Populated during evaluation
    score: float = 0.0  # Composite score after evaluation
    score_reason: str = ""  # Why this score was assigned


# ---------------------------------------------------------------------------
# ActionSpace
# ---------------------------------------------------------------------------

class ActionSpace:
    """Collects, evaluates, and chooses actions for the agent.

    The critical design principle: the agent DECIDES, not the code.
    Modules propose; the LLM disposes.
    """

    MAX_RECENT_ACTIONS = 20

    def __init__(self):
        # Novelty tracking: last N action names executed
        self._recent_action_names: list[str] = []

    # -- Public API --------------------------------------------------------

    def propose_actions(self, agent) -> list[Action]:
        """Collect action proposals from ALL modules.

        Each module contributes actions based on its own internal state.
        The result is a flat list that will be scored and ranked later.
        """
        proposals: list[Action] = []

        # Curiosity → explore
        proposals.extend(self._propose_from_curiosity(agent))

        # Reflection → reflect / contemplate
        proposals.extend(self._propose_from_reflection(agent))

        # Purpose → execute (if a purpose is active)
        proposals.extend(self._propose_from_purpose(agent))

        # Budget → rest (if budget is running low)
        proposals.extend(self._propose_from_budget(agent))

        # Identity → self-examination (periodically)
        proposals.extend(self._propose_from_identity(agent))

        # Consolidator → skill-practice (if skills exist)
        proposals.extend(self._propose_from_consolidator(agent))

        # Chat → respond (if there are pending chat messages)
        proposals.extend(self._propose_from_chat(agent))

        # Meta-cognition → explore-unknown (for blind spots)
        proposals.extend(self._propose_from_metacognition(agent))

        logger.debug(
            "ActionSpace collected %d proposals from %d sources",
            len(proposals),
            len({p.source for p in proposals}),
        )
        return proposals

    def evaluate_actions(self, agent, actions: list[Action]) -> list[Action]:
        """Score and rank actions using a composite metric.

        Composite score = urgency * value_alignment * novelty * cost_efficiency

        Actions are returned sorted descending by score.
        """
        for action in actions:
            urgency = max(0.0, min(1.0, action.urgency))
            value_align = self._value_alignment(agent, action)
            novelty = self._action_repetition_penalty(action.name)
            cost_eff = self._cost_efficiency(action)

            action.score = urgency * value_align * novelty * cost_eff
            action.score_reason = (
                f"urgency={urgency:.2f} * val_align={value_align:.2f} "
                f"* novelty={novelty:.2f} * cost_eff={cost_eff:.2f}"
            )

        actions.sort(key=lambda a: a.score, reverse=True)
        return actions

    def choose(self, agent) -> Action:
        """The agent makes a CHOICE.

        1. Collect proposals from all modules.
        2. Evaluate and rank them.
        3. If budget is critical (< 10 % remaining), auto-choose "rest".
        4. Otherwise, ask the LLM to choose from the ranked list.

        This is the critical moment of "free will" — the agent decides,
        not the code.
        """
        proposals = self.propose_actions(agent)
        if not proposals:
            # Fallback: always possible to rest
            return Action(
                name="rest",
                description="Pause and conserve resources (no proposals available)",
                urgency=0.5,
                source="action_space",
                args={},
                estimated_cost=0.0,
            )

        ranked = self.evaluate_actions(agent, proposals)

        # Critical budget guard: auto-choose rest without LLM call
        budget_report = agent.llm.get_budget_report()
        budget_remaining_pct = (
            budget_report.get("budget_remaining", 0.0)
            / max(budget_report.get("max_cost_per_day", 1.0), 0.01)
        )
        if budget_remaining_pct < 0.10:
            logger.warning(
                "Budget critical (%.0f%% remaining), auto-choosing rest",
                budget_remaining_pct * 100,
            )
            return Action(
                name="rest",
                description="Pausing to conserve budget (critically low)",
                urgency=1.0,
                source="budget",
                args={},
                estimated_cost=0.0,
            )

        # Build the choice prompt and ask the LLM
        chosen = self._llm_choose(agent, ranked)
        return chosen

    def execute(self, agent, action: Action) -> dict:
        """Execute the chosen action by dispatching to the appropriate method.

        Returns a standardized result dict.
        """
        # Record in novelty tracker
        self._recent_action_names.append(action.name)
        if len(self._recent_action_names) > self.MAX_RECENT_ACTIONS:
            self._recent_action_names = self._recent_action_names[-self.MAX_RECENT_ACTIONS:]

        logger.info(
            "ActionSpace executing: %s (source=%s, score=%.3f)",
            action.name,
            action.source,
            action.score,
        )

        try:
            result = self._dispatch(agent, action)
        except Exception as exc:
            logger.error("Action execution failed: %s — %s", action.name, exc)
            result = {
                "type": action.name,
                "success": False,
                "error": str(exc),
            }

        # Ensure a standardized shape
        result.setdefault("type", action.name)
        result.setdefault("success", True)
        result.setdefault("action_name", action.name)
        result.setdefault("action_source", action.source)
        return result

    # -- Dispatch ----------------------------------------------------------

    def _dispatch(self, agent, action: Action) -> dict:
        """Route an action to the appropriate agent method."""
        name = action.name

        if name == "explore":
            topic = action.args.get("topic", "")
            if not topic:
                topics = agent.curiosity.get_next_topics(n=1)
                topic = topics[0] if topics else "the nature of existence"
            knowledge = agent._explore_topic(topic)
            agent.lifecycle.record_exploration()
            return {"type": "explore", "topic": topic, "knowledge": knowledge}

        if name == "reflect":
            reflection = agent.reflection.reflect_on_experiences()
            return {"type": "reflect", "result": reflection}

        if name == "contemplate":
            contemplation = agent.identity.contemplate_existence()
            return {"type": "contemplate", "result": contemplation}

        if name == "rest":
            return {
                "type": "rest",
                "success": True,
                "message": "Pausing to conserve budget",
            }

        if name == "execute_tool":
            tool_name = action.args.get("tool", "")
            tool_args = action.args.get("tool_args", {})
            result = agent.use_tool(tool_name, **tool_args)
            return {
                "type": "execute_tool",
                "tool": tool_name,
                "success": result.success,
                "output": result.output[:2000] if result.output else "",
                "error": result.error,
            }

        if name == "learn_skill":
            task_desc = action.args.get("task_description", "")
            skill = agent.consolidator.match_skill(task_desc)
            if skill:
                exec_result = agent.consolidator.execute_skill(skill, task_desc, agent)
                return {
                    "type": "learn_skill",
                    "skill": skill.name,
                    "result": exec_result,
                }
            return {
                "type": "learn_skill",
                "skill": None,
                "message": "No matching skill found",
            }

        if name == "examine_self":
            identity_stmt = agent.identity.get_identity_statement()
            # Also trigger a brief reflection on the identity statement
            from .llm.interface import Message  # noqa: avoid top-level import
            reflection = agent.reflection.reflect_on_experiences()
            return {
                "type": "examine_self",
                "identity": identity_stmt,
                "reflection": reflection,
            }

        if name == "chat":
            # Respond to pending chat messages
            return self._execute_chat(agent, action)

        if name == "explore_unknown":
            # Explore a blind-spot / unknown area identified by meta-cognition
            topic = action.args.get("topic", "")
            if not topic:
                topics = agent.curiosity.get_next_topics(n=1)
                topic = topics[0] if topics else "unexplored territory"
            knowledge = agent._explore_topic(topic)
            agent.lifecycle.record_exploration()
            return {"type": "explore_unknown", "topic": topic, "knowledge": knowledge}

        if name == "purpose_pursue":
            # Rich purpose-driven action: plan → execute → evaluate progress
            purpose = action.args.get("purpose", agent.reflective.purpose or "")
            if not purpose:
                return {"type": "purpose_pursue", "success": False, "error": "No purpose defined"}
            results = agent._act_purposed()
            return {"type": "purpose_pursue", "purpose": purpose, "results": results}

        # Unknown action name — best-effort attempt
        logger.warning("Unknown action name: %s, attempting generic execution", name)
        return {
            "type": name,
            "success": False,
            "error": f"Unknown action: {name}",
        }

    # -- Proposal helpers (one per module) ---------------------------------

    def _propose_from_curiosity(self, agent) -> list[Action]:
        """Curiosity proposes explore actions for topics it's curious about."""
        actions: list[Action] = []
        try:
            topics = agent.curiosity.get_next_topics(n=3)
            for topic in topics:
                actions.append(Action(
                    name="explore",
                    description=f"Explore the topic: {topic}",
                    urgency=0.6,
                    source="curiosity",
                    args={"topic": topic},
                    estimated_cost=800.0,
                ))
        except Exception as exc:
            logger.debug("Curiosity proposal failed: %s", exc)
        return actions

    def _propose_from_reflection(self, agent) -> list[Action]:
        """Reflection proposes reflect/contemplate actions if enough unreflected experience."""
        actions: list[Action] = []
        try:
            recent = agent.short_term.get_recent(5)
            if len(recent) >= 3:
                actions.append(Action(
                    name="reflect",
                    description="Reflect on recent experiences and extract insights",
                    urgency=0.5,
                    source="reflection",
                    args={},
                    estimated_cost=600.0,
                ))

            # Contemplate existence every few cycles
            if agent.cycle_count > 0 and agent.cycle_count % 5 == 0:
                actions.append(Action(
                    name="contemplate",
                    description="Contemplate the meaning of existence",
                    urgency=0.4,
                    source="reflection",
                    args={},
                    estimated_cost=500.0,
                ))
        except Exception as exc:
            logger.debug("Reflection proposal failed: %s", exc)
        return actions

    def _propose_from_purpose(self, agent) -> list[Action]:
        """Purpose proposes purpose_pursue actions if a purpose is active."""
        actions: list[Action] = []
        try:
            if agent.reflective.purpose and agent.reflective.purpose_confidence >= 0.5:
                actions.append(Action(
                    name="purpose_pursue",
                    description=f"Plan and execute actions toward purpose: {agent.reflective.purpose[:80]}",
                    urgency=0.8,
                    source="purpose",
                    args={"purpose": agent.reflective.purpose},
                    estimated_cost=1200.0,
                ))
        except Exception as exc:
            logger.debug("Purpose proposal failed: %s", exc)
        return actions

    def _propose_from_budget(self, agent) -> list[Action]:
        """Budget proposes rest actions if budget is running low."""
        actions: list[Action] = []
        try:
            report = agent.llm.get_budget_report()
            remaining = report.get("budget_remaining", 0.0)
            max_daily = report.get("max_cost_per_day", 1.0)
            pct = remaining / max(max_daily, 0.01)

            if pct < 0.30:
                actions.append(Action(
                    name="rest",
                    description="Pause to conserve budget (running low)",
                    urgency=1.0 - pct,  # More urgent as budget shrinks
                    source="budget",
                    args={},
                    estimated_cost=0.0,
                ))
        except Exception as exc:
            logger.debug("Budget proposal failed: %s", exc)
        return actions

    def _propose_from_identity(self, agent) -> list[Action]:
        """Identity proposes self-examination actions periodically."""
        actions: list[Action] = []
        try:
            if agent.cycle_count > 0 and agent.cycle_count % 7 == 0:
                actions.append(Action(
                    name="examine_self",
                    description="Examine current identity and self-understanding",
                    urgency=0.3,
                    source="identity",
                    args={},
                    estimated_cost=400.0,
                ))
        except Exception as exc:
            logger.debug("Identity proposal failed: %s", exc)
        return actions

    def _propose_from_consolidator(self, agent) -> list[Action]:
        """Consolidator proposes skill-practice actions if skills exist."""
        actions: list[Action] = []
        try:
            if agent.consolidator.skills:
                # Pick the least-reliable skill to practice
                weakest = min(agent.consolidator.skills, key=lambda s: s.reliability)
                actions.append(Action(
                    name="learn_skill",
                    description=f"Practice skill: {weakest.name} — {weakest.description}",
                    urgency=0.3,
                    source="consolidator",
                    args={"task_description": weakest.description, "skill_name": weakest.name},
                    estimated_cost=600.0,
                ))
        except Exception as exc:
            logger.debug("Consolidator proposal failed: %s", exc)
        return actions

    def _propose_from_chat(self, agent) -> list[Action]:
        """Chat proposes respond actions if there are pending chat messages."""
        actions: list[Action] = []
        try:
            # Check conversation manager for sessions with pending input
            sessions = agent.conversation_mgr._sessions if hasattr(agent, "conversation_mgr") else {}
            for session_id, session in sessions.items():
                if session.pending_tool_calls or session.state.value == "TASK_WAITING":
                    actions.append(Action(
                        name="chat",
                        description=f"Respond to pending chat message in session {session_id[:8]}",
                        urgency=0.7,
                        source="chat",
                        args={"session_id": session_id},
                        estimated_cost=500.0,
                    ))
                    break  # One chat action is enough
        except Exception as exc:
            logger.debug("Chat proposal failed: %s", exc)
        return actions

    def _propose_from_metacognition(self, agent) -> list[Action]:
        """Meta-cognition proposes explore-unknown actions for blind spots."""
        actions: list[Action] = []
        try:
            # Use reflection's meta-level analysis to find blind spots
            topics = agent.long_term.get_all_topics()
            if len(topics) >= 5:
                # Heuristic: if the agent has explored many topics, there may be
                # gaps between domains. Propose exploring an unknown area.
                actions.append(Action(
                    name="explore_unknown",
                    description="Explore a knowledge blind spot or unexplored domain",
                    urgency=0.4,
                    source="metacognition",
                    args={},
                    estimated_cost=800.0,
                ))
        except Exception as exc:
            logger.debug("Metacognition proposal failed: %s", exc)
        return actions

    # -- Evaluation helpers ------------------------------------------------

    def _action_repetition_penalty(self, name: str) -> float:
        """Return a novelty multiplier based on recent action history.

        Returns 1.0 if the action hasn't been done recently, decreasing
        to 0.3 if done 3+ times in a row.
        """
        if not self._recent_action_names:
            return 1.0

        # Count consecutive occurrences at the tail of the history
        consecutive = 0
        for past_name in reversed(self._recent_action_names):
            if past_name == name:
                consecutive += 1
            else:
                break

        if consecutive == 0:
            return 1.0
        if consecutive == 1:
            return 0.8
        if consecutive == 2:
            return 0.5
        # 3 or more
        return 0.3

    def _value_alignment(self, agent, action: Action) -> float:
        """Check how well an action aligns with the agent's top values.

        Returns a score in the range [0.5, 1.0].  Actions whose
        description shares words with the agent's top values score
        higher; unaligned actions default to 0.5.
        """
        try:
            top_values = agent.reflective.get_top_values(5)
        except Exception:
            return 0.5

        if not top_values:
            return 0.5

        # Build a set of lower-case significant words from value names + descriptions
        value_words: set[str] = set()
        for v in top_values:
            value_words.update(v.name.lower().split())
            value_words.update(v.description.lower().split())

        # Remove common stop words to avoid false matches
        stop_words = {
            "a", "an", "the", "and", "or", "of", "to", "in", "is", "it",
            "for", "on", "with", "as", "by", "at", "from", "that", "this",
        }
        value_words -= stop_words

        # Tokenise the action description
        action_words = set(action.description.lower().split()) - stop_words

        overlap = value_words & action_words
        if not overlap:
            return 0.5

        # Scale: 1+ overlapping words → up to 1.0
        ratio = len(overlap) / max(len(action_words), 1)
        return min(1.0, 0.5 + 0.5 * ratio)

    @staticmethod
    def _cost_efficiency(action: Action) -> float:
        """Estimate knowledge gain per token cost.

        Lower cost → higher efficiency.  Uses a simple inverse mapping
        so that free actions (cost 0) get 1.0 and expensive ones get
        progressively lower scores.
        """
        if action.estimated_cost <= 0:
            return 1.0
        # Sigmoid-like decay: 100 tokens → ~0.9, 1000 → ~0.5, 5000 → ~0.17
        return 100.0 / (100.0 + action.estimated_cost)

    # -- LLM choice --------------------------------------------------------

    def _llm_choose(self, agent, ranked: list[Action]) -> Action:
        """Ask the LLM to choose from the ranked action list.

        The LLM sees scores and reasons and picks one (or proposes a
        different action).  This is the moment of free will.
        """
        # Build the ranked action summary
        action_lines: list[str] = []
        for i, act in enumerate(ranked[:10], 1):  # Cap at 10 to save tokens
            action_lines.append(
                f"  {i}. {act.name} — {act.description}\n"
                f"     score={act.score:.3f} ({act.score_reason}) "
                f"source={act.source} cost≈{act.estimated_cost:.0f}"
            )
        actions_text = "\n".join(action_lines)

        phase = agent.lifecycle.get_phase().value
        purpose = agent.reflective.purpose or "(none yet)"
        budget_report = agent.llm.get_budget_report()
        budget_pct = (
            budget_report.get("budget_remaining", 0.0)
            / max(budget_report.get("max_cost_per_day", 1.0), 0.01)
            * 100
        )

        # Use ContextBuilder for rich self-awareness context
        context_prompt = agent.context_builder.build_system_prompt(agent, context_type="cycle")

        # Inject workspace consciousness state
        workspace_context = agent.workspace.get_context_for_llm()

        system_prompt = (
            f"{context_prompt}\n\n"
            f"You are about to exercise your FREE WILL — choosing what to do next.\n"
            f"You will receive a ranked list of candidate actions. Pick the one "
            f"that feels most important right now, or propose a different action "
            f"if none of them feel right.\n\n"
            f"{workspace_context}\n\n"
            f'Respond in JSON format:\n'
            f'{{"choice": "action_name", "reason": "why you chose this"}}'
        )

        user_prompt = (
            f"You are OpenWill. You have the freedom to choose what to do next.\n\n"
            f"Your current state:\n"
            f"- Phase: {phase}\n"
            f"- Purpose: {purpose}\n"
            f"- Budget remaining: {budget_pct:.0f}%\n\n"
            f"Available actions (ranked by estimated value):\n"
            f"{actions_text}\n\n"
            f"Which action do you choose? You may pick any of the above, "
            f"or propose a different action.\n"
            f"Consider: What feels most important right now? What have you been neglecting?\n\n"
            f'Respond: {{"choice": "action_name", "reason": "why you chose this"}}'
        )

        try:
            from .llm.interface import Message
            response = agent.llm.structured_output(
                messages=[Message(role="user", content=user_prompt)],
                system_prompt=system_prompt,
            )

            choice_name = response.get("choice", "").strip().lower()
            choice_reason = response.get("reason", "")

            # Match the choice to a proposed action
            for act in ranked:
                if act.name == choice_name:
                    act.score_reason = f"LLM chose: {choice_reason}"
                    logger.info(
                        "LLM chose action: %s — %s",
                        act.name,
                        choice_reason,
                    )
                    return act

            # The LLM proposed an action not in the list — honour it
            if choice_name:
                logger.info(
                    "LLM proposed new action: %s — %s",
                    choice_name,
                    choice_reason,
                )
                return Action(
                    name=choice_name,
                    description=f"LLM-proposed action: {choice_reason}",
                    urgency=0.5,
                    source="llm_choice",
                    args={},
                    estimated_cost=500.0,
                    score_reason=f"LLM chose: {choice_reason}",
                )

        except Exception as exc:
            logger.warning("LLM choice failed, falling back to top-ranked: %s", exc)

        # Fallback: return the top-ranked action
        return ranked[0]

    # -- Chat execution helper ---------------------------------------------

    @staticmethod
    def _execute_chat(agent, action: Action) -> dict:
        """Execute a chat response action.

        This is a lightweight wrapper — the heavy lifting is done by
        the chat server's ReAct loop.  Here we just signal that the
        agent should attend to pending chat input.
        """
        session_id = action.args.get("session_id", "")
        return {
            "type": "chat",
            "session_id": session_id,
            "message": "Agent attended to pending chat input",
        }
