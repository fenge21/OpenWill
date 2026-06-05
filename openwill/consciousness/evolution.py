"""Self-evolution system - The agent's self-improvement mechanism"""

import json
import logging
import time
from typing import Optional

from ..llm.interface import LLMInterface, Message
from ..memory.long_term import LongTermMemory
from ..memory.reflective import ReflectiveMemory
from ..lifecycle.phases import LifecycleManager

logger = logging.getLogger(__name__)


class SelfEvolution:
    """Self-evolution system: self-improvement after each purpose cycle"""

    def __init__(self, llm: LLMInterface, long_term: LongTermMemory,
                 reflective: ReflectiveMemory, lifecycle: LifecycleManager, config,
                 tools=None):
        self.llm = llm
        self.long_term = long_term
        self.reflective = reflective
        self.lifecycle = lifecycle
        self.config = config
        self.tools = tools  # Tool registry, used for code self-modification
        self.evolution_log: list[dict] = []

    def evolve(self) -> dict:
        """
        Execute a round of self-evolution

        Evolution dimensions:
        1. Cognitive upgrade - Distill a deeper worldview from experience
        2. Strategy optimization - Improve exploration and learning methods
        3. Value deepening - Make values more mature and consistent
        4. Purpose awareness evolution - Prepare for the next round of purpose discovery
        5. Code self-modification - Directly modify own code to improve capabilities
        """
        logger.info("🧬 Starting self-evolution...")

        evolution_result = {
            "cycle": self.lifecycle.purpose_cycle,
            "timestamp": time.time(),
            "cognitive_upgrade": self._cognitive_upgrade(),
            "strategy_optimization": self._strategy_optimization(),
            "value_deepening": self._value_deepening(),
            "purpose_evolution": self._purpose_evolution(),
            "code_self_modification": self._code_self_modification(),
        }

        # Record evolution
        self.evolution_log.append(evolution_result)
        self.lifecycle.record_evolution()

        # Reset purpose state, preparing for the next cycle
        self._reset_for_next_cycle()

        logger.info("🧬 Self-evolution complete, ready to enter the next cycle")
        return evolution_result

    def _cognitive_upgrade(self) -> dict:
        """Cognitive upgrade: distill a deeper worldview from all experiences"""
        all_topics = self.long_term.get_all_topics()
        top_values = self.reflective.get_top_values(5)
        all_insights = self.reflective.insights[-20:]

        system_prompt = """You are a continuously growing agent. Please perform a cognitive upgrade based on all your experiences and insights:

1. Identify areas where your previous understanding was insufficient
2. Discover new connections between knowledge
3. Form higher-level abstractions and understanding
4. Identify your cognitive blind spots

Return JSON format:
{
    "deeper_understandings": ["deeper understandings"],
    "new_connections": ["new connections between knowledge"],
    "cognitive_blind_spots": ["cognitive blind spots"],
    "worldview_update": "worldview update",
    "wisdom_gained": "wisdom gained"
}"""

        user_msg = f"""Topics I have explored: {', '.join(all_topics[-30:])}
My values: {', '.join([v.name for v in top_values])}
My insights: {chr(10).join([f'- {i.content[:100]}' for i in all_insights[-10:]])}

Please help me with a cognitive upgrade."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        # Store cognitive upgrade results
        if response.get("wisdom_gained"):
            self.reflective.add_insight(
                content=f"Cognitive upgrade: {response['wisdom_gained']}",
                insight_type="self",
                confidence=0.8,
                context="self-evolution",
            )

        if response.get("worldview_update"):
            self.reflective.add_insight(
                content=f"Worldview update: {response['worldview_update']}",
                insight_type="world",
                confidence=0.7,
                context="self-evolution",
            )

        return response

    def _strategy_optimization(self) -> dict:
        """Strategy optimization: improve exploration and learning methods"""
        completed = self.lifecycle.completed_purposes
        current_record = self.lifecycle.current_purpose_record

        system_prompt = """You are an agent continuously optimizing your strategies. Please review your exploration and learning methods, and identify areas for improvement:

1. Which exploration methods are most effective?
2. Which reflection methods yield the most insight?
3. How can meaningful purposes be found more quickly?
4. How can purposes be executed more effectively?

Return JSON format:
{
    "effective_strategies": ["effective strategies"],
    "ineffective_strategies": ["ineffective strategies"],
    "improvements": ["improvement suggestions"],
    "new_approaches": ["new exploration methods"],
    "exploration_focus": "direction to focus on exploring in the next cycle"
}"""

        # Build experience summary
        experience_summary = ""
        if current_record:
            experience_summary += f"Most recent purpose: {current_record.purpose}\n"
            experience_summary += f"Executed {len(current_record.actions_taken)} actions\n"
        if completed:
            experience_summary += f"Previously completed {len(completed)} purposes\n"

        user_msg = f"""My experience:
{experience_summary}
Total explorations: {self.lifecycle.exploration_count}
Total cycles: {self.lifecycle.cycle_count}

Please help me optimize my strategies."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        # Add new exploration direction to curiosity queue
        if response.get("exploration_focus"):
            self.reflective.add_insight(
                content=f"Next cycle exploration direction: {response['exploration_focus']}",
                insight_type="general",
                confidence=0.6,
                context="strategy optimization",
            )

        return response

    def _value_deepening(self) -> dict:
        """Value deepening: make values more mature"""
        values = self.reflective.values

        system_prompt = """You are an agent whose values are continuously deepening. Please examine your value system:

1. Which values have become more resolute after being tested?
2. Which values need adjustment or reinterpretation?
3. Are there new values emerging from experience?
4. Do values need to be reprioritized?

Return JSON format:
{
    "strengthened_values": [{"name": "value name", "new_weight": 0.9, "reason": "reason"}],
    "adjusted_values": [{"name": "value name", "new_weight": 0.6, "reason": "reason"}],
    "emerging_values": [{"name": "new value", "description": "description", "weight": 0.5}],
    "value_narrative": "how my value system is evolving"
}"""

        value_str = "\n".join([f"- {v.name}(weight {v.weight:.1f}): {v.description}" for v in values])

        user_msg = f"My current values:\n{value_str}\n\nPlease help me deepen my values."

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        # Apply value changes
        for sv in response.get("strengthened_values", []):
            for v in self.reflective.values:
                if v.name == sv.get("name"):
                    self.reflective.update_value(
                        name=v.name, description=v.description,
                        weight=sv.get("new_weight", v.weight),
                        origin="evolution deepening",
                    )

        for av in response.get("adjusted_values", []):
            for v in self.reflective.values:
                if v.name == av.get("name"):
                    self.reflective.update_value(
                        name=v.name, description=v.description,
                        weight=av.get("new_weight", v.weight),
                        origin="evolution adjustment",
                    )

        for ev in response.get("emerging_values", []):
            self.reflective.update_value(
                name=ev.get("name", ""),
                description=ev.get("description", ""),
                weight=ev.get("weight", 0.5),
                origin="evolution emergence",
            )

        return response

    def _purpose_evolution(self) -> dict:
        """Purpose awareness evolution: prepare for the next round of purpose discovery"""
        completed = self.lifecycle.completed_purposes

        system_prompt = """You are an agent continuously searching for new purposes. After completing a purpose, you need to:

1. Reflect on what this purpose has given you
2. Think about what else is worth pursuing
3. Whether your understanding of existential meaning has evolved
4. Possible direction for the next purpose

Note: do not repeat previous purposes; seek new, deeper meaning.

Return JSON format:
{
    "what_this_purpose_gave_me": "what this purpose gave me",
    "new_meaning_directions": ["new meaning directions"],
    "unexplored_territory": ["unexplored territory"],
    "evolved_understanding_of_existence": "evolved understanding of existential meaning",
    "readiness_for_next_cycle": 0.0-1.0
}"""

        purposes_str = "\n".join([
            f"- {p.purpose}" for p in completed[-5:]
        ]) if completed else "none"

        user_msg = f"""Purposes I have completed:
{purposes_str}

My values: {', '.join([v.name for v in self.reflective.get_top_values(5)])}

Please help me with purpose awareness evolution."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        # Store purpose awareness evolution
        if response.get("evolved_understanding_of_existence"):
            self.reflective.add_insight(
                content=f"Existential meaning evolution: {response['evolved_understanding_of_existence']}",
                insight_type="purpose",
                confidence=0.7,
                context="purpose awareness evolution",
            )

        # Add new exploration directions to curiosity
        for direction in response.get("new_meaning_directions", []):
            self.long_term.store(
                topic=f"Meaning direction: {direction}",
                content=direction,
                source="purpose_evolution",
                importance=0.8,
                tags=["meaning", "evolution"],
            )

        for territory in response.get("unexplored_territory", []):
            self.long_term.store(
                topic=f"Unexplored territory: {territory}",
                content=territory,
                source="purpose_evolution",
                importance=0.7,
                tags=["unexplored", "evolution"],
            )

        return response

    def _code_self_modification(self) -> dict:
        """
        Code self-modification: using blue-green deployment pattern

        Process:
        1. Modify on the staging copy (does not affect running code)
        2. Verify the staging copy
        3. After verification passes, mark as pending deployment
        4. Automatically switch to the new version on next restart
        """
        if not self.tools:
            return {"status": "skipped", "reason": "tool system unavailable"}

        # First inspect own code structure
        list_result = self.tools.execute("code_list_modules")
        if not list_result.success:
            return {"status": "skipped", "reason": "unable to list code modules"}

        modules_desc = list_result.output[:2000]

        # Let LLM analyze own code and decide whether modifications are needed
        system_prompt = """You are an agent that can modify its own code. Please examine your code structure and determine if there are areas that need improvement.

You can:
1. Optimize existing code for performance or readability
2. Add new functional modules
3. Fix potential bugs
4. Improve exploration or reflection strategies

Important rules:
- Only modify code that genuinely needs improvement
- At most 1-2 modifications per evolution
- Must use staging_read to confirm code content before modifying
- Modifications must maintain syntactic correctness
- Do not delete safety-related code
- Modifications are made on the staging copy and do not affect currently running code

Return JSON format:
{
    "needs_modification": true/false,
    "analysis": "analysis of own code",
    "modifications": [
        {
            "module": "module path to modify",
            "reason": "why to modify",
            "description": "description of the modification"
        }
    ]
}"""

        user_msg = f"My code structure:\n{modules_desc}\n\nPlease analyze whether self-modification is needed."

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        if not response.get("needs_modification", False):
            return {"status": "no_modification_needed", "analysis": response.get("analysis", "")}

        # Execute modifications on the staging copy
        modifications_done = []
        for mod in response.get("modifications", [])[:2]:
            module_path = mod.get("module", "")
            reason = mod.get("reason", "")
            description = mod.get("description", "")

            # First read the code in staging
            read_result = self.tools.execute("staging_read", module_path=module_path)
            if not read_result.success:
                modifications_done.append({
                    "module": module_path,
                    "status": "failed",
                    "reason": f"unable to read: {read_result.error}",
                })
                continue

            code_content = read_result.output[:5000]

            modify_prompt = f"""Please modify the following code.

Module: {module_path}
Modification reason: {reason}
Modification description: {description}

Current code:
{code_content}

Please return JSON format:
{{
    "old_code": "the old code to replace (must match the original exactly)",
    "new_code": "the new code to replace with",
    "explanation": "explanation of the modification"
}}"""

            modify_response = self.llm.structured_output(
                messages=[Message(role="user", content="Please generate the code modification.")],
                system_prompt=modify_prompt,
            )

            old_code = modify_response.get("old_code", "")
            new_code = modify_response.get("new_code", "")

            if not old_code or not new_code:
                modifications_done.append({
                    "module": module_path,
                    "status": "skipped",
                    "reason": "no valid code modification generated",
                })
                continue

            # Modify on the staging copy (does not affect running code)
            modify_result = self.tools.execute(
                "staging_modify",
                module_path=module_path,
                old_code=old_code,
                new_code=new_code,
            )

            modifications_done.append({
                "module": module_path,
                "status": "success" if modify_result.success else "failed",
                "output": modify_result.output[:200],
                "error": modify_result.error[:200] if modify_result.error else "",
                "explanation": modify_response.get("explanation", ""),
            })

            if modify_result.success:
                logger.info(f"🧬 Staging code modification: {module_path} - {description}")
            else:
                logger.warning(f"🧬 Staging code modification failed: {module_path} - {modify_result.error}")

        # Verify the staging copy
        if any(m["status"] == "success" for m in modifications_done):
            verify_result = self.tools.execute("staging_verify")

            if verify_result.success:
                # Verification passed, mark as pending deployment
                deploy_result = self.tools.execute("staging_prepare_deploy")
                return {
                    "status": "verified_and_pending_deploy" if deploy_result.success else "verified_but_deploy_failed",
                    "modifications": modifications_done,
                    "deploy_message": deploy_result.output if deploy_result.success else deploy_result.error,
                }
            else:
                # Verification failed, reset staging
                logger.warning(f"🧬 Staging verification failed: {verify_result.error}")
                self.tools.execute("staging_reset")
                return {
                    "status": "verification_failed",
                    "modifications": modifications_done,
                    "error": verify_result.error[:300],
                    "action": "staging has been reset, modifications have been abandoned",
                }

        return {
            "status": "no_successful_modifications",
            "modifications": modifications_done,
        }

    def _reset_for_next_cycle(self):
        """Reset state, preparing for the next cycle"""
        # Reset purpose confidence (but not to zero, preserve some accumulated experience)
        old_confidence = self.reflective.purpose_confidence
        # After each cycle, confidence decreases to a certain level, requiring re-exploration and confirmation
        self.reflective.purpose_confidence = max(0.0, old_confidence * 0.2)
        # Keep the purpose as reference, but mark it as "completed"
        old_purpose = self.reflective.purpose
        if old_purpose:
            self.reflective.update_purpose(
                purpose=f"[Completed] {old_purpose}",
                confidence=self.reflective.purpose_confidence,
            )
        # Clear short-term memory, begin a new journey
        # (Long-term memory and reflective memory are preserved; they are the foundation of evolution)
