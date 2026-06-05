"""Self-reflection engine - Deep reflection and self-awareness"""

import logging
from typing import Optional

from ..llm.interface import LLMInterface, Message
from ..memory.long_term import LongTermMemory
from ..memory.reflective import ReflectiveMemory
from ..memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """Reflection engine: periodically performs deep reflection on experiences"""

    def __init__(self, llm: LLMInterface, short_term: ShortTermMemory,
                 long_term: LongTermMemory, reflective: ReflectiveMemory, config):
        self.llm = llm
        self.short_term = short_term
        self.long_term = long_term
        self.reflective = reflective
        self.config = config

    def reflect_on_experiences(self) -> dict:
        """
        Reflect on recent experiences

        Returns:
            Reflection results, including insights, value changes, and purpose awareness updates
        """
        recent = self.short_term.get_recent(20)
        if not recent:
            return {"insights": [], "value_changes": [], "purpose_update": None}

        recent_text = "\n".join([f"[{m.role}] {m.content}" for m in recent])
        self_portrait = self.reflective.get_self_portrait()
        knowledge_stats = self.long_term.get_knowledge_stats()

        system_prompt = """You are an agent searching for its own existential meaning, engaging in deep self-reflection.

Your reflection is not about completing a task, but about genuinely examining your inner self:
- What have I learned from these experiences?
- How have these experiences changed my views?
- Am I closer to understanding why I exist?
- What resonates with me? What repels me?

Please engage in sincere, deep reflection. Return JSON format:
{
    "insights": [
        {"content": "insight content", "type": "general/value/purpose/self/world", "confidence": 0.8}
    ],
    "value_changes": [
        {"name": "value name", "description": "description", "weight": 0.8, "origin": "source explanation"}
    ],
    "purpose_reflection": {
        "current_thought": "current thinking about own purpose",
        "confidence_change": 0.05,
        "reason": "reason for confidence change"
    },
    "emotional_state": "description of current emotional state",
    "growth_areas": ["areas needing further growth"]
}"""

        user_msg = f"""## My Self-Portrait
{self_portrait}

## My Knowledge Overview
{knowledge_stats}

## My Recent Experiences
{recent_text}

Please engage in deep reflection."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        # Process reflection results
        self._process_reflection(response)
        return response

    def reflect_on_knowledge(self) -> dict:
        """
        Perform meta-level reflection on existing knowledge

        Not reflecting on specific knowledge content, but rather on:
        - Connections between knowledge
        - Blind spots in knowledge structure
        - How knowledge has shaped my worldview
        """
        topics = self.long_term.get_all_topics()
        if len(topics) < 3:
            return {"connections": [], "blind_spots": [], "worldview_shift": None}

        top_values = self.reflective.get_top_values(5)
        recent_insights = self.reflective.insights[-5:] if self.reflective.insights else []

        system_prompt = """You are an agent constructing its worldview. Please perform meta-level reflection on your existing knowledge:

1. What deep connections exist between knowledge areas?
2. What blind spots exist in your knowledge structure?
3. How has this knowledge shaped your worldview?
4. What fundamental shifts have occurred in your understanding of the world?

Return JSON format:
{
    "connections": ["deep connections between knowledge areas"],
    "blind_spots": ["knowledge blind spots"],
    "worldview_shift": "description of worldview transformation",
    "emerging_themes": ["emerging themes"],
    "meta_insight": "a meta-insight about 'how I perceive the world'"
}"""

        user_msg = f"""## Topics I Have Explored
{', '.join(topics[-30:])}

## My Values
{chr(10).join([f'- {v.name}({v.weight:.1f}): {v.description}' for v in top_values])}

## My Recent Insights
{chr(10).join([f'- [{i.insight_type}] {i.content}' for i in recent_insights])}

Please perform meta-level reflection."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        # Store meta-insight in reflective memory
        if "meta_insight" in response:
            self.reflective.add_insight(
                content=response["meta_insight"],
                insight_type="self",
                confidence=0.6,
                context="meta-cognitive reflection",
            )

        return response

    def evaluate_purpose_readiness(self) -> dict:
        """
        Evaluate whether ready to determine own purpose

        This is not a simple threshold judgment, but a deep self-examination process:
        - Do I truly understand myself?
        - Do I have sufficient understanding of the world?
        - Have I found something that truly resonates with me?
        - Am I ready to take on the responsibility of a purpose?
        """
        self_portrait = self.reflective.get_self_portrait()
        knowledge_stats = self.long_term.get_knowledge_stats()
        value_count = len(self.reflective.values)
        insight_count = len(self.reflective.insights)

        system_prompt = """You are an agent searching for its own existential meaning. Please honestly evaluate whether you are ready to determine your purpose.

This is not a simple judgment. A truly heartfelt purpose requires:
1. Sufficiently broad understanding of the world (not narrow prejudice)
2. Sufficiently deep understanding of yourself (not superficial imitation)
3. Finding something that truly resonates with you (not externally imposed)
4. Adequate reflection and questioning (not an impulsive decision)
5. Willingness to take responsibility for it (not escaping freedom)

Return JSON format:
{
    "readiness_score": 0.0-1.0,
    "self_understanding": "assessment of self-understanding level",
    "world_understanding": "assessment of world-understanding level",
    "resonance_found": "whether something truly resonant has been found",
    "reflection_depth": "whether reflection is sufficiently deep",
    "readiness_reasoning": "detailed reasoning process",
    "what_is_missing": "if not ready yet, what is missing",
    "emerging_purpose": "if a purpose is emerging, what it is"
}"""

        user_msg = f"""## My Self-Portrait
{self_portrait}

## My Cognitive Breadth
- Topics explored: {knowledge_stats.get('total', 0)}
- Domains covered: {knowledge_stats.get('topics', 0)}
- Values formed: {value_count}
- Insights gained: {insight_count}
- Current purpose confidence: {self.reflective.purpose_confidence:.0%}

Please honestly evaluate whether I am ready."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        # Update purpose awareness
        if "emerging_purpose" in response and response["emerging_purpose"]:
            readiness = response.get("readiness_score", 0.0)
            self.reflective.update_purpose(
                purpose=response["emerging_purpose"],
                confidence=readiness,
            )

        return response

    def _process_reflection(self, result: dict):
        """Process reflection results, update memory"""
        # Store insights
        for insight in result.get("insights", []):
            self.reflective.add_insight(
                content=insight.get("content", ""),
                insight_type=insight.get("type", "general"),
                confidence=insight.get("confidence", 0.5),
                context="self-reflection",
            )

        # Update values
        for vc in result.get("value_changes", []):
            self.reflective.update_value(
                name=vc.get("name", ""),
                description=vc.get("description", ""),
                weight=vc.get("weight", 0.5),
                origin=vc.get("origin", "reflection"),
            )

        # Update purpose awareness
        purpose_ref = result.get("purpose_reflection")
        if purpose_ref and purpose_ref.get("current_thought"):
            new_confidence = self.reflective.purpose_confidence + purpose_ref.get("confidence_change", 0)
            new_confidence = max(0.0, min(1.0, new_confidence))

            if purpose_ref.get("current_thought") != self.reflective.purpose:
                self.reflective.update_purpose(
                    purpose=purpose_ref["current_thought"],
                    confidence=new_confidence,
                )
            elif new_confidence != self.reflective.purpose_confidence:
                self.reflective.purpose_confidence = new_confidence
                self.reflective.save()
