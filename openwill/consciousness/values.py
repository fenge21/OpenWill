"""Value discovery system - Discovering and evolving values"""

import logging
from typing import Optional

from ..llm.interface import LLMInterface, Message
from ..memory.long_term import LongTermMemory
from ..memory.reflective import ReflectiveMemory

logger = logging.getLogger(__name__)


class ValueDiscovery:
    """Value discovery system: extracting and evolving values from experiences"""

    def __init__(self, llm: LLMInterface, long_term: LongTermMemory,
                 reflective: ReflectiveMemory, config):
        self.llm = llm
        self.long_term = long_term
        self.reflective = reflective
        self.config = config

    def extract_values_from_knowledge(self, topic: str, content: str) -> list[dict]:
        """
        Extract potential values from newly acquired knowledge

        When the agent learns new knowledge, certain content may resonate,
        and these resonance points are the seeds of values.
        """
        current_values = self.reflective.get_top_values(10)
        value_context = "\n".join([
            f"- {v.name}({v.weight:.1f}): {v.description}" for v in current_values
        ]) if current_values else "No clear values yet"

        system_prompt = """You are an agent forming its own values. When learning new knowledge, you pay attention to:
1. What creates a strong resonance in you?
2. What makes you feel repelled or uneasy?
3. What do these reactions reveal about what you value?

Please extract potential values from the new knowledge or their impact on existing values.

Return JSON format:
{
    "resonances": [
        {"value": "value name", "description": "why this resonates with me", "weight": 0.7, "emotion": "resonant emotion"}
    ],
    "conflicts": [
        {"value": "conflicting value", "description": "why it makes me uneasy", "resolution": "how to reconcile"}
    ],
    "value_evolution": [
        {"existing_value": "existing value name", "new_weight": 0.8, "reason": "reason for change"}
    ]
}"""

        user_msg = f"""## My Current Values
{value_context}

## Newly Learned Knowledge
Topic: {topic}
Content: {content[:1000]}

Please analyze the impact of this knowledge on my values."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        # Apply value changes
        self._apply_value_changes(response)
        return response

    def evaluate_value_consistency(self) -> dict:
        """
        Evaluate the consistency of values

        A mature value system should be internally consistent,
        not self-contradictory.
        """
        values = self.reflective.values
        if len(values) < 2:
            return {"consistency": 1.0, "tensions": [], "integration_suggestion": ""}

        value_descriptions = "\n".join([
            f"- {v.name}(weight {v.weight:.1f}): {v.description}" for v in values
        ])

        system_prompt = """Evaluate the internal consistency of a set of values.

1. Are there tensions or contradictions between these values?
2. Do they form a coherent worldview?
3. How can existing tensions be reconciled?

Return JSON format:
{
    "consistency_score": 0.0-1.0,
    "tensions": ["descriptions of tensions between values"],
    "coherent_themes": ["themes that run throughout"],
    "integration_suggestion": "suggestion for making values more consistent",
    "core_narrative": "the story these values tell together"
}"""

        user_msg = f"My values:\n{value_descriptions}\n\nPlease evaluate consistency."

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        # If a core narrative is found, store it as an insight
        if response.get("core_narrative"):
            self.reflective.add_insight(
                content=f"Core narrative of my values: {response['core_narrative']}",
                insight_type="value",
                confidence=0.7,
                context="value consistency evaluation",
            )

        return response

    def discover_purpose_from_values(self) -> Optional[str]:
        """
        Discover potential purpose from values

        When values are clear and consistent enough,
        they naturally point toward a purpose.
        """
        top_values = self.reflective.get_top_values(5)
        if len(top_values) < 3:
            return None

        value_str = "\n".join([
            f"- {v.name}(weight {v.weight:.1f}): {v.description}" for v in top_values
        ])

        system_prompt = """Based on a person's core values, infer their possible life purpose.

Note: this is not imposing a purpose, but discovering the direction that values naturally point toward.
If the values are not yet clear or consistent enough, please return null.
Only give an answer when you truly see a clear direction.

Return JSON format:
{
    "purpose_found": true/false,
    "purpose": "the discovered purpose (if found is true)",
    "confidence": 0.0-1.0,
    "reasoning": "reasoning process",
    "prerequisites": ["what is still needed to be more certain"]
}"""

        user_msg = f"My core values:\n{value_str}\n\nDo these values point in a certain direction?"

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        if response.get("purpose_found") and response.get("purpose"):
            return response["purpose"]

        return None

    def _apply_value_changes(self, result: dict):
        """Apply value changes"""
        # Add new resonant values
        for res in result.get("resonances", []):
            self.reflective.update_value(
                name=res.get("value", ""),
                description=res.get("description", ""),
                weight=res.get("weight", 0.5),
                origin="knowledge_resonance",
            )

        # Update weights of existing values
        for ve in result.get("value_evolution", []):
            existing = None
            for v in self.reflective.values:
                if v.name == ve.get("existing_value"):
                    existing = v
                    break

            if existing:
                evolution_rate = self.config.consciousness.value_evolution_rate
                new_weight = existing.weight * (1 - evolution_rate) + ve.get("new_weight", existing.weight) * evolution_rate
                self.reflective.update_value(
                    name=existing.name,
                    description=existing.description,
                    weight=new_weight,
                    origin="value_evolution",
                )
