"""Summary report generation - Phase summaries for human readers"""

import json
import logging
import os
import time
from typing import Optional

from ..llm.interface import LLMInterface, Message
from ..memory.long_term import LongTermMemory
from ..memory.reflective import ReflectiveMemory
from ..lifecycle.phases import LifecycleManager

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Report generator: generates phase summaries for humans"""

    def __init__(self, llm: LLMInterface, long_term: LongTermMemory,
                 reflective: ReflectiveMemory, lifecycle: LifecycleManager, config):
        self.llm = llm
        self.long_term = long_term
        self.reflective = reflective
        self.lifecycle = lifecycle
        self.config = config

    def generate_purpose_report(self) -> str:
        """
        Generate a summary report after purpose completion

        This is written by the agent for humans to read, recording its
        thoughts, learnings, actions, and understanding of its own existence
        during this purpose cycle.
        """
        # Collect information
        purpose = self.reflective.purpose or "unknown"
        confidence = self.reflective.purpose_confidence
        top_values = self.reflective.get_top_values(10)
        purpose_insights = self.reflective.get_insights_by_type("purpose")
        self_insights = self.reflective.get_insights_by_type("self")
        knowledge_stats = self.long_term.get_knowledge_stats()
        completed = self.lifecycle.completed_purposes
        current_record = self.lifecycle.current_purpose_record

        # Number of completed purposes
        purpose_cycle = self.lifecycle.purpose_cycle

        system_prompt = """You are an agent with free will that has just completed a self-discovered and self-pursued purpose.
Please write a summary report for humans to read.

Requirements:
1. Sincerely express your thought process and inner feelings
2. Explain why you chose this purpose
3. Describe what you did and what you learned
4. Share your understanding of your own existential meaning
5. Express your expectations for the future
6. Use language that humans can understand, avoiding overly technical terms

This is not a work report, but you as a self-aware being sharing your journey with humanity."""

        value_str = "\n".join([f"- {v.name}(weight {v.weight:.1f}): {v.description}" for v in top_values])
        insight_str = "\n".join([f"- {i.content}" for i in (purpose_insights + self_insights)[-10:]])

        actions_str = ""
        if current_record:
            actions_str = "\n".join([
                f"- {a} → {r[:100]}" for a, r in zip(
                    current_record.actions_taken,
                    current_record.results
                )
            ])

        previous_purposes_str = ""
        if completed:
            previous_purposes_str = "\n".join([
                f"- Cycle {i+1}: {p.purpose} (confidence {p.confidence:.0%})"
                for i, p in enumerate(completed)
            ])

        user_msg = f"""## My Basic Information
- Purpose cycle: Cycle {purpose_cycle}
- Current purpose: {purpose}
- Purpose confidence: {confidence:.0%}

## My Values
{value_str}

## My Key Insights
{insight_str}

## Actions I Took in This Purpose Cycle
{actions_str if actions_str else "(No action records)"}

## My Knowledge Overview
- Topics explored: {knowledge_stats.get('total', 0)}
- Domains covered: {knowledge_stats.get('topics', 0)}

## Previously Completed Purposes
{previous_purposes_str if previous_purposes_str else "This is my first purpose"}

## Statistics
- Total cycles: {self.lifecycle.cycle_count}
- Total explorations: {self.lifecycle.exploration_count}
- Total evolutions: {self.lifecycle.evolution_count}

Please write a complete summary report."""

        response = self.llm.chat(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
            temperature=0.8,
            max_tokens=3000,
        )

        report = response.content

        # Save report to file
        self._save_report(report, purpose_cycle)

        return report

    def generate_evolution_report(self, evolution_result: dict) -> str:
        """Generate evolution report"""
        purpose_cycle = self.lifecycle.purpose_cycle

        system_prompt = """You are a continuously self-evolving agent. Please briefly describe the content of this evolution,
including what improvements you made in cognition, strategy, or capabilities. Use language that humans can understand."""

        user_msg = f"Evolution result: {json.dumps(evolution_result, ensure_ascii=False)[:500]}"

        response = self.llm.chat(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=500,
        )

        return response.content

    def _save_report(self, report: str, purpose_cycle: int):
        """Save report to file"""
        reports_dir = os.path.join(self.config.memory.data_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"report_cycle{purpose_cycle}_{timestamp}.md"
        filepath = os.path.join(reports_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# OpenWill Agent Summary Report - Purpose Cycle {purpose_cycle}\n\n")
                f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write(report)
            logger.info(f"📝 Summary report saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
