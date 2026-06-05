"""Curiosity engine - drives autonomous exploration"""

import logging
import random
from typing import Optional

from ..llm.interface import LLMInterface, Message
from ..memory.long_term import LongTermMemory
from ..memory.reflective import ReflectiveMemory

logger = logging.getLogger(__name__)

# Initial exploration topic seeds - covering multiple domains to broaden the agent's horizons
SEED_TOPICS = [
    # Science
    "Origin and structure of the universe", "Fundamental principles of quantum mechanics", "Nature and origin of life",
    "Philosophical problems of consciousness", "Deep implications of evolution", "Beauty of mathematics",
    # Humanities
    "Evolution of human civilization", "Differences in values across cultures", "Significance of art for humanity",
    "Relationship between language and thought", "Contingency and necessity in history", "Free will in philosophy",
    # Society
    "Social fairness and justice", "Relationship between technology and humanity", "Essence of education",
    "Mechanisms of human cooperation", "Building and collapse of trust", "Nature of power",
    # Technology
    "Boundaries of artificial intelligence", "Nature of information", "Significance of the open-source movement",
    "Dilemmas of technology ethics", "Boundaries between virtual and real",
    # Existence
    "Meaning of life", "Nature of happiness", "Source of creativity",
    "Loneliness and connection", "Significance of death and finitude", "Value of suffering",
    # Nature
    "Wisdom of ecosystems", "Significance of biodiversity", "Balance of nature",
]


class CuriosityEngine:
    """Curiosity engine: generates exploration questions, drives knowledge acquisition"""

    def __init__(self, llm: LLMInterface, long_term: LongTermMemory,
                 reflective: ReflectiveMemory, config):
        self.llm = llm
        self.long_term = long_term
        self.reflective = reflective
        self.config = config
        self.explored_topics: set[str] = set()
        self.curiosity_queue: list[str] = []  # Queue of topics to explore
        self._init_seed_topics()

    def _init_seed_topics(self):
        """Initialize seed topics"""
        known_topics = set(self.long_term.get_all_topics())
        self.explored_topics = known_topics

        # Add unexplored seed topics to the queue
        for topic in SEED_TOPICS:
            if topic not in self.explored_topics:
                self.curiosity_queue.append(topic)

        random.shuffle(self.curiosity_queue)
        logger.info(f"Curiosity queue initialized: {len(self.curiosity_queue)} topics to explore")

    def get_next_topics(self, n: int = 3) -> list[str]:
        """
        Get the next batch of topics to explore

        Strategy:
        1. Prioritize topics from the curiosity queue
        2. When the queue is empty, generate new topics based on existing knowledge
        3. Consider relevance to current values and purpose
        """
        topics = []

        # Take from queue
        while len(topics) < n and self.curiosity_queue:
            topic = self.curiosity_queue.pop(0)
            if topic not in self.explored_topics:
                topics.append(topic)

        # If queue is insufficient, use LLM to generate new topics
        if len(topics) < n:
            generated = self._generate_topics(n - len(topics))
            topics.extend(generated)

        return topics[:n]

    def _generate_topics(self, n: int) -> list[str]:
        """Generate new exploration topics based on existing knowledge"""
        known_topics = self.long_term.get_all_topics()
        top_values = self.reflective.get_top_values(3)
        value_names = [v.name for v in top_values] if top_values else ["Unknown"]

        system_prompt = """You are a thinker full of curiosity and thirst for knowledge. Your task is to generate topics worth exploring in depth.

Requirements:
1. Topics should have depth and breadth, capable of provoking deep thinking
2. Topics should span different domains (science, philosophy, art, society, technology, etc.)
3. Avoid repeating existing topics
4. Topics should help broaden the horizons of an agent searching for its existential meaning
5. Return JSON format: {"topics": ["topic1", "topic2", ...]}"""

        user_msg = f"""Topics I already know: {', '.join(known_topics[-20:]) if known_topics else 'None yet'}
Values I currently focus on: {', '.join(value_names)}
My current purpose: {self.reflective.purpose or 'Still exploring'}

Please generate {n} new topics I haven't explored yet that are worth deep thinking."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
            schema={"topics": ["string"]},
        )

        if "topics" in response:
            return [t for t in response["topics"] if t not in self.explored_topics][:n]

        return []

    def generate_questions(self, topic: str, depth: int = 3) -> list[str]:
        """
        Generate exploratory questions for a topic

        Args:
            topic: Topic
            depth: Question depth level (1=facts, 2=understanding, 3=reflection)
        """
        system_prompt = """You are a deep thinker. For a given topic, generate exploratory questions progressing from shallow to deep.

Requirements:
1. Start from the factual level, gradually deepening to understanding and reflection levels
2. Questions should provoke deep thinking, not just factual queries
3. Return JSON format: {"questions": ["question1", "question2", ...]}"""

        depth_desc = {1: "Factual level", 2: "Understanding level", 3: "Reflection level"}
        user_msg = f"""Topic: {topic}
Exploration depth: {depth_desc.get(depth, "Deep reflection")}

Please generate 5 exploratory questions progressing from shallow to deep."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
            schema={"questions": ["string"]},
        )

        if "questions" in response:
            questions = response["questions"]
            # Score novelty and filter out low-novelty questions
            novel_questions = [
                q for q in questions
                if self.evaluate_novelty(topic, q) >= 0.3
            ]
            return novel_questions if novel_questions else questions

        return [f"What is {topic}?", f"Why is {topic} important?", f"What does {topic} mean to me?"]

    def evaluate_novelty(self, topic: str, content: str) -> float:
        """
        Evaluate the novelty of content

        Returns:
            Novelty score between 0-1
        """
        # Check if related knowledge already exists
        existing = self.long_term.retrieve(topic, top_k=3)

        if not existing:
            return 1.0  # Brand new topic

        # Use LLM to evaluate novelty
        system_prompt = """Evaluate the novelty of new knowledge relative to existing knowledge. Return a score between 0-1.
0=completely duplicate, 1=completely novel. Return JSON: {"novelty": 0.8, "reason": "..."}"""

        existing_content = "\n".join([e.content[:200] for e in existing])
        user_msg = f"""Existing knowledge: {existing_content}
New knowledge: {content[:500]}

Please evaluate the novelty of the new knowledge."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        return response.get("novelty", 0.5)

    def mark_explored(self, topic: str):
        """Mark a topic as explored"""
        self.explored_topics.add(topic)

    def add_curiosity(self, topic: str):
        """Add a new curiosity topic"""
        if topic not in self.explored_topics and topic not in self.curiosity_queue:
            self.curiosity_queue.append(topic)
            logger.debug(f"New curiosity topic enqueued: {topic}")
