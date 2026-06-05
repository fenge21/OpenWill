"""Knowledge Graph and Meta-Cognition - concept relationship network and self-awareness"""

import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Common English stop words for lightweight concept extraction
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about", "up",
    "also", "what", "which", "who", "whom", "this", "that", "these",
    "those", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "am", "any",
})


class KnowledgeGraph:
    """Lightweight concept relationship network with capacity management."""

    MAX_NODES = 500  # Capacity upper limit

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.nodes: dict[str, dict] = {}  # concept -> {"connections": {other: relation_type}, "mention_count": int, "last_mentioned": float}
        self._incoming: dict[str, set] = {}  # Reverse index: concept -> set of nodes that point to it
        self._load()

    def _get_filepath(self) -> str:
        return os.path.join(self.data_dir, "runtime", "knowledge_graph.json")

    def _load(self):
        """Load knowledge graph from disk"""
        filepath = self._get_filepath()
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.nodes = data.get("nodes", {})
                # Rebuild reverse index from loaded nodes
                self._rebuild_incoming_index()
                logger.info(f"Loaded knowledge graph: {len(self.nodes)} nodes")
            except Exception as e:
                logger.error(f"Failed to load knowledge graph: {e}")
                self.nodes = {}

    def _rebuild_incoming_index(self):
        """Rebuild the reverse index from the current nodes dict."""
        self._incoming = {}
        for source, data in self.nodes.items():
            for target in data.get("connections", {}):
                if target not in self._incoming:
                    self._incoming[target] = set()
                self._incoming[target].add(source)

    def save(self):
        """Persist knowledge graph to disk"""
        filepath = self._get_filepath()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            data = {"nodes": self.nodes}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved knowledge graph: {len(self.nodes)} nodes")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")

    def _ensure_node(self, concept: str):
        """Create node entry if it does not exist"""
        if concept not in self.nodes:
            self.nodes[concept] = {
                "connections": {},
                "mention_count": 0,
                "last_mentioned": 0.0,
            }

    def _prune(self):
        """Prune least-relevant nodes when capacity is exceeded.

        Relevance score = mention_count * (outgoing + incoming connections).
        Nodes with the lowest relevance are removed first.
        Removes 10% of nodes (the least relevant) to avoid frequent pruning.
        """
        if len(self.nodes) <= self.MAX_NODES:
            return

        # Score each node by relevance
        scored = []
        for concept, data in self.nodes.items():
            outgoing = len(data["connections"])
            incoming = len(self._incoming.get(concept, set()))
            relevance = data["mention_count"] * (outgoing + incoming + 1)
            scored.append((concept, relevance))

        # Sort by relevance ascending (least relevant first)
        scored.sort(key=lambda x: x[1])

        # Remove bottom 10% of nodes
        to_remove_count = max(1, len(self.nodes) - int(self.MAX_NODES * 0.9))
        removed = 0
        for concept, _ in scored:
            if removed >= to_remove_count:
                break
            # Remove outgoing connections from other nodes
            for target in list(self.nodes[concept]["connections"].keys()):
                if target in self._incoming:
                    self._incoming[target].discard(concept)
            # Remove from incoming index
            for source in self._incoming.get(concept, set()):
                if source in self.nodes:
                    self.nodes[source]["connections"].pop(concept, None)
            # Remove the node itself
            del self.nodes[concept]
            self._incoming.pop(concept, None)
            removed += 1

        logger.info("KnowledgeGraph pruned: removed %d nodes (%d remaining)",
                     removed, len(self.nodes))

    def add_relation(self, subject: str, predicate: str, obj: str):
        """Add a triple: subject --predicate--> obj"""
        now = time.time()

        # Ensure both nodes exist
        self._ensure_node(subject)
        self._ensure_node(obj)

        # Add the directed edge
        self.nodes[subject]["connections"][obj] = predicate

        # Maintain reverse index
        if obj not in self._incoming:
            self._incoming[obj] = set()
        self._incoming[obj].add(subject)

        # Increment mention counts and update timestamps
        self.nodes[subject]["mention_count"] += 1
        self.nodes[obj]["mention_count"] += 1
        self.nodes[subject]["last_mentioned"] = now
        self.nodes[obj]["last_mentioned"] = now

        # Enforce capacity: prune least-relevant nodes if over limit
        if len(self.nodes) > self.MAX_NODES:
            self._prune()

    def add_from_text(self, text: str, source: str = ""):
        """Extract key concepts and relations from text using simple heuristics.

        Lightweight alternative to full NLP:
        - Split into sentences
        - Extract noun phrases (capitalized words, skip stop words)
        - Create "co-occurs_with" relations between concepts in the same sentence
        """
        # Split into sentences
        sentences = re.split(r'[.!?;]\s*', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Extract candidate concepts: words that are capitalized or longer than 5 chars
            words = re.findall(r'\b[A-Za-z]+\b', sentence)
            concepts = []
            for word in words:
                lower = word.lower()
                if lower in STOP_WORDS:
                    continue
                # Accept capitalized words or words longer than 5 characters
                if word[0].isupper() or len(word) > 5:
                    concepts.append(word)

            # Deduplicate while preserving order
            seen = set()
            unique_concepts = []
            for c in concepts:
                if c not in seen:
                    seen.add(c)
                    unique_concepts.append(c)

            # Create co-occurs_with relations between all pairs in the same sentence
            for i in range(len(unique_concepts)):
                for j in range(i + 1, len(unique_concepts)):
                    self.add_relation(unique_concepts[i], "co-occurs_with", unique_concepts[j])

    def query(self, concept: str, depth: int = 2) -> set[str]:
        """Semantic expansion: find all concepts within `depth` hops.

        Uses reverse index for O(V+E) instead of O(V*E) incoming edge lookup.
        Returns the expanded concept set for improved retrieval.
        """
        if concept not in self.nodes:
            return {concept}

        visited = {concept}
        frontier = {concept}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                if node in self.nodes:
                    # Follow outgoing edges
                    for neighbor in self.nodes[node]["connections"]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                    # Follow incoming edges via reverse index (O(1) lookup)
                    for source_node in self._incoming.get(node, set()):
                        if source_node not in visited:
                            visited.add(source_node)
                            next_frontier.add(source_node)
            frontier = next_frontier
            if not frontier:
                break

        return visited

    def get_central_concepts(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Return concepts sorted by centrality (mention_count * connection_count).

        These are the agent's "core knowledge".
        """
        scored = []
        for concept, data in self.nodes.items():
            connection_count = len(data["connections"])
            # Also count incoming connections
            incoming = sum(
                1 for other_data in self.nodes.values()
                if concept in other_data["connections"]
            )
            total_connections = connection_count + incoming
            centrality = data["mention_count"] * total_connections
            scored.append((concept, centrality))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_relations(self, concept: str) -> list[tuple[str, str, str]]:
        """Return all (subject, predicate, obj) triples involving this concept."""
        triples = []

        # Outgoing relations: concept -> other
        if concept in self.nodes:
            for obj, predicate in self.nodes[concept]["connections"].items():
                triples.append((concept, predicate, obj))

        # Incoming relations: other -> concept
        for other_concept, data in self.nodes.items():
            if other_concept == concept:
                continue
            if concept in data["connections"]:
                predicate = data["connections"][concept]
                triples.append((other_concept, predicate, concept))

        return triples

    def get_stats(self) -> dict:
        """Total nodes, total edges, top 10 central concepts."""
        total_nodes = len(self.nodes)
        total_edges = sum(len(data["connections"]) for data in self.nodes.values())
        top_concepts = self.get_central_concepts(top_k=10)

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "top_central_concepts": [
                {"concept": c, "centrality": score} for c, score in top_concepts
            ],
        }


class MetaCognition:
    """The agent's awareness of its own knowledge state."""

    def __init__(self, knowledge_graph: KnowledgeGraph, long_term_memory):
        self.kg = knowledge_graph
        self.ltm = long_term_memory

    def estimate_knowledge_coverage(self, topic: str) -> float:
        """Estimate how well the agent knows a topic (0-1).

        Formula: (kg_connections / 10 + ltm_mentions / 5) / 2, capped at 1.0
        """
        # Knowledge graph connections
        kg_connections = 0
        if topic in self.kg.nodes:
            kg_connections = len(self.kg.nodes[topic]["connections"])
            # Also count incoming connections
            incoming = sum(
                1 for data in self.kg.nodes.values()
                if topic in data["connections"]
            )
            kg_connections += incoming

        # Long-term memory mentions
        ltm_mentions = 0
        if self.ltm is not None:
            for entry in self.ltm.entries:
                text = (entry.topic + " " + entry.content).lower()
                if topic.lower() in text:
                    ltm_mentions += 1

        coverage = (kg_connections / 10 + ltm_mentions / 5) / 2
        return min(coverage, 1.0)

    def identify_blind_spots(self) -> list[str]:
        """Find concepts that are mentioned but poorly understood.

        - Concepts in knowledge graph with very few connections (< 2)
        - Concepts that appear in LTM entries but have NO knowledge graph node
        Returns top 10 by mention_count (most referenced but least understood).
        """
        blind_spots = {}

        # Known unknowns: KG nodes with < 2 connections
        for concept, data in self.kg.nodes.items():
            outgoing = len(data["connections"])
            incoming = sum(
                1 for d in self.kg.nodes.values()
                if concept in d["connections"]
            )
            total_connections = outgoing + incoming
            if total_connections < 2:
                blind_spots[concept] = data["mention_count"]

        # Unknown unknowns: concepts in LTM but not in KG
        if self.ltm is not None:
            for entry in self.ltm.entries:
                # Extract candidate concepts from LTM entries
                words = re.findall(r'\b[A-Za-z]+\b', (entry.topic + " " + entry.content))
                for word in words:
                    lower = word.lower()
                    if lower in STOP_WORDS:
                        continue
                    if word[0].isupper() or len(word) > 5:
                        if word not in self.kg.nodes:
                            blind_spots[word] = blind_spots.get(word, 0) + 1

        # Sort by mention_count descending, return top 10
        sorted_spots = sorted(blind_spots.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in sorted_spots[:10]]

    def should_explore(self, topic: str) -> tuple[bool, str]:
        """Based on information gain: should the agent explore this topic?

        Returns (bool, reason).
        High gain: topic is a blind spot or low coverage.
        Low gain: topic is already well understood.
        """
        coverage = self.estimate_knowledge_coverage(topic)
        blind_spots = self.identify_blind_spots()

        if topic in blind_spots:
            return True, f"'{topic}' is a blind spot — known to exist but poorly understood. High information gain expected."

        if coverage < 0.3:
            return True, f"'{topic}' has low knowledge coverage ({coverage:.2f}). Exploring would significantly expand understanding."

        if coverage < 0.6:
            return True, f"'{topic}' has moderate coverage ({coverage:.2f}). Further exploration could fill gaps."

        return False, f"'{topic}' is already well understood (coverage: {coverage:.2f}). Limited information gain from further exploration."

    def get_knowledge_report(self) -> dict:
        """Full meta-cognitive report for dashboard.

        Includes: total known concepts, blind spots, well-understood topics, coverage distribution.
        """
        total_concepts = len(self.kg.nodes)
        blind_spots = self.identify_blind_spots()

        # Well-understood topics: high coverage
        well_understood = []
        for concept in self.kg.nodes:
            coverage = self.estimate_knowledge_coverage(concept)
            if coverage >= 0.6:
                well_understood.append({"concept": concept, "coverage": round(coverage, 3)})

        well_understood.sort(key=lambda x: x["coverage"], reverse=True)

        # Coverage distribution
        coverage_buckets = {"high": 0, "medium": 0, "low": 0, "none": 0}
        for concept in self.kg.nodes:
            coverage = self.estimate_knowledge_coverage(concept)
            if coverage >= 0.6:
                coverage_buckets["high"] += 1
            elif coverage >= 0.3:
                coverage_buckets["medium"] += 1
            elif coverage > 0:
                coverage_buckets["low"] += 1
            else:
                coverage_buckets["none"] += 1

        # Also count LTM topics not in KG
        ltm_only = 0
        if self.ltm is not None:
            for entry in self.ltm.entries:
                words = re.findall(r'\b[A-Za-z]+\b', entry.topic)
                for word in words:
                    if word[0].isupper() and word not in self.kg.nodes:
                        ltm_only += 1
                        break

        return {
            "total_known_concepts": total_concepts,
            "blind_spots": blind_spots,
            "well_understood_topics": well_understood[:10],
            "coverage_distribution": coverage_buckets,
            "ltm_topics_not_in_kg": ltm_only,
        }

    def suggest_exploration_topics(self, n: int = 3) -> list[dict]:
        """Suggest topics worth exploring based on meta-cognitive analysis.

        Each suggestion includes: topic, reason, estimated_gain (0-1).
        Prioritizes blind spots and low-coverage topics.
        """
        candidates = []

        # Blind spots get highest priority
        blind_spots = self.identify_blind_spots()
        for topic in blind_spots:
            coverage = self.estimate_knowledge_coverage(topic)
            gain = 1.0 - coverage  # Lower coverage means higher gain
            candidates.append({
                "topic": topic,
                "reason": "Blind spot — known to exist but poorly understood",
                "estimated_gain": round(gain, 3),
            })

        # Low-coverage KG concepts
        for concept in self.kg.nodes:
            if concept in blind_spots:
                continue  # Already added
            coverage = self.estimate_knowledge_coverage(concept)
            if coverage < 0.5:
                gain = 1.0 - coverage
                candidates.append({
                    "topic": concept,
                    "reason": f"Low coverage ({coverage:.2f}) — further exploration recommended",
                    "estimated_gain": round(gain, 3),
                })

        # Sort by estimated_gain descending
        candidates.sort(key=lambda x: x["estimated_gain"], reverse=True)

        return candidates[:n]
