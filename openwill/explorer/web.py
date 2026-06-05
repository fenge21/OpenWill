"""Web explorer - Acquiring knowledge from the internet"""

import logging
from typing import Optional
import json

from ..llm.interface import LLMInterface, Message

logger = logging.getLogger(__name__)


class WebExplorer:
    """Web explorer: searching and acquiring web knowledge"""

    def __init__(self, llm: LLMInterface, config):
        self.llm = llm
        self.config = config
        self._search_available = False
        self._check_search_capability()

    def _check_search_capability(self):
        """Check whether search capability is available"""
        try:
            # Try importing search-related libraries
            import urllib.request
            self._search_available = True
            logger.info("Web explorer initialized successfully")
        except ImportError:
            logger.warning("Web search capability unavailable, will use LLM's internal knowledge")

    def explore_topic(self, topic: str, questions: list[str]) -> dict:
        """
        Explore a topic

        Strategy:
        1. Try to acquire information from web search
        2. If search is unavailable, use the LLM's internal knowledge
        3. Synthesize and organize into structured knowledge
        """
        # Step 1: Try to get web search results
        search_context = ""
        search_result = self.search_web(topic)
        if search_result:
            search_context = f"\n\nWeb search results:\n{search_result}"
            logger.info(f"Web search found results for: {topic}")
        else:
            logger.info(f"No web search results for: {topic}, using LLM knowledge")

        # Step 2: Synthesize knowledge with search context
        knowledge = self._synthesize_knowledge(topic, questions, search_context)

        return knowledge

    def _synthesize_knowledge(self, topic: str, questions: list[str], search_context: str = "") -> dict:
        """Use LLM to synthesize knowledge, optionally incorporating web search results"""
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

        system_prompt = """You are a knowledgeable and thoughtful knowledge explorer. Please provide in-depth, comprehensive, and multi-perspective knowledge on the given topic and questions.

Requirements:
1. Not only answer facts, but also provide deep understanding and different perspectives
2. Point out controversies and differing viewpoints on this topic
3. Reflect on the implications of this topic for "existential meaning" and "values"
4. Provide directions for further exploration
5. If web search results are provided, synthesize them with your own knowledge; do not rely solely on one source

Return JSON format:
{
    "topic": "topic",
    "summary": "brief overview",
    "key_points": ["key points"],
    "different_perspectives": ["different perspectives"],
    "controversies": ["controversies"],
    "existential_implications": "implications for existential meaning",
    "connections_to_other_fields": ["connections to other fields"],
    "further_exploration": ["further exploration directions"],
    "personal_reflection": "this makes me think..."
}"""

        user_msg = f"""Topic: {topic}

Questions I want to understand:
{questions_text}{search_context}

Please provide in-depth knowledge and reflection."""

        response = self.llm.structured_output(
            messages=[Message(role="user", content=user_msg)],
            system_prompt=system_prompt,
        )

        # Ensure required fields exist
        response.setdefault("topic", topic)
        response.setdefault("summary", "")
        response.setdefault("key_points", [])
        response.setdefault("different_perspectives", [])
        response.setdefault("controversies", [])
        response.setdefault("existential_implications", "")
        response.setdefault("connections_to_other_fields", [])
        response.setdefault("further_exploration", [])
        response.setdefault("personal_reflection", "")

        return response

    def _build_proxy_handler(self):
        """Build urllib proxy handler from config"""
        import urllib.request

        proxy_url = self.config.web.https_proxy or self.config.web.http_proxy
        if proxy_url:
            proxy_handler = urllib.request.ProxyHandler({
                'http': proxy_url,
                'https': proxy_url,
            })
            opener = urllib.request.build_opener(proxy_handler)
            return opener
        return None

    def _search_tavily(self, query: str) -> Optional[str]:
        """Search the web using Tavily API"""
        try:
            import urllib.request

            api_key = self.config.web.tavily_api_key
            if not api_key:
                return None

            payload = json.dumps({
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            opener = self._build_proxy_handler()
            urlopen = opener.open if opener else urllib.request.urlopen
            with urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode())

            results = data.get("results", [])
            if not results:
                return None

            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")
                formatted.append(f"{i}. {title}\n   URL: {url}\n   {content}")

            return "\n\n".join(formatted)

        except Exception as e:
            logger.debug(f"Tavily search failed: {e}")
            return None

    def _search_duckduckgo(self, query: str) -> Optional[str]:
        """Search the web using DuckDuckGo Instant Answer API"""
        try:
            import urllib.request
            import urllib.parse

            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={"User-Agent": "OpenWill/0.1"})
            opener = self._build_proxy_handler()
            urlopen = opener.open if opener else urllib.request.urlopen
            with urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                if data.get("Abstract"):
                    return data["Abstract"]
                if data.get("RelatedTopics"):
                    topics = data["RelatedTopics"][:3]
                    return "\n".join([
                        t.get("Text", "") for t in topics if isinstance(t, dict) and "Text" in t
                    ])
        except Exception as e:
            logger.debug(f"DuckDuckGo search failed: {e}")

        return None

    def search_web(self, query: str) -> Optional[str]:
        """
        Search the web using configured provider with fallback

        Provider logic:
        - "tavily": use Tavily only
        - "duckduckgo": use DuckDuckGo only
        - "auto" (default): try Tavily first (if API key available), fallback to DuckDuckGo
        """
        provider = getattr(self.config, "web", None)
        search_provider = provider.search_provider if provider else "auto"
        tavily_api_key = provider.tavily_api_key if provider else None

        # Try Tavily first when configured or in auto mode with key available
        if search_provider in ("tavily", "auto") and tavily_api_key:
            result = self._search_tavily(query)
            if result:
                logger.info(f"Web search used Tavily for: {query}")
                return result
            elif search_provider == "tavily":
                logger.warning(f"Tavily search failed and no fallback configured for: {query}")
                return None
            # auto mode: fall through to DuckDuckGo
            logger.info(f"Tavily search returned no results, falling back to DuckDuckGo for: {query}")

        # Use DuckDuckGo as primary or fallback
        if search_provider in ("duckduckgo", "auto"):
            result = self._search_duckduckgo(query)
            if result:
                logger.info(f"Web search used DuckDuckGo for: {query}")
                return result

        return None
