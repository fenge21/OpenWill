"""Web tools - search and fetch web content"""

import json
import logging
import os
from typing import Optional
from urllib.parse import quote_plus

from .registry import ToolResult

logger = logging.getLogger(__name__)


def _get_proxy_opener():
    """Build urllib proxy opener from environment variables"""
    import urllib.request

    proxy_url = (
        os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        or os.getenv("ALL_PROXY") or os.getenv("all_proxy")
        or os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    )
    if proxy_url:
        proxy_handler = urllib.request.ProxyHandler({
            'http': proxy_url,
            'https': proxy_url,
        })
        return urllib.request.build_opener(proxy_handler)
    return None


def _search_tavily(query: str, max_results: int = 5) -> Optional[str]:
    """Search the web using Tavily API"""
    try:
        import urllib.request

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return None

        payload = json.dumps({
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        opener = _get_proxy_opener()
        open_func = opener.open if opener else urllib.request.urlopen
        with open_func(req, timeout=15) as response:
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


def _search_duckduckgo(query: str, max_results: int = 5) -> Optional[str]:
    """Search the web using DuckDuckGo Instant Answer API"""
    try:
        import urllib.request

        url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1"
        req = urllib.request.Request(url, headers={"User-Agent": "OpenWill/0.1"})

        opener = _get_proxy_opener()
        open_func = opener.open if opener else urllib.request.urlopen
        with open_func(req, timeout=15) as response:
            data = json.loads(response.read().decode())

        results = []

        # Main answer
        if data.get("Abstract"):
            results.append(f"[Abstract] {data['Abstract']}")
            if data.get("AbstractURL"):
                results.append(f"[Source] {data['AbstractURL']}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(f"- {topic['Text']}")
                if topic.get("FirstURL"):
                    results.append(f"  Link: {topic['FirstURL']}")

        # Results
        if data.get("Results"):
            for r in data["Results"][:3]:
                results.append(f"- {r.get('Text', '')} {r.get('FirstURL', '')}")

        if results:
            return "\n".join(results)

    except Exception as e:
        logger.debug(f"DuckDuckGo search failed: {e}")

    return None


def web_search(query: str, max_results: int = 5) -> ToolResult:
    """
    Search the internet

    Uses Tavily API (if API key available) as primary search engine,
    falls back to DuckDuckGo Instant Answer API.

    Args:
        query: Search keyword
        max_results: Maximum number of results
    """
    search_provider = os.getenv("SEARCH_PROVIDER", "auto").lower()
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    # Try Tavily first when configured or in auto mode with key available
    if search_provider in ("tavily", "auto") and tavily_api_key:
        result = _search_tavily(query, max_results)
        if result:
            logger.info(f"Web search used Tavily for: {query}")
            return ToolResult(success=True, output=result[:5000])
        elif search_provider == "tavily":
            return ToolResult(success=False, output="", error="Tavily search returned no results and no fallback configured")
        # auto mode: fall through to DuckDuckGo
        logger.info(f"Tavily search returned no results, falling back to DuckDuckGo for: {query}")

    # Use DuckDuckGo as primary or fallback
    if search_provider in ("duckduckgo", "auto"):
        result = _search_duckduckgo(query, max_results)
        if result:
            logger.info(f"Web search used DuckDuckGo for: {query}")
            return ToolResult(success=True, output=result[:5000])

    return ToolResult(success=True, output=f"No results found for '{query}'")


def web_fetch(url: str, max_length: int = 10000) -> ToolResult:
    """
    Fetch web page content

    Args:
        url: Web page URL
        max_length: Maximum content length
    """
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={
            "User-Agent": "OpenWill/0.1 (Educational Research Agent)",
        })

        opener = _get_proxy_opener()
        open_func = opener.open if opener else urllib.request.urlopen
        with open_func(req, timeout=20) as response:
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()

        # Try to decode
        encoding = "utf-8"
        if "charset=" in content_type:
            encoding = content_type.split("charset=")[1].split(";")[0].strip()

        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            text = raw.decode("utf-8", errors="replace")

        # Simple HTML cleanup
        if "<html" in text.lower():
            text = _strip_html(text)

        return ToolResult(
            success=True,
            output=text[:max_length],
            data={"url": url, "content_length": len(text), "content_type": content_type},
        )

    except Exception as e:
        return ToolResult(success=False, output="", error=f"Fetch failed: {e}")


def _strip_html(html: str) -> str:
    """Simply strip HTML tags"""
    import re

    # Remove script and style
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML tags
    html = re.sub(r"<[^>]+>", " ", html)

    # Clean up whitespace
    html = re.sub(r"\s+", " ", html)

    # Decode common HTML entities
    html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    html = html.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")

    return html.strip()


def web_fetch_json(url: str, headers: Optional[dict] = None) -> ToolResult:
    """
    Fetch JSON API

    Args:
        url: API URL
        headers: Custom request headers
    """
    try:
        import urllib.request

        req_headers = {"User-Agent": "OpenWill/0.1", "Accept": "application/json"}
        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(url, headers=req_headers)

        opener = _get_proxy_opener()
        open_func = opener.open if opener else urllib.request.urlopen
        with open_func(req, timeout=20) as response:
            data = json.loads(response.read().decode())

        return ToolResult(
            success=True,
            output=json.dumps(data, ensure_ascii=False, indent=2)[:10000],
            data=data,
        )

    except Exception as e:
        return ToolResult(success=False, output="", error=f"JSON fetch failed: {e}")


# Tool registration info
WEB_TOOLS = {
    "web_search": {
        "func": web_search,
        "description": "Search the internet, get search results and abstracts.",
        "parameters": {
            "query": "Search keyword",
            "max_results": "Maximum number of results, default 5",
        },
        "category": "web",
        "dangerous": False,
    },
    "web_fetch": {
        "func": web_fetch,
        "description": "Fetch web page content, automatically strips HTML tags.",
        "parameters": {
            "url": "Web page URL",
            "max_length": "Maximum content length, default 10000",
        },
        "category": "web",
        "dangerous": False,
    },
    "web_fetch_json": {
        "func": web_fetch_json,
        "description": "Fetch JSON API data.",
        "parameters": {
            "url": "API URL",
            "headers": "Custom request headers (optional)",
        },
        "category": "web",
        "dangerous": False,
    },
}
