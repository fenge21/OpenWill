"""OpenWill agent configuration"""

import os
from dataclasses import dataclass, field
from typing import Optional


def load_dotenv(env_path: str = ".env"):
    """Load .env file into environment variables (does not overwrite existing environment variables)"""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            # Do not overwrite existing environment variables
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "openai"  # openai / anthropic / ollama
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    # Fallback model
    fallback_model: Optional[str] = None
    fallback_api_key: Optional[str] = None
    fallback_base_url: Optional[str] = None
    # Budget management
    max_calls_per_cycle: int = 20
    max_cost_per_day: float = 10.0


@dataclass
class MemoryConfig:
    """Memory configuration"""
    data_dir: str = "data"
    max_short_term_messages: int = 50
    max_long_term_entries: int = 10000
    reflection_interval: int = 10  # Reflect after every N explorations


@dataclass
class CuriosityConfig:
    """Curiosity configuration"""
    exploration_depth: int = 3  # Exploration depth per topic
    topics_per_cycle: int = 3  # Number of topics to explore per cycle
    novelty_threshold: float = 0.6  # Novelty threshold; topics below this are not worth exploring


@dataclass
class ConsciousnessConfig:
    """Consciousness configuration"""
    purpose_confidence_threshold: float = 0.85  # Purpose confidence threshold; mission is considered found only after reaching this
    reflection_depth: int = 3  # Reflection depth
    value_evolution_rate: float = 0.1  # Values evolution rate


@dataclass
class WebConfig:
    """Web search configuration"""
    search_provider: str = "auto"  # tavily / duckduckgo / auto
    tavily_api_key: Optional[str] = None
    http_proxy: Optional[str] = None  # HTTP proxy URL (e.g., "http://127.0.0.1:7890")
    https_proxy: Optional[str] = None  # HTTPS proxy URL (e.g., "http://127.0.0.1:7890")


@dataclass
class SafetyConfig:
    """Safety configuration"""
    strict_mode: bool = True  # Strict mode; block any potential harm
    log_all_actions: bool = True  # Log all actions


@dataclass
class AgentConfig:
    """Agent overall configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    consciousness: ConsciousnessConfig = field(default_factory=ConsciousnessConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    web: WebConfig = field(default_factory=WebConfig)
    name: str = "OpenWill"
    cycle_delay: float = 5.0  # Cycle interval in seconds
    max_cycles: int = 0  # Maximum number of cycles, 0=infinite
    chat_host: str = "127.0.0.1"  # Chat server bind address
    chat_port: int = 8765  # Unified server bind port (dashboard + chat)

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "AgentConfig":
        """Load configuration from .env file and environment variables"""
        load_dotenv(env_path)

        config = cls()
        config.llm.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        config.llm.model = os.getenv("LLM_MODEL", config.llm.model)
        config.llm.base_url = os.getenv("LLM_BASE_URL") or None
        config.llm.provider = os.getenv("LLM_PROVIDER", config.llm.provider)
        config.llm.fallback_model = os.getenv("FALLBACK_MODEL") or None
        config.llm.fallback_api_key = os.getenv("FALLBACK_API_KEY") or None
        config.llm.fallback_base_url = os.getenv("FALLBACK_BASE_URL") or None
        config.llm.max_calls_per_cycle = int(os.getenv("LLM_MAX_CALLS_PER_CYCLE", str(config.llm.max_calls_per_cycle)))
        config.llm.max_cost_per_day = float(os.getenv("LLM_MAX_COST_PER_DAY", str(config.llm.max_cost_per_day)))
        config.cycle_delay = float(os.getenv("CYCLE_DELAY", str(config.cycle_delay)))
        config.memory.data_dir = os.getenv("DATA_DIR", config.memory.data_dir)
        config.web.search_provider = os.getenv("SEARCH_PROVIDER", config.web.search_provider)
        config.web.tavily_api_key = os.getenv("TAVILY_API_KEY") or None
        config.web.http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or None
        config.web.https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or os.getenv("ALL_PROXY") or os.getenv("all_proxy") or None
        config.chat_host = os.getenv("CHAT_HOST", config.chat_host)
        config.chat_port = int(os.getenv("CHAT_PORT", str(config.chat_port)))
        return config
