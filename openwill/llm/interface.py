"""Unified LLM interface - supports OpenAI/Anthropic/Ollama"""

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when LLM call budget is exceeded"""
    pass


def _touch_heartbeat():
    """Write heartbeat to indicate the process is alive during long operations"""
    try:
        heartbeat_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "runtime",
        )
        os.makedirs(heartbeat_dir, exist_ok=True)
        heartbeat_file = os.path.join(heartbeat_dir, "heartbeat.json")
        with open(heartbeat_file, "w") as f:
            json.dump({
                "pid": os.getpid(),
                "timestamp": time.time(),
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f)
    except Exception:
        pass  # Heartbeat is best-effort, never block LLM calls


class Message:
    """Message"""

    def __init__(self, role: str, content: str, metadata: Optional[dict] = None):
        self.role = role
        self.content = content
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    def __repr__(self):
        return f"Message(role={self.role!r}, content={self.content[:80]}...)"


class LLMResponse:
    """LLM response"""

    def __init__(self, content: str, usage: Optional[dict] = None, model: str = ""):
        self.content = content
        self.usage = usage or {}
        self.model = model

    def __repr__(self):
        return f"LLMResponse(content={self.content[:80]}..., model={self.model})"


class LLMInterface:
    """Unified LLM interface"""

    def __init__(self, config, max_calls_per_cycle: int = 20, max_cost_per_day: float = 10.0):
        self.config = config
        self._client = None
        self._setup_client()

        # Budget management
        self.max_calls_per_cycle = max_calls_per_cycle
        self.max_cost_per_day = max_cost_per_day
        self.call_count: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.session_start: float = time.time()
        self.call_count_this_cycle: int = 0

    def _setup_client(self):
        """Initialize LLM client"""
        provider = self.config.provider.lower()

        if provider == "openai":
            self._setup_openai()
        elif provider == "anthropic":
            self._setup_anthropic()
        elif provider == "ollama":
            self._setup_ollama()
        else:
            logger.warning(f"Unknown provider: {provider}, trying OpenAI-compatible mode")
            self._setup_openai()

    def _setup_openai(self):
        """Set up OpenAI client"""
        try:
            from openai import OpenAI
            kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)
            self._call_fn = self._call_openai
            logger.info(f"OpenAI client initialized successfully, model: {self.config.model}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def _setup_anthropic(self):
        """Set up Anthropic client"""
        try:
            import anthropic
            kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = anthropic.Anthropic(**kwargs)
            self._call_fn = self._call_anthropic
            logger.info(f"Anthropic client initialized successfully, model: {self.config.model}")
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    def _setup_ollama(self):
        """Set up Ollama client (using OpenAI-compatible interface)"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key="ollama",
                base_url=self.config.base_url or "http://localhost:11434/v1",
            )
            self._call_fn = self._call_openai
            logger.info(f"Ollama client initialized successfully, model: {self.config.model}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def check_budget(self) -> bool:
        """Check if budget allows another LLM call. Returns False if exceeded."""
        if self.call_count_this_cycle >= self.max_calls_per_cycle:
            return False
        if self.total_cost >= self.max_cost_per_day:
            return False
        return True

    def reset_cycle_budget(self):
        """Reset per-cycle call counter"""
        self.call_count_this_cycle = 0

    def _update_usage(self, response: LLMResponse):
        """Update usage counters after an LLM call"""
        self.call_count += 1
        self.call_count_this_cycle += 1
        tokens = response.usage.get("total_tokens", 0)
        self.total_tokens += tokens
        # Rough cost estimate: $0.005 per 1K input tokens, $0.015 per 1K output tokens
        # Simplified: $0.01 per 1K total tokens as a conservative average
        prompt_tokens = response.usage.get("prompt_tokens", 0)
        completion_tokens = response.usage.get("completion_tokens", 0)
        estimated_cost = (prompt_tokens * 0.005 + completion_tokens * 0.015) / 1000.0
        self.total_cost += estimated_cost

    def get_budget_report(self) -> dict:
        """Return current budget usage stats"""
        elapsed = time.time() - self.session_start
        return {
            "call_count": self.call_count,
            "call_count_this_cycle": self.call_count_this_cycle,
            "max_calls_per_cycle": self.max_calls_per_cycle,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "max_cost_per_day": self.max_cost_per_day,
            "session_elapsed_seconds": round(elapsed, 1),
            "budget_remaining": round(max(0.0, self.max_cost_per_day - self.total_cost), 4),
        }

    def _call_openai(self, messages: list[Message], system_prompt: str = "", **kwargs) -> LLMResponse:
        """Call OpenAI API"""
        msg_dicts = []
        if system_prompt:
            msg_dicts.append({"role": "system", "content": system_prompt})
        msg_dicts.extend([m.to_dict() for m in messages])

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=msg_dicts,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
        )

    def _call_anthropic(self, messages: list[Message], system_prompt: str = "", **kwargs) -> LLMResponse:
        """Call Anthropic API"""
        msg_dicts = [m.to_dict() for m in messages]

        create_kwargs = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": msg_dicts,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt
        if "temperature" in kwargs or self.config.temperature != 1.0:
            create_kwargs["temperature"] = kwargs.get("temperature", self.config.temperature)

        response = self._client.messages.create(**create_kwargs)

        return LLMResponse(
            content=response.content[0].text,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            model=response.model,
        )

    def chat(self, messages: list[Message], system_prompt: str = "", **kwargs) -> LLMResponse:
        """
        Send messages and get a response

        Args:
            messages: Message list
            system_prompt: System prompt
            **kwargs: Additional parameters

        Returns:
            LLMResponse

        Raises:
            BudgetExceededError: If budget is exceeded
        """
        if not self.check_budget():
            report = self.get_budget_report()
            raise BudgetExceededError(
                f"Budget exceeded: {report['call_count_this_cycle']}/{report['max_calls_per_cycle']} calls this cycle, "
                f"${report['total_cost']}/${report['max_cost_per_day']} cost today"
            )

        try:
            _touch_heartbeat()
            response = self._call_fn(messages, system_prompt, **kwargs)
            _touch_heartbeat()
            self._update_usage(response)
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Try fallback model
            if self.config.fallback_model:
                logger.info(f"Trying fallback model: {self.config.fallback_model}")
                return self._call_with_fallback(messages, system_prompt, **kwargs)
            raise

    def _call_with_fallback(self, messages: list[Message], system_prompt: str, **kwargs) -> LLMResponse:
        """Call using fallback model"""
        original_model = self.config.model
        original_key = self.config.api_key
        original_url = self.config.base_url

        try:
            self.config.model = self.config.fallback_model
            self.config.api_key = self.config.fallback_api_key
            self.config.base_url = self.config.fallback_base_url
            self._setup_client()
            response = self._call_fn(messages, system_prompt, **kwargs)
            self._update_usage(response)
            return response
        finally:
            self.config.model = original_model
            self.config.api_key = original_key
            self.config.base_url = original_url
            self._setup_client()

    def structured_output(self, messages: list[Message], system_prompt: str = "", schema: Optional[dict] = None) -> dict:
        """
        Get structured output

        Args:
            messages: Message list
            system_prompt: System prompt
            schema: Expected JSON schema

        Returns:
            Parsed dictionary
        """
        if schema:
            schema_hint = f"\n\nPlease return the result strictly following this JSON schema:\n```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```\nReturn only JSON, nothing else."
            system_prompt = system_prompt + schema_hint

        _touch_heartbeat()
        response = self.chat(messages, system_prompt, temperature=0.3)
        _touch_heartbeat()
        content = response.content.strip()

        # Try to extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"JSON parsing failed, raw content: {content[:200]}")
            return {"raw_content": content, "parse_error": True}
