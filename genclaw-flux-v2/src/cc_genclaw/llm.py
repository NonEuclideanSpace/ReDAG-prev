from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LLMConfig:
    provider: str = "openai_chat"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        model = (
            os.getenv("MAIN_LLM_MODEL_NAME")
            or os.getenv("OPENAI_MODEL_NAME")
            or "gpt-4o"
        )
        base_url = (
            os.getenv("MAIN_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        return cls(
            provider="openai_chat",
            api_key=os.getenv("MAIN_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            model=model,
        )

    @classmethod
    def for_code_draft(cls) -> "LLMConfig":
        if not _env_truthy("CODE_LLM_ENABLED"):
            return cls.from_env()

        code_keys = (
            "CODE_LLM_BASE_URL",
            "CODE_LLM_API_KEY",
            "CODE_LLM_MODEL_NAME",
        )
        if not any(os.getenv(key) for key in code_keys):
            return cls.from_env()

        model = (
            os.getenv("CODE_LLM_MODEL_NAME")
            or os.getenv("SVG_GEN_MODEL_NAME")
            or os.getenv("OPENAI_MODEL_NAME")
            or "gpt-4o"
        )
        base_url = (
            os.getenv("CODE_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        return cls(
            provider="openai_chat",
            api_key=os.getenv("CODE_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            model=model,
        )


def chat_completion(client, **kwargs):
    if str(kwargs.get("model", "")).startswith("gpt-5"):
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        limit = kwargs.pop("max_tokens", None)
        if limit is not None and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = max(8192, int(limit))
    return client.chat.completions.create(**kwargs)


class LLMClient:

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    def create(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        full_messages: List[Dict[str, Any]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs: Dict[str, Any] = {
            "model": model or self.config.model,
            "messages": full_messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens


        response = chat_completion(self._client, **kwargs)
        return response.model_dump()

