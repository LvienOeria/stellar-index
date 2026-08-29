from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import Settings


@dataclass
class LLMReply:
    content: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None


@dataclass
class LLMJSONReply:
    data: dict
    input_tokens: int
    output_tokens: int


class DeepSeekClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key or "not-set",
            base_url=settings.base_url or "https://api.deepseek.com",
            timeout=300,
            max_retries=3,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMReply:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LLMReply(
            content=choice.message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=choice.finish_reason,
        )

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        return self.complete_json_reply(messages, **kwargs).data

    def complete_json_reply(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMJSONReply:
        reply = self.complete(messages, json_mode=True, **kwargs)
        text = reply.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"_raw": text}
        return LLMJSONReply(data=data, input_tokens=reply.input_tokens, output_tokens=reply.output_tokens)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model.startswith("deepseek-v4-pro"):
        return input_tokens / 1_000_000 * 0.66 + output_tokens / 1_000_000 * 1.98
    return input_tokens / 1_000_000 * 0.22 + output_tokens / 1_000_000 * 0.66
