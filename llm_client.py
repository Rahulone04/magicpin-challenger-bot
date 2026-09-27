"""
llm_client.py — thin, dependency-light LLM wrapper for the composer.

Design goals:
  * Zero required dependencies (uses `urllib` from the stdlib, not `requests`,
    so the bot still boots in a bare container).
  * Deterministic: every call is made with temperature=0 as the challenge
    requires ("Must be deterministic given the same inputs").
  * Provider-agnostic entrypoint: `get_llm()` picks a provider from env vars
    so the same code runs locally (no key -> heuristic fallback in
    composer.py) and in whatever the grading environment provides.
  * Fails soft: any network/parse error raises `LLMError`, which composer.py
    catches and falls back to the deterministic heuristic composer. The bot
    must never 500 or hang because an LLM call failed.

Supported providers (set LLM_PROVIDER, default "anthropic"):
  - "anthropic": needs ANTHROPIC_API_KEY. Model: LLM_MODEL or
    "claude-sonnet-4-5-20250929".
  - "openai": needs OPENAI_API_KEY. Model: LLM_MODEL or "gpt-4o-mini".
  - "none": always raises LLMError immediately -> composer uses the
    heuristic engine. Useful for local dev / judge_simulator.py without
    burning API credits, and it's what runs if no key is configured.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional

TIMEOUT_S = 25  # keep comfortably under the judge's 30s/call budget


class LLMError(Exception):
    """Raised whenever the LLM path can't be used; caller should fall back."""


class LLMClient:
    def complete_json(self, system: str, user: str) -> dict:
        """Send system+user prompt, return parsed JSON dict from the reply.

        Raises LLMError on any failure (network, non-200, unparseable JSON).
        """
        raise NotImplementedError


class NoneProvider(LLMClient):
    def complete_json(self, system: str, user: str) -> dict:
        raise LLMError("no LLM provider configured (LLM_PROVIDER=none)")


class AnthropicProvider(LLMClient):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete_json(self, system: str, user: str) -> dict:
        body = json.dumps({
            "model": self.model,
            "max_tokens": 800,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"anthropic request failed: {e}") from e
        try:
            text = data["content"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected anthropic response shape: {data}") from e
        return _extract_json(text)


class OpenAIProvider(LLMClient):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete_json(self, system: str, user: str) -> dict:
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMError(f"openai request failed: {e}") from e
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected openai response shape: {data}") from e
        return _extract_json(text)


def _extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of a model's text reply."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMError(f"no JSON object found in LLM reply: {text[:200]!r}")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"could not parse JSON from LLM reply: {e}") from e


def get_llm() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "none":
        return NoneProvider()
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return NoneProvider()
        model = os.environ.get("LLM_MODEL", "claude-sonnet-4-5-20250929")
        return AnthropicProvider(key, model)
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return NoneProvider()
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        return OpenAIProvider(key, model)
    return NoneProvider()
