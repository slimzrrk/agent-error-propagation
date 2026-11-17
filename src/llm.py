"""Thin wrapper around any OpenAI-compatible chat endpoint (Ollama, mlx_lm.server, vLLM)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.request
from dataclasses import asdict, dataclass

import openai
from openai import OpenAI

log = logging.getLogger(__name__)

# Errors worth retrying: the server is busy, restarting or briefly unreachable.
RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:1.5b-instruct-fp16"
    api_key: str = "ollama"  # ignored by local servers, but the client requires one
    temperature: float = 0.7
    max_tokens: int = 768
    timeout: float = 180.0
    max_retries: int = 4
    retry_backoff: float = 2.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "LLMConfig":
        d = d or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown llm config keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("api_key")
        return d


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency: float  # seconds, including retries
    finish_reason: str | None = None
    attempts: int = 1


def derive_seed(*parts: object, base: int = 0) -> int:
    """Stable 31-bit seed from arbitrary parts (Python's hash() is salted per process)."""
    key = ":".join(str(p) for p in (base, *parts))
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


class LLMClient:
    def __init__(self, config: LLMConfig, client: OpenAI | None = None):
        self.config = config
        # Retries are handled here so that latency and attempts are measured consistently.
        self._client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=0,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        cfg = self.config
        params = dict(
            model=cfg.model,
            messages=messages,
            temperature=cfg.temperature if temperature is None else temperature,
            max_tokens=max_tokens or cfg.max_tokens,
        )
        if seed is not None:
            params["seed"] = seed

        start = time.perf_counter()
        for attempt in range(1, cfg.max_retries + 2):
            try:
                resp = self._client.chat.completions.create(**params)
                break
            except RETRYABLE_ERRORS as exc:
                if attempt > cfg.max_retries:
                    raise
                wait = cfg.retry_backoff * 2 ** (attempt - 1)
                log.warning("LLM call failed (%s), retry %d in %.1fs", type(exc).__name__, attempt, wait)
                time.sleep(wait)

        choice = resp.choices[0]
        usage = resp.usage
        return Completion(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency=time.perf_counter() - start,
            finish_reason=choice.finish_reason,
            attempts=attempt,
        )

    def server_info(self) -> dict:
        """Best-effort backend metadata (Ollama exposes version and model digest)."""
        info: dict = {"model": self.config.model, "base_url": self.config.base_url}
        root = self.config.base_url.rstrip("/").removesuffix("/v1")

        def get(path):
            with urllib.request.urlopen(f"{root}{path}", timeout=5) as r:
                return json.load(r)

        try:
            info["ollama_version"] = get("/api/version").get("version")
            for m in get("/api/tags").get("models", []):
                if m.get("name") == self.config.model:
                    info["model_digest"] = m.get("digest")
                    info["model_details"] = m.get("details")
            for m in get("/api/ps").get("models", []):
                if m.get("name") == self.config.model:
                    info["context_length"] = m.get("context_length")
        except (OSError, ValueError):
            pass  # not an Ollama server
        return info
