"""LLM clients. Both take OpenAI-style `messages` and return the model's text, so the agent loop
(prompt-based JSON tool protocol) is identical whichever provider is configured.

  LLM_PROVIDER=openai     -> OpenAI SDK, any OpenAI-compatible Chat Completions API (e.g. llama.cpp)
  LLM_PROVIDER=anthropic  -> Anthropic SDK, the Claude Messages API (directly or through a proxy)
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import anthropic
import openai
from anthropic import Anthropic
from openai import OpenAI

from .config import Settings

log = logging.getLogger(__name__)

TRANSIENT_ATTEMPTS = 3
MODEL_SKIP_SECONDS = 60


def _model_unavailable(exc: Exception) -> bool:
    return "not available for your organization" in str(exc)


class LLMUnavailable(RuntimeError):
    pass


class ChatModel(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


def create_llm(settings: Settings) -> "OpenAICompatibleLLM | AnthropicLLM":
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings)
    return OpenAICompatibleLLM(settings)


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        # llama.cpp ignores the key unless started with --api-key, but the SDK requires a value.
        self.client = OpenAI(
            api_key=settings.llm_api_key or "no-key",
            base_url=settings.llm_base_url or None,
            timeout=settings.llm_timeout_seconds,
            max_retries=1,
        )
        self._json_mode = settings.llm_json_mode

    def complete(self, messages: list[dict]) -> str:
        kwargs = dict(
            model=self.settings.llm_model,
            messages=messages,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        try:
            if self._json_mode:
                try:
                    response = self.client.chat.completions.create(
                        **kwargs, response_format={"type": "json_object"}
                    )
                except openai.BadRequestError as exc:
                    log.warning("Server rejected response_format=json_object (%s); disabling JSON mode.", exc)
                    self._json_mode = False
                    response = self.client.chat.completions.create(**kwargs)
            else:
                response = self.client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise LLMUnavailable(f"LLM request to {self.client.base_url} failed: {exc}") from exc
        choice = response.choices[0]
        if choice.finish_reason == "length":
            log.warning("LLM reply hit LLM_MAX_TOKENS (%s) and was cut off.", self.settings.llm_max_tokens)
        return choice.message.content or ""

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self.client.models.list().data]
        except openai.APIError as exc:
            raise LLMUnavailable(str(exc)) from exc


class AnthropicLLM:
    """Claude via the Messages API. There is no JSON response mode, but Claude follows the JSON
    protocol in the system prompt reliably and the parser tolerates stray text around the object.
    Current Claude models/SDKs do not accept sampling parameters, so LLM_TEMPERATURE is not sent."""

    def __init__(self, settings: Settings):
        self.settings = settings
        base_url = (settings.llm_base_url or "").rstrip("/")
        if base_url.endswith("/v1"):  # the SDK appends /v1/messages itself
            log.info("Stripping trailing /v1 from LLM_BASE_URL for the Anthropic SDK.")
            base_url = base_url[: -len("/v1")]
        credentials = ({"auth_token": settings.llm_api_key, "api_key": None}
                       if settings.llm_auth_style == "bearer" else {"api_key": settings.llm_api_key})
        self.client = Anthropic(
            **credentials,
            base_url=base_url or None,
            timeout=settings.llm_timeout_seconds,
            max_retries=1,
        )
        self.models = [settings.llm_model] + [m for m in settings.llm_fallback_models if m != settings.llm_model]
        self._skip_until: dict[str, float] = {}  # model -> monotonic time it may be tried again

    def complete(self, messages: list[dict]) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [{"role": m["role"], "content": m["content"] or "(empty)"}  # the API rejects empty content
                for m in messages if m["role"] != "system"]
        now = time.monotonic()
        candidates = [m for m in self.models if self._skip_until.get(m, 0) <= now] or self.models
        last_error: Exception | None = None
        for index, model in enumerate(candidates):
            is_last = index == len(candidates) - 1
            try:
                response = self._create(model, system, chat, attempts=TRANSIENT_ATTEMPTS if is_last else 1)
            except anthropic.BadRequestError as exc:
                if not _model_unavailable(exc):
                    raise LLMUnavailable(f"LLM request to {self.client.base_url} failed: {exc}") from exc
                # Vocareum's gateway intermittently reports enabled models as "not available for your
                # organization". Skip this model for a while and fall back to the next one.
                self._skip_until[model] = time.monotonic() + MODEL_SKIP_SECONDS
                last_error = exc
                if not is_last:
                    log.warning("Model '%s' unavailable on the gateway; falling back to '%s'.",
                                model, candidates[index + 1])
                continue
            except anthropic.APIError as exc:
                raise LLMUnavailable(f"LLM request to {self.client.base_url} failed: {exc}") from exc
            if model != self.settings.llm_model:
                log.info("Reply generated by fallback model '%s'.", model)
            if getattr(response, "stop_reason", None) == "max_tokens":
                log.warning("LLM reply hit LLM_MAX_TOKENS (%s) and was cut off.", self.settings.llm_max_tokens)
            return "".join(block.text for block in response.content if block.type == "text")
        raise LLMUnavailable(f"LLM request to {self.client.base_url} failed for model(s) "
                             f"{', '.join(candidates)}: {last_error}") from last_error

    def _create(self, model: str, system: str, chat: list[dict], attempts: int):
        """One model, retrying the gateway's transient "not available" error `attempts` times."""
        for attempt in range(attempts):
            try:
                return self.client.messages.create(model=model, max_tokens=self.settings.llm_max_tokens,
                                                   system=system, messages=chat)
            except anthropic.BadRequestError as exc:
                if not _model_unavailable(exc) or attempt == attempts - 1:
                    raise
                log.warning("Gateway reported model '%s' unavailable (attempt %d); retrying.", model, attempt + 1)
                time.sleep(1.0 * (attempt + 1))

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self.client.models.list().data]
        except (anthropic.NotFoundError, anthropic.PermissionDeniedError, anthropic.BadRequestError):
            # Some proxies don't expose /v1/models (Vocareum answers 400, treating "models" as a model
            # name): prove connectivity and that LLM_MODEL is enabled with a tiny request instead.
            self.complete([{"role": "user", "content": "Reply with OK."}])
            return self.models
        except anthropic.APIError as exc:
            raise LLMUnavailable(str(exc)) from exc
