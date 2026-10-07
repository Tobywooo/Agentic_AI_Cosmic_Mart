"""Thin wrapper over the OpenAI SDK pointed at an OpenAI-compatible server (llama.cpp `llama-server`)."""

from __future__ import annotations

import logging
from typing import Protocol

import openai
from openai import OpenAI

from .config import Settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class ChatModel(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        # llama.cpp ignores the key unless started with --api-key, but the SDK requires a value.
        self.client = OpenAI(
            api_key=settings.openai_api_key or "no-key",
            base_url=settings.openai_base_url,
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
            raise LLMUnavailable(f"LLM request to {self.settings.openai_base_url} failed: {exc}") from exc
        return response.choices[0].message.content or ""

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self.client.models.list().data]
        except openai.APIError as exc:
            raise LLMUnavailable(str(exc)) from exc
