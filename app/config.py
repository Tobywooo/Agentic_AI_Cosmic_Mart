"""Application settings, loaded from environment variables and the project's .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment variable among `names`."""
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    # LLM. provider "openai": any OpenAI-compatible Chat Completions API (llama.cpp, OpenAI, proxies).
    #      provider "anthropic": the Anthropic Messages API (Claude), directly or via a proxy.
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_base_url: str = "http://localhost:8080/v1"  # "" = the SDK's default endpoint
    # Anthropic only: "api_key" sends the x-api-key header; "bearer" sends Authorization: Bearer <key>.
    llm_auth_style: str = "api_key"
    llm_model: str = "local-model"
    # Anthropic only: tried in order when the gateway reports LLM_MODEL as unavailable.
    llm_fallback_models: tuple[str, ...] = ()
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2000
    llm_timeout_seconds: float = 120.0
    # Ask the server to constrain output to JSON (llama.cpp turns this into a grammar).
    llm_json_mode: bool = True

    # Business rules / agent authority
    currency: str = "USD"
    return_authority_limit: float = 150.0
    return_window_days: int = 30
    non_returnable_categories: tuple[str, ...] = ("gift_card", "perishable", "personalised")
    pickup_days_ahead: int = 7
    max_verification_attempts: int = 3

    # Agent behaviour
    frustration_threshold: float = 0.6
    max_agent_steps: int = 8
    history_messages: int = 20

    # Storage
    memory_max_rows: int = 50
    memory_path: Path = PROJECT_ROOT / "runtime" / "memory.xlsx"
    state_path: Path = PROJECT_ROOT / "runtime" / "state.json"
    seed_path: Path = PROJECT_ROOT / "data" / "seed_data.json"

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = ("*",)
    follow_up_poll_seconds: int = 30
    # Open cases with nothing outstanding are resolved after this much inactivity (0 disables).
    auto_resolve_hours: float = 72.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        d = cls()
        categories = _env("NON_RETURNABLE_CATEGORIES")
        origins = _env("CORS_ORIGINS")
        provider = (_env("LLM_PROVIDER", default=d.llm_provider) or "").strip().lower()
        if provider not in ("openai", "anthropic"):
            raise ValueError(f"LLM_PROVIDER must be 'openai' or 'anthropic', got '{provider}'.")
        # Generic LLM_* names win; the provider-specific names are accepted as fallbacks.
        if provider == "anthropic":
            api_key = _env("LLM_API_KEY", "ANTHROPIC_API_KEY", default="")
            base_url = _env("LLM_BASE_URL", "ANTHROPIC_BASE_URL", default="")
        else:
            api_key = _env("LLM_API_KEY", "OPENAI_API_KEY", default="")
            base_url = _env("LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", default=d.llm_base_url)
        auth_style = (_env("LLM_AUTH_STYLE", default=d.llm_auth_style) or "").strip().lower()
        if auth_style not in ("api_key", "bearer"):
            raise ValueError(f"LLM_AUTH_STYLE must be 'api_key' or 'bearer', got '{auth_style}'.")
        return cls(
            llm_provider=provider,
            llm_api_key=api_key,
            llm_base_url=base_url,
            llm_auth_style=auth_style,
            llm_model=_env("LLM_MODEL", default=d.llm_model),
            llm_fallback_models=tuple(m.strip() for m in (_env("LLM_FALLBACK_MODELS") or "").split(",") if m.strip()),
            llm_temperature=float(_env("LLM_TEMPERATURE", default=str(d.llm_temperature))),
            llm_max_tokens=int(_env("LLM_MAX_TOKENS", default=str(d.llm_max_tokens))),
            llm_timeout_seconds=float(_env("LLM_TIMEOUT_SECONDS", default=str(d.llm_timeout_seconds))),
            llm_json_mode=_bool(_env("LLM_JSON_MODE"), d.llm_json_mode),
            currency=_env("CURRENCY", default=d.currency),
            return_authority_limit=float(_env("RETURN_AUTHORITY_LIMIT", default=str(d.return_authority_limit))),
            return_window_days=int(_env("RETURN_WINDOW_DAYS", default=str(d.return_window_days))),
            non_returnable_categories=(
                tuple(c.strip() for c in categories.split(",") if c.strip())
                if categories
                else d.non_returnable_categories
            ),
            pickup_days_ahead=int(_env("PICKUP_DAYS_AHEAD", default=str(d.pickup_days_ahead))),
            max_verification_attempts=int(
                _env("MAX_VERIFICATION_ATTEMPTS", default=str(d.max_verification_attempts))
            ),
            frustration_threshold=float(_env("FRUSTRATION_THRESHOLD", default=str(d.frustration_threshold))),
            max_agent_steps=int(_env("MAX_AGENT_STEPS", default=str(d.max_agent_steps))),
            history_messages=int(_env("HISTORY_MESSAGES", default=str(d.history_messages))),
            memory_max_rows=int(_env("MEMORY_MAX_ROWS", default=str(d.memory_max_rows))),
            memory_path=_path(_env("MEMORY_PATH"), d.memory_path),
            state_path=_path(_env("STATE_PATH"), d.state_path),
            seed_path=_path(_env("SEED_PATH"), d.seed_path),
            host=_env("HOST", default=d.host),
            port=int(_env("PORT", default=str(d.port))),
            cors_origins=(
                tuple(o.strip() for o in origins.split(",") if o.strip()) if origins else d.cors_origins
            ),
            follow_up_poll_seconds=int(_env("FOLLOW_UP_POLL_SECONDS", default=str(d.follow_up_poll_seconds))),
            auto_resolve_hours=float(_env("AUTO_RESOLVE_HOURS", default=str(d.auto_resolve_hours))),
        )
