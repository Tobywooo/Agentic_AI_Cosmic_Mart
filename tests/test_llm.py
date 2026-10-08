from dataclasses import replace
from types import SimpleNamespace

import pytest

import app.config as config
from app.config import Settings
from app.llm import AnthropicLLM, LLMUnavailable, OpenAICompatibleLLM, create_llm

LLM_VARS = ["LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "LLM_AUTH_STYLE", "LLM_MODEL", "LLM_FALLBACK_MODELS",
            "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate from the developer's real .env and shell."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for var in LLM_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_openai_provider_keeps_legacy_variable_names(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "old-key")
    clean_env.setenv("OPENAI_BASE_URL", "http://llama:8080/v1")
    s = Settings.from_env()
    assert (s.llm_provider, s.llm_api_key, s.llm_base_url) == ("openai", "old-key", "http://llama:8080/v1")
    assert isinstance(create_llm(s), OpenAICompatibleLLM)


def test_anthropic_provider_prefers_generic_names(clean_env):
    clean_env.setenv("LLM_PROVIDER", "Anthropic")
    clean_env.setenv("ANTHROPIC_API_KEY", "fallback-key")
    clean_env.setenv("LLM_API_KEY", "generic-key")
    s = Settings.from_env()
    assert (s.llm_provider, s.llm_api_key, s.llm_base_url) == ("anthropic", "generic-key", "")
    assert isinstance(create_llm(s), AnthropicLLM)


def test_invalid_provider_is_rejected(clean_env):
    clean_env.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings.from_env()


def anthropic_settings(**overrides):
    values = {"llm_provider": "anthropic", "llm_api_key": "k", "llm_model": "claude-sonnet-5-5", **overrides}
    return replace(Settings(), **values)


def fake_response(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def test_anthropic_request_shape_and_url(monkeypatch):
    llm = AnthropicLLM(anthropic_settings(llm_base_url="https://proxy.example.com/v1/"))
    assert str(llm.client.base_url).rstrip("/") == "https://proxy.example.com"  # /v1 stripped
    assert llm.client.api_key == "k"

    calls = []
    monkeypatch.setattr(llm.client.messages, "create",
                        lambda **kw: calls.append(kw) or fake_response('{"final_answer": "Hi"}'))
    out = llm.complete([
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "again"},
    ])
    assert out == '{"final_answer": "Hi"}'
    kw = calls[0]
    assert kw["system"] == "SYSTEM PROMPT" and kw["model"] == "claude-sonnet-5-5"
    assert [m["role"] for m in kw["messages"]] == ["user", "assistant", "user"]
    assert kw["messages"][1]["content"] == "(empty)"  # the API rejects empty content


def test_anthropic_bearer_auth_style():
    llm = AnthropicLLM(anthropic_settings(llm_auth_style="bearer"))
    assert llm.client.auth_token == "k"


def test_anthropic_only_sends_arguments_the_sdk_accepts():
    """Guards against passing parameters (e.g. temperature) that the installed SDK has dropped."""
    import inspect

    accepted = set(inspect.signature(AnthropicLLM(anthropic_settings()).client.messages.create).parameters)
    assert {"model", "max_tokens", "system", "messages"} <= accepted


def test_anthropic_connection_error_becomes_llm_unavailable():
    llm = AnthropicLLM(anthropic_settings(llm_base_url="http://127.0.0.1:9", llm_timeout_seconds=2))
    with pytest.raises(LLMUnavailable):
        llm.complete([{"role": "user", "content": "x"}])


def _not_available_error():
    import anthropic
    import httpx

    response = httpx.Response(400, request=httpx.Request("POST", "https://gateway.example.com/v1/messages"))
    return anthropic.BadRequestError("Model 'x' is not available for your organization.", response=response,
                                     body=None)


def test_anthropic_retries_transient_model_not_available(monkeypatch):
    import app.llm as llm_module

    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    llm = AnthropicLLM(anthropic_settings())
    attempts = []

    def create(**kw):
        attempts.append(kw)
        if len(attempts) == 1:
            raise _not_available_error()
        return fake_response("ok")

    monkeypatch.setattr(llm.client.messages, "create", create)
    assert llm.complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(attempts) == 2


def test_anthropic_gives_up_after_repeated_model_not_available(monkeypatch):
    import app.llm as llm_module

    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    llm = AnthropicLLM(anthropic_settings())
    attempts = []

    def create(**kw):
        attempts.append(kw)
        raise _not_available_error()

    monkeypatch.setattr(llm.client.messages, "create", create)
    with pytest.raises(LLMUnavailable, match="not available for your organization"):
        llm.complete([{"role": "user", "content": "x"}])
    assert len(attempts) == llm_module.TRANSIENT_ATTEMPTS


def test_anthropic_falls_back_when_primary_model_unavailable(monkeypatch):
    import app.llm as llm_module

    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    llm = AnthropicLLM(anthropic_settings(llm_model="claude-haiku-4-5", llm_fallback_models=("claude-opus-5-5",)))
    used = []

    def create(**kw):
        used.append(kw["model"])
        if kw["model"] == "claude-haiku-4-5":
            raise _not_available_error()
        return fake_response(f"answered by {kw['model']}")

    monkeypatch.setattr(llm.client.messages, "create", create)
    assert llm.complete([{"role": "user", "content": "x"}]) == "answered by claude-opus-5-5"
    assert used == ["claude-haiku-4-5", "claude-opus-5-5"]  # primary tried once, no slow retries

    # The unavailable primary is skipped for a while, so the next call goes straight to the fallback.
    used.clear()
    llm.complete([{"role": "user", "content": "y"}])
    assert used == ["claude-opus-5-5"]

    # After the skip window, the primary is tried again.
    monkeypatch.setattr(llm_module.time, "monotonic", lambda: 10**9)
    used.clear()
    llm.complete([{"role": "user", "content": "z"}])
    assert used[0] == "claude-haiku-4-5"


def test_fallback_models_setting(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    clean_env.setenv("LLM_MODEL", "claude-haiku-4-5")
    clean_env.setenv("LLM_FALLBACK_MODELS", " claude-opus-5-5 , claude-sonnet-4-5 ")
    s = Settings.from_env()
    assert s.llm_fallback_models == ("claude-opus-5-5", "claude-sonnet-4-5")
    assert AnthropicLLM(s).models == ["claude-haiku-4-5", "claude-opus-5-5", "claude-sonnet-4-5"]
