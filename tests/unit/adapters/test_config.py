import json

import pytest
from pydantic import SecretStr, ValidationError

from planner.adapters.config import LOCAL_GPU, Endpoint, Settings


def test_defaults_point_at_local_ollama() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_endpoints == (LOCAL_GPU,)
    assert settings.max_rejections == 3


def test_environment_overrides_endpoints_and_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoints = [
        {"name": "gpu", "base_url": "http://localhost:11434/v1", "model": "m"},
        {"name": "cloud", "base_url": "https://example.test/v1", "model": "big", "api_key": "k"},
    ]
    monkeypatch.setenv("PLANNER_LLM_ENDPOINTS", json.dumps(endpoints))
    monkeypatch.setenv("PLANNER_MAX_REJECTIONS", "5")
    settings = Settings(_env_file=None)
    assert [e.name for e in settings.llm_endpoints] == ["gpu", "cloud"]
    assert settings.llm_endpoints[1].api_key.get_secret_value() == "k"
    assert settings.max_rejections == 5


def test_api_key_is_hidden_in_repr() -> None:
    endpoint = Endpoint(
        name="x", base_url="http://x/v1", model="m", api_key=SecretStr("secret-value")
    )
    assert "secret-value" not in repr(endpoint)


def test_an_empty_endpoint_list_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANNER_LLM_ENDPOINTS", "[]")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_limits_and_model_settings_have_documented_defaults() -> None:
    s = Settings(_env_file=None)
    assert (s.concurrency_limit, s.queue_max_size, s.custom_timeout_s) == (4, 20, 30)
    assert (s.message_max_chars, s.import_max_bytes, s.state_ttl_s) == (4000, 1_000_000, 3600)
    assert s.zerogpu_model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert (s.zerogpu_model_revision, s.zerogpu_max_new_tokens) == (None, 1024)
    assert s.zerogpu_gpu_duration_s == 60
    assert (s.tool_call_open, s.tool_call_close) == ("<tool_call>", "</tool_call>")


def test_limits_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANNER_MESSAGE_MAX_CHARS", "50")
    monkeypatch.setenv("PLANNER_ZEROGPU_MODEL_REVISION", "abc123")
    s = Settings(_env_file=None)
    assert (s.message_max_chars, s.zerogpu_model_revision) == (50, "abc123")


def test_a_limit_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLANNER_QUEUE_MAX_SIZE", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_the_upload_backstop_must_exceed_the_import_limit() -> None:
    assert Settings(_env_file=None).upload_max_bytes == 10_000_000
    with pytest.raises(ValidationError, match="upload_max_bytes"):
        Settings(_env_file=None, import_max_bytes=2_000_000, upload_max_bytes=2_000_000)
