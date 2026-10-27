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
