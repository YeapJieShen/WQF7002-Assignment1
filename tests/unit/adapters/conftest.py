import pytest

# Gradio's LoginButton, outside a Space, mocks OAuth via the private
# `gradio.oauth._get_mocked_oauth_info`, which needs an HF login and a network
# call to `whoami()`. That makes UI tests pass only on a logged-in machine, so
# replace it with a fixed fake profile.
_FAKE_OAUTH_INFO = {
    "access_token": "mock-oauth-token-for-tests",
    "token_type": "bearer",
    "expires_in": 3600,
    "id_token": "test",
    "scope": "openid profile",
    "expires_at": 0,
    "userinfo": {
        "sub": "0",
        "name": "Test User",
        "preferred_username": "test-user",
        "profile": "https://huggingface.co/test-user",
        "picture": "",
    },
}


@pytest.fixture(autouse=True)
def _mock_gradio_oauth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gradio.oauth._get_mocked_oauth_info", lambda: dict(_FAKE_OAUTH_INFO))
