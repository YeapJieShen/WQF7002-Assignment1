# SPDX-FileCopyrightText: 2026 Jie Shen Yeap
# SPDX-License-Identifier: MIT
import socket
from collections.abc import Callable

import pytest

PUBLIC_IP = "93.184.216.34"

type Resolver = Callable[..., list[tuple[int, int, int, str, tuple[str, int]]]]


def answer(*ips: str) -> Resolver:
    """A stand-in for `socket.getaddrinfo` that resolves every name to `ips`."""

    def getaddrinfo(
        _host: str, port: int, *_args: object, **_: object
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        family = socket.AF_INET6 if ":" in ips[0] else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]

    return getaddrinfo


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every name to a public address, so no test touches the network."""
    monkeypatch.setattr(socket, "getaddrinfo", answer(PUBLIC_IP))


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
