import socket

import httpx2
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from planner.adapters.backends import AppBackend, ProviderError, _Redacting, build_backend
from planner.adapters.config import Endpoint, Settings
from planner.adapters.endpoint_guard import BlockedEndpointError, check_url, guarded_client

from .conftest import PUBLIC_IP, answer

pytestmark = pytest.mark.usefixtures("public_dns")
KEY = "sk-secret-0123456789"
REQUEST = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")


def nothing(_: object) -> AIMessage:
    return AIMessage("")


def backend(*, allow_private: bool = False) -> AppBackend:
    return AppBackend(
        RunnableLambda(nothing),
        zerogpu=False,
        allow_private=allow_private,
        error_max_chars=300,
    )


def endpoint(url: str) -> Endpoint:
    return Endpoint(name="mine", base_url=url, model="m", api_key=SecretStr(KEY))


def literal_or_public(
    host: str, port: int, **_: object
) -> list[tuple[int, int, int, str, tuple[str, int]]]:
    """Resolve IP literals and localhost as themselves and any other name to a public address."""
    if host == "localhost":
        return answer("127.0.0.1")(host, port)
    if host.replace(".", "").isdigit() or ":" in host:
        return answer(host)(host, port)
    return answer(PUBLIC_IP)(host, port)


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/v1",
        "https://127.0.0.1/v1",
        "https://169.254.169.254/v1",
        "https://10.0.0.5/v1",
        "https://192.168.1.1/v1",
        "https://172.16.0.1/v1",
        "https://[::1]/v1",
        "https://[fc00::1]/v1",
        "https://[fe80::1]/v1",
        "https://[::ffff:127.0.0.1]/v1",
        "https://[::ffff:10.0.0.1]/v1",
        "https://0.0.0.0/v1",
        "https://224.0.0.1/v1",
        "https://240.0.0.1/v1",
        "https://100.64.0.1/v1",
        "https:///x",
        "https://",
        "https://user:pass@api.example.com/v1",
        "https://user@api.example.com/v1",
        "https://api.example.com:8443/v1",
        "https://api.example.com:notaport/v1",
        "http://api.example.com/v1",
        "ftp://api.example.com/v1",
        "api.example.com",
    ],
)
def test_unsafe_urls_are_rejected(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", literal_or_public)
    with pytest.raises(BlockedEndpointError) as caught:
        backend().model(endpoint(url))
    assert str(caught.value)
    assert KEY not in str(caught.value)


@pytest.mark.parametrize(
    "ip", ["127.0.0.1", "169.254.169.254", "10.1.2.3", "::1", "fc00::1", "::ffff:192.168.0.1"]
)
def test_a_name_that_resolves_inside_is_rejected_with_the_address(
    ip: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", answer(ip))
    with pytest.raises(BlockedEndpointError, match="inside the server's network"):
        check_url("https://sneaky.example.com/v1", allow_private=False)


def test_one_private_answer_among_public_ones_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", answer(PUBLIC_IP, "10.0.0.1"))
    with pytest.raises(BlockedEndpointError):
        check_url("https://mixed.example.com/v1", allow_private=False)


def test_an_unknown_host_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object, **__: object) -> None:
        raise socket.gaierror

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(BlockedEndpointError, match="could not be found"):
        check_url("https://nope.example.com/v1", allow_private=False)


@pytest.mark.parametrize("url", ["https://api.example.com/v1", "https://api.example.com:443/v1"])
def test_a_public_https_url_is_accepted(url: str) -> None:
    assert backend().model(endpoint(url)) is not None


def test_a_public_ipv6_address_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", answer("2606:4700:4700::1111"))
    check_url("https://api.example.com/v1", allow_private=False)


# --- use time ------------------------------------------------------------------------------


def test_a_redirect_is_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def handle(_self: object, request: httpx2.Request) -> httpx2.Response:
        seen.append(str(request.url))
        return httpx2.Response(302, headers={"Location": "https://10.0.0.1/secret"})

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", handle)
    response = guarded_client().get("https://api.example.com/v1")
    assert response.status_code == 302
    assert seen == ["https://api.example.com/v1"]


def test_a_name_that_turns_private_after_entry_is_rejected_at_request_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = backend().model(endpoint("https://rebind.example.com/v1"))  # resolves public
    monkeypatch.setattr(socket, "getaddrinfo", answer("169.254.169.254"))
    with pytest.raises(ProviderError, match="inside the server's network") as caught:
        model.invoke([HumanMessage("hi")])
    assert KEY not in str(caught.value)


def test_a_vetted_request_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", lambda *_: httpx2.Response(200))
    assert guarded_client().get("https://api.example.com/v1").status_code == 200


# --- allow_private_endpoints ---------------------------------------------------------------


@pytest.mark.parametrize(
    "url", ["http://localhost:11434/v1", "https://127.0.0.1:8443/v1", "http://10.0.0.5/v1"]
)
def test_private_endpoints_are_accepted_when_allowed(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", literal_or_public)
    assert backend(allow_private=True).model(endpoint(url)) is not None


@pytest.mark.parametrize("url", ["http:///x", "http://user:pw@localhost/v1"])
def test_malformed_urls_are_rejected_even_when_private_is_allowed(url: str) -> None:
    with pytest.raises(BlockedEndpointError):
        backend(allow_private=True).model(endpoint(url))


def test_the_settings_reach_the_backend() -> None:
    plain = build_backend(Settings(_env_file=None))
    assert not plain.allow_private
    assert plain.error_max_chars == 300
    tuned = build_backend(
        Settings(_env_file=None, allow_private_endpoints=True, provider_error_max_chars=50)
    )
    assert tuned.allow_private
    assert tuned.error_max_chars == 50


# --- the provider's error text -------------------------------------------------------------


def test_a_long_provider_error_is_cut_and_the_key_stays_out() -> None:
    def boom(_: object) -> None:
        raise openai.BadRequestError(
            f"{KEY} " + "x" * 5000, response=httpx2.Response(400, request=REQUEST), body=None
        )

    with pytest.raises(ProviderError) as caught:
        _Redacting(RunnableLambda(boom), KEY, 30, 300).invoke([])  # type: ignore[arg-type]
    text = str(caught.value)
    assert len(text) == 303
    assert text.endswith("...")
    assert KEY not in text
