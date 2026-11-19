import inspect
from collections.abc import Iterator

import gradio as gr
import httpx2
import openai
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from planner.adapters import zerogpu
from planner.adapters.backend import Backend
from planner.adapters.backends import (
    BACKEND_DOWN,
    BILLING,
    ZEROGPU_DOCS,
    AppBackend,
    EndpointTimeoutError,
    ProviderError,
    _Redacting,
    build_backend,
)
from planner.adapters.config import Endpoint, Settings
from planner.adapters.endpoint_guard import NOT_HTTPS, BlockedEndpointError
from planner.adapters.session import Session, Token, Usage
from planner.domain.models import BaseData, SessionState

pytestmark = pytest.mark.usefixtures("public_dns")
KEY = "sk-secret-0123456789"
CUSTOM = Endpoint(
    name="mine", base_url="https://api.example.com/v1", model="m", api_key=SecretStr(KEY)
)
REQUEST = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")
DEFAULT: RunnableLambda[object, AIMessage] = RunnableLambda(lambda _: AIMessage("default"))


def _rejects_key() -> openai.AuthenticationError:
    return openai.AuthenticationError(
        f"Incorrect API key provided: {KEY}",
        response=httpx2.Response(401, request=REQUEST),
        body=None,
    )


def _fails(error: Exception) -> RunnableLambda[object, AIMessage]:
    def raise_it(_: object) -> AIMessage:
        raise error

    return RunnableLambda(raise_it)


def local() -> AppBackend:
    return AppBackend(DEFAULT, zerogpu=False, allow_private=False, error_max_chars=300)


def gpu() -> AppBackend:
    return AppBackend(DEFAULT, zerogpu=True, allow_private=False, error_max_chars=300)


@pytest.mark.parametrize("name", ["model", "explain", "cost"])
def test_it_matches_the_backend_seam(name: str) -> None:
    ours, seam = getattr(AppBackend, name), getattr(Backend, name)
    assert inspect.signature(ours) == inspect.signature(seam)


def test_the_default_model_is_used_without_an_endpoint() -> None:
    assert local().model(None) is DEFAULT


# --- model ---------------------------------------------------------------------------------


def test_a_custom_endpoint_gets_its_own_client() -> None:
    model = local().model(CUSTOM)
    assert isinstance(model, _Redacting)
    assert model.chat.kwargs["tool_choice"] == "auto"  # type: ignore[attr-defined]
    assert KEY not in repr(model)


@pytest.mark.parametrize("url", ["http://api.example.com/v1", "api.example.com", "ftp://x"])
def test_a_custom_endpoint_must_use_https(url: str) -> None:
    with pytest.raises(BlockedEndpointError, match="https") as caught:
        local().model(CUSTOM.model_copy(update={"base_url": url}))
    assert str(caught.value) == NOT_HTTPS
    assert KEY not in str(caught.value)


def test_provider_errors_lose_the_key() -> None:
    model = _Redacting(_fails(_rejects_key()), KEY, 30, 300)
    with pytest.raises(ProviderError) as caught:
        model.invoke([HumanMessage("hi")])
    assert KEY not in str(caught.value)
    assert "[key removed]" in str(caught.value)
    assert caught.value.__suppress_context__
    explained = local().explain(caught.value, signed_in=False, custom=True)
    assert explained is not None
    assert KEY not in explained
    assert "Incorrect API key" in explained


def test_an_empty_key_is_not_redacted_everywhere() -> None:
    error = openai.APIConnectionError(request=REQUEST)
    with pytest.raises(ProviderError, match=r"^Connection error\.$"):
        _Redacting(_fails(error), "", 30, 300).invoke([])


def test_other_errors_pass_through_a_custom_client() -> None:
    with pytest.raises(ValueError, match="bug"):
        _Redacting(_fails(ValueError("bug")), KEY, 30, 300).invoke([])


def test_a_custom_client_still_streams_tokens(state: SessionState, base: BaseData) -> None:
    chat = GenericFakeChatModel(messages=iter([AIMessage("Hello there")]))
    session = Session(_Redacting(chat, KEY, 30, 300), base, "rules", state, max_rejections=3)
    tokens = [e.text for e in session.stream("hi") if isinstance(e, Token)]
    assert len(tokens) > 1
    assert "".join(tokens) == "Hello there"


# --- explain -------------------------------------------------------------------------------


def test_local_backend_trouble_says_the_backend_is_down() -> None:
    error = openai.APIConnectionError(request=REQUEST)
    assert local().explain(error, signed_in=False, custom=False) == BACKEND_DOWN


@pytest.mark.parametrize("backend", [local(), gpu()])
@pytest.mark.parametrize("custom", [True, False])
def test_other_errors_are_not_explained(backend: AppBackend, *, custom: bool) -> None:
    assert backend.explain(ValueError("bug"), signed_in=True, custom=custom) is None


def test_a_gpu_error_from_the_local_backend_is_not_explained() -> None:
    assert local().explain(gr.Error("quota"), signed_in=False, custom=False) is None


def test_gpu_trouble_when_signed_out_suggests_signing_in() -> None:
    text = gpu().explain(
        gr.Error("You have exceeded your GPU quota"), signed_in=False, custom=False
    )
    assert text is not None
    assert "You have exceeded your GPU quota" in text
    assert "Sign in" in text
    assert "own endpoint" in text
    assert ZEROGPU_DOCS in text
    assert BILLING not in text


def test_gpu_trouble_when_signed_in_links_to_billing() -> None:
    text = gpu().explain(gr.Error("No GPU was available"), signed_in=True, custom=False)
    assert text is not None
    assert BILLING in text
    assert ZEROGPU_DOCS in text
    assert "Sign in" not in text


def test_quota_numbers_are_never_made_up() -> None:
    text = gpu().explain(gr.Error("busy"), signed_in=False, custom=False)
    assert text is not None
    assert not any(c.isdigit() for c in text.replace(ZEROGPU_DOCS, ""))


# --- cost ----------------------------------------------------------------------------------


def test_custom_endpoints_show_tokens() -> None:
    usage = Usage(input_tokens=1234, output_tokens=56)
    assert local().cost(usage, CUSTOM) == "1,234 in / 56 out tokens"


def test_the_default_model_shows_no_cost() -> None:
    assert gpu().cost(Usage(1, 2), None) is None


# --- build_backend -------------------------------------------------------------------------


def test_endpoints_by_default() -> None:
    backend = build_backend(Settings(_env_file=None))
    assert not backend.zerogpu


class _Engine:
    def render(self, _chat: object, _tools: object, /) -> str:
        return ""

    def count(self, _text: str, /) -> int:
        return 0

    def generate(self, _prompt: str, /) -> Iterator[str]:
        yield ""


def test_zerogpu_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Settings] = []

    def load(settings: Settings) -> _Engine:
        seen.append(settings)
        return _Engine()

    monkeypatch.setattr(zerogpu, "load_engine", load)
    settings = Settings(
        _env_file=None, backend="zerogpu", tool_call_open="[[", tool_call_close="]]"
    )
    backend = build_backend(settings)
    assert backend.zerogpu
    assert seen == [settings]
    assert isinstance(backend.default, zerogpu.TemplateChat)
    assert (backend.default.call_open, backend.default.call_close) == ("[[", "]]")


def test_a_custom_endpoint_that_times_out_says_how_long_it_was_given() -> None:
    error = openai.APITimeoutError(request=REQUEST)
    with pytest.raises(EndpointTimeoutError) as caught:
        _Redacting(_fails(error), KEY, 30, 300).invoke([])
    explained = local().explain(caught.value, signed_in=False, custom=True)
    assert explained == (
        "Your endpoint didn't answer within 30 s. Check that it is up, or try a faster model."
    )


def test_the_custom_client_is_built_with_its_endpoint_timeout() -> None:
    model = local().model(CUSTOM.model_copy(update={"timeout_s": 7}))
    assert isinstance(model, _Redacting)
    assert model.timeout_s == 7
