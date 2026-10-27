import httpx2  # the openai SDK's HTTP layer; its errors are built from these types
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from planner.adapters.config import LOCAL_GPU, Settings
from planner.adapters.llm import build_model, chain, tool_specs
from planner.app.tools.args import TOOLS_BY_NAME

REQUEST = httpx2.Request("POST", "http://backend.test/v1/chat/completions")


def _answers(text: str) -> RunnableLambda[object, AIMessage]:
    return RunnableLambda(lambda _: AIMessage(text))


def _fails(error: Exception) -> RunnableLambda[object, AIMessage]:
    def raise_it(_: object) -> AIMessage:
        raise error

    return RunnableLambda(raise_it)


def test_every_tool_is_described_under_its_tool_name() -> None:
    specs = tool_specs()
    assert [s["function"]["name"] for s in specs] == list(TOOLS_BY_NAME)
    delete = next(s for s in specs if s["function"]["name"] == "delete_task")
    assert "confirmation" in delete["function"]["description"]
    assert delete["function"]["parameters"]["required"] == ["task_id"]


def test_build_model_makes_one_client_per_endpoint() -> None:
    one = build_model(Settings(_env_file=None))
    two = build_model(
        Settings(
            _env_file=None, llm_endpoints=(LOCAL_GPU, LOCAL_GPU.model_copy(update={"name": "b"}))
        )
    )
    assert not hasattr(one, "fallbacks")
    assert len(two.fallbacks) == 1  # type: ignore[attr-defined]


def test_a_single_model_is_used_as_is() -> None:
    only = _answers("one")
    assert chain([only]) is only


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(request=REQUEST),
        openai.APITimeoutError(request=REQUEST),
        openai.RateLimitError(
            "slow down", response=httpx2.Response(429, request=REQUEST), body=None
        ),
        openai.InternalServerError(
            "boom", response=httpx2.Response(500, request=REQUEST), body=None
        ),
    ],
)
def test_backend_trouble_moves_to_the_next_endpoint(error: Exception) -> None:
    model = chain([_fails(error), _answers("fallback")])
    assert model.invoke("hi").content == "fallback"


def test_other_errors_are_not_hidden_by_a_fallback() -> None:
    model = chain([_fails(ValueError("bug")), _answers("fallback")])
    with pytest.raises(ValueError, match="bug"):
        model.invoke("hi")
