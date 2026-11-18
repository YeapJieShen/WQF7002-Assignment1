"""Qwen3 running in-process with transformers, on a Hugging Face ZeroGPU Space.

The tools go in through the model's chat template; the model answers with plain text or with
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`, which `parse_reply` turns into a
LangChain tool call, or an invalid one when it is malformed so the graph rejects it visibly.

Generation stops at the first `</tool_call>`, so a reply carries at most one call (the graph
handles one per turn); anything after it is never generated, and `parse_reply` ignores it too.

`TemplateChat` implements `_generate` only, and reports reply text through `on_llm_new_token`
itself. A `_stream` override would make LangChain merge chunks and re-parse tool arguments with
its lenient partial-JSON parser, which accepts truncated JSON and drops the parse error.
"""

from __future__ import annotations

import json
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import invalid_tool_call, tool_call
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.utils import LC_ID_PREFIX
from pydantic import ConfigDict

from planner.adapters.llm import tool_specs

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from langchain_core.callbacks import CallbackManagerForLLMRun

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS = 1024
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"

type Chat = list[dict[str, Any]]


@runtime_checkable  # pydantic checks the field with isinstance
class Engine(Protocol):
    """A tokenizer and a generator for one model."""

    def render(self, chat: Chat, tools: list[dict[str, Any]], /) -> str:
        """Apply the chat template, ending with the assistant's turn to speak."""
        ...

    def count(self, text: str, /) -> int:
        """Return how many tokens `text` is."""
        ...

    def generate(self, prompt: str, /) -> Iterator[str]:
        """Yield the reply to `prompt` in pieces, stopping after the first `</tool_call>`."""
        ...


def to_chat(messages: Sequence[BaseMessage]) -> Chat:
    """Convert LangChain messages to the role/content dicts a chat template takes.

    Raises:
        TypeError: For a message type the agent never sends.
    """
    return [_turn(m) for m in messages]


def _turn(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.text}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.text}
    if isinstance(message, ToolMessage):
        return {"role": "tool", "content": message.text}
    if isinstance(message, AIMessage):
        # A malformed call is replayed as the model wrote it, so the rejection that follows fits.
        calls = [{"name": c["name"], "arguments": c["args"]} for c in message.tool_calls] + [
            {"name": c["name"] or "", "arguments": c["args"] or ""}
            for c in message.invalid_tool_calls
        ]
        turn: dict[str, Any] = {"role": "assistant", "content": message.text}
        if calls:
            turn["tool_calls"] = [{"type": "function", "function": c} for c in calls]
        return turn
    msg = f"cannot send a {type(message).__name__} to the model"
    raise TypeError(msg)


def visible_end(text: str, *, final: bool) -> int:
    """Return how much of `text` is reply text that is safe to show.

    Everything from `<tool_call>` on is hidden. Until the reply is `final`, a tail that could be
    the start of `<tool_call>` is held back too.
    """
    cut = text.find(CALL_OPEN)
    if cut >= 0:
        return cut
    if not final:
        for size in range(min(len(CALL_OPEN) - 1, len(text)), 0, -1):
            if text.endswith(CALL_OPEN[:size]):
                return len(text) - size
    return len(text)


def parse_reply(text: str) -> AIMessage:
    """Turn the raw reply into a message: text, plus the first tool call if there is one."""
    head, found, rest = text.partition(CALL_OPEN)
    if not found:
        return AIMessage(content=text.strip())
    body = rest.partition(CALL_CLOSE)[0].strip()
    call_id = f"call_{uuid4().hex}"
    try:
        call = json.loads(body)
    except json.JSONDecodeError as exc:
        bad = invalid_tool_call(name=None, args=body, id=call_id, error=f"not valid JSON: {exc}")
        return AIMessage(content=head.strip(), invalid_tool_calls=[bad])
    name = call.get("name") if isinstance(call, dict) else None
    args = call.get("arguments") if isinstance(call, dict) else None
    if not isinstance(name, str) or not isinstance(args, dict):
        error = 'expected {"name": <string>, "arguments": <object>}'
        bad = invalid_tool_call(
            name=name if isinstance(name, str) else None, args=body, id=call_id, error=error
        )
        return AIMessage(content=head.strip(), invalid_tool_calls=[bad])
    return AIMessage(content=head.strip(), tool_calls=[tool_call(name=name, args=args, id=call_id)])


class TemplateChat(BaseChatModel):
    """A chat model over an `Engine`, with the seven tools always offered.

    Attributes:
        engine: Renders prompts, counts tokens and generates.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    engine: Engine

    @property
    def _llm_type(self) -> str:
        return "chat-template"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002 - generation stops at </tool_call> only
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,  # noqa: ANN401, ARG002 - BaseChatModel's signature; no per-call options
    ) -> ChatResult:
        prompt = self.engine.render(to_chat(messages), tool_specs())
        message_id = f"{LC_ID_PREFIX}-{run_manager.run_id}" if run_manager else None
        text, shown = "", 0
        for piece in chain(self.engine.generate(prompt), [None]):  # None: the reply is complete
            text += piece or ""
            end = visible_end(text, final=piece is None)
            if run_manager and end > shown:
                # The final message has the same id, so stream consumers do not see it twice.
                chunk = AIMessageChunk(content=text[shown:end], id=message_id)
                run_manager.on_llm_new_token(
                    text[shown:end], chunk=ChatGenerationChunk(message=chunk)
                )
            shown = max(shown, end)
        reply = parse_reply(text)
        reply.id = message_id
        used_in, used_out = self.engine.count(prompt), self.engine.count(text)
        reply.usage_metadata = {
            "input_tokens": used_in,
            "output_tokens": used_out,
            "total_tokens": used_in + used_out,
        }
        return ChatResult(generations=[ChatGeneration(message=reply)])


class _Transformers:  # pragma: no cover - needs the model weights and a GPU
    def __init__(self, tokenizer: Any, generate: Any) -> None:  # noqa: ANN401 - untyped libraries
        self._tokenizer = tokenizer
        self.generate = generate

    def render(self, chat: Chat, tools: list[dict[str, Any]], /) -> str:
        return str(
            self._tokenizer.apply_chat_template(
                chat, tools=tools, tokenize=False, add_generation_prompt=True
            )
        )

    def count(self, text: str, /) -> int:
        return len(self._tokenizer(text, add_special_tokens=False).input_ids)


def load_engine(model_id: str = MODEL_ID) -> Engine:  # pragma: no cover - needs a GPU
    """Load the model onto the GPU and register its ZeroGPU generate function.

    Call this once while the Space starts: ZeroGPU wants models placed on cuda at startup and
    `@spaces.GPU` functions defined before the app launches. Needs the "space" dependencies.
    """
    from threading import Thread  # noqa: PLC0415

    import spaces  # noqa: PLC0415 - optional; must load before torch starts CUDA
    from transformers import (  # noqa: PLC0415 - optional, heavy
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Sampling comes from the model's generation_config, the settings Qwen recommends.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to("cuda")

    @spaces.GPU  # type: ignore[untyped-decorator]  # spaces is untyped
    def generate(prompt: str) -> Iterator[str]:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        options = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": MAX_NEW_TOKENS,
            "stop_strings": [CALL_CLOSE],
            "tokenizer": tokenizer,
        }
        Thread(target=model.generate, kwargs=options, daemon=True).start()
        yield from streamer

    return _Transformers(tokenizer, generate)
