from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from planner.adapters.llm import tool_specs
from planner.adapters.session import Done, Reply, Session, Stage, Token, Usage
from planner.adapters.zerogpu import TemplateChat, parse_reply, to_chat, visible_end
from planner.domain.models import BaseData, SessionState

CALL = '<tool_call>\n{"name": "add_task", "arguments": {"title": "Post parcel"}}\n</tool_call>'


class FakeEngine:
    """Replies from a script, in fixed pieces; counts words as tokens."""

    def __init__(self, *replies: list[str]) -> None:
        self._replies = iter(replies)
        self.prompts: list[tuple[Any, Any]] = []

    def render(self, chat: Any, tools: Any, /) -> str:
        self.prompts.append((chat, tools))
        return "one two three"

    def count(self, text: str, /) -> int:
        return len(text.split())

    def generate(self, prompt: str, /) -> Iterator[str]:  # noqa: ARG002 - fixed script
        yield from next(self._replies)


# --- parse_reply ---------------------------------------------------------------------------


def test_plain_text_is_the_reply() -> None:
    reply = parse_reply("  Hello there.\n")
    assert reply.content == "Hello there."
    assert reply.tool_calls == []
    assert reply.invalid_tool_calls == []


def test_a_tool_call_is_parsed_with_its_lead_in_text() -> None:
    reply = parse_reply("Adding it now.\n" + CALL)
    assert reply.content == "Adding it now."
    [call] = reply.tool_calls
    assert call["name"] == "add_task"
    assert call["args"] == {"title": "Post parcel"}
    assert call["id"]
    assert call["id"].startswith("call_")


def test_only_the_first_call_is_kept() -> None:
    second = CALL.replace("add_task", "delete_task")
    reply = parse_reply(CALL + "\n" + second)
    assert [c["name"] for c in reply.tool_calls] == ["add_task"]


def test_a_call_cut_off_before_its_closing_tag_still_parses() -> None:
    reply = parse_reply(CALL.removesuffix("</tool_call>"))
    assert reply.tool_calls[0]["args"] == {"title": "Post parcel"}


@pytest.mark.parametrize(
    "body",
    [
        '{"name": "add_task", "arguments": {"title": "Post',  # truncated
        "add_task(title='x')",  # not JSON
        "",  # empty
    ],
)
def test_malformed_json_is_an_invalid_call(body: str) -> None:
    reply = parse_reply(f"<tool_call>{body}</tool_call>")
    assert reply.tool_calls == []
    [bad] = reply.invalid_tool_calls
    assert bad["name"] is None
    assert bad["args"] == body
    assert bad["error"]
    assert bad["error"].startswith("not valid JSON")
    assert bad["id"]


@pytest.mark.parametrize(
    ("body", "name"),
    [
        ('{"name": "add_task", "arguments": "title=x"}', "add_task"),
        ('{"name": "add_task"}', "add_task"),
        ('{"arguments": {}}', None),
        ('{"name": 3, "arguments": {}}', None),
        ('["add_task", {}]', None),
    ],
)
def test_wrong_shape_is_an_invalid_call(body: str, name: str | None) -> None:
    [bad] = parse_reply(f"<tool_call>{body}</tool_call>").invalid_tool_calls
    assert bad["name"] == name
    assert bad["args"] == body
    assert bad["error"]
    assert "arguments" in bad["error"]


# --- visible_end ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "final", "end"),
    [
        ("Hello", False, 5),
        ("Hello <", False, 6),  # could be the start of <tool_call>
        ("Hello <tool_c", False, 6),
        ("Hello <tool_c", True, 13),  # the reply ended, so it was text after all
        ("Hello <b>", False, 9),
        ("Hi <tool_call>{", False, 3),
        ("Hi <tool_call>{", True, 3),
        ("", False, 0),
    ],
)
def test_visible_end_hides_tool_calls_and_holds_back_a_possible_start(
    text: str, *, final: bool, end: int
) -> None:
    assert visible_end(text, final=final) == end


# --- to_chat -------------------------------------------------------------------------------


def test_messages_become_template_turns() -> None:
    asked = AIMessage("", tool_calls=[{"name": "add_task", "args": {"title": "A"}, "id": "c1"}])
    chat = to_chat(
        [
            SystemMessage("rules"),
            HumanMessage("hi"),
            AIMessage("Hello"),
            asked,
            ToolMessage("Added T6.", tool_call_id="c1"),
        ]
    )
    assert chat == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "add_task", "arguments": {"title": "A"}}}
            ],
        },
        {"role": "tool", "content": "Added T6."},
    ]


def test_a_malformed_call_is_replayed_as_written() -> None:
    bad = parse_reply("<tool_call>oops</tool_call>")
    [turn] = to_chat([bad])
    assert turn["tool_calls"] == [
        {"type": "function", "function": {"name": "", "arguments": "oops"}}
    ]


def test_unknown_message_types_are_refused() -> None:
    with pytest.raises(TypeError, match="ChatMessage"):
        to_chat([ChatMessage(content="x", role="critic")])


# --- TemplateChat --------------------------------------------------------------------------


def test_invoke_offers_the_tools_and_reports_usage() -> None:
    engine = FakeEngine(["Hello ", "there."])
    reply = TemplateChat(engine=engine).invoke([HumanMessage("hi")])
    assert reply.content == "Hello there."
    assert reply.usage_metadata == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    [(chat, tools)] = engine.prompts
    assert chat == [{"role": "user", "content": "hi"}]
    assert tools == tool_specs()


def test_invoke_returns_a_tool_call() -> None:
    reply = TemplateChat(engine=FakeEngine([CALL])).invoke([HumanMessage("add")])
    assert reply.content == ""
    assert reply.tool_calls[0]["name"] == "add_task"


def test_a_session_streams_reply_text_but_never_tool_call_markup(
    state: SessionState, base: BaseData
) -> None:
    pieces = ["Adding", " it.", "\n<", "tool_", "call>", CALL.removeprefix("<tool_call>")]
    engine = FakeEngine(pieces, ["Done", " <b>", "!"])
    session = Session(TemplateChat(engine=engine), base, "rules", state)
    events = list(session.stream("add a parcel task"))
    tokens = "".join(e.text for e in events if isinstance(e, Token))
    assert tokens == "Adding it.\nDone <b>!"
    assert [e.record.stage for e in events if isinstance(e, Stage)] == ["check", "act"]
    done = events[-1]
    assert isinstance(done, Done)
    assert done.reply == Reply("Done <b>!")
    words_out = len("".join(pieces).split()) + 2  # "Done", "<b>!"
    assert done.usage == Usage(input_tokens=6, output_tokens=words_out)
    assert session.state.task("T6") is not None


def test_a_malformed_call_is_rejected_by_the_graph(state: SessionState, base: BaseData) -> None:
    engine = FakeEngine(['<tool_call>{"name": "add_task", "arguments": {</tool_call>'], ["Sorry."])
    session = Session(TemplateChat(engine=engine), base, "rules", state)
    events = list(session.stream("add"))
    [stage] = [e.record for e in events if isinstance(e, Stage)]
    assert stage.outcome.startswith("rejected")
    assert "not valid JSON" in stage.outcome
    # the second prompt replays the bad call and the rejection
    retry_chat, _ = engine.prompts[1]
    assert retry_chat[-3]["tool_calls"][0]["function"]["arguments"].startswith('{"name"')
    assert retry_chat[-2]["content"].startswith("Rejected:")
