from collections.abc import Iterator, Sequence

from langchain_core.messages import AIMessage, AnyMessage

from planner.adapters.session import Reply, Session
from planner.domain.models import BaseData, SessionState


class ScriptedModel:
    def __init__(self, *script: AIMessage) -> None:
        self._script: Iterator[AIMessage] = iter(script)
        self.history_sizes: list[int] = []

    def invoke(self, messages: Sequence[AnyMessage], /) -> AIMessage:
        self.history_sizes.append(len(messages))
        return next(self._script)


def call(name: str, args: dict[str, object]) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": "c1"}])


def test_send_returns_the_assistant_text(state: SessionState, base: BaseData) -> None:
    session = Session(ScriptedModel(AIMessage("Hello")), base, "rules", state)
    assert session.send("hi") == Reply("Hello")


def test_state_is_the_start_state_until_a_tool_changes_it(
    state: SessionState, base: BaseData
) -> None:
    model = ScriptedModel(call("add_task", {"title": "Post parcel"}), AIMessage("Added."))
    session = Session(model, base, "rules", state)
    assert session.state == state
    session.send("add a parcel task")
    assert session.state.task("T6") is not None


def test_later_turns_keep_the_history(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(AIMessage("One"), AIMessage("Two"))
    session = Session(model, base, "rules", state)
    session.send("first")
    assert session.send("second") == Reply("Two")
    # rules + snapshot around 1 message, then around 3 (user, assistant, user)
    assert model.history_sizes == [3, 5]


def test_delete_asks_then_resume_decides(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Deleted."))
    session = Session(model, base, "rules", state)
    asked = session.send("delete T4")
    assert asked.text == ""
    assert asked.confirm is not None
    assert "T4" in asked.confirm
    assert session.resume(approved=True) == Reply("Deleted.")
    assert session.state.task("T4") is None


def test_declining_keeps_the_task(state: SessionState, base: BaseData) -> None:
    model = ScriptedModel(call("delete_task", {"task_id": "T4"}), AIMessage("Kept."))
    session = Session(model, base, "rules", state)
    session.send("delete T4")
    session.resume(approved=False)
    assert session.state == state
