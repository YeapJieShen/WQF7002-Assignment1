"""The agent loop as a LangGraph: model -> check -> confirm -> act, with a cap on repeated errors.

One tool call is handled per model turn; the LLM client must disable parallel tool calls.
Resume value for a paused confirmation is a bool: True runs the call, anything else cancels it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Protocol, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from planner.app.guardrails import confirmation_question
from planner.app.tools.args import ToolArgs  # noqa: TC001 - LangGraph reads annotations at runtime
from planner.app.tools.dispatch import run_tool
from planner.app.validation import Accepted, reject_malformed, validate_call
from planner.domain.models import SessionState  # noqa: TC001 - LangGraph reads annotations at runtime

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langgraph.graph.state import CompiledStateGraph

    from planner.domain.models import BaseData

DEFAULT_MAX_REJECTIONS = 3
GAVE_UP = "I could not complete that request. Please rephrase it or try something simpler."
DECLINED = "The user declined, so nothing was changed."


class Model(Protocol):
    """Anything that turns the conversation so far into the next assistant message."""

    def invoke(self, messages: Sequence[AnyMessage], /) -> AIMessage:
        """Return the next assistant message, which may carry a tool call."""
        ...


class AgentState(TypedDict):
    """What the graph remembers between steps of one conversation.

    Attributes:
        messages: The conversation, including tool results.
        session: The user's tasks, location and clock.
        pending: A validated call waiting to run, if any.
        rejections: Consecutive rejected calls; reset by any accepted call or plain reply.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    session: SessionState
    pending: ToolArgs | None
    rejections: int


def build_graph(
    model: Model, base: BaseData, *, max_rejections: int = DEFAULT_MAX_REJECTIONS
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Assemble and compile the agent graph.

    Args:
        model: Produces the assistant messages.
        base: The read-only reference data.
        max_rejections: How many rejected calls in a row end the turn with an apology.

    Returns:
        A compiled graph with an in-memory checkpointer, so a confirmation can pause and resume.
    """
    steps = _Steps(model, base, max_rejections)
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    graph.add_node("call_model", steps.call_model)
    graph.add_node("check", steps.check)
    graph.add_node("confirm", steps.confirm)
    graph.add_node("act", steps.act)
    graph.add_node("give_up", steps.give_up)
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", steps.after_model, ["check", END])
    graph.add_conditional_edges("check", steps.after_check, ["confirm", "give_up", "call_model"])
    graph.add_conditional_edges("confirm", steps.after_confirm, ["act", "call_model"])
    graph.add_edge("act", "call_model")
    graph.add_edge("give_up", END)
    return graph.compile(checkpointer=InMemorySaver())


class _Steps:
    """The graph's nodes and routing functions, sharing the model, data and cap."""

    def __init__(self, model: Model, base: BaseData, max_rejections: int) -> None:
        self._model = model
        self._base = base
        self._max_rejections = max_rejections

    def call_model(self, state: AgentState) -> dict[str, object]:
        reply = self._model.invoke(state["messages"])
        asked = reply.tool_calls or reply.invalid_tool_calls
        return {"messages": [reply], "rejections": state["rejections"] if asked else 0}

    def check(self, state: AgentState) -> dict[str, object]:
        last = _last_ai(state)
        if last.tool_calls:
            call = last.tool_calls[0]
            verdict = validate_call(call["name"], call["args"], state["session"], self._base)
            call_id = call["id"]
        else:
            bad = last.invalid_tool_calls[0]
            verdict = reject_malformed(bad["name"], bad["error"] or "")
            call_id = bad["id"]
        if isinstance(verdict, Accepted):
            return {"pending": verdict.args, "rejections": 0}
        reply = ToolMessage(f"Rejected: {verdict.message()}", tool_call_id=call_id or "")
        return {"messages": [reply], "pending": None, "rejections": state["rejections"] + 1}

    def confirm(self, state: AgentState) -> dict[str, object]:
        pending = state["pending"]
        if pending is None:  # pragma: no cover - routing only sends accepted calls here
            return {}
        question = confirmation_question(pending, state["session"])
        # interrupt() must stay first: on resume this node runs again from the top.
        if question is not None and interrupt(question) is not True:
            return {"messages": [_tool_reply(state, DECLINED)], "pending": None}
        return {}

    def act(self, state: AgentState) -> dict[str, object]:
        pending = state["pending"]
        if pending is None:  # pragma: no cover - routing only sends accepted calls here
            return {}
        result = run_tool(pending, state["session"], self._base)
        return {
            "session": result.state,
            "messages": [_tool_reply(state, result.message)],
            "pending": None,
        }

    def give_up(self, state: AgentState) -> dict[str, object]:  # noqa: ARG002 - node signature
        return {"messages": [AIMessage(GAVE_UP)], "rejections": 0}

    def after_model(self, state: AgentState) -> str:
        last = _last_ai(state)
        return "check" if last.tool_calls or last.invalid_tool_calls else END

    def after_check(self, state: AgentState) -> str:
        if state["pending"] is not None:
            return "confirm"
        return "give_up" if state["rejections"] >= self._max_rejections else "call_model"

    def after_confirm(self, state: AgentState) -> str:
        return "act" if state["pending"] is not None else "call_model"


def _last_ai(state: AgentState) -> AIMessage:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):  # pragma: no cover - routing only follows model turns
        msg = f"expected an assistant message, got {type(last).__name__}"
        raise TypeError(msg)
    return last


def _tool_reply(state: AgentState, content: str) -> ToolMessage:
    return ToolMessage(content, tool_call_id=_last_ai(state).tool_calls[0]["id"] or "")
