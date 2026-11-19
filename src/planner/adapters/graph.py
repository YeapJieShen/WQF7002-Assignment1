"""The agent loop as a LangGraph: model -> check -> confirm -> act, with a cap on repeated errors.

One tool call is handled per model turn; the LLM client must disable parallel tool calls.
Resume value for a paused confirmation is a bool: True runs the call, anything else cancels it.
"""

from __future__ import annotations

import json
import logging
import operator
import time
from functools import wraps
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from planner.adapters.logs import elapsed_ms, emit
from planner.app.calls import CallRecord
from planner.app.guardrails import confirmation_question
from planner.app.prompt import session_snapshot
from planner.app.tools.args import TOOLS_BY_NAME, TaskPatch, ToolArgs
from planner.app.tools.dispatch import run_tool
from planner.app.validation import Accepted, reject_malformed, validate_call
from planner.domain.models import (
    Pace,
    SessionState,
    Task,
    TaskStatus,
    TaskType,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langgraph.graph.state import CompiledStateGraph

    from planner.domain.models import BaseData

# Only our own state types may be rebuilt from a checkpoint; LangGraph will block the rest.
_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        SessionState,
        CallRecord,
        Task,
        Pace,
        TaskType,
        TaskStatus,
        TaskPatch,
        *TOOLS_BY_NAME.values(),
    ]
)
LOG = logging.getLogger("planner.graph")
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
        call_log: Every stage every tool call has passed through, oldest first.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    session: SessionState
    pending: ToolArgs | None
    rejections: int
    call_log: Annotated[list[CallRecord], operator.add]


def build_graph(
    model: Model, base: BaseData, rules: str, *, max_rejections: int
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Assemble and compile the agent graph.

    Args:
        model: Produces the assistant messages.
        base: The read-only reference data.
        rules: The fixed system prompt, sent first on every model call.
        max_rejections: How many rejected calls in a row end the turn with an apology.

    Returns:
        A compiled graph with an in-memory checkpointer, so a confirmation can pause and resume.
    """
    steps = _Steps(model, base, rules, max_rejections)
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    graph.add_node("call_model", _timed("call_model", steps.call_model))
    graph.add_node("check", _timed("check", steps.check))
    graph.add_node("confirm", _timed("confirm", steps.confirm))
    graph.add_node("act", _timed("act", steps.act))
    graph.add_node("give_up", _timed("give_up", steps.give_up))
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", steps.after_model, ["check", END])
    graph.add_conditional_edges("check", steps.after_check, ["confirm", "give_up", "call_model"])
    graph.add_conditional_edges("confirm", steps.after_confirm, ["act", "call_model"])
    graph.add_edge("act", "call_model")
    graph.add_edge("give_up", END)
    return graph.compile(checkpointer=InMemorySaver(serde=_SERDE))


class _Steps:
    """The graph's nodes and routing functions, sharing the model, data and cap."""

    def __init__(self, model: Model, base: BaseData, rules: str, max_rejections: int) -> None:
        self._model = model
        self._base = base
        self._rules = rules
        self._max_rejections = max_rejections

    def call_model(self, state: AgentState) -> dict[str, object]:
        # Rules and snapshot are rebuilt on every call and never stored in `messages`.
        prompt = [
            SystemMessage(self._rules),
            *state["messages"],
            HumanMessage(session_snapshot(state["session"], self._base)),
        ]
        reply = self._model.invoke(prompt)
        asked = reply.tool_calls or reply.invalid_tool_calls
        return {"messages": [reply], "rejections": state["rejections"] if asked else 0}

    def check(self, state: AgentState) -> dict[str, object]:
        last = _last_ai(state)
        args: object
        if last.tool_calls:
            call = last.tool_calls[0]
            verdict = validate_call(call["name"], call["args"], state["session"], self._base)
            call_id = call["id"]
            tool, args = call["name"], call["args"]
        else:
            bad = last.invalid_tool_calls[0]
            verdict = reject_malformed(bad["name"], bad["error"] or "")
            call_id = bad["id"]
            tool, args = bad["name"] or "", bad["args"]
        if LOG.isEnabledFor(logging.DEBUG):  # tool arguments are user data: DEBUG only
            emit(LOG, "tool_args", level=logging.DEBUG, tool=tool, args=_record_args(args))
        if isinstance(verdict, Accepted):
            emit(LOG, "tool_check", tool=tool, accepted=True, issue_codes=[], rejections=0)
            entry = _record("check", tool, args, "accepted")
            return {"pending": verdict.args, "rejections": 0, "call_log": [entry]}
        emit(
            LOG,
            "tool_check",
            tool=tool,
            accepted=False,
            issue_codes=[str(i.code) for i in verdict.issues],
            rejections=state["rejections"] + 1,
        )
        entry = _record("check", tool, args, f"rejected: {verdict.message()}")
        reply = ToolMessage(f"Rejected: {verdict.message()}", tool_call_id=call_id or "")
        return {
            "messages": [reply],
            "pending": None,
            "rejections": state["rejections"] + 1,
            "call_log": [entry],
        }

    def confirm(self, state: AgentState) -> dict[str, object]:
        pending = state["pending"]
        if pending is None:  # pragma: no cover - routing only sends accepted calls here
            return {}
        question = confirmation_question(pending, state["session"])
        # interrupt() must stay first: on resume this node runs again from the top.
        if question is None:
            return {}
        approved = interrupt(question) is True
        emit(LOG, "confirm_answered", tool=_last_ai(state).tool_calls[0]["name"], approved=approved)
        entry = _logged(state, "confirm", "approved" if approved else "declined")
        if approved:
            return {"call_log": [entry]}
        return {
            "messages": [_tool_reply(state, DECLINED)],
            "pending": None,
            "call_log": [entry],
        }

    def act(self, state: AgentState) -> dict[str, object]:
        pending = state["pending"]
        if pending is None:  # pragma: no cover - routing only sends accepted calls here
            return {}
        tool = _last_ai(state).tool_calls[0]["name"]
        started = time.perf_counter()
        try:
            result = run_tool(pending, state["session"], self._base)
        except Exception:
            emit(LOG, "tool_act", level=logging.ERROR, exc_info=True, tool=tool, success=False)
            raise
        emit(
            LOG,
            "tool_act",
            tool=tool,
            success=True,
            duration_ms=elapsed_ms(started),
        )
        return {
            "session": result.state,
            "messages": [_tool_reply(state, result.message)],
            "pending": None,
            "call_log": [_logged(state, "act", result.message)],
        }

    def give_up(self, state: AgentState) -> dict[str, object]:
        emit(LOG, "give_up", rejections=state["rejections"], max_rejections=self._max_rejections)
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


def _timed[**P, R](name: str, node: Callable[P, R]) -> Callable[P, R]:
    """Wrap a node so its duration is logged at DEBUG, even when it pauses or fails."""

    @wraps(node)
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        try:
            return node(*args, **kwargs)
        finally:
            emit(LOG, "graph_step", level=logging.DEBUG, node=name, duration_ms=elapsed_ms(started))

    return run


def _record_args(args: object) -> str:
    return args if isinstance(args, str) else json.dumps(args, default=str)


def _last_ai(state: AgentState) -> AIMessage:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):  # pragma: no cover - routing only follows model turns
        msg = f"expected an assistant message, got {type(last).__name__}"
        raise TypeError(msg)
    return last


def _record(
    stage: Literal["check", "confirm", "act"], tool: str, args: object, outcome: str
) -> CallRecord:
    return CallRecord(stage=stage, tool=tool, args=_record_args(args), outcome=outcome)


def _logged(state: AgentState, stage: Literal["confirm", "act"], outcome: str) -> CallRecord:
    call = _last_ai(state).tool_calls[0]
    return _record(stage, call["name"], call["args"], outcome)


def _tool_reply(state: AgentState, content: str) -> ToolMessage:
    return ToolMessage(content, tool_call_id=_last_ai(state).tool_calls[0]["id"] or "")
