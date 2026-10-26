"""The argument models are the one source for both shape validation and the LLM schema."""

from planner.app.tools.args import (
    TOOLS_BY_NAME,
    AddTaskArgs,
    TaskPatch,
    UpdateTaskArgs,
)


def test_exactly_the_seven_tools() -> None:
    assert set(TOOLS_BY_NAME) == {
        "add_task",
        "update_task",
        "delete_task",
        "travel_time",
        "check_feasibility",
        "recommend_now",
        "plan_day",
    }


def test_every_tool_has_a_description_for_the_model() -> None:
    for tool in TOOLS_BY_NAME.values():
        assert tool.model_json_schema().get("description"), tool.tool_name


def test_schema_forbids_extra_properties() -> None:
    for tool in TOOLS_BY_NAME.values():
        assert tool.model_json_schema()["additionalProperties"] is False, tool.tool_name


def test_model_cannot_supply_a_task_id_on_add() -> None:
    assert "id" not in AddTaskArgs.model_fields


def test_task_id_is_not_patchable() -> None:
    assert "id" not in TaskPatch.model_fields
    assert UpdateTaskArgs.model_fields["fields"].annotation is TaskPatch
