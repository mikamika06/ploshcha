import pytest

from ploshcha_sim.adapters.tools_fake import FakeToolbox
from ploshcha_sim.domain.gate import FINAL_TOOL
from ploshcha_sim.ports.tool import (
    BAD_ARGS,
    NOT_JSON,
    NO_TOOL_FIELD,
    TOOL_FIELD_FORBIDDEN,
    UNKNOWN_TOOL,
)


@pytest.fixture
def tools():
    return FakeToolbox()


@pytest.fixture
def data_tool(tools):
    return next(s.name for s in tools.specs() if s.name != FINAL_TOOL)


def test_args_schema_has_no_tool_field(tools, data_tool):
    schema = tools.args_schema(data_tool)
    assert "tool" not in schema["properties"]
    assert schema["additionalProperties"] is False
    assert schema["type"] == "object"


def test_args_schema_keeps_required_of_the_tool(tools, data_tool):
    spec = next(s for s in tools.specs() if s.name == data_tool)
    assert tools.args_schema(data_tool)["required"] == sorted(spec.params.get("required", []))


def test_args_schema_unknown_tool_raises(tools):
    with pytest.raises(KeyError):
        tools.args_schema("немає-такого")


def test_choice_schema_is_one_field_without_final_answer(tools):
    schema = tools.choice_schema(exclude=(FINAL_TOOL,))
    assert list(schema["properties"]) == ["tool"]
    assert FINAL_TOOL not in schema["properties"]["tool"]["enum"]
    assert schema["properties"]["tool"]["enum"]


def test_parse_locked_injects_the_fixed_tool(tools, data_tool):
    spec = next(s for s in tools.specs() if s.name == data_tool)
    args = {k: "1918" for k in spec.params.get("required", [])}
    call, reason = tools.parse_locked(args, data_tool)
    assert reason is None
    assert call.tool == data_tool


def test_parse_locked_rejects_an_attempt_to_choose(tools, data_tool):
    spec = next(s for s in tools.specs() if s.name == data_tool)
    args = {k: "1918" for k in spec.params.get("required", [])}
    call, reason = tools.parse_locked({**args, "tool": "щось-інше"}, data_tool)
    assert call is None
    assert reason == TOOL_FIELD_FORBIDDEN


def test_parse_locked_reports_missing_required(tools, data_tool):
    spec = next(s for s in tools.specs() if s.name == data_tool)
    if not spec.params.get("required"):
        pytest.skip("інструмент без обовʼязкових аргументів")
    call, reason = tools.parse_locked({}, data_tool)
    assert call is None and reason == BAD_ARGS


def test_parse_locked_on_non_dict(tools, data_tool):
    call, reason = tools.parse_locked("не json", data_tool)
    assert call is None and reason == NOT_JSON


def test_parse_choice_happy_and_sad(tools, data_tool):
    name, reason = tools.parse_choice({"tool": data_tool}, exclude=(FINAL_TOOL,))
    assert (name, reason) == (data_tool, None)
    assert tools.parse_choice({"tool": FINAL_TOOL}, exclude=(FINAL_TOOL,)) == (None, UNKNOWN_TOOL)
    assert tools.parse_choice({}, exclude=(FINAL_TOOL,)) == (None, NO_TOOL_FIELD)
    assert tools.parse_choice("x", exclude=(FINAL_TOOL,)) == (None, NOT_JSON)


def test_locked_executor_cannot_reach_final_answer(tools, data_tool):
    assert FINAL_TOOL not in tools.args_schema(data_tool)["properties"]
    assert FINAL_TOOL not in tools.choice_schema(exclude=(FINAL_TOOL,))["properties"]["tool"]["enum"]
    assert FINAL_TOOL in [s.name for s in tools.specs()]
    assert FINAL_TOOL in tools.wire_schema()["properties"]["tool"]["enum"]


def test_free_mode_specs_are_unchanged_by_the_new_axis():
    from evalkit.conditions import CONDITIONS

    free = CONDITIONS["hetero-plan@8"]
    locked = CONDITIONS["hetero-plan-locked@8"]
    assert free.executor == "free"
    assert locked.executor == "locked"
    assert free.sha256 != locked.sha256
    assert free.with_(executor="locked").sha256 == locked.sha256
