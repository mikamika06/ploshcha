import jsonschema
import pytest

from ploshcha_sim.adapters import FakeToolbox
from ploshcha_sim.ports.tool import BAD_ARGS, NO_TOOL_FIELD, NOT_JSON, UNKNOWN_TOOL, ToolCall


@pytest.fixture
def box():
    return FakeToolbox()


def test_wire_schema_is_grammar_compilable(box):
    s = box.wire_schema()
    assert s["required"] == ["tool"]
    assert s["additionalProperties"] is False
    assert set(s["properties"]["tool"]["enum"]) == {"check_date", "lookup_fact", "calc", "final_answer"}


def test_wire_schema_merges_all_tool_fields(box):
    props = box.wire_schema()["properties"]
    for field in ("year", "event", "entity", "expr", "text"):
        assert field in props


def test_strict_schema_variant_per_tool_with_required(box):
    variants = box.strict_schema()["oneOf"]
    assert len(variants) == 4
    cd = next(v for v in variants if v["properties"]["tool"]["const"] == "check_date")
    assert set(cd["required"]) == {"tool", "year", "event"}
    assert cd["additionalProperties"] is False


def test_strict_schema_validates_real_call(box):
    jsonschema.validate({"tool": "lookup_fact", "entity": "Тарас Шевченко"}, box.strict_schema())


def test_strict_schema_rejects_missing_arg(box):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"tool": "check_date", "year": 1648}, box.strict_schema())


def test_native_tools_openai_format(box):
    native = box.native()
    fn = next(t["function"] for t in native if t["function"]["name"] == "check_date")
    assert fn["parameters"]["required"] == ["year", "event"]


def test_parse_valid_call(box):
    call, reason = box.parse({"tool": "lookup_fact", "entity": "Іван Мазепа"})
    assert reason is None and call.tool == "lookup_fact" and call.args == {"entity": "Іван Мазепа"}


def test_parse_drops_foreign_fields(box):
    call, _ = box.parse({"tool": "lookup_fact", "entity": "X", "year": 5})
    assert call.args == {"entity": "X"}


def test_parse_unknown_tool(box):
    call, reason = box.parse({"tool": "delete_everything", "entity": "X"})
    assert call is None and reason == UNKNOWN_TOOL


def test_parse_missing_tool_field(box):
    call, reason = box.parse({"entity": "X"})
    assert call is None and reason == NO_TOOL_FIELD


def test_parse_not_dict(box):
    call, reason = box.parse("не JSON")
    assert call is None and reason == NOT_JSON


def test_parse_missing_required_arg(box):
    call, reason = box.parse({"tool": "check_date", "year": 1648})
    assert call is None and reason == BAD_ARGS


def test_call_check_date_correct(box):
    r = box.call(ToolCall(tool="check_date", args={"year": 1648, "event": "початок Хмельниччини"}))
    assert r.ok and r.value["matches"] is True and r.value["actual_year"] == 1648


def test_call_check_date_wrong_year(box):
    r = box.call(ToolCall(tool="check_date", args={"year": 1700, "event": "початок Хмельниччини"}))
    assert r.ok and r.value["matches"] is False


def test_call_lookup_fact_known_and_unknown(box):
    assert box.call(ToolCall(tool="lookup_fact", args={"entity": "Тарас Шевченко"})).value["known"] is True
    assert box.call(ToolCall(tool="lookup_fact", args={"entity": "Хтось"})).value["known"] is False


def test_call_calc(box):
    assert box.call(ToolCall(tool="calc", args={"expr": "2 + 2 * 3"})).value["result"] == 8


def test_call_calc_rejects_code(box):
    r = box.call(ToolCall(tool="calc", args={"expr": "__import__('os')"}))
    assert r.ok is False and r.error


def test_call_final_answer(box):
    assert box.call(ToolCall(tool="final_answer", args={"text": "готово"})).value["text"] == "готово"


def test_call_unknown_tool(box):
    r = box.call(ToolCall(tool="nope"))
    assert r.ok is False and r.error == "unknown_tool"


def test_call_bad_args_type(box):
    r = box.call(ToolCall(tool="check_date", args={"year": "не число", "event": "x"}))
    assert r.ok is False and r.error
