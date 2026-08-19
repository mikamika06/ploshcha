import json
import time

import pytest

from ploshcha_sim.adapters import FakeLlm, FakeToolbox, PresetEffort
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.adapters.tools_remote import RemoteToolbox
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.ports.tool import TOOL_FIELD_FORBIDDEN, ToolCall

MANIFEST = [
    {"name": "lookup_fact", "description": "Знайти факт про сутність.",
     "params": {"type": "object", "properties": {"entity": {"type": "string"}},
                "required": ["entity"]}},
    {"name": "final_answer", "description": "Завершити й повернути фінальну відповідь.",
     "params": {"type": "object", "properties": {"text": {"type": "string"}},
                "required": ["text"]}},
]


def transport_ok(name, args):
    if name == "final_answer":
        return {"text": args.get("text", "")}
    return {"відомо": True, "fact": f"факт про {args.get('entity')}"}


def box(transport=transport_ok, **kw):
    return RemoteToolbox(MANIFEST, transport, **kw)


# ── маніфест і схеми ──────────────────────────────────────────────────────────

def test_manifest_becomes_specs():
    assert [s.name for s in box().specs()] == ["lookup_fact", "final_answer"]


def test_schemas_build_exactly_as_for_a_local_toolbox():
    remote, local = box(), FakeToolbox()
    assert "tool" in remote.wire_schema()["properties"]
    assert "entity" in remote.wire_schema()["properties"]
    assert set(remote.wire_schema()) == set(local.wire_schema())


def test_locked_args_schema_has_no_tool_field():
    schema = box().args_schema("lookup_fact")
    assert "tool" not in schema["properties"]
    assert schema["required"] == ["entity"]


def test_locked_parse_rejects_an_attempt_to_choose():
    call, reason = box().parse_locked({"entity": "X", "tool": "інше"}, "lookup_fact")
    assert call is None and reason == TOOL_FIELD_FORBIDDEN


def test_parameters_key_is_accepted_too():
    manifest = [{"name": "t", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}]
    assert "a" in RemoteToolbox(manifest, transport_ok).wire_schema()["properties"]


def test_missing_description_falls_back_to_the_name():
    assert RemoteToolbox([{"name": "t"}], transport_ok).specs()[0].description == "t"


# ── виклик ────────────────────────────────────────────────────────────────────

def test_call_reaches_the_transport_with_arguments():
    tb = box()
    result = tb.call(ToolCall(tool="lookup_fact", args={"entity": "мешти"}))
    assert result.ok is True
    assert tb.calls == [("lookup_fact", {"entity": "мешти"})]
    assert result.value["fact"] == "факт про мешти"


def test_unknown_tool_never_reaches_the_transport():
    tb = box()
    result = tb.call(ToolCall(tool="немає", args={}))
    assert result.ok is False and result.error == "unknown_remote_tool"
    assert tb.calls == []


def test_transport_exception_becomes_a_tool_result_not_a_crash():
    def boom(name, args):
        raise RuntimeError("мережа впала")

    result = box(boom).call(ToolCall(tool="lookup_fact", args={"entity": "X"}))
    assert result.ok is False
    assert "RuntimeError" in result.error and "мережа впала" in result.error


def test_timeout_is_reported_not_hung():
    def slow(name, args):
        time.sleep(0.5)
        return {"відомо": True}

    result = box(slow, timeout_s=0.05).call(ToolCall(tool="lookup_fact", args={"entity": "X"}))
    assert result.ok is False and result.error == "remote_timeout"


def test_error_field_in_the_payload_is_honoured():
    result = box(lambda n, a: {"error": "нема доступу"}).call(
        ToolCall(tool="lookup_fact", args={"entity": "X"}))
    assert result.ok is False and result.error == "нема доступу"


def test_latency_is_measured():
    assert box().call(ToolCall(tool="lookup_fact", args={"entity": "X"})).latency_ms >= 0


# ── тризначний `found` ────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload,expected", [
    ({"відомо": True}, True),
    ({"відомо": False}, False),
    ({"found": True}, True),
    ({"known": False}, False),
    ({"result": 42}, None),
    ("не словник", None),
])
def test_found_stays_three_valued(payload, expected):
    result = box(lambda n, a: payload).call(ToolCall(tool="lookup_fact", args={"entity": "X"}))
    assert result.found is expected, "«не знайшов» ≠ «зламався» ≠ «незастосовно»"


# ── ядро не змінюється ────────────────────────────────────────────────────────

def tc(tool, **args):
    return json.dumps({"tool": tool, **args}, ensure_ascii=False)


def _run(tools):
    llm = FakeLlm([tc("lookup_fact", entity="мешти"), tc("final_answer", text="взуття")],
                  model="fake")
    return Orchestrator(single_model_router(llm), PresetEffort(), tools,
                        verifier=False, run_id="r").run("що таке мешти")


def test_the_same_run_works_over_a_remote_toolbox():
    remote = _run(box())
    assert remote.answer == "взуття"
    assert remote.accepted is True
    assert [x["call"]["tool"] for x in remote.scratch] == ["lookup_fact"]


def test_remote_over_the_local_toolbox_is_indistinguishable():
    """Транспорт обгортає сам FakeToolbox: дані ті самі, отже будь-яка різниця = дефект порту."""
    local = FakeToolbox()

    def via_local(name, args):
        result = local.call(ToolCall(tool=name, args=args))
        return result.value if result.ok else {"error": result.error}

    remote_manifest = [{"name": s.name, "description": s.description, "params": s.params}
                       for s in local.specs()]
    remote = _run(RemoteToolbox(remote_manifest, via_local))
    plain = _run(FakeToolbox())

    assert remote.answer == plain.answer
    assert remote.outcome == plain.outcome, "той самий інструмент через порт мусить дати той самий стан"
    assert remote.evidence == plain.evidence, "тризначний `found` мусить пережити транспорт"
    assert remote.steps == plain.steps
    assert [x["call"] for x in remote.scratch] == [x["call"] for x in plain.scratch]


def test_a_broken_remote_tool_does_not_kill_the_loop():
    def boom(name, args):
        if name == "lookup_fact":
            raise TimeoutError("зависло")
        return {"text": args.get("text", "")}

    result = _run(box(boom))
    assert result.steps >= 1
    assert any("tool_error" in i for i in result.incidents) or result.degraded
