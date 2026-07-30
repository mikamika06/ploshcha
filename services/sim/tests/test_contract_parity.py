"""Паритет ядра й контракту подій: енуми не мають розійтися між Python і TS.

Розходження тут не абстрактне. K9 показав, що коли шар не знає про стан «чесна відмова», він
трактує його як помилку — саме це сталось у метриці (страта відмов читалась 0/8 при 8/8 правильних
відповідях). Якщо контракт втратить `abstain`, ту саму помилку зробить UI.

Джерело правди — `contracts/ploshcha-events.schema.json` (мовно-нейтральне); TS-дзеркало стережеться
власним самотестом (`pnpm --filter @ploshcha/contract-ts test`).
"""

import json
from pathlib import Path
from typing import get_args

import pytest

from ploshcha_sim.agents.verify import ACCEPTING, VerdictKind
from ploshcha_sim.domain.evidence import Outcome
from ploshcha_sim.ports.router import ModelLane

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "contracts" / "ploshcha-events.schema.json"
VERSION = ROOT / "contracts" / "VERSION"


@pytest.fixture(scope="module")
def schema() -> dict:
    if not SCHEMA.exists():
        pytest.skip(f"немає {SCHEMA}")
    return json.loads(SCHEMA.read_text(encoding="utf-8"))


def _enum(schema: dict, name: str) -> set[str]:
    return set(schema["$defs"][name]["enum"])


def test_verdict_kinds_match_the_core(schema):
    assert _enum(schema, "VerdictKind") == set(get_args(VerdictKind))


def test_run_outcomes_match_the_core(schema):
    assert _enum(schema, "RunOutcome") == set(get_args(Outcome))


def test_the_lane_enum_matches_the_core(schema):
    """Ярус — не політика маршрутизації: `hetero` це політика (`Routing`), яруси лише lapa/mamay."""
    assert _enum(schema, "ModelLane") == set(get_args(ModelLane))


def test_abstain_is_a_state_not_an_error(schema):
    """Найважливіший інваріант: відмова мусить бути окремим станом і в контракті теж."""
    outcomes = _enum(schema, "RunOutcome")
    assert {"abstain", "failure"} <= outcomes and "abstain" != "failure"
    assert "abstain" in _enum(schema, "VerdictKind")
    assert "abstain" in ACCEPTING, "відмова, підтверджена доказом відсутності, приймається"


def test_the_contract_carries_the_verdict_kind_not_a_boolean(schema):
    """Булеве «прийнято» вже приховало був цілий стан — тому в події мусить бути ВИД."""
    payload = _event_payload(schema, "verify.verdict")
    assert "kind" in payload["properties"] and "kind" in payload["required"]


def test_the_tool_result_carries_typed_absence(schema):
    payload = _event_payload(schema, "tool.result")
    found = payload["properties"]["found"]
    target = found.get("$ref", "").rsplit("/", 1)[-1] or None
    types = set(schema["$defs"][target]["type"]) if target else set(found["type"])
    assert types == {"boolean", "null"}, "«нема даних» ≠ «зламався» ≠ «незастосовно»"


def test_the_outcome_event_exists_and_declares_evidence(schema):
    payload = _event_payload(schema, "task.outcome")
    assert "outcome" in payload["required"]
    assert "evidence" in payload["properties"]


def test_the_protocol_version_is_bumped_for_the_new_layer():
    assert VERSION.read_text(encoding="utf-8").strip() == "1.1.0"


def _event_payload(schema: dict, type_name: str) -> dict:
    for name, spec in schema["$defs"].items():
        branches = spec.get("allOf")
        if not branches:
            continue
        props = branches[-1].get("properties", {})
        if props.get("type", {}).get("const") == type_name:
            return props["payload"]
    raise AssertionError(f"у схемі немає події «{type_name}»")


def test_every_cognition_event_the_core_can_emit_is_declared(schema):
    """Оголошуємо лише те, що ядро віддає СЬОГОДНІ: спекулятивні типи в контракті — брехня."""
    for type_name in ("route.decided", "tool.called", "tool.result", "memory.recalled",
                      "verify.verdict", "plan.revised", "task.outcome"):
        assert _event_payload(schema, type_name)
