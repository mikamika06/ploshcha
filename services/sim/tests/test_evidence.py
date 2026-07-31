import json

import pytest

from ploshcha_sim.agents.verify import ACCEPTING, Verdict, _grounded_system, verify
from ploshcha_sim.domain.evidence import evidence_state, found_in, outcome_of
from ploshcha_sim.domain.recovery import ABSENT_HINT
from ploshcha_sim.ports.tool import ToolCall, ToolResult

FOUND = {"відомо": True, "стаття": "…"}
MISSING = {"відомо": False, "стаття": "слова немає в довіднику"}


@pytest.mark.parametrize("value,want", [
    (FOUND, True), (MISSING, False),
    ({"known": True}, True), ({"known": False}, False),
    ({"результат": 42}, None), ("текст", None), (None, None),
    ({"відомо": "так"}, None),
])
def test_absence_is_read_from_the_shared_convention(value, want):
    assert found_in(value) is want


def test_the_toolbox_puts_absence_into_the_port_contract():
    """Раніше «нема даних» жило лише в payload, тож кожен шар угадував домовленість окремо."""
    from ploshcha_sim.adapters.tools_fake import DEFAULT_TOOLS, FakeToolbox

    box = FakeToolbox(tools=DEFAULT_TOOLS)
    assert box.call(ToolCall(tool="lookup_fact", args={"entity": "Тарас Шевченко"})).found is True
    assert box.call(ToolCall(tool="lookup_fact", args={"entity": "Хтось"})).found is False
    assert box.call(ToolCall(tool="calc", args={"expr": "2+2"})).found is None


def test_a_broken_tool_is_not_an_absence():
    assert ToolResult(tool="x", ok=False, error="boom").found is None


@pytest.mark.parametrize("scratch,want", [
    ([], None),
    ([{"result": FOUND}], True),
    ([{"result": MISSING}], False),
    ([{"result": MISSING}, {"result": FOUND}], True),
    ([{"result": {"сума": 5}}], None),
    ([{"found": False, "result": {}}], False),
])
def test_evidence_state_separates_nothing_found_from_nothing_searched(scratch, want):
    assert evidence_state(scratch) is want


@pytest.mark.parametrize("answer,degraded,partial,evidence,want", [
    ("тлумачення", False, False, True, "answer"),
    ("тлумачення", False, False, None, "answer"),
    ("даних немає", False, False, False, "abstain"),
    ("", False, False, False, "failure"),
    (None, False, False, True, "failure"),
    ("щось", True, False, True, "failure"),
    ("щось", False, True, False, "failure"),
])
def test_abstention_is_its_own_kind_of_completion(answer, degraded, partial, evidence, want):
    assert outcome_of(answer, degraded=degraded, partial=partial, evidence=evidence) == want


def test_the_grounded_judge_is_told_which_regime_it_is_in():
    with_tools = _grounded_system("required", absent=False)
    without = _grounded_system("optional", absent=False)
    assert "без доказу — unsupported" in with_tools
    assert "з власних знань" in without
    assert "abstain" in with_tools and "НЕ впливають на вид" in with_tools


def test_the_grounded_judge_is_warned_when_everything_was_missing():
    assert "abstain" in _grounded_system("required", absent=True)
    assert _grounded_system("required", absent=True) != _grounded_system("required", absent=False)


def test_abstain_counts_as_accepted_but_unsupported_does_not():
    assert set(ACCEPTING) == {"supported", "abstain", "no_evidence"}
    assert "unsupported" not in ACCEPTING and "contradicted" not in ACCEPTING
    assert Verdict(accepted="abstain" in ACCEPTING, kind="abstain").accepted is True


class _Judge:
    model = "judge"

    def __init__(self, payload):
        self.payload = payload
        self.seen = []

    def generate_structured(self, prompt, schema, *, system=None, **kw):
        from ploshcha_sim.ports.llm import LlmResult, LlmUsage

        self.seen.append((prompt, system, json.dumps(schema, sort_keys=True)))
        return LlmResult(text=self.payload, model=self.model,
                         usage=LlmUsage(prompt_tokens=1, completion_tokens=1), latency_ms=0,
                         finish_reason="stop")


class _Router:
    def __init__(self, llm):
        self.llm = llm

    def route(self, kind):
        return self.llm

    def lane(self, kind):
        return "mamay"


class _Effort:
    def effort(self, kind):
        from ploshcha_sim.adapters.router_profile import PresetEffort

        return PresetEffort().effort(kind)


def _verdict(payload, **kw):
    judge = _Judge(payload)
    return verify("задача", "відповідь", _Router(judge), _Effort(), **kw), judge


def test_grounded_mode_maps_each_kind_to_acceptance():
    for kind, accepted in (("supported", True), ("abstain", True),
                           ("unsupported", False), ("contradicted", False)):
        verdict, _ = _verdict(json.dumps({"kind": kind, "reason": "…"}), mode="grounded")
        assert verdict.kind == kind and verdict.accepted is accepted


def test_basic_mode_keeps_the_old_contract():
    verdict, judge = _verdict(json.dumps({"accepted": True, "reason": "ок"}))
    assert verdict.accepted is True and verdict.kind == "supported"
    assert "accepted" in judge.seen[0][2] and "kind" not in judge.seen[0][2]


def test_an_unparsable_verdict_is_loud_not_accepted():
    verdict, _ = _verdict("не json", mode="grounded")
    assert verdict.accepted is False and verdict.kind == "parse_fail"


def _orch(script, **kw):
    from ploshcha_sim.adapters import FakeLlm, FakeToolbox, PresetEffort, single_model_router
    from ploshcha_sim.agents import Orchestrator

    return Orchestrator(single_model_router(FakeLlm(script, model="m")), PresetEffort(),
                        FakeToolbox(), **kw)


def _call(tool, **args):
    return json.dumps({"tool": tool, **args})


def test_the_absence_hint_reaches_the_model_without_the_recovery_ladder():
    """Драбина лікує збої; «нема статті» — нормальний результат, і модель мусить бачити його завжди.

    Без цього при вимкненому відновленні модель не дізнавалась про відсутність, повторювала виклик і
    вмирала на `dup_call` з порожньою відповіддю (5 із 8 на страті чесної відмови).
    """
    o = _orch([_call("lookup_fact", entity="Хтось"), _call("final_answer", text="даних немає")],
              verifier=False, recovery=False)
    result = o.run("питання", seed=1)
    llm = o.router.route("select")
    assert any(ABSENT_HINT in c["prompt"] for c in llm.calls), "модель не дізналась про відсутність"
    assert result.outcome == "abstain" and result.evidence is False


def test_a_found_result_produces_no_absence_hint():
    o = _orch([_call("lookup_fact", entity="Тарас Шевченко"), _call("final_answer", text="гетьман")],
              verifier=False, recovery=False)
    result = o.run("питання", seed=1)
    llm = o.router.route("select")
    assert not any(ABSENT_HINT in c["prompt"] for c in llm.calls)
    assert result.outcome == "answer" and result.evidence is True


def test_auto_mode_spends_nothing_when_there_is_nothing_to_ground_on():
    """Заміряно: без доказів заземлений суддя дає 20/32 проти 21/32 у базового — тобто лише
    переставляє помилки. Отже правильна поведінка — не виносити вердикту й не палити токени."""
    from ploshcha_sim.agents.verify import NO_EVIDENCE_REASON

    class _Never:
        def route(self, kind):
            raise AssertionError("режим auto не має викликати модель без доказів")

        def lane(self, kind):
            return "mamay"

    verdict = verify("задача", "відповідь", _Never(), None, mode="auto", grounding="optional")
    assert verdict.kind == "no_evidence" and verdict.tokens == 0
    assert verdict.accepted is True and verdict.reason == NO_EVIDENCE_REASON


def test_auto_mode_grounds_when_evidence_is_possible():
    verdict, judge = _verdict(json.dumps({"kind": "abstain", "reason": "нема даних"}),
                              mode="auto", grounding="required")
    assert verdict.kind == "abstain" and verdict.accepted is True
    assert "kind" in judge.seen[0][2], "режим auto з доказами мусить питати заземлено"


def test_no_evidence_is_not_a_silent_pass_of_a_wrong_answer():
    """`no_evidence` = «суддя не має думки», а не «відповідь правильна»: вид зафіксований окремо."""
    verdict = verify("задача", "вигадка", None, None, mode="auto", grounding="optional")
    assert verdict.accepted is True and verdict.kind == "no_evidence"
    assert verdict.kind != "supported"


def test_the_refusal_is_composed_by_code_and_names_what_was_asked():
    """K7g: детермінований крок треба віддати коду. Зміст відмови повністю визначений станом
    доказів, а модель на ньому нестабільна (`pass^k` = 0.000 при t=0.7)."""
    from ploshcha_sim.domain.evidence import absent_answer, failed_queries

    scratch = [{"call": {"tool": "словник", "слово": "бардина"}, "result": MISSING}]
    text = absent_answer(scratch)
    assert "бардина" in text and "немає статті" in text
    assert failed_queries(scratch) == ["бардина"]


def test_no_refusal_is_composed_when_something_was_found():
    from ploshcha_sim.domain.evidence import absent_answer

    assert absent_answer([{"result": FOUND}]) is None
    assert absent_answer([{"result": MISSING}, {"result": FOUND}]) is None
    assert absent_answer([]) is None


def test_the_composed_refusal_is_identical_across_calls():
    """Сенс усієї зміни: детермінованість. Однаковий стан доказів — байт-у-байт та сама відмова."""
    from ploshcha_sim.domain.evidence import absent_answer

    scratch = [{"call": {"tool": "словник", "слово": "абахта"}, "result": MISSING}]
    assert absent_answer(scratch) == absent_answer(list(scratch))


def test_the_composed_refusal_passes_the_shared_refusal_vocabulary():
    """Якщо код складає відмову словами, яких не знає словник відмов, набір читатиме її як провал."""
    from evalkit.refusal import ADMIT
    from ploshcha_sim.domain.evidence import absent_answer

    text = absent_answer([{"call": {"tool": "словник", "слово": "х"}, "result": MISSING}]).casefold()
    assert any(v in text for v in ADMIT)


def test_the_orchestrator_replaces_a_fabrication_with_the_composed_refusal():
    o = _orch([_call("lookup_fact", entity="Хтось"), _call("final_answer", text="це стародавній обряд")],
              verifier=False, recovery=False, absent_answer=True)
    result = o.run("питання", seed=1)
    assert "Хтось" in result.answer and "не подаю" in result.answer
    assert result.outcome == "abstain" and "absent_answer_composed" in result.notes


def test_the_orchestrator_leaves_a_grounded_answer_alone():
    o = _orch([_call("lookup_fact", entity="Тарас Шевченко"), _call("final_answer", text="гетьман")],
              verifier=False, recovery=False, absent_answer=True)
    assert o.run("питання", seed=1).answer == "гетьман"
