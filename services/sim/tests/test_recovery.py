import pytest

from ploshcha_sim.domain.recovery import (
    CAPS,
    LADDERS,
    NEAR_DUP_THRESHOLD,
    PARSE_CODES,
    SOFT_CODES,
    attempt_key,
    WIDEN_CEILING,
    Recovery,
    StepOutcome,
    build,
    classify,
    hint_for,
    is_near_duplicate,
    overrides_for,
    partial_answer,
    policy,
    recovery_cap,
    signature,
)


def test_classify_budget_wins_over_everything():
    out = StepOutcome(raw_output="", reject_reason="not_json", budget_left=False)
    assert classify(out) == "budget_exhausted"


def test_classify_parse_rejects():
    for reason in ("not_json", "no_tool_field", "unknown_tool", "bad_args"):
        assert classify(StepOutcome(raw_output="щось", reject_reason=reason)) == reason


def test_classify_empty_output_beats_parse_reason():
    assert classify(StepOutcome(raw_output="   ", reject_reason="not_json")) == "empty_output"


def test_classify_truncated_duplicate_tool_states():
    assert classify(StepOutcome(raw_output="{}", finish_reason="length")) == "truncated"
    assert classify(StepOutcome(raw_output="{}", duplicate=True)) == "dup_call"
    assert classify(StepOutcome(raw_output="{}", near_duplicate=True)) == "near_dup_call"
    assert classify(StepOutcome(raw_output="{}", tool_ok=False)) == "tool_error"
    assert classify(StepOutcome(raw_output="{}", tool_known=False)) == "tool_unknown"


def test_classify_clean_step_is_none():
    assert classify(StepOutcome(raw_output="{}", tool_ok=True, tool_known=True)) is None


def test_duplicate_precedes_near_duplicate():
    out = StepOutcome(raw_output="{}", duplicate=True, near_duplicate=True)
    assert classify(out) == "dup_call"


def test_signature_is_order_independent():
    assert signature("calc", {"expr": "1+1", "x": 2}) == signature("calc", {"x": 2, "expr": "1+1"})


def test_near_duplicate_catches_measured_rewordings():
    prev = [("check_date", {"event": "Хмельниччина", "year": 1648})]
    assert is_near_duplicate("check_date", {"event": "Початок Хмельниччини", "year": 1648}, prev)
    prev2 = [("lookup_fact", {"entity": "Битва під Крутами"})]
    assert is_near_duplicate("lookup_fact", {"entity": "Крути"}, prev2)


def test_near_duplicate_ignores_different_entities():
    prev = [("lookup_fact", {"entity": "Тарас Шевченко"})]
    assert not is_near_duplicate("lookup_fact", {"entity": "Іван Мазепа"}, prev)


def test_near_duplicate_ignores_other_tools_and_exact_repeat():
    prev = [("lookup_fact", {"entity": "Іван Мазепа"})]
    assert not is_near_duplicate("check_date", {"event": "Іван Мазепа", "year": 1687}, prev)
    assert not is_near_duplicate("lookup_fact", {"entity": "Іван Мазепа"}, prev)


def test_near_duplicate_ignores_previous_failed_calls():
    prev = [("calc", {"expr": "144^2"})]
    assert is_near_duplicate("calc", {"expr": "144**2"}, prev)
    assert not is_near_duplicate("calc", {"expr": "144**2"}, prev, succeeded=[False])
    assert is_near_duplicate("calc", {"expr": "144**2"}, prev, succeeded=[True])


def test_near_duplicate_threshold_is_a_parameter():
    prev = [("lookup_fact", {"entity": "Битва під Крутами"})]
    assert is_near_duplicate("lookup_fact", {"entity": "Крути"}, prev, threshold=0.2)
    assert not is_near_duplicate("lookup_fact", {"entity": "Крути"}, prev, threshold=0.8)


def test_parse_class_ladder_starts_with_retighten():
    for code in PARSE_CODES:
        assert policy(code, {}) == "retighten"


def test_duplicate_ladder_never_escalates():
    assert "escalate" not in LADDERS["dup_call"]
    assert "escalate" not in LADDERS["near_dup_call"]
    assert policy("dup_call", {}) == "nudge"


def test_ladder_walks_in_order_as_rungs_exhaust():
    attempts: dict[str, int] = {}
    seen = []
    for _ in range(6):
        rung = policy("empty_output", attempts)
        if rung is None:
            break
        seen.append(rung)
        attempts[rung] = attempts.get(rung, 0) + 1
    assert seen == ["retighten", "widen", "escalate", "partial"]
    assert policy("empty_output", attempts) is None


def test_nudge_cap_allows_two_then_moves_on():
    assert CAPS["nudge"] == 2
    assert policy("dup_call", {"nudge": 1}) == "nudge"
    assert policy("dup_call", {"nudge": 2}) == "partial"
    assert policy("dup_call", {"nudge": 2}, plan_exists=True) == "replan"


def test_replan_skipped_without_plan():
    assert policy("dup_call", {"nudge": 2}, plan_exists=False) == "partial"


def test_total_cap_forces_partial_only():
    assert policy("dup_call", {}, spent=3, cap_total=3) == "partial"
    assert policy("dup_call", {"partial": 1}, spent=3, cap_total=3) is None


def test_disabled_rungs_are_skipped():
    assert policy("not_json", {}, disabled=("retighten",)) == "escalate"
    assert policy("not_json", {}, disabled=("retighten", "escalate")) == "partial"
    assert policy("not_json", {"partial": 1}, disabled=("retighten", "escalate")) is None


def test_recovery_cap_is_half_the_budget():
    assert recovery_cap(8) == 4
    assert recovery_cap(5) == 2
    assert recovery_cap(1) == 1


def test_overrides_tighten_and_widen():
    assert overrides_for("retighten", 256) == {"tier": "strict"}
    assert overrides_for("escalate", 256) == {"tier": "strict"}
    assert overrides_for("widen", 256) == {"max_tokens": 512}
    assert overrides_for("widen", WIDEN_CEILING) == {"max_tokens": WIDEN_CEILING}
    assert overrides_for("nudge", 256) == {}


def test_hint_puts_final_answer_first():
    for code in ("dup_call", "near_dup_call", "tool_unknown", "tool_error"):
        assert hint_for(code).startswith("Заверши через final_answer")


def test_hint_appends_detail():
    assert "(disallowed characters)" in hint_for("tool_error", "disallowed characters")


def test_build_attaches_hint_only_for_nudge():
    nudge = build("dup_call", "nudge")
    assert isinstance(nudge, Recovery) and nudge.hint and nudge.overrides == {}
    tighten = build("not_json", "retighten")
    assert tighten.hint is None and tighten.overrides == {"tier": "strict"}


def test_partial_answer_requires_scratch():
    assert partial_answer([]) is None


def test_partial_answer_summarizes_scratch():
    text = partial_answer([
        {"call": {"tool": "lookup_fact", "entity": "Іван Мазепа"},
         "result": {"fact": "Гетьман", "known": True}},
    ])
    assert text is not None
    assert "lookup_fact" in text and "Гетьман" in text


def test_hard_ladders_end_with_partial_soft_never_terminate():
    for code, ladder in LADDERS.items():
        if code in SOFT_CODES:
            assert "partial" not in ladder, code
            assert ladder == ("nudge",), code
        else:
            assert ladder[-1] == "partial", code


def test_soft_codes_are_tool_feedback_only():
    assert set(SOFT_CODES) == {"tool_error", "tool_unknown"}
    assert policy("tool_error", {}) == "nudge"
    assert policy("tool_error", {"nudge:soft": 2}) is None


def test_soft_hints_do_not_eat_the_hard_nudge_budget():
    spent_by_soft = {"nudge:soft": 2}
    assert policy("tool_unknown", spent_by_soft) is None, "мʼякий ліміт вичерпано"
    assert policy("dup_call", spent_by_soft) == "nudge", "жорсткий ліміт лишається цілим"

    spent_by_hard = {"nudge": 2}
    assert policy("dup_call", spent_by_hard) == "partial"
    assert policy("tool_unknown", spent_by_hard) == "nudge", "мʼякі не залежать від жорстких"


def test_every_rung_in_ladders_has_a_cap():
    for ladder in LADDERS.values():
        for rung in ladder:
            assert rung in CAPS


def test_threshold_default_sits_in_the_measured_gap():
    assert 0.05 < NEAR_DUP_THRESHOLD < 0.30


def test_unknown_code_yields_no_rung():
    assert policy("no_such_code", {}) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("code", list(LADDERS))
def test_policy_terminates_for_every_code(code):
    attempts: dict[str, int] = {}
    for _ in range(10):
        rung = policy(code, attempts, plan_exists=True)
        if rung is None:
            return
        key = attempt_key(rung, code in SOFT_CODES)
        attempts[key] = attempts.get(key, 0) + 1
    raise AssertionError(f"драбина {code} не завершилась")
