"""Importance: евристика проти оцінки моделлю; провал LLM лишається метрикою."""

import json

from ploshcha_sim.adapters import FakeLlm, InMemoryTrace
from ploshcha_sim.agents import heuristic_importance, rate_importance
from ploshcha_sim.agents.importance import (
    BATCH_TOO_SMALL,
    COUNT_MISMATCH,
    OUT_OF_RANGE,
    SCHEMA_INVALID,
    TRUNCATED,
)

MUNDANE = "did (Свирид) підійшов."
ADDRESSED = "mati (Оксана) сказав тобі: «ходи їсти»"
BIG = "На Дошці зʼявилось: «весілля в неділю»"


# ── евристика ────────────────────────────────────────────────────────────────


def test_plain_observation_is_middling():
    assert heuristic_importance("Свирид стоїть коло криниці") == 3


def test_addressed_speech_ranks_higher_than_overheard():
    assert heuristic_importance("гей", addressed=True) > heuristic_importance("гей")


def test_salient_event_lifts_score():
    assert heuristic_importance(BIG) > heuristic_importance("підкова готова")


def test_mundane_movement_is_discounted():
    assert heuristic_importance(MUNDANE) < 3


def test_score_is_clamped_to_range():
    loud = heuristic_importance("весілля і пожежа і похорон", addressed=True)
    assert 1 <= loud <= 10


def test_inflected_marker_still_matches():
    """«хворий» мусить ловитись маркером «хвор»."""
    assert heuristic_importance("Свирид хворий") > heuristic_importance("Свирид спить")


def test_batch_autodetects_addressed_form():
    ratings = rate_importance([ADDRESSED, MUNDANE]).ratings
    assert ratings[0] > ratings[1]


def test_empty_input_returns_empty():
    assert rate_importance([]).ratings == []


# ── стратегія LLM ────────────────────────────────────────────────────────────


def test_llm_ratings_are_used_when_valid():
    llm = FakeLlm([json.dumps({"ratings": [9, 1]})])
    r = rate_importance([ADDRESSED, MUNDANE], strategy="llm", llm=llm, min_batch=1)
    assert r.ratings == [9, 1] and r.strategy == "llm" and not r.fallback


def test_count_mismatch_falls_back_and_is_named():
    llm = FakeLlm([json.dumps({"ratings": [5]})])
    r = rate_importance([ADDRESSED, MUNDANE], strategy="llm", llm=llm, min_batch=1)
    assert r.fallback and r.reject_reason == COUNT_MISMATCH and len(r.ratings) == 2


def test_out_of_range_falls_back():
    llm = FakeLlm([json.dumps({"ratings": [0, 11]})])
    r = rate_importance([ADDRESSED, MUNDANE], strategy="llm", llm=llm, min_batch=1)
    assert r.fallback and r.reject_reason == OUT_OF_RANGE


def test_broken_json_falls_back():
    r = rate_importance([ADDRESSED], strategy="llm", llm=FakeLlm(["не JSON"]), min_batch=1)
    assert r.fallback and r.reject_reason == SCHEMA_INVALID


def test_missing_llm_degrades_to_heuristic_silently():
    r = rate_importance([ADDRESSED], strategy="llm", llm=None)
    assert r.strategy == "heuristic" and not r.fallback


def test_schema_sent_to_model_is_grammar_compilable():
    """Без `required` бекенд тихо ігнорує схему."""
    llm = FakeLlm([json.dumps({"ratings": [5]})])
    rate_importance([ADDRESSED], strategy="llm", llm=llm, min_batch=1)
    schema = llm.calls[0]["schema"]
    assert schema["required"] == ["ratings"]
    assert schema["additionalProperties"] is False


def test_trace_records_failure_as_data():
    trace = InMemoryTrace()
    llm = FakeLlm([json.dumps({"ratings": [1, 2, 3]})])
    rate_importance(
        [ADDRESSED, MUNDANE], strategy="llm", llm=llm, trace=trace, run_id="r1", min_batch=1,
        tick=7, agent_id="koval",
    )
    rec = trace.records[0]
    assert rec.stage == "importance" and rec.tick == 7 and rec.agent == "koval"
    assert rec.schema_valid is False and rec.reject_reason == COUNT_MISMATCH
    assert rec.prompt and rec.raw_output


# ── якорі шкали й обрізання ──────────────────────────────────────────────────


def test_anchors_are_sent_by_default():
    """Без опор Lapa-12B віддає константу."""
    llm = FakeLlm([json.dumps({"ratings": [5]})])
    rate_importance([ADDRESSED], strategy="llm", llm=llm, min_batch=1)
    assert "згоріла хата" in llm.calls[0]["system"]


def test_anchors_can_be_switched_off_for_ablation():
    llm = FakeLlm([json.dumps({"ratings": [5]})])
    rate_importance([ADDRESSED], strategy="llm", llm=llm, anchored=False, min_batch=1)
    assert "згоріла хата" not in llm.calls[0]["system"]


def test_truncation_is_not_counted_as_schema_error():
    llm = FakeLlm(['{"ratings":[5, 4, 3'], finish_reason="length")
    r = rate_importance([ADDRESSED, MUNDANE], strategy="llm", llm=llm, min_batch=1)
    assert r.fallback and r.reject_reason == TRUNCATED


def test_tiny_batch_uses_heuristic_and_says_why():
    """Пакет з одного = вироджена оцінка; краще евристика, ніж мовчазні одиниці."""
    llm = FakeLlm([json.dumps({"ratings": [9]})])
    r = rate_importance([ADDRESSED], strategy="llm", llm=llm)
    assert r.strategy == "heuristic" and r.reject_reason == BATCH_TOO_SMALL
    assert llm.calls == []


def test_min_batch_is_overridable():
    llm = FakeLlm([json.dumps({"ratings": [9]})])
    r = rate_importance([ADDRESSED], strategy="llm", llm=llm, min_batch=1)
    assert r.strategy == "llm" and r.ratings == [9]


def test_full_batch_reaches_the_model():
    llm = FakeLlm([json.dumps({"ratings": [9, 1, 5, 3]})])
    r = rate_importance([ADDRESSED, MUNDANE, BIG, "щось"], strategy="llm", llm=llm)
    assert r.strategy == "llm" and len(llm.calls) == 1
