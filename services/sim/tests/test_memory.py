"""Скор памʼяті: recency · importance · relevance."""

from ploshcha_sim.domain import MemoryItem, normalized_importance, recency, score, top_k


def test_recency_decays_with_age():
    assert recency(10, 10) == 1.0
    assert recency(10, 9) > recency(10, 5) > recency(10, 0)


def test_recency_clamps_future_items():
    assert recency(5, 99) == 1.0


def test_normalized_importance_scales_to_unit():
    m = MemoryItem(id="x", tick=0, kind="observation", text="t", importance=10)
    assert normalized_importance(m) == 1.0


def test_relevance_dominates_when_others_equal(memories):
    m = memories[2]
    assert score(m, 5, relevance=1.0) > score(m, 5, relevance=0.0)


def test_top_k_ranks_by_combined_score(memories):
    """decay=0.99: importance домінує над свіжістю."""
    rel = {m.id: 0.5 for m in memories}
    ranked = top_k(memories, now_tick=10, relevances=rel, k=3)
    assert [m.id for m in ranked] == ["m_old_imp", "m_mid", "m_new_low"]


def test_decay_is_a_real_knob(memories):
    """decay=0.7: свіжість перемагає importance."""
    rel = {m.id: 0.5 for m in memories}
    ranked = top_k(memories, now_tick=10, relevances=rel, k=3, decay=0.7)
    assert ranked[0].id == "m_new_low"


def test_top_k_respects_relevance(memories):
    rel = {"m_old_imp": 1.0, "m_new_low": 0.0, "m_mid": 0.0}
    ranked = top_k(memories, now_tick=10, relevances=rel, k=1)
    assert ranked[0].id == "m_old_imp"


def test_top_k_is_deterministic(memories):
    rel = {m.id: 0.5 for m in memories}
    a = [m.id for m in top_k(memories, 10, rel, 3)]
    b = [m.id for m in top_k(list(reversed(memories)), 10, rel, 3)]
    assert a == b


def test_top_k_bounds():
    assert top_k([], 0, {}, 5) == []


def test_importance_range_enforced():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MemoryItem(id="x", tick=0, kind="observation", text="t", importance=11)
