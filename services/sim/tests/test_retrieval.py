"""Retrieval: релевантність (три способи), нормалізація, детермінізм."""

from ploshcha_sim.domain import (
    MemoryItem,
    char_grams,
    relevance_chargram,
    relevance_cosine,
    relevance_jaccard,
    retrieve,
    tokens,
)


def mem(id_: str, tick: int, text: str, importance: int = 5) -> MemoryItem:
    return MemoryItem(id=id_, tick=tick, kind="observation", text=text, importance=importance)


# ── нормалізація тексту ───────────────────────────────────────────────────────


def test_apostrophe_variants_fold_together():
    assert tokens("мʼясо") == tokens("м'ясо") == tokens("м’ясо")


def test_tokens_ignore_punctuation_and_case():
    assert tokens("Оксана, гей!") == {"оксана", "гей"}


# ── ключове: відмінки ────────────────────────────────────────────────────────


def test_chargram_survives_inflection_where_jaccard_dies():
    """«Оксані» і «Оксана» — те саме для нас, різне для токенного Jaccard."""
    assert relevance_jaccard("Оксана", "сказав Оксані") == 0.0
    assert relevance_chargram("Оксана", "сказав Оксані") > 0.3


def test_chargram_beats_jaccard_on_declined_role():
    q = "коваль"
    assert relevance_jaccard(q, "прийшов до коваля") == 0.0
    assert relevance_chargram(q, "прийшов до коваля") > 0.3


def test_dice_dilutes_with_longer_text():
    """Дайс симетричний: те саме влучання в довшому тексті дає менший скор.

    Це компроміс, а не баг — саме тому cosine на embeddings є в абляції.
    """
    short = relevance_chargram("коваль", "коваля")
    long = relevance_chargram("коваль", "уранці прийшов до коваля по підкови")
    assert short > long > 0.0


def test_chargram_does_not_match_unrelated_words():
    assert relevance_chargram("весілля", "підкова") < 0.15


def test_identical_text_is_full_relevance():
    assert relevance_jaccard("толока в селі", "толока в селі") == 1.0
    assert relevance_chargram("толока в селі", "толока в селі") == 1.0


def test_char_grams_pad_word_boundaries():
    assert " ко" in char_grams("коваль")


def test_empty_side_is_zero():
    assert relevance_jaccard("", "текст") == 0.0
    assert relevance_chargram("", "текст") == 0.0


# ── косинус ──────────────────────────────────────────────────────────────────


def test_cosine_identical_and_orthogonal():
    assert relevance_cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert relevance_cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_guards_missing_and_mismatched():
    assert relevance_cosine(None, [1.0]) == 0.0
    assert relevance_cosine([1.0, 0.0], [1.0]) == 0.0
    assert relevance_cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── скоринг ──────────────────────────────────────────────────────────────────


def test_retrieve_returns_at_most_k():
    items = [mem(f"m{i}", i, f"подія {i}") for i in range(5)]
    assert len(retrieve(items, "подія", 10, 2)) == 2


def test_retrieve_components_are_normalized_to_unit():
    items = [mem("a", 0, "весілля"), mem("b", 5, "підкова"), mem("c", 10, "толока")]
    hits = retrieve(items, "весілля", 10, 3)
    for h in hits:
        for v in (h.recency, h.importance, h.relevance):
            assert 0.0 <= v <= 1.0


def test_normalization_flattens_non_discriminative_component():
    """Усі однаково важливі -> importance не впливає на порядок."""
    items = [mem("a", 0, "перше", 5), mem("b", 1, "друге", 5)]
    hits = retrieve(items, "щось", 1, 2)
    assert all(h.importance == 0.0 for h in hits)


def test_mode_none_zeroes_relevance():
    items = [mem("a", 0, "весілля"), mem("b", 1, "весілля")]
    hits = retrieve(items, "весілля", 1, 2, mode="none")
    assert all(h.relevance == 0.0 for h in hits)


def test_relevance_wins_when_weighted():
    items = [mem("fresh", 10, "підкова"), mem("old", 0, "весілля у Оксани")]
    hits = retrieve(items, "весілля", 10, 1, w_recency=0.0, w_importance=0.0, w_relevance=1.0)
    assert hits[0].item.id == "old"


def test_recency_wins_when_weighted():
    items = [mem("fresh", 10, "підкова"), mem("old", 0, "весілля")]
    hits = retrieve(items, "весілля", 10, 1, w_relevance=0.0, w_importance=0.0)
    assert hits[0].item.id == "fresh"


def test_retrieve_is_order_independent():
    items = [mem("a", 0, "весілля"), mem("b", 5, "толока"), mem("c", 9, "підкова")]
    a = [h.item.id for h in retrieve(items, "толока", 10, 3)]
    b = [h.item.id for h in retrieve(list(reversed(items)), "толока", 10, 3)]
    assert a == b


def test_retrieve_bounds():
    assert retrieve([], "q", 0, 5) == []
    assert retrieve([mem("a", 0, "т")], "q", 0, 0) == []


def test_unnormalized_recency_is_raw_decay():
    items = [mem("a", 10, "т"), mem("b", 0, "т")]
    hits = retrieve(items, "т", 10, 2, normalize=False)
    assert hits[0].recency == 1.0
