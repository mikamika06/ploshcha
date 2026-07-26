from ploshcha_sim.adapters import (
    HashEmbedder,
    HybridRetriever,
    ModeRetriever,
    WorldStateMemory,
    default_hybrid,
    reciprocal_rank_fusion,
)
from ploshcha_sim.domain import MemoryItem


def mem(id_, text, tick=0):
    return MemoryItem(id=id_, tick=tick, kind="observation", text=text, importance=5)


ITEMS = [
    mem("m1", "Оксана готує борщ на вечерю"),
    mem("m2", "коваль Остап кує підкови в кузні"),
    mem("m3", "у неділю весілля в Ганни"),
    mem("m4", "Свирид пішов по воду до криниці"),
]


def test_mode_retriever_chargram_finds_inflected():
    r = ModeRetriever(mode="chargram")
    hits = r.rank("Оксані", ITEMS, 1)
    assert hits[0].item.id == "m1"


def test_mode_retriever_respects_k():
    assert len(ModeRetriever(mode="chargram").rank("село", ITEMS, 2)) == 2


def test_mode_retriever_empty_and_zero_k():
    r = ModeRetriever(mode="chargram")
    assert r.rank("x", [], 3) == []
    assert r.rank("x", ITEMS, 0) == []


def test_cosine_mode_fills_embeddings():
    r = ModeRetriever(mode="cosine", embedder=HashEmbedder(dim=32))
    r.rank("весілля", ITEMS, 2)
    assert all(m.embedding is not None for m in ITEMS)


def test_rrf_rewards_agreement():
    a = [ITEMS[0], ITEMS[1], ITEMS[2]]
    b = [ITEMS[0], ITEMS[2], ITEMS[1]]
    fused = reciprocal_rank_fusion([a, b], k=3)
    assert fused[0].id == "m1"


def test_rrf_bounds_to_k():
    fused = reciprocal_rank_fusion([ITEMS, list(reversed(ITEMS))], k=2)
    assert len(fused) == 2


def test_hybrid_is_order_independent():
    r = HybridRetriever([ModeRetriever(mode="chargram"), ModeRetriever(mode="jaccard")])
    a = [h.item.id for h in r.rank("весілля Ганни", ITEMS, 3)]
    b = [h.item.id for h in r.rank("весілля Ганни", list(reversed(ITEMS)), 3)]
    assert a == b


def test_default_hybrid_without_embedder_uses_lexical_pair():
    r = default_hybrid()
    hits = r.rank("підкови", ITEMS, 1)
    assert hits[0].item.id == "m2"


def test_default_hybrid_with_embedder_runs():
    r = default_hybrid(embedder=HashEmbedder(dim=32))
    assert len(r.rank("вода криниця", ITEMS, 2)) == 2


def test_worldstate_memory_add_and_retrieve():
    store = WorldStateMemory()
    for it in ITEMS:
        store.add(it)
    got = store.retrieve("хто кує підкови", 1)
    assert got[0].id == "m2"


def test_worldstate_memory_wraps_existing_list():
    items = list(ITEMS)
    store = WorldStateMemory(items=items)
    store.add(mem("m5", "нова подія"))
    assert items[-1].id == "m5" and len(store.retrieve("подія", 5)) <= 5
