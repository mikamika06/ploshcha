"""Recall: пригадування під ситуацію + траса покомпонентних скорів."""

from ploshcha_sim.adapters import HashEmbedder, InMemoryTrace
from ploshcha_sim.agents import build_recall_query, recall
from ploshcha_sim.domain import MemoryItem


def load(world, agent_id: str, *texts: str):
    world.agents[agent_id].memory = [
        MemoryItem(id=f"m{i}", tick=i, kind="observation", text=t, importance=5)
        for i, t in enumerate(texts)
    ]


def test_query_describes_current_situation(world):
    world.agents["koval"].location = "kuznya"
    q = build_recall_query(world, "koval", observations=["дзвін ударив"])
    assert "Остап" in q and "Кузня" in q and "дзвін ударив" in q


def test_query_lists_only_people_present(world):
    world.agents["koval"].location = "kuznya"
    world.agents["mati"].location = "ploshcha"
    assert "Оксана" not in build_recall_query(world, "koval")


def test_recall_returns_at_most_k(world):
    load(world, "koval", "весілля", "підкова", "толока", "дощ")
    assert len(recall(world, "koval", k=2)) == 2


def test_recall_on_empty_memory_is_empty(world):
    assert recall(world, "koval") == []


def test_recall_finds_declined_mention(world):
    """Прив'язка до відмінків, а не до точного слова."""
    load(world, "koval", "підкова готова", "Оксані потрібна вода")
    hits = recall(world, "koval", query="Оксана", k=1, w_recency=0.0, w_importance=0.0)
    assert hits[0].item.text == "Оксані потрібна вода"


def test_trace_keeps_component_scores(world):
    load(world, "koval", "весілля", "підкова")
    trace = InMemoryTrace()
    recall(world, "koval", k=2, trace=trace, run_id="r1")
    rec = trace.records[0]
    assert rec.stage == "recall" and rec.model == "retrieval:chargram"
    assert rec.parsed["candidates"] == 2 and rec.parsed["k"] == 2
    first = rec.parsed["retrieved"][0]
    assert set(first) == {"id", "recency", "importance", "relevance", "total"}


def test_cosine_mode_fills_missing_embeddings_once(world):
    load(world, "koval", "весілля в неділю", "підкова готова")
    emb = HashEmbedder(dim=32)
    recall(world, "koval", k=2, mode="cosine", embedder=emb)
    assert all(m.embedding is not None for m in world.agents["koval"].memory)
    assert emb.calls == 2  # один пакет на запит, один на памʼять


def test_cosine_mode_reuses_stored_embeddings(world):
    load(world, "koval", "весілля")
    emb = HashEmbedder(dim=32)
    recall(world, "koval", mode="cosine", embedder=emb)
    calls_after_first = emb.calls
    recall(world, "koval", mode="cosine", embedder=emb)
    assert emb.calls == calls_after_first + 1  # тільки запит


def test_cosine_mode_ranks_by_surface_similarity(world):
    load(world, "koval", "весілля в неділю", "підкова готова")
    emb = HashEmbedder(dim=128)
    hits = recall(
        world, "koval", query="весілля", k=1, mode="cosine", embedder=emb,
        w_recency=0.0, w_importance=0.0,
    )
    assert hits[0].item.text == "весілля в неділю"


def test_chargram_mode_needs_no_embedder(world):
    load(world, "koval", "весілля")
    assert recall(world, "koval", mode="chargram")[0].item.text == "весілля"


def test_trace_names_embedder_when_cosine(world):
    load(world, "koval", "весілля")
    trace = InMemoryTrace()
    recall(world, "koval", mode="cosine", embedder=HashEmbedder(), trace=trace)
    assert trace.records[0].model == "hash-3gram"
