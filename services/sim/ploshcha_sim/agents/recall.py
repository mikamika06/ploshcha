"""Крок `recall`: пригадування під ситуацію."""

from ..domain.memory import DECAY, W_IMPORTANCE, W_RECENCY, W_RELEVANCE
from ..domain.retrieval import RelevanceMode, Retrieved, retrieve
from ..domain.state import WorldState
from ..ports.embedding import EmbeddingPort
from ..ports.trace import StepRecord, TracePort


def build_recall_query(
    world: WorldState,
    agent_id: str,
    observations: list[str] | None = None,
) -> str:
    me = world.agents[agent_id]
    poi = world.pois.get(me.location)
    parts = [
        me.persona.role,
        me.persona.name,
        poi.name if poi else me.location,
        world.time_of_day,
    ]
    parts += [world.agents[a].persona.name for a in world.agents_at(me.location) if a != agent_id]
    parts += observations or []
    return " ".join(p for p in parts if p)


def recall(
    world: WorldState,
    agent_id: str,
    *,
    observations: list[str] | None = None,
    query: str | None = None,
    k: int = 5,
    mode: RelevanceMode = "chargram",
    embedder: EmbeddingPort | None = None,
    normalize: bool = True,
    w_recency: float = W_RECENCY,
    w_importance: float = W_IMPORTANCE,
    w_relevance: float = W_RELEVANCE,
    decay: float = DECAY,
    trace: TracePort | None = None,
    run_id: str = "",
    ablation: dict | None = None,
) -> list[Retrieved]:
    me = world.agents[agent_id]
    text = query if query is not None else build_recall_query(world, agent_id, observations)

    query_embedding = None
    if mode == "cosine" and embedder is not None:
        query_embedding = embedder.embed([text])[0]
        missing = [m for m in me.memory if m.embedding is None]
        if missing:
            for item, vec in zip(missing, embedder.embed([m.text for m in missing])):
                item.embedding = vec

    hits = retrieve(
        me.memory,
        text,
        world.tick,
        k,
        mode=mode,
        query_embedding=query_embedding,
        normalize=normalize,
        w_recency=w_recency,
        w_importance=w_importance,
        w_relevance=w_relevance,
        decay=decay,
    )

    if trace is not None:
        trace.emit(
            StepRecord(
                run_id=run_id,
                tick=world.tick,
                agent=agent_id,
                stage="recall",
                model=embedder.model if (mode == "cosine" and embedder) else f"retrieval:{mode}",
                prompt=text,
                raw_output="",
                # покомпонентні скори — без них не відрізнити «факту не було в
                # памʼяті» від «був, але скоринг його не підняв»
                parsed={
                    "k": k,
                    "mode": mode,
                    "candidates": len(me.memory),
                    "retrieved": [
                        {
                            "id": h.item.id,
                            "recency": round(h.recency, 4),
                            "importance": round(h.importance, 4),
                            "relevance": round(h.relevance, 4),
                            "total": round(h.total, 4),
                        }
                        for h in hits
                    ],
                },
                schema_valid=True,
                world_valid=True,
                ablation=ablation or {},
            )
        )

    return hits
