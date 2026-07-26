"""Жива проба S2: чи тримають реальні моделі схеми importance та reflect.

Важливо не «чи гарні висновки», а три вимірювані речі:
  1. чи компілюється вкладена схема insights (масив обʼєктів)
  2. чи вкладається модель у кількість оцінок (count_mismatch)
  3. чи цитує РЕАЛЬНІ номери згадок, а не вигадані (dropped_citations)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")

from ploshcha_sim.adapters import InMemoryTrace  # noqa: E402
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm  # noqa: E402
from ploshcha_sim.agents import perceive, rate_importance, reflect, to_memories  # noqa: E402
from ploshcha_sim.domain import (  # noqa: E402
    POI,
    AgentState,
    Persona,
    Utterance,
    WorldState,
)

DAY = [
    ("mati", ["koval"], "Остапе, у неділю весілля в Ганни, підкуй коней."),
    ("did", [], "Чув, чужий чоловік ходив коло криниці."),
    ("mati", ["koval"], "Свирид знову кашляє, зовсім хворий."),
    ("did", ["koval"], "Позич сокиру, дровець нарубати."),
]


def build_world() -> WorldState:
    return WorldState(
        tick=0,
        time_of_day="morning",
        pois={
            "ploshcha": POI(id="ploshcha", name="Площа", kind="square"),
            "kuznya": POI(id="kuznya", name="Кузня", kind="forge"),
            "krynytsia": POI(id="krynytsia", name="Криниця", kind="well"),
        },
        agents={
            "koval": AgentState(
                id="koval",
                persona=Persona(role="koval", name="Остап", bio="Сільський коваль."),
                location="ploshcha",
            ),
            "mati": AgentState(
                id="mati",
                persona=Persona(role="mati", name="Оксана", bio="Мати Остапа."),
                location="ploshcha",
            ),
            "did": AgentState(
                id="did",
                persona=Persona(role="did", name="Свирид", bio="Старий сусід."),
                location="ploshcha",
            ),
        },
    )


def run(label: str, model: str, base_url: str, api_key: str) -> None:
    llm = OpenAICompatLlm(model=model, base_url=base_url, api_key=api_key, timeout=180)
    world = build_world()
    agent = world.agents["koval"]
    trace = InMemoryTrace()

    print(f"\n{'=' * 74}\n{label}   {model}\n{'=' * 74}")

    print("— крок 1: importance (евристика vs модель)")
    for tick, (speaker, to, text) in enumerate(DAY):
        world.tick = tick
        world.utterances.append(
            Utterance(tick=tick, speaker=speaker, to=to, text=text, poi="ploshcha")
        )
        observed = perceive(world, "koval")
        if not observed:
            continue
        heur = rate_importance(observed).ratings
        res = rate_importance(
            observed, strategy="llm", llm=llm, trace=trace, tick=tick, agent_id="koval"
        )
        flag = "" if not res.fallback else f"  ВІДКАТ:{res.reject_reason}"
        print(f"  t{tick} евристика={heur} модель={res.ratings}{flag}  «{observed[0][:52]}»")
        agent.memory += to_memories(
            observed, res.ratings, tick, "koval", seq_start=len(agent.memory)
        )

    print(f"— накопичено памʼятей: {len(agent.memory)}")

    print("— крок 2: reflect (питання -> висновки з посиланнями)")
    world.tick = len(DAY)
    r = reflect(world, "koval", llm, trace=trace)
    print(f"  схема ок: {r.schema_valid}   відкид: {r.reject_reason}   викликів: {r.llm_calls}")
    print(f"  вигаданих посилань: {r.dropped_citations}   повторів викинуто: {r.duplicates_dropped}")
    for q in r.questions:
        print(f"  ? {q}")
    for item in r.items:
        print(f"  = {item.text[:88]}  [важливість {item.importance}, підстав {len(item.evidence)}]")
    if r.items:
        # глибина синтезу: 1 підстава = переказ спостереження, 2+ = звʼязування
        depths = [len(i.evidence) for i in r.items]
        synth = sum(1 for d in depths if d >= 2) / len(depths)
        print(f"  глибина синтезу: {sum(depths) / len(depths):.2f} підстав/висновок, "
              f"{synth:.0%} висновків звʼязують 2+")
    print(f"  токенів: {r.usage.total}, {r.latency_ms} мс")

    bad = [rec for rec in trace.records if not rec.schema_valid]
    print(f"— трас: {len(trace.records)}, з них невалідних: {len(bad)}")
    for rec in bad:
        print(f"    {rec.stage}: {rec.reject_reason}  «{rec.raw_output[:60]}»")


def main() -> int:
    if os.environ.get("LAPA_API_KEY"):
        run("ХОСТОВАНИЙ Lapathoniia", os.environ["LAPA_MODEL"],
            os.environ["LAPA_BASE_URL"], os.environ["LAPA_API_KEY"])
    if os.environ.get("MAMAY_MODEL") and os.environ.get("LAPA_API_KEY"):
        run("ХОСТОВАНИЙ Lapathoniia", os.environ["MAMAY_MODEL"],
            os.environ["LAPA_BASE_URL"], os.environ["LAPA_API_KEY"])
    if os.environ.get("LOCAL_BASE_URL"):
        run("ЛОКАЛЬНИЙ", os.environ.get("LOCAL_MODEL", "local"),
            os.environ["LOCAL_BASE_URL"], os.environ.get("LOCAL_API_KEY", "EMPTY"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
