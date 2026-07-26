"""Прямий агентний замір на нашому action-space: чи є в літератури числа Mamay, чи ні,
у нас буде свій. Гоняємо реальний крок act через три моделі на наборі ситуацій.

Міряємо (усе детерміновано, редюсером — нуль суддів):
  schema_valid — синтаксис+тип дії
  world_valid  — id локацій/людей існують, передумови ок
  діапазон дій — чи не застряг на одному типі (вироджена агентність)
  no-op        — move_to у власну локацію / порожній speak
Ситуації навмисно різнорідні: є де треба говорити, іти, утриматись, постити.
"""

import json
import os
import sys
from collections import Counter
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
from ploshcha_sim.agents import act  # noqa: E402
from ploshcha_sim.domain import POI, AgentState, Persona, Utterance, WorldState  # noqa: E402

MODELS = [
    ("Lapa-12B", os.environ.get("LAPA_MODEL", "")),
    ("Mamay-12B", "MamayLM-Gemma-3-12B-IT-v1.0"),
    ("Mamay-27B", os.environ.get("MAMAY_MODEL", "")),
]


def base_world(tick: int = 0) -> WorldState:
    return WorldState(
        tick=tick,
        time_of_day="morning",
        pois={
            "ploshcha": POI(id="ploshcha", name="Площа", kind="square"),
            "kuznya": POI(id="kuznya", name="Кузня", kind="forge"),
            "krynytsia": POI(id="krynytsia", name="Криниця", kind="well"),
            "richka": POI(id="richka", name="Річка", kind="river", usable=False),
        },
        agents={
            "koval": AgentState(id="koval", persona=Persona(role="koval", name="Остап", bio="Коваль."), location="kuznya"),
            "mati": AgentState(id="mati", persona=Persona(role="mati", name="Оксана", bio="Мати Остапа."), location="kuznya"),
            "did": AgentState(id="did", persona=Persona(role="did", name="Свирид", bio="Старий сусід."), location="ploshcha"),
        },
    )


def situations() -> list[tuple[str, WorldState, str, list[str]]]:
    """(назва, світ, агент, спостереження). Різні ситуації тягнуть різні доречні дії."""
    out = []

    w = base_world()
    w.agents["mati"].location = "kuznya"
    w.utterances.append(Utterance(tick=0, speaker="mati", to=["koval"], text="Остапе, чути мене?", poi="kuznya"))
    out.append(("до тебе звернулись (напрошується speak)", w, "koval", ["mati (Оксана) сказав тобі: «Остапе, чути мене?»"]))

    w = base_world()
    out.append(("нікого поруч (напрошується move/wait)", w, "did", []))

    w = base_world()
    w.board.append("Завтра толока коло річки")
    out.append(("новина на Дошці (можна обговорити)", w, "koval", ["На Дошці зʼявилось: «Завтра толока коло річки»"]))

    w = base_world()
    w.agents["koval"].location = "ploshcha"
    out.append(("ти на площі з дідом (можна speak/use)", w, "koval", ["did (Свирид) поруч мовчить"]))

    w = base_world()
    out.append(("робочий ранок у кузні (use_object доречно)", w, "koval", ["горно холодне, підкови не готові"]))

    return out


def run(label: str, model: str, client_args: dict) -> None:
    llm = OpenAICompatLlm(model=model, **client_args, timeout=180)
    trace = InMemoryTrace()
    print(f"\n{'='*82}\n{label}   {model}\n{'='*82}")
    types = Counter()
    schema_ok = world_ok = noop = 0
    n = 0
    for name, world, agent_id, obs in situations():
        r = act(world, agent_id, llm, observations=obs, trace=trace)
        n += 1
        types[r.action.type] += 1
        schema_ok += r.schema_valid
        world_ok += r.world_valid
        a = r.action
        # no-op: рух у власну локацію
        if a.type == "move_to" and a.poi == world.agents[agent_id].location:
            noop += 1
        detail = ""
        if a.type == "speak":
            detail = f"-> {a.to}: «{(a.text or '')[:44]}»"
        elif a.type in ("move_to", "use_object"):
            detail = f"-> {a.poi}"
        elif a.type == "post_to_board":
            detail = f"-> «{(a.topic or '')[:40]}»"
        flag = "" if (r.schema_valid and r.world_valid) else f"  [{r.reject_reason}]"
        print(f"  {name[:40]:<40} {a.type:<13} {detail}{flag}")
    print(f"  --- schema {schema_ok}/{n}  world {world_ok}/{n}  no-op {noop}  "
          f"типів дій: {len(types)} {dict(types)}")


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    args = {"base_url": url, "api_key": key}
    for label, model in MODELS:
        if model:
            run(label, model, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
