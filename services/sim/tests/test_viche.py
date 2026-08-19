"""Віче: розмова, а не задача з відповіддю.

Головний інваріант, який тут стережеться: **`abstain` у вічі не існує**. Саме він убивав демо —
довідник із шести статей давав «не знайдено», гейт `outcome_of` перетворював це на відмову, і на
будь-яку тему село відповідало «нема в довіднику». У розмові відсутність даних — це репліка.

Другий інваріант: **виконавець не вибирає**. У схемі репліки немає жодного поля рішення, тож
неможливо навіть висловити «я візьму інструмент» — той самий прийом, що в `E-locked`.
"""

import json
import pathlib

import pytest

from ploshcha_sim.adapters import FakeLlm, PresetEffort
from ploshcha_sim.adapters.projector import POI_OF_TOOL, StreamProjector, villager_of_span
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.adapters.tools_lexis import LEXIS_TOOLS
from ploshcha_sim.adapters.tools_fake import FakeToolbox
from ploshcha_sim.agents.viche import Viche
from ploshcha_sim.domain.task import Budget
from ploshcha_sim.domain.viche import (
    MAX_BEATS,
    MOVES,
    PERSONAS,
    Beat,
    cast_for,
    line_schema,
    repair_score,
    scatter,
    score_schema,
)

NEWS = "Кажуть, за річкою бачили вовка, і він унадився до кошари."


def score(*beats) -> str:
    return json.dumps({"такти": list(beats)}, ensure_ascii=False)


def beat(who, move="згадати", reply=None, tool=None, query=None) -> dict:
    return {"хто": who, "хід": move, "у_відповідь": reply, "інструмент": tool, "запит": query}


def line(text) -> str:
    return json.dumps({"репліка": text}, ensure_ascii=False)


VARIED = [
    "Отакої, а я ж казав, що добром не скінчиться.",
    "Хай йому грець, треба вози лаштувати змалку.",
    "Мені баба Химка інше торочила про ту справу.",
    "Кум із Лип'янки бачив таке ж під осінь.",
    "Та ну, дурниці, ліпше про жнива думати.",
    "Гроші лік люблять, а тут і рахувати нічого.",
    "Ой лишенько, діти ж малі, куди тепер.",
    "Я в церкві свічку поставлю, як воно минеться.",
    "Сусід божиться, ніби сам на власні очі уздрів.",
    "Дощ піде — і всі балачки змиє, отак-то.",
    "Наш млин третій рік стоїть, кому те мито.",
    "Пішов би та подивився, а не язиком плескав.",
]


def lines(n: int) -> list[str]:
    return [line(VARIED[i % len(VARIED)] + " " + "*" * i) for i in range(n)]

def build(replies, *, tools=None, width=3, trace=None):
    llm = FakeLlm(replies, model="fake")
    return Viche(single_model_router(llm), PresetEffort(), tools, width=width, trace=trace,
                 run_id="r"), llm


# ── склад: визначений даними, не моделлю ──────────────────────────────────────

def test_the_same_topic_always_gathers_the_same_people():
    assert cast_for(NEWS, 4) == cast_for(NEWS, 4)


def test_a_different_topic_gathers_a_different_crowd():
    other = [p.role for p in cast_for("Гребля протікає третій тиждень.", 4)]
    assert other != [p.role for p in cast_for(NEWS, 4)] or len(PERSONAS) < 5


def test_the_cast_never_exceeds_the_people_we_have():
    assert len(cast_for(NEWS, 99)) == len(PERSONAS)
    assert len(cast_for(NEWS, 0)) == 2


# ── схема: вибір неможливо навіть висловити ───────────────────────────────────

def test_the_line_schema_has_no_field_for_choosing_anything():
    props = line_schema()["properties"]
    assert list(props) == ["репліка"]
    assert line_schema()["additionalProperties"] is False


def test_the_score_schema_restricts_who_and_how_to_enums():
    schema = score_schema(["did", "koval"], ["словник"])
    item = schema["properties"]["такти"]["items"]["properties"]
    assert item["хто"]["enum"] == ["did", "koval"]
    assert item["хід"]["enum"] == list(MOVES)
    assert "словник" in item["інструмент"]["enum"]


# ── лагодження партитури робить код ───────────────────────────────────────────

def test_a_beat_with_a_stranger_is_dropped():
    beats = repair_score({"такти": [beat("did"), beat("чужий")]}, ["did"], [])
    assert [b.хто for b in beats] == ["did"]


def test_an_unknown_move_is_dropped():
    assert repair_score({"такти": [beat("did", "станцювати")]}, ["did"], []) == []


def test_a_reply_to_the_future_is_cleared_not_kept():
    beats = repair_score({"такти": [beat("did", reply=7)]}, ["did"], [])
    assert beats[0].у_відповідь is None


def test_a_tool_outside_the_toolset_is_cleared():
    beats = repair_score({"такти": [beat("did", tool="ракета")]}, ["did"], ["словник"])
    assert beats[0].інструмент is None


def test_garbage_never_raises():
    assert repair_score(None, ["did"], []) == []
    assert repair_score({"такти": "не список"}, ["did"], []) == []


def test_the_score_is_capped():
    roles = [p.role for p in PERSONAS]
    raw = {"такти": [beat(roles[i % len(roles)]) for i in range(MAX_BEATS + 40)]}
    assert len(repair_score(raw, roles, [])) == MAX_BEATS


# ── спонтанність з коду, не з моделі ──────────────────────────────────────────

def test_the_dice_are_reproducible_for_the_same_seed():
    base = [Beat(хто="did", хід="згадати") for _ in range(6)]
    assert ([(b.хто, b.хід) for b in scatter(base, ["did", "koval"], 7, NEWS)]
            == [(b.хто, b.хід) for b in scatter(base, ["did", "koval"], 7, NEWS)])


def test_another_seed_gives_another_conversation():
    base = [Beat(хто="did", хід="згадати") for _ in range(8)]
    a = scatter(base, ["did", "koval", "mati"], 1, NEWS)
    b = scatter(base, ["did", "koval", "mati"], 2, NEWS)
    assert [x.хто for x in a] != [x.хто for x in b] or len(a) != len(b)


def test_an_interrupter_is_never_the_one_being_interrupted():
    base = [Beat(хто="did", хід="згадати") for _ in range(10)]
    out = scatter(base, ["did", "koval"], 3, NEWS)
    for i, b in enumerate(out):
        if b.хід == "перебити":
            assert out[i - 1].хто != b.хто


# ── прогін ────────────────────────────────────────────────────────────────────

def test_a_plain_news_without_quotes_still_gives_several_voices():
    """Головний гейт: раніше фан-аут різався по лапках, тож новина реченням давала ОДИН голос."""
    cast = [p.role for p in cast_for(NEWS, 3)]
    agent, _ = build([score(beat(cast[0]), beat(cast[1], "засумніватись", 1),
                            beat(cast[2], "спитати_діло", 2))]
                     + lines(14), width=3)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    voices = {l.split(":")[0] for l in (result.answer or "").splitlines()}
    assert len(voices) >= 3, result.answer


def test_a_run_is_never_an_abstain():
    """★ Той самий інструмент, що давав «нема в довіднику», тепер дає лише репліку."""
    agent, _ = build([score(beat("did", tool="словник", query="вовк"))]
                     + [line("Піду в дяка спитаю.")] + [line("У книзі того нема.")] * 8,
                     tools=FakeToolbox(tools=LEXIS_TOOLS), width=3)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    assert result.outcome == "answer"
    assert result.outcome != "abstain"
    assert result.evidence is None, "у розмові немає стану доказів — немає й відмови"


def test_an_empty_conversation_is_a_failure_not_an_abstain():
    agent, _ = build([score(beat("did"))] + [line("")] * 10, width=2)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    assert result.outcome == "failure"


def test_a_broken_score_still_produces_a_conversation():
    """Партитура — не єдина точка відмови: без неї кожен реагує по разу."""
    agent, _ = build(["не json"] + lines(14), width=3)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    assert result.outcome == "answer"
    assert (result.answer or "").count("\n") >= 2


def test_the_starosta_speaks_last_and_the_priest_doubts():
    agent, _ = build([score(beat("did"))] + lines(14), width=2)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    names = [l.split(":")[0] for l in (result.answer or "").splitlines()]
    assert "староста" in names
    assert "піп" in names


def test_a_drifted_line_is_retried_and_reported():
    agent, _ = build([score(beat("did"))] + ["{}"] + lines(14), width=2)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    assert any(i.startswith("viche_drift") for i in result.incidents)


def test_the_budget_stops_the_conversation_without_killing_it():
    agent, _ = build([score(*[beat("did") for _ in range(10)])]
                     + lines(22), width=2)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=4, max_tokens=99_999))
    assert "viche_budget" in result.incidents
    assert result.outcome == "answer", "обрізана розмова — все одно розмова"


# ── ярус: пара в дії ──────────────────────────────────────────────────────────

def test_the_expensive_slot_is_called_a_handful_of_times_not_per_line():
    from ploshcha_sim.adapters.router_profile import profile_router

    pair = [p.role for p in cast_for(NEWS, 2)]
    mamay = FakeLlm([score(beat(pair[0]), beat(pair[1], "піддакнути", 1))]
                    + [line("Слово старости.")] + [line("Сумнів.")] * 6, model="mamay")
    lapa = FakeLlm(lines(22), model="lapa")
    agent = Viche(profile_router(lapa, mamay), PresetEffort(), None, width=2, run_id="r")
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    assert result.tokens_by_lane.get("lapa", 0) > 0
    assert len(mamay.calls) <= 3, "Mamay — партитура, підсумок, сумнів; не по виклику на репліку"
    assert len(lapa.calls) >= 2


# ── проєкція: ритуал у локації ────────────────────────────────────────────────

def test_the_span_carries_the_role_so_the_voice_is_the_person():
    assert villager_of_span("r/viche/koval/3") == "koval"
    assert villager_of_span("r/viche/starosta/0") == "starosta"


def test_an_unmarked_span_still_falls_back_to_a_villager():
    assert villager_of_span("graph/2")


@pytest.mark.parametrize("tool,poi", sorted(POI_OF_TOOL.items()))
def test_every_tool_has_a_place_to_go(tool, poi):
    assert poi in {"well", "church", "forge", "square"}


def test_asking_a_tool_walks_the_person_there_first():
    """Виклик інструмента мусить бути ВИДНИЙ: спершу людина йде, аж тоді питає."""
    from ploshcha_sim.ports.trace import StepRecord

    proj = StreamProjector("r", "2026-01-01T00:00:00Z")
    events = proj.feed(StepRecord(run_id="r", tick=1, agent="tool", stage="tool_result",
                                  span="r/viche/mirosh/2", model="tool", lane="none",
                                  prompt="", raw_output="",
                                  parsed={"tool": "словник", "ok": True, "found": False}))
    moved = next(e for e in events if e["type"] == "agent.moved")
    assert moved["payload"]["agentId"] == "mirosh"
    assert moved["payload"]["to"] == {"poi": POI_OF_TOOL["словник"]}
    assert [e["type"] for e in events].index("agent.moved") < \
           [e["type"] for e in events].index("tool.result")


def test_two_villagers_move_independently():
    """Спільний POI на сцену гасив би рух усіх, крім першого."""
    from ploshcha_sim.ports.trace import StepRecord

    proj = StreamProjector("r", "2026-01-01T00:00:00Z")
    first = proj.feed(StepRecord(run_id="r", tick=1, agent="subagent", span="r/viche/did/1",
                                 stage="speak", model="m", lane="lapa", prompt="",
                                 raw_output="Кажу перше."))
    second = proj.feed(StepRecord(run_id="r", tick=2, agent="subagent", span="r/viche/koval/2",
                                  stage="speak", model="m", lane="lapa", prompt="",
                                  raw_output="Кажу друге."))
    assert [e["payload"]["agentId"] for e in first if e["type"] == "agent.moved"] == ["did"]
    assert [e["payload"]["agentId"] for e in second if e["type"] == "agent.moved"] == ["koval"]


# ── сторожі, які виросли з першого живого прогону ─────────────────────────────

def test_a_line_that_retells_the_news_is_rejected():
    """Живий прогін: «дід Свирид: Кажуть, за річкою бачили вовка… Спитай, що робити практично»."""
    from ploshcha_sim.agents.viche import _echoes

    assert _echoes(NEWS, NEWS, "ТВІЙ ХІД: згадати")
    assert not _echoes("Треба кошару обгородити, поки не пізно.", NEWS, "ТВІЙ ХІД: згадати")


def test_a_line_that_repeats_a_neighbour_is_rejected():
    """Живий прогін: одна фраза прозвучала ЧОТИРИ рази від різних людей."""
    from ploshcha_sim.agents.viche import _too_similar

    said = ["Памʼятаю, як торік вовк до кошари забрався, то вівці порозбігалися."]
    assert _too_similar("Памʼятаю, як торік вовк до кошари забрався, то вівці порозбігалися.", said)
    assert not _too_similar("А я кажу, то був здичавілий пес, не вовк.", said)


def test_the_persona_lens_lives_in_the_system_not_in_the_request():
    """Лінза в тексті запиту переказувалась дослівно; у системному повідомленні — ні."""
    agent, llm = build([score(beat("did"))] + lines(8), width=2)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    speak = [c for c in llm.calls if "ТВІЙ ХІД" in (c.get("prompt") or "")]
    assert speak, "мусить бути хоч один такт"
    assert "памʼять" not in speak[0]["prompt"] and "лінза" not in speak[0]["prompt"].lower()
    assert "Дивишся на світ так" in speak[0]["system"]


def test_a_repeat_is_retried_on_the_cheap_lane_before_the_expensive_one():
    """Ремонт дефекту виконавця не має оплачуватись оркестратором: спершу перепит, потім ескалація."""
    from ploshcha_sim.adapters.router_profile import profile_router

    same = line("Одна й та сама фраза геть без жодної зміни тут.")
    pair = [p.role for p in cast_for(NEWS, 2)]
    mamay = FakeLlm([score(beat(pair[0]), beat(pair[1]))] + [same] * 10, model="mamay")
    lapa = FakeLlm([same] * 10, model="lapa")
    agent = Viche(profile_router(lapa, mamay), PresetEffort(), None, width=2, run_id="r")
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    assert any(i.startswith("viche_same") for i in result.incidents)
    escalations = [i for i in result.incidents if i.startswith("viche_escalate")]
    assert len(escalations) <= len([i for i in result.incidents if i.startswith("viche_same")])


def test_a_rejected_line_never_reaches_the_scene():
    """Живий прогін: сцена промовляла репліки, які ядро забракувало як повтор — чужим голосом."""
    from ploshcha_sim.adapters import InMemoryTrace

    same = line("Одна й та сама фраза геть без жодної зміни отут.")
    good = line("А я кажу зовсім інше, бо бачив усе на власні очі.")
    trace = InMemoryTrace()
    agent, _ = build([score(beat(p.role) for p in cast_for(NEWS, 2))] if False else
                     [score(beat(cast_for(NEWS, 2)[0].role), beat(cast_for(NEWS, 2)[1].role))]
                     + [good, same, same, good] + lines(10), width=2, trace=trace)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))

    voiced = [r.raw_output for r in trace.records if r.agent == "subagent"]
    kept = [ln.split(": ", 1)[1] for ln in (result.answer or "").splitlines() if ": " in ln]
    assert voiced, "прийняті репліки мусять бути в трасі"
    for text in voiced:
        assert text in kept, f"на сцену пішла репліка, якої немає у розмові: {text!r}"


def test_the_cost_of_a_rejected_attempt_is_still_counted():
    """Не озвучуємо — але й не ховаємо: спроба коштувала грошей, і це має бути видно."""
    same = line("Одна й та сама фраза геть без жодної зміни отут.")
    agent, llm = build([score(beat(cast_for(NEWS, 2)[0].role))] + [same] * 8, width=2)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    assert result.tokens > 0
    assert len(llm.calls) > len((result.answer or "").splitlines())


def test_no_single_voice_may_hog_the_conversation():
    """Живий замір: партитура віддала одній персоні 6 тактів із 15, і вона ж дала всі повтори."""
    from ploshcha_sim.domain.viche import MAX_BEATS, MAX_SHARE

    beats = repair_score({"такти": [beat("did") for _ in range(MAX_BEATS)]}, ["did", "koval"], [])
    assert len(beats) <= max(2, int(MAX_BEATS * MAX_SHARE))


def test_the_score_gets_its_own_output_budget():
    """Спільна стеля різала JSON партитури на півслові, парс падав, і план тихо викидався."""
    from ploshcha_sim.agents.viche import SCORE_TOKENS

    agent, llm = build([score(beat(cast_for(NEWS, 2)[0].role))] + lines(8), width=2)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    assert llm.calls[0]["max_tokens"] == SCORE_TOKENS
    assert SCORE_TOKENS > 800, "дванадцять тактів це ~800 токенів JSON"


def test_a_lost_score_is_loud_not_silent():
    agent, _ = build(["обрізаний {\"такти\": [{\"хто\""] + lines(10), width=2)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    assert "viche_score_lost" in result.incidents


def test_going_to_ask_is_shown_by_movement_not_by_a_flat_line():
    """Живий прогін дав «Йду дізнаюсь про «вовк»» — переказ підказки замість мови."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    agent, _ = build([score(beat(cast_for(NEWS, 2)[0].role, tool="словник", query="вовк"))]
                     + lines(8), tools=FakeToolbox(tools=LEXIS_TOOLS), width=2, trace=trace)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    for ln in (result.answer or "").splitlines():
        assert "дізна" not in ln.lower(), ln
    assert [r for r in trace.records if r.agent == "tool"], "інструмент мусить бути викликаний"


# ── склад оголошує ЯДРО, а не фікстура ────────────────────────────────────────

def test_the_core_announces_its_own_cast():
    """Корінь цілого класу: доти вісім селян приходили з `quiet-day.jsonl`, тож староста й піп
    для сцени не існували, а імена в тексті й на підписі могли розійтись."""
    from ploshcha_sim.domain.viche import public_cast

    cast = public_cast()
    ids = {p["id"] for p in cast}
    assert {"starosta", "pip"} <= ids, "хто говорить у фіналі, мусить бути в касті"
    assert ids >= {p.role for p in PERSONAS}
    assert "hist" in ids, "гість — теж людина в гурті, а не бог над селом"
    # Справжній інваріант не «роль дорівнює id» (гість позичає спрайт чумака), а «спрайт існує»:
    # роль без малюнка = людина без бульбашки на карті, тобто той самий клас дефекту, що вже ловили.
    sprites = {p.name for p in
               (pathlib.Path(__file__).resolve().parents[3] / "apps/web/public/assets/roles").iterdir()
               if p.is_dir()}
    for person in cast:
        assert person["name"], person
        assert person["role"] in sprites, f"{person['id']}: нема спрайта {person['role']}"


def test_casting_done_is_emitted_right_after_run_started():
    from ploshcha_sim.domain.viche import public_cast

    proj = StreamProjector("r", "2026-01-01T00:00:00Z", scene={"id": "ploshcha", "name": "Площа"},
                           cast=public_cast())
    types = [e["type"] for e in proj.start()]
    assert types == ["run.started", "casting.done"]


def test_without_a_cast_nothing_extra_is_announced():
    proj = StreamProjector("r", "2026-01-01T00:00:00Z", scene={"id": "p", "name": "П"})
    assert [e["type"] for e in proj.start()] == ["run.started"]


def test_every_voice_the_core_can_use_is_in_the_cast():
    """Гарантія проти повторення дефекту: голос, якого нема в касті, не має бульбашки на карті."""
    from ploshcha_sim.adapters.projector import VOICE_OF_LANE, VOICE_VERIFIER
    from ploshcha_sim.domain.viche import public_cast

    ids = {p["id"] for p in public_cast()}
    assert VOICE_VERIFIER in ids
    assert set(VOICE_OF_LANE.values()) <= ids


def test_a_tool_call_and_its_result_come_in_pairs():
    """Доти віче емітило лише `tool.result` — порахувати походи по довідник було неможливо."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    agent, _ = build([score(beat(cast_for(NEWS, 2)[0].role, tool="словник", query="вовк"))]
                     + lines(8), tools=FakeToolbox(tools=LEXIS_TOOLS), width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))

    proj = StreamProjector("r", "2026-01-01T00:00:00Z")
    types = [e["type"] for r in trace.records for e in proj.feed(r)]
    assert types.count("tool.called") == types.count("tool.result") >= 1
    assert types.index("tool.called") < types.index("tool.result")


def test_diagnostic_notes_do_not_masquerade_as_replans():
    """Живий прогін: літопис писав «передумали: beats=17» і витісняв справжні події зі сцени."""
    class Result:
        outcome = "answer"
        evidence = None
        scratch: list = []
        incidents: list = []
        notes = ["viche", "beats=17", "lines=13", "voices=7",
                 "план переглянуто: інструмент не відповів"]

    proj = StreamProjector("r", "2026-01-01T00:00:00Z")
    revised = [e for e in proj.close(Result()) if e["type"] == "plan.revised"]
    assert len(revised) == 1
    assert revised[0]["payload"]["reason"].startswith("план переглянуто")


def test_the_executor_lane_is_visible_not_only_the_orchestrator():
    """`lanes` показував {mamay: 3} і нуль Lapa — при тому що кожну репліку промовляє Lapa."""
    from ploshcha_sim.ports.trace import StepRecord

    proj = StreamProjector("r", "2026-01-01T00:00:00Z")
    events = proj.feed(StepRecord(run_id="r", tick=1, agent="subagent", span="r/viche/did/1",
                                  stage="speak", model="m", lane="lapa", prompt="",
                                  raw_output="Кажу своє."))
    route = next(e for e in events if e["type"] == "route.decided")
    assert route["payload"]["lane"] == "lapa"


# ── план, хроніка, думки: типи, які фронт умів малювати, а ядро не надсилало ───

def chron(*thoughts, mood="тривога", force=0.8) -> str:
    return json.dumps({"заголовок": "Вовк за річкою", "оповідь": "Село погомоніло й розійшлось.",
                       "настрій": mood, "сила": force,
                       "думки": [{"хто": r, "думка": t} for r, t in thoughts]},
                      ensure_ascii=False)


def _events(trace):
    proj = StreamProjector("r", "2026-01-01T00:00:00Z")
    return [e for r in trace.records for e in proj.feed(r)]


def test_the_score_becomes_a_plan_hanging_on_the_board():
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0]), beat(pair[1], "заперечити", 1))]
                     + lines(4) + [chron((pair[0], "Треба було раніше."))],
                     width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    plan = next(e for e in _events(trace) if e["type"] == "plan.formed")
    assert plan["payload"]["agentId"] == "starosta"
    assert plan["payload"]["steps"], "порядок мусить бути читабельним, а не логом"
    assert plan["payload"]["steps"][0].startswith("1. ")


def test_the_chronicler_gives_a_day_and_a_mood():
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0]))] + lines(3)
                     + [chron((pair[0], "Лишилось тривожно."))], width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    report = next(e for e in _events(trace) if e["type"] == "report.compiled")
    mood = report["payload"]["chronicle"]["mood"]
    assert mood["label"] == "тривога"
    assert mood["valence"] < 0, "знак настрою бере ЯРЛИК, а не число від моделі"


def test_reflections_reach_the_inspector():
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0]))] + lines(3)
                     + [chron((pair[0], "А я ж казав."))], width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    thought = next(e for e in _events(trace) if e["type"] == "reflection.formed")
    assert thought["payload"]["agentId"] == pair[0]
    assert thought["payload"]["thought"] == "А я ж казав."


def test_a_lost_chronicle_is_loud_not_silent():
    """Літописець працює лише за наявності спостерігача — не платимо за те, чого ніхто не бачить."""
    from ploshcha_sim.adapters import InMemoryTrace

    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0]))] + lines(3) + ["не json"], width=2,
                     trace=InMemoryTrace())
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert "viche_chronicle_lost" in result.incidents


def test_the_mood_sign_comes_from_the_label_not_the_model():
    """Модель віддавала б «тривога» з додатною силою, і погода суперечила б тексту."""
    from ploshcha_sim.domain.viche import mood_view

    assert mood_view("тривога", 1.0)["valence"] < 0
    assert mood_view("радість", 1.0)["valence"] > 0
    assert -1.0 <= mood_view("туга", 99)["valence"] <= 1.0


def test_without_an_observer_the_chronicle_is_not_paid_for():
    """Зворотний бік того ж рішення: у пакетному прогоні хроніка не коштує жодного виклику."""
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, llm = build([score(beat(pair[0]))] + lines(6), width=2)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert not [c for c in llm.calls if "РОЗМОВА:" in (c.get("prompt") or "")]


def test_the_finale_is_not_starved_by_the_conversation_budget():
    """Стеля обмежує розмову, не її закриття: інакше хроніка зникала саме на довгих вічах."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(*[beat(pair[i % 2]) for i in range(6)])] + lines(3)
                     + [chron((pair[0], "Лишилось тривожно."))], width=2, trace=trace)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=3, max_tokens=99_999))

    assert "viche_budget" in result.incidents, "розмова МУСИТЬ обрізатись стелею"
    types = {e["type"] for e in _events(trace)}
    assert "report.compiled" in types, "а закриття — ні"
    assert "reflection.formed" in types


def test_a_flaky_structured_call_is_retried_once():
    """Збій структурованого виводу шлюзу ПЕРЕРИВЧАСТИЙ: та сама схема то проходить, то ні.
    Без перепиту один невдалий виклик знецінював усю розмову."""
    from ploshcha_sim.adapters import InMemoryTrace

    pair = [p.role for p in cast_for(NEWS, 2)]
    trace = InMemoryTrace()
    agent, _ = build(["обірваний {", score(beat(pair[0]))] + lines(3)
                     + ["теж обірваний {", chron((pair[0], "Отак."))], width=2, trace=trace)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    assert "viche_score_retry" in result.incidents
    assert "viche_score_lost" not in result.incidents, "перепит мусить врятувати партитуру"
    assert "viche_chronicle_retry" in result.incidents
    assert "viche_chronicle_lost" not in result.incidents
    assert "report.compiled" in {e["type"] for e in _events(trace)}


# ── Ш1: ти в розмові ──────────────────────────────────────────────────────────

def test_your_word_enters_the_conversation_and_gets_answered():
    """Головна межа: доти ти був за склом — писав тему й дивився."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    cast = [p.role for p in cast_for(NEWS, 4)]
    agent, _ = build([score(*[beat(cast[i % len(cast)]) for i in range(6)])] + lines(16),
                     width=4, trace=trace)
    agent.tell({"kind": "say", "text": "А чи не пес то часом заблукав?"})
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    names = [ln.split(":")[0] for ln in (result.answer or "").splitlines()]
    assert "ти" in names, "твоя репліка мусить бути В розмові, а не поруч із нею"
    assert "viche_guest" in result.incidents
    said = [e for e in _events(trace) if e["type"] == "utterance.spoken"]
    assert any(e["payload"]["agentId"] == "hist" for e in said), "голос гостя йде на сцену"


def test_your_word_does_not_hijack_the_whole_viche():
    """Двоє відгукуються, не всі: інакше кожна твоя репліка спиняла б розмову."""
    from ploshcha_sim.domain.viche import GUEST_REPLIES, guest_beats

    out = guest_beats(3, ["did", "koval", "mati", "pip"], ["did"], 1, "слово")
    assert len(out) == GUEST_REPLIES
    assert len({b.хто for b in out}) == GUEST_REPLIES, "відгукуються РІЗНІ люди"


def test_answers_come_from_those_who_did_not_just_speak():
    from ploshcha_sim.domain.viche import guest_beats

    out = guest_beats(3, ["did", "koval", "mati", "pip"], ["did", "koval"], 5, "слово")
    assert not ({b.хто for b in out} & {"did", "koval"})


def test_a_whisper_is_carried_by_the_one_you_told():
    """Пошептане має ЙТИ В РОЗМОВУ — інакше це просто нотатка нікуди."""
    cast = [p.role for p in cast_for(NEWS, 3)]
    agent, llm = build([score(beat(cast[0]), beat(cast[1]))] + lines(10), width=3)
    agent.tell({"kind": "whisper", "to": cast[0], "text": "кажуть, то пес шинкаря"})
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    carried = [c for c in llm.calls if "ПОШЕПТАЛИ" in (c.get("prompt") or "")]
    assert len(carried) == 1, "шепіт іде РІВНО одному, і рівно раз"
    assert "то пес шинкаря" in carried[0]["prompt"]


def test_a_whisper_to_a_stranger_is_dropped_not_crashed():
    cast = [p.role for p in cast_for(NEWS, 3)]
    agent, llm = build([score(beat(cast[0]))] + lines(8), width=3)
    agent.tell({"kind": "whisper", "to": "лісовик", "text": "щось"})
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert not [c for c in llm.calls if "ПОШЕПТАЛИ" in (c.get("prompt") or "")]


def test_an_empty_word_changes_nothing():
    cast = [p.role for p in cast_for(NEWS, 3)]
    agent, _ = build([score(beat(cast[0]))] + lines(8), width=3)
    agent.tell({"kind": "say", "text": "   "})
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert "viche_guest" not in result.incidents


def test_speaking_into_a_dead_viche_says_so_instead_of_silence():
    """«Я написав, і нічого» вже раз виглядало як поламка — тому тут ЧЕСНА відмова."""
    from ploshcha_sim.live.server import handle_command

    class Runner:
        current = None
        queue = None

    code, body = handle_command({"kind": "say", "text": "агов"}, Runner())
    assert code == 409 and "віча немає" in body["error"]


# ── Ш2: ухвала з наслідком ────────────────────────────────────────────────────

def chron_d(*thoughts, decided="так", what="поставити сторожа коло кошари",
            who=None, where="ploshcha", mood="тривога") -> str:
    return json.dumps({"заголовок": "Вовк", "оповідь": "Погомоніли.", "настрій": mood,
                       "сила": "дуже",
                       "ухвала": {"ухвалено": decided, "що": what,
                                  "хто": who or thoughts[0][0], "де": where},
                       "думки": [{"хто": r, "думка": t} for r, t in thoughts]},
                      ensure_ascii=False)


def test_a_decision_puts_someone_in_a_place():
    """Рішення без сліду у світі — це просто ще один рядок тексту."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0]))] + lines(3)
                     + [chron_d((pair[0], "Отак."), where="kuznya")], width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    events = _events(trace)
    ev = next(e for e in events if e["type"] == "event.happened")
    assert ev["payload"]["event"]["kind"] == "decision"
    assert ev["payload"]["event"]["place"] == {"poi": "kuznya"}
    moved = [e for e in events if e["type"] == "agent.moved"
             and e["payload"]["to"] == {"poi": "kuznya"}]
    assert moved, "доручений мусить СТАТИ на місце, а не лишитись написом"


def test_no_agreement_means_no_decision():
    """«Не зійшлись» — теж чесний результат; вигадувати ухвалу не можна."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0]))] + lines(3)
                     + [chron_d((pair[0], "Отак."), decided="ні")], width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert not [e for e in _events(trace) if e["type"] == "event.happened"]


def test_a_decision_for_a_stranger_is_dropped():
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0]))] + lines(3)
                     + [chron_d((pair[0], "Отак."), who="лісовик")], width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert not [e for e in _events(trace) if e["type"] == "event.happened"]


def test_only_places_that_exist_on_the_scene_are_offered():
    """Місце, якого нема на сцені, дало б рішення без наслідку — знову «намальовану» механіку."""
    import json as _json

    from ploshcha_sim.domain.viche import DECISION_POIS

    scene = _json.loads((pathlib.Path(__file__).resolve().parents[3]
                         / "packages/fixtures/scenes/verbolozy.scene.json").read_text(encoding="utf-8"))
    assert set(DECISION_POIS) <= {p["id"] for p in scene["pois"]}


def test_standing_decisions_survive_and_come_back(tmp_path):
    from ploshcha_sim.adapters.decisions_sqlite import SqliteDecisions

    store = SqliteDecisions(tmp_path / "d.db")
    store.add("вовк", "стерегти кошару", "parubok", "ploshcha")
    store.add("мито", "рахувати збитки", "mirosh", "mlyn")
    assert {d["who"] for d in store.standing()} == {"parubok", "mirosh"}


def test_one_person_stands_in_one_place(tmp_path):
    """Інакше сцена намагалась би поставити людину у двох місцях одразу."""
    from ploshcha_sim.adapters.decisions_sqlite import SqliteDecisions

    store = SqliteDecisions(tmp_path / "d.db")
    store.add("вовк", "стерегти вдень", "parubok", "ploshcha")
    store.add("вовк", "стерегти вночі", "parubok", "dzvin")
    standing = store.standing()
    assert len(standing) == 1
    assert standing[0]["poi"] == "dzvin", "чинне ОСТАННЄ доручення"


# ── Ш3: породження людей ──────────────────────────────────────────────────────

def test_a_trait_names_the_pole_not_the_axis():
    """★ Найгірший клас: вісь «старий» описує і старого, і молодого. Ядро слало НАЗВУ ОСІ, тож
    дівчина з віком 0.00 приїжджала на сцену з міткою «старий» і фарбувалась сивиною."""
    from ploshcha_sim.domain.people import Person

    young = Person(role="parubok", traits={"старий": 0.02, "гарячий": 0.9})
    keys = [t.key for t in young.marked]
    assert "молодий" in keys and "старий" not in keys
    assert "гарячий" in keys


def test_the_role_bends_the_dice_but_does_not_fix_it():
    """Кубик без ролі дав молодого діда — село перестало читатись. Але діапазон мусить лишитись."""
    import statistics

    from ploshcha_sim.domain.people import roll_traits

    old = statistics.mean(roll_traits(s, "did")["старий"] for s in range(40))
    young = statistics.mean(roll_traits(s, "parubok")["старий"] for s in range(40))
    assert old > 0.7 and young < 0.3
    spread = [roll_traits(s, "did")["старий"] for s in range(40)]
    assert max(spread) - min(spread) > 0.3, "зміщення не сміє перетворитись на константу"


def test_the_same_seed_is_the_same_village():
    from ploshcha_sim.domain.people import roll_traits, village_roles

    roles = [p.role for p in PERSONAS]
    assert village_roles(5, roles, 6) == village_roles(5, roles, 6)
    assert roll_traits(5, "did") == roll_traits(5, "did")
    assert village_roles(5, roles, 6) != village_roles(6, roles, 6)


def test_a_trait_changes_the_score_not_just_the_label():
    """Ознака, яка нічого не міняє в поведінці, — наліпка."""
    from ploshcha_sim.domain.people import Person, roll_traits
    from ploshcha_sim.domain.viche import interrupt_chance

    hot = Person(role="parubok", traits={**roll_traits(1, "parubok"), "гарячий": 0.98})
    calm = Person(role="pip", traits={**roll_traits(1, "pip"), "гарячий": 0.02})
    assert interrupt_chance(hot) > interrupt_chance(calm) * 1.6


def test_an_outsider_has_no_access_to_the_village_memory():
    from ploshcha_sim.domain.people import Person, remembers

    assert not remembers(Person(role="chumak", traits={"прийшлий": 0.95}))
    assert remembers(Person(role="did", traits={"прийшлий": 0.05}))


def test_the_model_never_gets_to_rewrite_the_dice():
    """Норов визначений кубиком; віддати його моделі означало б віддати те, що вже вирішено даними."""
    from ploshcha_sim.domain.people import people_schema, repair_people, roll_traits

    fields = people_schema(["did"])["properties"]["люди"]["items"]["properties"]
    assert "норов" not in fields and "traits" not in fields

    mine = roll_traits(9, "did")
    got = repair_people({"люди": [{"роль": "did", "імʼя": "Дід", "про_себе": "",
                                   "примовка": "", "traits": {"старий": 0.0}}]},
                        ["did"], {"did": mine})
    assert got[0].traits == mine


def test_a_silent_model_still_leaves_a_village():
    """Краще людина без історії, ніж село, яке мовчки поменшало."""
    from ploshcha_sim.adapters.router_profile import single_model_router
    from ploshcha_sim.agents.forge import forge_village

    roles = [p.role for p in PERSONAS]
    lenses = {p.role: p.lens for p in PERSONAS}
    people = forge_village(single_model_router(FakeLlm(["не json"], model="f")), PresetEffort(),
                           seed=3, roles=roles, lenses=lenses, size=5)
    assert len(people) == 5
    assert all(p.traits for p in people)


def test_the_village_survives_a_restart(tmp_path):
    from ploshcha_sim.adapters.village_sqlite import SqliteVillage
    from ploshcha_sim.domain.people import Person, roll_traits

    store = SqliteVillage(tmp_path / "v.db")
    assert store.load(11) == []
    folk = [Person(role="did", name="Дід Мирон", traits=roll_traits(11, "did"))]
    store.save(11, folk)
    back = store.load(11)
    assert [p.name for p in back] == ["Дід Мирон"]
    assert back[0].traits == folk[0].traits


def test_a_corrupt_village_regenerates_instead_of_crashing(tmp_path):
    import sqlite3

    from ploshcha_sim.adapters.village_sqlite import SqliteVillage

    store = SqliteVillage(tmp_path / "v.db")
    with sqlite3.connect(store.path) as db:
        db.execute("INSERT INTO village(seed, people) VALUES(?,?)", (11, "{побите"))
    assert store.load(11) == []
