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
from ploshcha_sim.agents.viche import MAX_WAVES, Viche
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

class WaveLlm(FakeLlm):
    """Фейк, що роздає відповіді ЗА ПРИЗНАЧЕННЯМ, а не однією чергою.

    Партитуру тепер просять кілька разів (хвилі), тож лінійний скрипт зсувався: хвиля зʼїдала
    рядок із реплік, і до літописця доїжджало не те. Тут кожен вид виклику має свою чергу —
    рівно як у справжнього шлюзу, де виклики незалежні. Партитура, коли черга вичерпалась,
    повторює останню вдалу: це й означає «хвиля не принесла нового», тобто розмова добігає кінця.
    """

    def __init__(self, responses, model: str = "fake", finish_reason: str = "stop",
                 strict: bool = False):
        super().__init__(responses, model=model, finish_reason=finish_reason, strict=strict)
        self.q: dict[str, list[str]] = {"score": [], "line": [], "chron": []}
        for r in responses:
            self.q[_kind_of(r)].append(r)

    def _next(self, prompt, system, structured, schema, seed, temperature=0.0, max_tokens=0):
        props = (schema or {}).get("properties") if isinstance(schema, dict) else None
        kind = ("score" if props and "такти" in props
                else "vote" if props and "голос" in props
                else "chron" if props and "заголовок" in props
                else "line")
        if kind == "vote":
            self._responses = ['{"голос": "за", "чому": "бо село так вирішило"}']
        elif kind == "score":
            self._responses = [self.q["score"].pop(0)] if self.q["score"] else [""]
        else:
            self._responses = [self.q[kind].pop(0)] if self.q[kind] else []
        return super()._next(prompt, system, structured, schema, seed, temperature, max_tokens)


def _kind_of(raw: str) -> str:
    if "такти" in raw:
        return "score"
    if '"репліка"' in raw:
        return "line"
    if "заголовок" in raw:
        return "chron"
    return "line"  # решта скрипта — репліки, зокрема навмисно биті («{}»)


def build(replies, *, tools=None, width=3, trace=None):
    llm = WaveLlm(replies, model="fake")
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
    # Економіка змінилась свідомо: партитура тепер не одна на прогін, а ХВИЛЯМИ — інакше
    # аргументи ні на що не впливали, бо черга була написана до першого слова. Але межа лишається
    # тією самою за суттю: оркестратор коштує кілька викликів на розмову, а не виклик на репліку.
    assert len(mamay.calls) <= MAX_WAVES + 3, "хвилі + підсумок + сумнів + літопис — і не більше"
    assert len(mamay.calls) < len(lapa.calls), "виконавця кличуть на кожну репліку, оркестратора — ні"
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
    agent, _ = build(["такти обірваний {", score(beat(pair[0]))] + lines(3)
                     + ["заголовок обірваний {", chron((pair[0], "Отак."))], width=2, trace=trace)
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


# ── Ш4: чутка й репутація ─────────────────────────────────────────────────────

def chron_r(*thoughts, rumour="так", who=None, claim="то не вовк, а пес шинкаря",
            ground="не було", decided="ні") -> str:
    return json.dumps({"заголовок": "Вовк", "оповідь": "Погомоніли.", "настрій": "тривога",
                       "сила": "помірно",
                       "чутка": {"є": rumour, "хто": who or thoughts[0][0], "що": claim,
                                 "підстава": ground},
                       "ухвала": {"ухвалено": decided, "що": "-", "хто": who or thoughts[0][0],
                                  "де": "ploshcha"},
                       "думки": [{"хто": r, "думка": t} for r, t in thoughts]},
                      ensure_ascii=False)


def test_a_claim_without_ground_becomes_a_rumour():
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0]))] + lines(3) + [chron_r((pair[0], "Отак."))],
                     width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    ev = next(e for e in _events(trace) if e["type"] == "event.happened")
    assert ev["payload"]["event"]["kind"] == "rumour"
    assert "пес шинкаря" in ev["payload"]["event"]["label"]


def test_a_claim_with_ground_is_not_a_rumour():
    """Якщо підстава була — це просто слово, і в обіг воно не йде."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0]))] + lines(3)
                     + [chron_r((pair[0], "Отак."), ground="була")], width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert not [e for e in _events(trace) if e["type"] == "event.happened"]


def test_being_refuted_costs_you_speaking_time():
    """★ Заради чого репутація й існує: не напис, а буквально менше тактів."""
    roles = [p.role for p in PERSONAS]
    raw = {"такти": [beat(roles[0]) for _ in range(MAX_BEATS)]}
    full = repair_score(raw, roles, [])
    punished = repair_score(raw, roles, [], {roles[0]: 0.4})
    assert len(punished) < len(full)
    assert punished, "людину не викидають із села за помилку"


def test_being_right_returns_trust_but_buys_nothing_extra(tmp_path):
    """Інакше один щасливий здогад робив би людину головною назавжди."""
    from ploshcha_sim.adapters.rumours_sqlite import SqliteRumours

    store = SqliteRumours(tmp_path / "r.db")
    store.add("t", "shynkar", "щось")
    store.settle(1, "спростована")
    hurt = store.standing("shynkar")
    store.add("t", "shynkar", "інше")
    store.settle(2, "підтверджена")
    assert hurt < 1.0
    assert store.standing("shynkar") == 1.0


def test_reputation_has_a_floor(tmp_path):
    from ploshcha_sim.adapters.rumours_sqlite import SqliteRumours

    store = SqliteRumours(tmp_path / "r.db")
    for i in range(1, 12):
        store.add("t", "did", f"чутка {i}")
        store.settle(i, "спростована")
    assert store.standing("did") >= 0.4


def test_open_rumours_reach_the_next_score():
    """Чутка мусить ХОДИТИ селом, інакше вона просто рядок у базі."""
    pair = [p.role for p in cast_for(NEWS, 2)]
    llm = FakeLlm([score(beat(pair[0]))] + lines(6), model="f")
    agent = Viche(single_model_router(llm), PresetEffort(), None, width=2, run_id="r",
                  rumours=[{"who": "shynkar", "claim": "то пес, а не вовк"}])
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert "то пес, а не вовк" in llm.calls[0]["prompt"]


def test_the_builder_forwards_every_parameter_viche_accepts():
    """★ `build_viche` мовчки ковтав `village`, `standing`, `rumours` і `place`: агент працював зі
    сталими персонами, поки сцена показувала породжені імена, а режим місця не доїжджав узагалі.
    Мовчазне ковтання kwargs — той самий клас, що вже коштував нам нетрасованого графа."""
    import inspect

    from ploshcha_sim.compose import VICHE_KWARGS

    base = {"router", "effort", "tools", "trace", "run_id", "width", "system",
            "prompt_id", "prompt_sha", "self"}
    accepted = set(inspect.signature(Viche.__init__).parameters) - base
    assert accepted == set(VICHE_KWARGS), f"розійшлось: {accepted ^ set(VICHE_KWARGS)}"


def test_the_place_actually_changes_the_run():
    """Режим, який не міняє нічого, крім підпису, не потрібен."""
    from ploshcha_sim.domain.modes import mode_for

    tavern, square = mode_for("shynok"), mode_for("ploshcha")
    assert tavern.summary is False and square.summary is True, "у шинку старости НЕМА"
    assert tavern.interrupts > square.interrupts
    assert mode_for("tserkva").width < square.width
    assert mode_for("tserkva").rumours is False, "сповідь не пускають селом"


def test_a_tavern_viche_has_no_elder_and_no_priest():
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0]))] + lines(8), width=4)
    agent.mode = __import__("ploshcha_sim.domain.modes", fromlist=["x"]).mode_for("shynok")
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    names = [ln.split(":")[0] for ln in (result.answer or "").splitlines()]
    assert "староста" not in names and "піп" not in names


# ── Ш6: послати когось ────────────────────────────────────────────────────────

class _Scout:
    """Дитина-агент: робить кілька кроків і вертається з висновком, а не з сирим полем."""

    def __init__(self, answer="у книзі писано, що грамота справжня", steps=2, outcome="answer"):
        self.answer, self.steps, self.outcome = answer, steps, outcome
        self.seen: list[str] = []

    def __call__(self, budget):
        self.budget = budget
        return self

    def run(self, task, seed=0, budget=None, depth=1):
        from ploshcha_sim.domain.task import TaskResult

        self.seen.append(task)
        return TaskResult(
            answer=self.answer, accepted=True, outcome=self.outcome, evidence=True,
            steps=self.steps, tokens=140,
            scratch=[{"call": {"tool": "словник", "запит": task}, "found": True}
                     for _ in range(self.steps)])


def _with_scout(scout, **kw):
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, llm = build([score(beat(pair[0], tool="словник", query="грамота"))] + lines(8),
                       tools=FakeToolbox(tools=LEXIS_TOOLS), width=2, **kw)
    agent.scout = scout
    return agent, llm, pair


def test_sending_someone_spawns_a_child_agent_not_a_tool_call():
    from ploshcha_sim.adapters import InMemoryTrace

    scout = _Scout()
    trace = InMemoryTrace()
    agent, _, _ = _with_scout(scout, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    assert scout.seen == ["грамота"], "посланому дають ЗАПИТ, а не всю тему"
    events = _events(trace)
    assert len([e for e in events if e["type"] == "tool.called"]) == scout.steps
    assert len([e for e in events if e["type"] == "tool.result"]) == scout.steps


def test_the_scouts_steps_are_shown_as_that_persons_own():
    """Інакше на сцені це робив би хтось інший, і глядач бачив би не те, що сталось."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    scout = _Scout()
    agent, _, pair = _with_scout(scout, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    moved = [e for e in _events(trace) if e["type"] == "agent.moved"]
    assert any(e["payload"]["agentId"] == pair[0] for e in moved)


def test_the_scouts_spending_lands_in_our_budget():
    """Інакше стеля прогону нічого не обмежувала б: дитина витрачала б повз облік."""
    scout = _Scout()
    agent, _, _ = _with_scout(scout)
    budget = Budget(max_steps=40, max_tokens=99_999)
    agent.run(NEWS, seed=1, budget=budget)
    assert budget.tokens_used >= 140


def test_the_scout_gets_a_divided_budget_not_the_whole_one():
    scout = _Scout()
    agent, _, _ = _with_scout(scout)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=48, max_tokens=99_999))
    assert scout.budget.max_steps < 48


def test_a_scout_that_found_nothing_says_so_instead_of_inventing():
    scout = _Scout(answer="", outcome="abstain")
    agent, _, _ = _with_scout(scout)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert "viche_scout_empty" in result.incidents


def test_a_broken_scout_does_not_kill_the_viche():
    class Boom:
        def __call__(self, budget):
            return self

        def run(self, *a, **kw):
            raise RuntimeError("зламався")

    agent, _, _ = _with_scout(Boom())
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert any(i.startswith("viche_scout_failed") for i in result.incidents)
    assert result.outcome == "answer", "розмова мусить іти далі"


def test_without_a_scout_it_is_still_one_tool_call():
    """Розвідник — доповнення, не заміна: віче мусить працювати й без нього."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, _ = build([score(beat(pair[0], tool="словник", query="грамота"))] + lines(8),
                     tools=FakeToolbox(tools=LEXIS_TOOLS), width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    assert len([e for e in _events(trace) if e["type"] == "tool.called"]) == 1


def test_the_scout_does_not_lose_the_three_valued_found():
    """«Не знайшов» ≠ «зламався» ≠ «незастосовно». У сліді оркестратора `found` не лежить готовим,
    тож посланий показував «незастосовно» там, де насправді знав."""
    from ploshcha_sim.adapters import InMemoryTrace
    from ploshcha_sim.domain.task import TaskResult

    class Knowing:
        def __call__(self, budget):
            return self

        def run(self, task, seed=0, budget=None, depth=1):
            return TaskResult(answer="знайшов", accepted=True, outcome="answer", steps=1,
                              tokens=10,
                              scratch=[{"call": {"tool": "словник"}, "result": {"відомо": False}}])

    trace = InMemoryTrace()
    agent, _, _ = _with_scout(Knowing(), trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    res = next(e for e in _events(trace) if e["type"] == "tool.result")
    assert res["payload"]["found"] is False, "шукав і НЕ знайшов — це не «незастосовно»"


# ── Ш7: памʼять, стосунки, літопис ────────────────────────────────────────────

def test_the_village_remembers_a_related_viche(tmp_path):
    from ploshcha_sim.adapters import InMemoryTrace
    from ploshcha_sim.adapters.memory_sqlite import SqliteMemory

    mem = SqliteMemory(tmp_path / "m.db")
    mem.remember("вовк коло кошари", "Вовча напасть", "Село погомоніло й розійшлось.", "тривога")
    mem.remember("гребля протікає", "Гребля", "Дощі обіцяють.", "спокій")

    trace = InMemoryTrace()
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, llm = build([score(beat(pair[0]))] + lines(8), width=2, trace=trace)
    agent.memory = mem
    agent.run("Знову вовк коло кошари.", seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    assert "Вовча напасть" in llm.calls[0]["prompt"]
    assert "Гребля" not in llm.calls[0]["prompt"], "пригадують СПОРІДНЕНЕ, а не все підряд"
    recalled = [e for e in _events(trace) if e["type"] == "memory.recalled"]
    assert recalled and recalled[0]["payload"]["items"] == ["Вовча напасть"]


def test_an_outsider_is_not_told_what_the_village_remembers(tmp_path):
    """Прийшлий бачить те, чого свої вже не помічають — але лише якщо йому не переказали."""
    from ploshcha_sim.adapters.memory_sqlite import SqliteMemory
    from ploshcha_sim.domain.people import Person, roll_traits

    mem = SqliteMemory(tmp_path / "m.db")
    mem.remember("вовк коло кошари", "Вовча напасть", "Було таке.", "тривога")
    pair = [p.role for p in cast_for(NEWS, 2)]
    stranger = Person(role=pair[0], name="Прийшлий",
                      traits={**roll_traits(1, pair[0]), "прийшлий": 0.95})
    agent, llm = build([score(beat(pair[0]))] + lines(8), width=2)
    agent.memory = mem
    agent.village = [stranger]
    agent._people = {stranger.role: stranger}
    agent.run("Знову вовк коло кошари.", seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    speak = [c for c in llm.calls if "ТВІЙ ХІД" in (c.get("prompt") or "")]
    assert speak and "Село памʼятає" not in speak[0]["system"]


def test_bonds_are_derived_from_the_score_not_asked_of_the_model():
    """Хто кому піддакнув — зблизились; хто заперечив — розійшлись. Це вже є в партитурі."""
    from ploshcha_sim.domain.viche import bonds_from

    got = bonds_from([Beat(хто="did", хід="згадати"),
                      Beat(хто="koval", хід="піддакнути", у_відповідь=1),
                      Beat(хто="pip", хід="заперечити", у_відповідь=1)])
    assert ("koval", "did", 1.0) in got
    assert ("pip", "did", -1.0) in got


def test_a_beat_answering_itself_makes_no_bond():
    from ploshcha_sim.domain.viche import bonds_from

    assert bonds_from([Beat(хто="did", хід="піддакнути", у_відповідь=1)]) == []


def test_a_quarrel_makes_you_likelier_to_cut_that_person_off(tmp_path):
    """★ Інакше сварка лишалась би записом у базі, а не поведінкою."""
    import collections

    from ploshcha_sim.domain.people import Person, roll_traits
    from ploshcha_sim.domain.viche import scatter

    people = {r: Person(role=r, traits=roll_traits(1, r)) for r in ("did", "koval", "pip")}
    base = [Beat(хто="did", хід="згадати") for _ in range(8)]
    quarrel = {("did", "pip"): -6.0}

    def who_cuts(bonds):
        c = collections.Counter()
        for s in range(200):
            for b in scatter(base, list(people), s, "тема", people, 1.0, bonds):
                if b.хід == "перебити":
                    c[b.хто] += 1
        return c

    calm, angry = who_cuts({}), who_cuts(quarrel)
    assert angry["pip"] / max(1, angry["koval"]) > calm["pip"] / max(1, calm["koval"])


def test_the_chronicle_accumulates(tmp_path):
    from ploshcha_sim.adapters.memory_sqlite import SqliteMemory

    mem = SqliteMemory(tmp_path / "m.db")
    for i in range(3):
        mem.remember(f"тема {i}", f"День {i}", "оповідь", "спокій")
    book = mem.chronicle()
    assert [r["title"] for r in book] == ["День 2", "День 1", "День 0"], "найсвіжіше перше"


def test_bonds_do_not_run_away(tmp_path):
    from ploshcha_sim.adapters.memory_sqlite import BOND_CAP, SqliteMemory

    mem = SqliteMemory(tmp_path / "m.db")
    for _ in range(40):
        mem.bond("did", "pip", -1.0)
    assert mem.between("did", "pip") == -BOND_CAP


# ── позиції й голос: віче мусить ЩОСЬ вирішувати ──────────────────────────────

def test_a_move_shifts_the_stance_by_code_not_by_judgement():
    """Позицію рухає КОД: це визначено ходом і фактом, а не судженням моделі про власну розмову."""
    from ploshcha_sim.domain.viche import stance_after, stance_start, stance_label

    st = stance_start(["koval", "pip"])
    st = stance_after(Beat(хто="koval", хід="заперечити"), st, {}, None)
    st = stance_after(Beat(хто="pip", хід="піддакнути"), st, {}, None)
    assert stance_label(st["koval"]) == "проти"
    assert st["pip"] > 0

    # знайдений факт важить більше за слово, ненайдений — тягне назад
    plus = stance_after(Beat(хто="koval", хід="порахувати"), stance_start(["koval"]), {}, True)
    minus = stance_after(Beat(хто="koval", хід="порахувати"), stance_start(["koval"]), {}, False)
    assert plus["koval"] > 0 > minus["koval"]


def test_reputation_scales_how_much_a_voice_moves_others():
    """Кому спростували чутку, того слухають менше — не метафорично, а меншим зрушенням позиції."""
    from ploshcha_sim.domain.viche import stance_after, stance_start

    strong = stance_after(Beat(хто="koval", хід="заперечити"), stance_start(["koval"]),
                          {"koval": 1.4}, None)
    weak = stance_after(Beat(хто="koval", хід="заперечити"), stance_start(["koval"]),
                        {"koval": 0.4}, None)
    assert abs(strong["koval"]) > abs(weak["koval"])


def test_the_decision_is_a_count_not_a_retelling():
    from ploshcha_sim.domain.viche import tally

    out = tally([("koval", "за"), ("pip", "за"), ("did", "проти")])
    assert out["ухвалено"] is True
    assert out["лічба"] == {"за": 2, "проти": 1, "утримуюсь": 0}
    assert "за 2" in out["підсумок"]
    assert tally([])["ухвалено"] is False


def test_the_next_wave_is_planned_KNOWING_what_was_already_said():
    """Головне в хвилях: друга партитура бачить стенограму й позиції. Інакше це та сама
    написана наперед черга, тільки в кілька викликів."""
    pair = [p.role for p in cast_for(NEWS, 2)]
    agent, llm = build([score(beat(pair[0])), score(beat(pair[1], "заперечити", 1))]
                       + lines(6) + [chron((pair[0], "Отак."))], width=2)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    scores = [c for c in llm.calls
              if isinstance(c["schema"], dict) and "такти" in (c["schema"].get("properties") or {})]
    assert len(scores) >= 2, "партитура мусить плануватись хвилями, а не одна на весь прогін"
    assert "ЩО ВЖЕ СКАЗАНО" in scores[1]["prompt"]
    assert "ПОЗИЦІЇ ЗАРАЗ" in scores[1]["prompt"]


def test_every_voice_votes_and_the_vote_is_spoken_aloud():
    from ploshcha_sim.adapters import InMemoryTrace

    pair = [p.role for p in cast_for(NEWS, 2)]
    trace = InMemoryTrace()
    agent, _ = build([score(beat(pair[0]), beat(pair[1], "піддакнути", 1))] + lines(6)
                     + [chron((pair[0], "Отак."))], width=2, trace=trace)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    said = [e["payload"]["text"] for e in _events(trace) if e["type"] == "utterance.spoken"]
    assert any(t.startswith(("за", "проти", "утримуюсь")) for t in said), \
        "голос мусить ЗВУЧАТИ: інакше підрахунок — ще одне приховане число"


# ── тихі шляхи помилок: збій мусить бути ЧУТНИЙ ───────────────────────────────
#
# Три з чотирьох схем уже мали гучний провал (`viche_score_lost`, `viche_chronicle_lost`,
# `viche_scout_failed`). Зведення старости й сумнів попа — не мали: вони йшли на сцену БЕЗ
# перевірки, тобто нерозбірний вивід шлюзу ставав голосом дослівно.


def _one_voice(script, *, system_match=None, finish="stop"):
    """Виче з ОДНИМ скриптованим викликом: тут перевіряється не розмова, а її шлях помилки."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    llm = FakeLlm(script, model="fake", finish_reason=finish)
    agent = Viche(single_model_router(llm), PresetEffort(), None, width=3, trace=trace, run_id="r")
    return agent, trace


SAID = [(PERSONAS[0], "Кажу вам, вовк то не жарт, і кошару треба латати негайно.")]


def test_a_truncated_summary_is_never_spoken_as_the_starostas_word():
    """Живий випадок: шлюз обірвав вивід на стелі, і староста промовив `{"репліка": "Отак воно і`.
    Перевірку мала кожна репліка, крім цієї, — тому дефект був невидимий саме в кінці розмови."""
    agent, trace = _one_voice(['{"репліка": "Отак воно і'])
    incidents: list[str] = []
    assert agent._summary(NEWS, SAID, 1, Budget(max_tokens=9999), incidents) is None
    assert incidents == ["viche_summary_lost"]
    assert not [e for e in _events(trace) if e["type"] == "utterance.spoken"], \
        "забракована спроба не має звучати на сцені"


def test_a_summary_the_gateway_never_sent_is_named_not_counted_as_a_voice():
    """Порожня відповідь давала німого мовця в стенограмі, який ще й накручував `voices=`."""
    agent, _ = _one_voice([""])
    incidents: list[str] = []
    assert agent._summary(NEWS, SAID, 1, Budget(max_tokens=9999), incidents) is None
    assert "viche_summary_lost" in incidents


def test_a_good_summary_still_gets_through_and_is_spoken():
    """Гучність не має коштувати робочого шляху: справне зведення лишається голосом старости."""
    agent, trace = _one_voice([line("Зійшлись на тому, що кошару треба латати всім гуртом.")])
    incidents: list[str] = []
    who, text = agent._summary(NEWS, SAID, 1, Budget(max_tokens=9999), incidents)
    assert who.role == "starosta" and "кошару" in text and incidents == []
    said = [e["payload"]["text"] for e in _events(trace) if e["type"] == "utterance.spoken"]
    assert said == [text]


def test_a_rejected_doubt_never_reaches_the_stage():
    """Сумнів емітився ДО перевірки: `{}` уже прозвучало вустами попа, а зі стенограми випадало —
    тобто сцена й підсумок розходились, і жодне число про це не казало."""
    agent, trace = _one_voice(["{}"])
    incidents: list[str] = []
    assert agent._doubt(NEWS, SAID, 1, Budget(max_tokens=9999), incidents) is None
    assert incidents == ["viche_doubt_lost"]
    assert not [e for e in _events(trace) if e["type"] == "utterance.spoken"]


def test_a_real_doubt_is_still_heard():
    agent, trace = _one_voice([line("А хто те бачив на власні очі? Самі перекази ходять.")])
    incidents: list[str] = []
    who, text = agent._doubt(NEWS, SAID, 1, Budget(max_tokens=9999), incidents)
    assert who.role == "pip" and incidents == []
    said = [e["payload"]["text"] for e in _events(trace) if e["type"] == "utterance.spoken"]
    assert said == [text]


def test_a_vote_the_gateway_garbled_is_named_not_silently_dropped():
    """Загублений голос МІНЯЄ ухвалу. Доти лічба чесно показувала нулі, але причина не лишалась
    ніде, і «віче не дійшло голосу» читалось як рішення села, а не як збій шлюзу."""
    from ploshcha_sim.domain.viche import stance_start

    agent, _ = _one_voice(["{обірвано"] * 6)
    cast = cast_for(NEWS, 3)
    said = [(p, "щось сказав про справу") for p in cast]
    incidents: list[str] = []
    out = agent._vote(NEWS, cast, said, stance_start([p.role for p in cast]), 1,
                      Budget(max_tokens=9999), incidents)

    assert out["підсумок"] == "віче не дійшло голосу"
    assert len(incidents) == len(cast), "кожен загублений голос мусить бути названий"
    assert all(i.startswith("viche_vote_lost:") for i in incidents)


def test_a_ceiling_cut_is_told_apart_from_a_model_writing_nonsense():
    """Обидва дають той самий нерозбірний JSON. Без `finish_reason` інцидент казав `score_lost`, а
    справжня причина — замала стеля виводу — не лишалась ніде, тобто лагодили не те."""
    agent, _ = _one_voice(['{"такти": [{"хто": "kova'], finish="length")
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=6, max_tokens=9999))
    assert any(i.startswith("viche_cut:") for i in result.incidents)


def test_a_gateway_that_answered_with_nothing_is_named_too():
    """Мовчання шлюзу й порожня відповідь моделі — різні поламки з однаковим наслідком."""
    agent, _ = _one_voice([""])
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=6, max_tokens=9999))
    assert any(i.startswith("viche_empty:") for i in result.incidents)


def test_a_healthy_call_raises_no_channel_flag():
    """Доказ, що прапорці каналу не вмикаються самі: інакше вони були б шумом, а не сигналом."""
    agent, _ = _one_voice([line("Кажу вам, лихо буде, і не мале зовсім.")])
    agent._call("speak", "п", "с", line_schema(), 1, Budget(max_tokens=9999))
    assert agent._flaws == []


def test_the_finish_reason_reaches_the_trace_not_just_the_incident():
    """Траса — прилад: якщо обрив видно лише в інцидентах, у пакетних вимірах його немає взагалі."""
    from ploshcha_sim.adapters import InMemoryTrace

    trace = InMemoryTrace()
    llm = FakeLlm([line("Кажу вам, лихо буде.")], model="fake", finish_reason="length")
    agent = Viche(single_model_router(llm), PresetEffort(), None, width=3, trace=trace, run_id="r")
    agent._call("speak", "п", "с", line_schema(), 1, Budget(max_tokens=9999), span="r/viche/did/1")
    assert [r.finish_reason for r in trace.records] == ["length"]


def test_a_lost_chronicle_still_closes_the_viche():
    """Одна невдала відповідь шлюзу не має лишати віче без кінця.

    Заміряно на живому прогоні: `viche_chronicle_lost` — і зникає геть усе закриття (ухвала,
    чутка, думки, настрій, підсумок). Глядач дочитував останню репліку й лишався ні з чим, ніби
    розмову обірвало. Підрахунок голосів рахує КОД, від моделі він не залежить, тож закрити віче
    можна завжди.
    """
    from ploshcha_sim.adapters import InMemoryTrace

    pair = [p.role for p in cast_for(NEWS, 2)]
    trace = InMemoryTrace()
    # літопис двічі віддає непотріб → `viche_chronicle_lost`
    agent, _ = build([score(beat(pair[0]), beat(pair[1], "піддакнути", 1))] + lines(6)
                     + ["заголовок битий {", "заголовок теж битий {"], width=2, trace=trace)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    assert "viche_chronicle_lost" in result.incidents
    events = _events(trace)
    report = [e for e in events if e["type"] == "report.compiled"]
    assert report, "без літопису сцена мусить дістати бодай сухий підсумок"
    assert report[0]["payload"]["chronicle"]["narration"], "підсумок не може бути порожнім"
    decision = [e for e in events if e["type"] == "event.happened"
                and e["payload"]["event"]["kind"] == "decision"]
    assert decision, "лічба голосів дає ухвалу навіть без літописця"
    assert "за" in decision[0]["payload"]["event"]["label"]


# ── розбір відповіді шлюзу: врятувати те, що вціліло ──────────────────────────

def test_a_truncated_answer_keeps_what_the_model_already_said():
    """Строгий розбір викидав УСЕ через одну незакриту дужку в хвості.

    Заміряно на живих прогонах: `viche_chronicle_lost` двічі з двох. Літопис — найбільша відповідь
    у прогоні, і саме вона приходила обрізаною; при цьому заголовок і оповідь лежали на самому
    початку й були цілі. Викидати їх разом із хвостом — це втрачати вже зроблену роботу.
    """
    from ploshcha_sim.agents.viche import _safe_json

    whole = _safe_json('{"заголовок":"Вовки","оповідь":"Село радилось."}')
    assert whole == {"заголовок": "Вовки", "оповідь": "Село радилось."}

    cut_array = _safe_json('{"заголовок":"Вовки","оповідь":"Радились.","думки":[{"хто":"koval","дум')
    assert cut_array and cut_array["заголовок"] == "Вовки" and cut_array["оповідь"] == "Радились."

    cut_string = _safe_json('{"заголовок":"Гребля","оповідь":"Ухвалили лагод')
    assert cut_string and cut_string["заголовок"] == "Гребля"

    in_prose = _safe_json('Ось хроніка:\n{"заголовок":"Гребля","оповідь":"Готово."}\nсподіваюсь')
    assert in_prose and in_prose["оповідь"] == "Готово."

    assert _safe_json("зовсім не json") is None
    assert _safe_json("") is None
