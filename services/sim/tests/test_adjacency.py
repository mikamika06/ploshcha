"""Суміжна пара: тип попередньої репліки обмежує ХІД у відповідь — і ніде не стає словом.

Механізм узятий з огляду поля (`docs/research/dialogue-mechanics-field.md`) і з нашого власного
інвентаря (`docs/research/dialogue-mechanics-ours.md`). Дві чужі цифри, на яких він стоїть: жодна
відвантажена ігрова система не передає наступному мовцеві ТЕКСТ попередньої репліки, лише символ,
а типізація попередньої репліки як першої частини суміжної пари підняла людську оцінку
багатосторонньої розмови з 4.61 до 7.45.

Наш закон пакета (заміряний шість разів на шести різних текстах: будь-який вільний текст, покладений
виконавцеві, вертається дослівно) забороняє везти тип словом. Тому він їде рівно тим, чим уже їде
хід: закритим енумом, який код призначає сам. Тут стережуться всі три половини цього:
тип виводиться лічбою й детерміновано; недозволений хід у відповідь не доходить до мовця;
жодне з семи слів типу не зʼявляється в тілі пакета.
"""

import json

from ploshcha_sim.adapters import PresetEffort
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.agents.viche import _CONTRAST, Viche, _stems
from ploshcha_sim.domain.task import Budget
from ploshcha_sim.domain.viche import (
    ANSWER_GUEST,
    ANSWER_MOVES,
    BOND_OF_MOVE,
    INTERRUPT_MOVE,
    KIND_OF_MOVE,
    MOVES,
    MOVE_HINT,
    NEWS_TIE,
    REPLY_KINDS,
    Beat,
    cast_for,
    paired_swap,
    reply_kind,
    reply_move,
)

from test_viche import NEWS, WaveLlm, beat, line, lines, score, speak_calls

WOLF = "Та де там вовк, то пес приблудний з хутора никав."


def build(replies, *, width=3, adjacency=True):
    llm = WaveLlm(replies, model="fake")
    return Viche(single_model_router(llm), PresetEffort(), None, width=width, run_id="r",
                 adjacency=adjacency), llm


# ── тип виводиться КОДОМ і детерміновано ──────────────────────────────────────

def test_the_kind_of_a_reply_is_counted_from_the_beat_and_never_asked():
    """Тип — похідна від полів, які вже є, тож на тому самому такті він той самий завжди.

    Це та сама вимога, що вже стоїть на складі учасників (хеш теми) і на перебивках (кубик за
    сідом): якщо рішення визначене станом, то в нього немає чого «розуміти», і виклик моделі
    додав би лише шанс збитись.
    """
    said = Beat(хто="did", хід="заперечити", мітка="т:1")
    first = reply_kind(said, WOLF)
    assert first == "заперечення"
    assert all(reply_kind(said, WOLF) == first for _ in range(20))
    assert set(KIND_OF_MOVE.values()) <= set(REPLY_KINDS)
    # Кожен хід партитури має тип: інакше правило пари мовчало б рівно там, де воно потрібне.
    assert all(move in KIND_OF_MOVE for move in MOVES)


def test_each_of_the_four_sources_moves_the_kind_exactly_once():
    """Хід, збіг основ із новиною, зсув позиції, питальний знак — і кожне робить одну роботу."""
    news = _stems(NEWS)
    memory = Beat(хто="did", хід="згадати", мітка="т:1")
    # 1. Хід дає основу.
    assert reply_kind(memory, "Торік у нас на Поділлі таке саме діялось.") == "спогад"
    # 2. Збіг основ із новиною робить спогад твердженням про саму справу.
    about = "Бачили того вовка коло кошари, і він унадився."
    assert len(_stems(about) & news) >= NEWS_TIE
    assert reply_kind(memory, about, stems=_stems(about), news=news) == "твердження"
    # 3. Зсув позиції відрізняє скаргу від погрози — за тим самим порогом, що й `stance_label`.
    grief = Beat(хто="did", хід="пожалітись", мітка="т:2")
    assert reply_kind(grief, "Мене це геть розорить.", stance=-0.2) == "скарга"
    assert reply_kind(grief, "Мене це геть розорить.", stance=-0.9) == "погроза"
    # 4. Питальний знак б'є все: борг відповіді лишається незалежно від задуму такту.
    assert reply_kind(grief, "А мені що тепер робити?", stance=-0.9) == "питання"


def test_a_beat_the_score_never_wrote_still_has_a_kind():
    """Перебивка й відгук гостю складає КОД, і без типу вони лишили б наступного мовця без пари."""
    assert reply_kind(Beat(хто="did", хід=INTERRUPT_MOVE, мітка="т:1+п")) == "заперечення"
    assert reply_kind(Beat(хто="did", хід=ANSWER_GUEST, мітка="гість:1:1")) == "твердження"


# ── недозволений хід у відповідь не проходить ─────────────────────────────────

def test_a_memory_is_never_an_answer_to_a_denial():
    """★ Головне правило: на заперечення відповідають запереченням, поступкою чи питанням.

    Спогад — саме той хід, який доти був найчастішою відповіддю на будь-що: на 70 тактах чотирьох
    живих прогонів у пакет доїхало 41 «пригадай схожий випадок з минулого села» з 70 (59%).
    """
    assert "згадати" not in ANSWER_MOVES["заперечення"]
    assert reply_move("згадати", "заперечення") == "піддакнути"
    assert reply_move("пожалітись", "заперечення") == "спитати_діло"


def test_no_forbidden_move_survives_the_pair_for_any_kind():
    """Вичерпний перебір: правило не має жодної дірки на восьми ходах і семи типах."""
    for kind in REPLY_KINDS:
        for move in MOVES:
            assert reply_move(move, kind) in ANSWER_MOVES[kind], (kind, move)


def test_the_pair_changes_the_move_but_not_the_sign_of_the_bond():
    """Правило пари судить, ЯКИМ ходом відповідати, а не чи людина за чи проти.

    Інакше воно переписувало б заразом і суперечку: заперечення, яке стало згодою, зсунуло б і
    позицію (`stance_after`), і стосунки (`bonds_from`) — тобто одна правка міняла б три стани.
    """
    def sign(move):
        delta = BOND_OF_MOVE.get(move, 0.0)
        return (delta > 0) - (delta < 0)

    for kind in REPLY_KINDS:
        for move in MOVES:
            fixed = reply_move(move, kind)
            if fixed != move and any(sign(m) == sign(move) for m in ANSWER_MOVES[kind]):
                assert sign(fixed) == sign(move), (kind, move, fixed)


def test_the_lever_is_not_asked_about_moves_the_code_composes_itself():
    """Перебивка — ланка вбік, а не друга частина пари; той самий виняток, що в `damp_chain`."""
    assert reply_move(INTERRUPT_MOVE, "заперечення") == INTERRUPT_MOVE
    assert reply_move(ANSWER_GUEST, "питання") == ANSWER_GUEST


def test_the_repair_ladder_keeps_the_pair_and_still_changes_the_move():
    """Ремонт іде на 68.7% тактів і міняє хід — отже без нього правило не дожило б до сцени."""
    for kind in REPLY_KINDS:
        for move in MOVES:
            swap = paired_swap(_CONTRAST, 0, kind, move)
            assert swap in ANSWER_MOVES[kind], (kind, move, swap)
            assert swap != move, (kind, move)
    # Порядок драбини лишається її власним: правило лише викидає з неї непарні ходи.
    assert paired_swap(_CONTRAST, 0, "", "згадати") == _CONTRAST[0]


# ── наскрізь: хід доїжджає, слово типу — ні ───────────────────────────────────

def _played(adjacency: bool):
    """Дві людини: перша заперечує, друга відповідає на неї спогадом."""
    trio = [p.role for p in cast_for(NEWS, 3)]
    sc = score(beat(trio[0], "заперечити"), beat(trio[1], "згадати", reply=1),
               beat(trio[2], "згадати", reply=1))
    agent, llm = build([sc] * 6 + lines(60), width=3, adjacency=adjacency)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    return result, llm


def answering(llm) -> list[list[str]]:
    """Ходи, сказані тим, у кого в пакеті СПРАВДІ стоїть адресат.

    Такт, чия ціль промовчала, сюди не потрапляє, і це не послаблення перевірки, а те саме
    правило: типу немає — немає й пари, бо відповідати нема на що (`ПЕРЕД ТОБОЮ ГОВОРИЛИ`
    замість `ТИ ВІДПОВІДАЄШ`).
    """
    out = []
    for call in speak_calls(llm):
        if "ТИ ВІДПОВІДАЄШ" not in (call.get("prompt") or ""):
            continue
        system = call.get("system") or ""
        out.append([move for move, hint in MOVE_HINT.items() if hint in system])
    return out


def test_the_speaker_is_never_told_to_remember_in_answer_to_a_denial():
    """Наскрізно: партитура написала спогад у відповідь на заперечення — до мовця він не доїхав."""
    result, llm = _played(adjacency=True)
    assert any(i.startswith("viche_pair:заперечення") for i in result.incidents), result.incidents
    told = answering(llm)
    assert told, told
    for moves in told:
        assert "згадати" not in moves, told
        assert all(m in ANSWER_MOVES["заперечення"] for m in moves), told

    # Плече «до» — той самий код і той самий сід, різниця лише в важелі.
    before, llm_before = _played(adjacency=False)
    assert not [i for i in before.incidents if i.startswith("viche_pair")]
    assert any("згадати" in moves for moves in answering(llm_before)), answering(llm_before)


def test_no_word_of_the_kind_ever_enters_the_body_of_the_packet():
    """★ Тип — четвертий текст, і в пакет він не кладеться СЛОВОМ: закон пакета його поверне.

    Заміряно тричі на трьох різних рядках: цитата сусіда 19 повторів із 29, підказка «почни з
    іншого слова» 12 реплік, підказка ходу 7 із 40 (17.5%). Тому тип живе там, де вже живе хід, —
    у призначенні, а звідти в системне повідомлення, де протік 1 із 36 (2.8%).
    """
    result, llm = _played(adjacency=True)
    assert any(i.startswith("viche_pair") for i in result.incidents), result.incidents
    for call in llm.calls:
        packet = call.get("prompt") or ""
        for kind in REPLY_KINDS:
            assert kind not in packet, (kind, packet)
    # І в системному його теж немає: туди їде ХІД, а не тип, яким його вибрали.
    for call in speak_calls(llm):
        system = call.get("system") or ""
        for kind in REPLY_KINDS:
            assert kind not in system, (kind, system)


def test_the_kind_is_written_only_by_a_beat_that_actually_spoke():
    """Промовчаний такт типу не має: відповідати на нього нема на що.

    Те саме правило, що вже стоїть на довіднику голосів (`_voices`) і тез (`_theses`), і саме воно
    робить викинутий такт нешкідливим: посилання на нього тихо гасне, а не тягне за собою хід.
    """
    trio = [p.role for p in cast_for(NEWS, 3)]
    sc = score(beat(trio[0], "заперечити"), beat(trio[1], "згадати", reply=1))
    # Порожня репліка на першому такті: він не прозвучав, отже типу не лишив.
    agent, llm = build([sc] * 6 + [json.dumps({"варіанти": ["", "", ""]})] + lines(16),
                       width=3)
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    # Довідник типів ніколи не ширший за довідник голосів: обидва наповнює лише сказаний такт.
    assert set(agent._kinds) <= set(agent._voices)
    assert agent._kinds, agent._kinds
    # А такт, чия ціль промовчала, дістає пакет БЕЗ адресата — і хід йому не переписують.
    orphan = [c for c in speak_calls(llm)
              if "ТИ ВІДПОВІДАЄШ" not in (c.get("prompt") or "")]
    assert orphan, "мовчазний такт мусив лишити наступного без адресата"


def _repaired(adjacency: bool):
    """Такт-відповідь, чия перша спроба — дослівний повтор: ремонт МУСИТЬ поміняти йому хід."""
    trio = [p.role for p in cast_for(NEWS, 3)]
    sc = score(beat(trio[0], "заперечити"), beat(trio[1], "згадати", reply=1),
               beat(trio[2], "піддакнути", reply=1))
    # Черга реплік вивірена: зачин, заперечення, перебивка — і четвертим ДОСЛІВНИЙ повтор
    # заперечення, сказаний тим, хто на нього відповідає. Саме на цьому такті вмикається ремонт.
    said = [line("Отакої, а я ж казав, що добром воно не скінчиться."),
            line(WOLF), line("Хай йому грець, треба вози лаштувати змалку."), line(WOLF)]
    agent, llm = build([sc] * 6 + said + lines(40), width=3, adjacency=adjacency)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))
    return result, llm


def test_the_repair_of_an_answer_stays_inside_the_pair():
    """★ Ремонт іде на 68.7% тактів і його єдиний важіль — зміна ходу.

    Тобто без цього правило пари вмирало б рівно там, де воно щойно спрацювало: партитура писала
    спогад у відповідь на заперечення, код міняв його на поступку, а перший же ремонт вертав у
    системне «поскаржся, як це вдарить по тобі» — хід, яким на заперечення не відповідають.
    """
    result, llm = _repaired(adjacency=True)
    assert any(i.startswith("viche_same") or i.startswith("viche_echo")
               for i in result.incidents), result.incidents
    told = answering(llm)
    assert told, told
    for moves in told:
        assert all(m in ANSWER_MOVES["заперечення"] for m in moves), told

    before, llm_before = _repaired(adjacency=False)
    assert any(any(m not in ANSWER_MOVES["заперечення"] for m in moves)
               for moves in answering(llm_before)), answering(llm_before)


# ── важіль: вимкнений за заміром, і це стережеться ────────────────────────────

def test_the_lever_is_off_in_production_until_the_gate_moves():
    """★ Дефолт `False` — це ЧИСЛО, а не обережність, тож він і стережеться числом.

    Живий шлюз 2026-08-30, прод-умова `viche`, сіди 1 і 2, теми «вовк» і «мито», по чотири віча
    в плечі: зчеплення 20 із 62 (32.3%) без важеля проти 18 із 60 (30.0%) з ним, підхоплення
    1 із 62 проти 1 із 60. Ворота, поставлені на підхопленні, не зрушені — отже на прод важіль
    не йде, як не пішли ні штраф повторення, ні згасання ланцюга, ні тези такту.

    Друга половина заміру сказала «так» і теж тримається тестом нижче: дослівне повернення не
    зросло (протіків ≥ 4 слова нуль в обох плечах, найдовший ланцюжок 3 слова).
    """
    from ploshcha_sim.domain.spec import AppSpec
    from evalkit.conditions import CONDITIONS

    assert AppSpec().viche_adjacency is False, "дефолт зберігає теперішню поведінку"
    assert AppSpec().sha256 != AppSpec().with_(viche_adjacency=True).sha256
    assert CONDITIONS["viche"].viche_adjacency is False
    assert CONDITIONS["viche-notools"].viche_adjacency is False


def test_the_lever_reaches_the_agent_only_through_the_specification():
    """Магічного прапорця всередині агента бути не може: його не видно ні в `sha256`, ні у звіті.

    Той самий припис, що вже стоїть на штрафі повторення й на згасанні ланцюга, і та сама
    застава — перелік `VICHE_KWARGS`, крізь який мовчазно ковтнутий kwarg уже одного разу лишив
    нам нетрасований граф і сталі персони при породженому селі.
    """
    from ploshcha_sim.adapters import FakeLlm
    from ploshcha_sim.compose import VICHE_KWARGS, build_viche
    from evalkit.conditions import CONDITIONS

    assert "adjacency" in VICHE_KWARGS
    llm = FakeLlm([""])
    spec = CONDITIONS["viche"]
    assert build_viche(spec, lapa=llm, mamay=llm).adjacency is False
    assert build_viche(spec.with_(viche_adjacency=True), lapa=llm, mamay=llm).adjacency is True
