"""Голий пакет: у тілі пакета мовця не лишається вільного тексту взагалі.

Шість кругів правок міняли, ЯКИЙ САМЕ текст лежить у пакеті, і щоразу він вертався дослівно в
репліці (`docs/research/dialogue-audit.md`, розділ 18: цитата сусіда 19 повторів із 29, «почни з
іншого слова» 12 реплік, підказка ходу 13 із 80, службовий рядок `ТИ ВІДПОВІДАЄШ: імʼя` — цілою
реплікою «Панас: Горпина баба»). Правило 7 того ж розділу написане нами й двічі порушене нами ж:
прибирання ОКРЕМОГО тексту не закриває канал. Тут перевіряється саме канал.

Числа плечей — у `docs/research/packet-strip.md` і в `Viche.__init__`. Тут стережеться три речі:
вимкнений важіль лишає пакет байт-у-байт таким, як був; ввімкнений не лишає в тілі жодного
службового рядка (і жодного вільного тексту ремонту); адреса при `mark` їде ЧИСЛОМ, а не імʼям, —
і сторож при цьому бачить увесь текст, хоч би де той лежав.
"""

from ploshcha_sim.adapters import PresetEffort
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.agents.viche import (
    LINE_ASK,
    PACKET_BARE,
    PACKET_MARK,
    Viche,
    _SERVICE_HEADS,
)
from ploshcha_sim.domain.task import Budget
from ploshcha_sim.domain.viche import BY_ROLE, Beat, cast_for

from test_viche import NEWS, WaveLlm, beat, line, lines, score, speak_calls


def build(replies, *, width=3, bare_packet=""):
    llm = WaveLlm(replies, model="fake")
    return Viche(single_model_router(llm), PresetEffort(), None, width=width, run_id="r",
                 bare_packet=bare_packet), llm


def talk(replies, *, width=3, bare_packet="", moves=("згадати", "порахувати", "пожалітись")):
    """Віче, у якому перший такт кожної хвилі відповідає другому: адресат є, і він не сам мовець."""
    trio = [p.role for p in cast_for(NEWS, width)]
    sc = score(beat(trio[0], moves[0]), beat(trio[1], moves[1], reply=1),
               beat(trio[2], moves[2], reply=1))
    agent, llm = build([sc] * 4 + list(replies), width=width, bare_packet=bare_packet)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    return agent, llm, result


def _bodies(llm):
    return [c.get("prompt") or "" for c in speak_calls(llm)]


def _systems(llm):
    return [c.get("system") or "" for c in speak_calls(llm)]


def test_the_packet_keeps_its_old_shape_while_the_lever_is_off():
    """Дефолт зберігає теперішню поведінку — інакше жоден порахований прогін не порівнюваний.

    Розбиття пакета на «озвучуване» й «службове» зроблене заради важеля, тож при вимкненому
    важелі складання мусить дати ТОЙ САМИЙ рядок: службові рядки в тому самому порядку, шепіт
    останнім, прохання про три варіанти — у тілі пакета, а не в системному.
    """
    agent, llm, _ = talk(lines(14))
    agent._cast_now = list(cast_for(NEWS, 3))
    packet = agent._packet(NEWS, BY_ROLE["did"], Beat(хто="did", хід="згадати"), [], None)

    assert packet.startswith("НА ВІЧІ ЩЕ: "), packet
    assert all(LINE_ASK not in system for system in _systems(llm))
    assert all(body.endswith(LINE_ASK) for body in _bodies(llm)), _bodies(llm)
    assert any(head in body.lower() for body in _bodies(llm) for head in _SERVICE_HEADS)


def test_with_the_lever_on_no_service_line_stays_in_the_packet():
    """★ Канал закривається лише тоді, коли в тілі пакета не лишається вільного тексту взагалі.

    Аудит прибрав рядок довідника — виліз хід; круг підказки прибрав хід — виліз єдиний лишений
    рядок з іменем адресата, і виліз ЦІЛОЮ реплікою. Тому перевіряється не окремий рядок, а всі
    службові зачини одразу — і те, що вони не зникли, а переїхали в системне.
    """
    _, llm, _ = talk(lines(14), bare_packet=PACKET_BARE)

    for body in _bodies(llm):
        flat = body.lower()
        assert not any(head in flat for head in _SERVICE_HEADS), body
        assert LINE_ASK not in body, body
    systems = _systems(llm)
    assert any("на вічі ще" in system.lower() for system in systems)
    assert all(system.endswith(LINE_ASK) for system in systems), systems[0]


def test_the_mark_points_at_the_addressee_by_place_not_by_name():
    """★ Адреса — МІСЦЕМ у пронумерованому переліку, тобто числом, яке не можна вимовити реплікою.

    Імʼя адресата в тілі пакета заміряне як найдорожчий протік: чотири службові слова виходили на
    сцену цілою реплікою двічі за одне віче, і прилад зараховував це ще й як ознаку звʼязку. Число
    на двох темах живого прогону не вилізло жодного разу (0 реплік із числом на 39).
    """
    _, llm, _ = talk(lines(14), bare_packet=PACKET_MARK)

    pointed = [(body, system) for body, system in zip(_bodies(llm), _systems(llm))
               if body.strip().isdigit()]
    assert pointed, _bodies(llm)
    for body, system in pointed:
        roster = next(ln for ln in system.splitlines() if ln.startswith("НА ВІЧІ: "))
        names = [n.strip() for n in roster.removeprefix("НА ВІЧІ: ").split(", ")]
        seat = names[int(body.strip()) - 1]
        assert seat.startswith(f"{body.strip()} "), (body, roster)
        assert seat.split(" ", 1)[1] not in body, (body, roster)


def test_the_repair_nudge_travels_in_the_system_too():
    """Підказка ремонту — теж вільний текст, отже теж не лишається в тілі голого пакета.

    Цитата сусіда в підказці — найгірший заміряний протік із шести: 19 повторів із 29 реплік.
    Важіль, який прибирає з пакета службові рядки й лишає там цитату, лікував би не канал, а знову
    окремий рядок.
    """
    same = "Отакої, а я ж казав, що добром не скінчиться."
    replies = [line(same)] * 2 + lines(12)
    _, llm, result = talk(replies, bare_packet=PACKET_BARE)

    assert any(inc.startswith("viche_echo") or inc.startswith("viche_same")
               for inc in result.incidents), result.incidents
    assert any("ТИ ЩОЙНО ПОВТОРИВ ЧУЖЕ" in system for system in _systems(llm)), _systems(llm)
    assert not any("ТИ ЩОЙНО ПОВТОРИВ ЧУЖЕ" in body for body in _bodies(llm)), _bodies(llm)

    _, off, _ = talk(replies)
    assert any("ТИ ЩОЙНО ПОВТОРИВ ЧУЖЕ" in body for body in _bodies(off)), _bodies(off)


def test_the_guard_still_sees_the_text_that_moved_to_the_system():
    """★ Сторожеві їде УВЕСЬ текст, який лежав перед мовцем, хоч би де він лежав.

    `_echoes` звіряє пʼятірки лише з пакетом, а по системному — тільки дослівно (свідоме рішення:
    інакше сторож ловив би власну примовку персони). Тож переїзд службових рядків у системне сам
    по собі ослабив би перевірку рівно на ту частину, яку переносимо, і замір плечей порівнював би
    різну сторожу, а не різні пакети.
    """
    names = [p.name for p in cast_for(NEWS, 3)][1:]
    tail = " ".join(("НА ВІЧІ ЩЕ: " + ", ".join(names)).split()[-5:])
    replies = [line(f"Отакої, {tail}, а я що казав.")] + lines(14)
    _, _, result = talk(replies, bare_packet=PACKET_BARE)

    assert any(inc.startswith("viche_echo") for inc in result.incidents), result.incidents


def test_the_lever_is_an_axis_of_the_run_not_an_ornament():
    """Рядок усередині агента був би невидимий ні в `sha256` умови, ні у звіті.

    Той самий припис, що вже стоїть на штрафі повторення, згасанні ланцюга й суміжній парі, і та
    сама застава — перелік `VICHE_KWARGS`, крізь який мовчазно ковтнутий kwarg уже одного разу
    лишив нам нетрасований граф і сталі персони при породженому селі.
    """
    from evalkit.conditions import CONDITIONS
    from ploshcha_sim.adapters import FakeLlm
    from ploshcha_sim.compose import VICHE_KWARGS, build_viche
    from ploshcha_sim.domain.spec import AppSpec

    assert AppSpec().viche_bare_packet == "", "дефолт зберігає теперішню поведінку"
    assert AppSpec().sha256 != AppSpec().with_(viche_bare_packet=PACKET_MARK).sha256

    assert "bare_packet" in VICHE_KWARGS
    llm = FakeLlm([""])
    # Умова прода вже несе `mark` (тест нижче), тож обидва кінці важеля питаємо в неї явно —
    # інакше «вимкнено» перевірялось би на конфігурації, якої в переліку більше немає.
    spec = CONDITIONS["viche"].with_(viche_bare_packet="")
    assert build_viche(spec, lapa=llm, mamay=llm).bare_packet == ""
    assert build_viche(spec.with_(viche_bare_packet=PACKET_MARK),
                       lapa=llm, mamay=llm).bare_packet == PACKET_MARK


def test_the_prod_condition_speaks_with_a_packet_free_of_service_text():
    """★ Важіль стоїть у ПРОД-УМОВІ, і доказ цьому — прогін, а не поле в специфікації.

    Проводка вся з коду й одна: `infra/server/deploy.sh` запускає `serve_ploshcha.py --condition
    viche`, той бере `CONDITIONS[умова]` і складає віче через `build_viche(spec, …)`, а
    `build_viche` кладе `spec.viche_bare_packet` у `Viche.bare_packet` (`VICHE_KWARGS`). Тому агент
    тут збирається САМЕ з умови прода, а не з ручних аргументів: доти обидві прод-умови стояли в
    дефолтному "" — важіль був написаний, покритий тестами й у живій конфігурації мертвий, як були
    свого часу мертві охорона й суддя.

    Числа підстави — `docs/research/packet-strip.md`, те саме віче на живому шлюзі: пар без ознаки
    звʼязку 80.0% → 60.0%, спроб мовця на такт 2.18 → 1.18, ремонтів 23 → 5, ескалацій 16 → 2,
    кроків 96 і 100 → 74 і 74 при стелі 80, токенів на два віча 36 679 → 28 530. Ціна названа:
    поле входить у `sha256` умови, тож звіти по ній до й після непорівнянні.
    """
    import pathlib

    from evalkit.conditions import CONDITIONS
    from ploshcha_sim.compose import build_viche

    for name in ("viche", "viche-notools"):
        assert CONDITIONS[name].viche_bare_packet == PACKET_MARK, name

    root = pathlib.Path(__file__).resolve().parents[3]
    deploy = (root / "infra" / "server" / "deploy.sh").read_text("utf-8")
    assert "serve_ploshcha.py" in deploy and "--condition viche" in deploy, "ланка deploy.sh"
    server = (pathlib.Path(__file__).resolve().parents[1] / "scripts"
              / "serve_ploshcha.py").read_text("utf-8")
    assert "spec = CONDITIONS.get(condition)" in server, "ланка serve_ploshcha"
    assert "build_viche(spec," in server, "ланка serve_ploshcha → build_viche"

    spec = CONDITIONS["viche"]
    trio = [p.role for p in cast_for(NEWS, spec.max_width)]
    sc = score(beat(trio[0], "згадати"), beat(trio[1], "порахувати", reply=1),
               beat(trio[2], "пожалітись", reply=1))
    llm = WaveLlm([sc] * 6 + lines(20), model="fake")
    agent = build_viche(spec, lapa=llm, mamay=llm)
    assert agent.bare_packet == PACKET_MARK
    agent.run(NEWS, seed=1, budget=Budget(max_steps=40, max_tokens=99_999))

    bodies = _bodies(llm)
    assert bodies, "село мусить заговорити, інакше перевіряти нічого"
    for body in bodies:
        flat = body.lower()
        assert not any(head in flat for head in _SERVICE_HEADS), body
        assert LINE_ASK not in body, body
        # Або порожньо (такт без адресата), або САМЕ ЧИСЛО — місце адресата в переліку, який
        # лишився в системному. Третього в тілі пакета бути не може: рядок, який можна вимовити
        # реплікою, і є той канал, що вісім кругів поспіль вертав службовий текст на сцену.
        assert body.strip() == "" or body.strip().isdigit(), body
    assert any(body.strip().isdigit() for body in bodies), bodies
    assert any("на вічі: 1 " in system.lower() for system in _systems(llm)), _systems(llm)[0]
