"""Памʼять села — своя в кожного гостя, і старі села прибираються.

Скарга, з якої це почалось: на публічному порті всі сховища сиділи на ОДНОМУ SQLite. Хто б не кинув
тему, мінялось те саме село — чужі ухвали, чужі чутки, чужий літопис. Памʼять і є те, заради чого
сюди вертаються, тож спільна памʼять означає, що вертатись немає куди.

Тут стережуться чотири речі, кожна з яких ламається ТИХО:
  • два гості не бачать памʼяті один одного — ні в базі, ні в потоці подій;
  • памʼять переживає перезапуск ядра, бо інакше вона не памʼять, а кеш;
  • забуте село зникає з диска, а не лежить там вічно;
  • ані вік, ані кількість файлів не можуть рости без межі.
"""

import json
import os
import time
import urllib.request

from ploshcha_sim.adapters.decisions_sqlite import SqliteDecisions
from ploshcha_sim.adapters.memory_sqlite import SqliteMemory
from ploshcha_sim.adapters.queue_sqlite import SqliteQueue
from ploshcha_sim.adapters.rumours_sqlite import SqliteRumours
from ploshcha_sim.domain.governor import Governor
from ploshcha_sim.live import EventBus, LiveRunner, handle_command, serve
from ploshcha_sim.live.server import RateGate, allow_new_session
from ploshcha_sim.live.sessions import (DEFAULT_MAX_SESSIONS, DEFAULT_TTL_DAYS, Session,
                                        SessionRegistry, clean_sid, max_sessions, ttl_seconds)
from ploshcha_sim.ports.trace import StepRecord

SID_A = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
SID_B = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
SCENE = {"id": "ploshcha", "name": "Площа"}


# ── підпори ───────────────────────────────────────────────────────────────────

class _Result:
    """Форма, яку читає `StreamProjector.close`. Нуль викликів моделі."""

    outcome = "answer"
    verdict_kind = None
    scratch: list = []
    notes: list = []
    incidents: list = []
    tokens = 7
    aux_tokens = 0
    steps = 1


class _CouncilAgent:
    """Агент, що ухвалює рішення ТИМ САМИМ шляхом, що справжнє віче.

    Запис іде в трасу, а в базу його переносить РАННЕР. Писати в базу з тесту означало б перевіряти
    не той код: саме проводка «траса → сховище сесії» і є те, що ми поламали б, змінюючи ядро.
    """

    def __init__(self, trace, run_id: str, who: str):
        self.trace, self.run_id, self.who = trace, run_id, who

    def run(self, task, seed=1, budget=None):
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=1, agent="council", stage="act", model="fake",
            prompt="", raw_output="", schema_valid=True,
            parsed={"label": f"ухвала: {task}", "who": self.who, "poi": "well"}))
        return _Result()


def _build_session(path: str, sid: str | None) -> Session:
    who = sid or "спільне"

    def make_agent(trace, run_id, place=None):
        return _CouncilAgent(trace, run_id, who)

    return Session(sid=sid, path=path, make_agent=make_agent,
                   cast=[{"id": "starosta", "name": "Староста", "role": "starosta"}],
                   decisions=SqliteDecisions(path), rumours=SqliteRumours(path),
                   memory=SqliteMemory(path))


def _registry(tmp_path, **kw) -> SessionRegistry:
    return SessionRegistry(tmp_path / "sessions", _build_session,
                           base=_build_session(str(tmp_path / "base.db"), None), **kw)


def _live(tmp_path, **kw):
    bus = EventBus()
    queue = SqliteQueue(str(tmp_path / "q.db"))
    registry = _registry(tmp_path, **kw)
    runner = LiveRunner(bus, queue, registry.base.make_agent, scene=SCENE,
                        governor=Governor(max_tokens=1_000_000), sessions=registry,
                        cast=registry.base.cast, decisions=registry.base.decisions,
                        rumours=registry.base.rumours)
    return bus, queue, registry, runner


def _drain(runner, queue) -> None:
    """Один айтем, синхронно. Детермінізм тут дорожчий за реалізм потоку: сам потік стережуть
    інші тести, а ці — про те, ЧИЄ село змінилось."""
    item = queue.lease("test")
    assert item is not None, "черга мусить віддати айтем"
    runner._run_one(item)


def _age(path, seconds: float, *, now: float) -> None:
    os.utime(path, (now - seconds, now - seconds))


# ── ідентифікатор ─────────────────────────────────────────────────────────────

def test_an_id_that_walks_out_of_the_folder_is_refused():
    """`sid` приходить із браузера й стає ІМʼЯМ ФАЙЛУ. Без перевірки `../../` писав би поза теку —
    тобто чужий гість керував би тим, які файли на диску ядра затираються."""
    for bad in ("../../etc/passwd", "a/b", "sid.db", "..", "a" * 65, "коротко", ""):
        assert clean_sid(bad) is None, bad


def test_a_browser_uuid_is_accepted_as_is():
    """Клієнт кладе в `localStorage` саме `crypto.randomUUID()`. Якби абетка його не пускала,
    кожен гість мовчки падав би у спільне село — тобто фіча була б вимкнена й ніхто б не помітив."""
    assert clean_sid(SID_A) == SID_A


def test_a_missing_id_means_the_shared_village(tmp_path):
    """CLI, соак і старий клієнт `sid` не шлють. Відмовляти їм означало б зламати запуск з консолі."""
    registry = _registry(tmp_path)
    assert registry.get(None) is registry.base
    assert registry.get("погань") is registry.base


# ── розділена памʼять ─────────────────────────────────────────────────────────

def test_two_sessions_keep_their_memory_apart(tmp_path):
    """Головне твердження. Доти ухвала одного гостя ставала ухвалою села для всіх."""
    _, queue, registry, runner = _live(tmp_path)
    queue.put("t1", {"task": "криниця", "sid": SID_A})
    _drain(runner, queue)
    queue.put("t2", {"task": "гребля", "sid": SID_B})
    _drain(runner, queue)

    assert [d["label"] for d in registry.get(SID_A).decisions.standing()] == ["ухвала: криниця"]
    assert [d["label"] for d in registry.get(SID_B).decisions.standing()] == ["ухвала: гребля"]


def test_each_session_writes_into_its_own_file(tmp_path):
    """Файл на сесію — не деталь реалізації, а те, що робить прибирання можливим: видалити село
    цілком можна лише тоді, коли воно ціле лежить в одному місці."""
    _, queue, registry, runner = _live(tmp_path)
    for key, sid in (("t1", SID_A), ("t2", SID_B)):
        queue.put(key, {"task": "тема", "sid": sid})
        _drain(runner, queue)
    files = sorted(p.name for p in (tmp_path / "sessions").glob("*.db"))
    assert files == sorted([f"{SID_A}.db", f"{SID_B}.db"])


def test_the_shared_village_stays_out_of_the_sessions_folder(tmp_path):
    """Тема без `sid` не має плодити файл: інакше кожен запуск з консолі з'їдав би стелю сесій."""
    _, queue, registry, runner = _live(tmp_path)
    queue.put("t", {"task": "тема"})
    _drain(runner, queue)
    assert registry.count() == 0
    assert registry.base.decisions.standing()


def test_village_memory_survives_a_core_restart(tmp_path):
    """Памʼять, що гине з процесом, — це кеш. Тест закриває ядро й піднімає нове на тій самій теці:
    інакше «своє село» трималось би лише на живих обʼєктах у RAM і зникало на кожен деплой."""
    _, queue, _, runner = _live(tmp_path)
    queue.put("t", {"task": "криниця", "sid": SID_A})
    _drain(runner, queue)

    fresh = _registry(tmp_path)
    assert [d["label"] for d in fresh.get(SID_A).decisions.standing()] == ["ухвала: криниця"]


def test_a_standing_decision_returns_only_to_its_own_session(tmp_path):
    """Чинні ухвали повертаються на сцену перед розмовою. Якби вони бралися зі спільної бази,
    гість бачив би на своїй площі сторожа, якого ставив хтось інший."""
    bus, queue, _, runner = _live(tmp_path)
    for key, sid in (("a1", SID_A), ("a2", SID_A), ("b1", SID_B)):
        queue.put(key, {"task": f"тема-{key}", "sid": sid})
        _drain(runner, queue)

    def standing(sid):
        return [e for e in bus.since(0, sid)[0]
                if e["type"] == "event.happened"
                and str(e["payload"]["event"]["id"]).startswith("standing-")]

    assert standing(SID_A), "друге віче гостя мусить памʼятати його ж ухвалу"
    assert standing(SID_B) == [], "чужа ухвала не має проступати в його село"


# ── потік ─────────────────────────────────────────────────────────────────────

def test_a_run_of_one_session_does_not_reach_another_stream(tmp_path):
    """Фільтрація потоку — друга половина розділення. Без неї бази роздільні, а на екрані гостя
    все одно йде чужа розмова: він бачить імена й репліки, яких у його селі не було."""
    bus, queue, _, runner = _live(tmp_path)
    for key, sid in (("a", SID_A), ("b", SID_B)):
        queue.put(key, {"task": key, "sid": sid})
        _drain(runner, queue)

    runs_a = {e["runId"] for e in bus.since(0, SID_A)[0]}
    runs_b = {e["runId"] for e in bus.since(0, SID_B)[0]}
    assert runs_a and runs_b
    assert runs_a & runs_b == set(), "жоден прогін не має бути видимий обом"


def test_core_events_without_a_session_reach_every_stream():
    """Стеля витрат і смерть робітника — стан ЯДРА, а не чиясь розмова: воно спинилось для всіх.
    Мовчати про це перед іншими гостями означало б лишити їх із написом «Село думу думає…»."""
    bus = EventBus()
    bus.publish({"type": "run.degraded"})
    assert len(bus.since(0, SID_A)[0]) == 1
    assert len(bus.since(0, SID_B)[0]) == 1


def test_an_inspector_without_a_session_sees_everything():
    """Глядач без `sid` — консоль і соак. Якби фільтр рахував його «чужим», `soak_ploshcha`
    перестав би бачити власні прогони й міряв би тишу."""
    bus = EventBus()
    bus.publish({"n": 1}, SID_A)
    bus.publish({"n": 2}, SID_B)
    assert [e["n"] for e in bus.since(0)[0]] == [1, 2]


def test_stream_ids_stay_absolute_after_filtering():
    """`id` у SSE — це позиція для реконекту. Після фільтрації пачка коротша за крок курсора, тож
    старий підрахунок «від довжини пачки» віддавав би позицію з минулого — і реконект тягнув би
    вже показане наново."""
    bus = EventBus()
    bus.publish({"n": 0}, SID_A)
    bus.publish({"n": 1}, SID_B)
    bus.publish({"n": 2}, SID_A)
    pairs, cursor = bus.since_ids(0, SID_B)
    assert [i for i, _ in pairs] == [1]
    assert cursor == 3, "курсор рухається по ВСІХ подіях, інакше глядач застрягне на чужій"
    assert bus.since_ids(2, SID_B)[0] == [], "реконект з віддану позицію не повторює нічого"


def test_a_late_viewer_of_a_session_still_gets_only_his_own():
    bus = EventBus()
    bus.publish({"n": 0}, SID_A)
    cursor = bus.tail_cursor()
    bus.publish({"n": 1}, SID_B)
    bus.publish({"n": 2}, SID_A)
    assert [e["n"] for e in bus.since(cursor, SID_A)[0]] == [2]


def test_history_of_a_shared_run_is_not_replayed_to_a_newcomer():
    """★ Спільна подія видима, доки вона ЙДЕ, але не відтворюється новому глядачеві.

    Фронт просить історію з нуля, щоб гість побачив своє село; на прогонах без сесії (CLI, мої
    перевірки) це означало, що кожен, хто відкривав ПЛОЩУ з будь-якого пристрою, діставав чужі
    прогони як власні — на Дошці висіла купа тем, яких він не кидав. Мітка «спільне» — про стан
    ядра ЗАРАЗ, а не про історію.
    """
    bus = EventBus()
    bus.publish({"n": 0}, None)   # чужий прогін без сесії — уже минув
    bus.publish({"n": 1}, SID_A)  # і чужа сесія теж
    joined = bus.tail_cursor()
    bus.publish({"n": 2}, None)   # а це вже за нього — стан ядра наживо
    seen = [e["n"] for e in bus.since(0, SID_B, shared_from=joined)[0]]
    assert seen == [2], "з історії гостеві належить лише СВОЄ, спільне — тільки наживо"
    mine = [e["n"] for e in bus.since(0, SID_A, shared_from=joined)[0]]
    assert mine == [1, 2], "власне село гість добирає з історії, як і доти"
    assert [e["n"] for e in bus.since(0, None, shared_from=joined)[0]] == [0, 1, 2], "інспектор бачить усе"


def test_the_http_stream_filters_by_the_query_parameter(tmp_path):
    """Розбір `?sid=` живе в хендлері, тобто поза межами тестів шини. Доти той самий рядок уже
    ламався один раз на `since=`, тож перевіряємо шлях цілком, а не лише його середину."""
    bus, queue, _, runner = _live(tmp_path)
    for key, sid in (("a", SID_A), ("b", SID_B)):
        queue.put(key, {"task": key, "sid": sid})
        _drain(runner, queue)
    httpd = serve(bus, runner, port=0)
    port = httpd.server_address[1]
    try:
        res = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/stream?sid={SID_A}&since=0", timeout=5)
        seen: list[dict] = []
        deadline = time.time() + 5.0
        while len(seen) < 6 and time.time() < deadline:
            line = res.readline().decode("utf-8")
            if line.startswith("data: "):
                seen.append(json.loads(line[6:]))
        res.close()
    finally:
        httpd.shutdown()
    assert seen, "потік мусить віддати хоч щось"
    assert {e["payload"].get("task") for e in seen if e["type"] == "task.outcome"} != {"b"}
    assert all("live-b" not in e["runId"] for e in seen), "чужий прогін не має доїхати"


# ── команди ───────────────────────────────────────────────────────────────────

def test_a_topic_carries_the_session_into_the_queue(tmp_path):
    """Виконавець ОДИН, і тема лежить у черзі хвилинами. Якби `sid` не їхав у самому айтемі, до
    моменту прогону ядро вже не знало б, чиє село міняти."""
    _, queue, _, runner = _live(tmp_path)
    code, _ = handle_command({"kind": "topic", "text": "Криниця", "sid": SID_A}, runner)
    assert code == 200
    assert queue.lease("w").payload["sid"] == SID_A


def test_a_topic_without_a_session_carries_none(tmp_path):
    _, queue, _, runner = _live(tmp_path)
    handle_command({"kind": "topic", "text": "Криниця"}, runner)
    assert "sid" not in queue.lease("w").payload


def test_a_bad_session_id_in_a_topic_is_not_written_to_the_queue(tmp_path):
    """Погань не відхиляється, а знеособлюється: інакше один зіпсований `localStorage` робив би
    ПЛОЩУ непрацездатною для гостя, замість тихо повернути його у спільне село."""
    _, queue, _, runner = _live(tmp_path)
    code, _ = handle_command({"kind": "topic", "text": "Криниця", "sid": "../etc"}, runner)
    assert code == 200
    assert "sid" not in queue.lease("w").payload


class _Talker:
    def __init__(self):
        self.heard: list[dict] = []

    def tell(self, msg):
        self.heard.append(msg)


def test_a_guest_cannot_speak_into_another_guests_viche(tmp_path):
    """Гість не бачить чужого прогону в потоці, тож слово, кинуте в нього, зникло б без сліду — а
    з боку іншого села прилетів би голос нізвідки."""
    _, _, _, runner = _live(tmp_path)
    talker = _Talker()
    runner._active[SID_A] = talker
    code, body = handle_command({"kind": "say", "text": "агов", "sid": SID_B}, runner)
    assert code == 409 and "error" in body
    assert talker.heard == []


def test_a_guest_can_speak_into_his_own_viche(tmp_path):
    _, _, _, runner = _live(tmp_path)
    talker = _Talker()
    runner._active[SID_A] = talker
    code, _ = handle_command({"kind": "say", "text": "агов", "sid": SID_A}, runner)
    assert code == 200
    assert talker.heard == [{"kind": "say", "text": "агов"}]


def test_a_guest_may_speak_into_the_shared_viche(tmp_path):
    """Спільне віче (тема з консолі) видиме ВСІМ, бо його події без мітки. Раз чути — значить і
    відповідати можна: інакше на екрані була б розмова, у якій поле вводу мовчки не працює."""
    _, _, _, runner = _live(tmp_path)
    runner._active[""] = _Talker()   # прогін без сесії — спільне віче
    assert handle_command({"kind": "say", "text": "агов", "sid": SID_A}, runner)[0] == 200


# ── стеля на створення ────────────────────────────────────────────────────────

def test_one_address_cannot_hatch_unlimited_sessions(tmp_path):
    """Стеля команд каже, як ЧАСТО можна просити; ця — скільки слідів на диску можна лишити.
    Без неї цикл із новим `sid` на кожен запит наплодив би сотні файлів, не перевищивши жодної
    старої межі."""
    registry = _registry(tmp_path)
    gate = RateGate(window=60.0, limit=2)
    made = [allow_new_session(gate, registry, f"gost-{i:04d}", "1.2.3.4")[0] for i in range(5)]
    assert made == [True, True, False, False, False]
    assert registry.count() == 2, "відмовлена сесія не має лишати файл"


def test_a_known_session_costs_nothing_from_the_new_session_budget(tmp_path):
    """Інакше гість, який просто працює, вигорав би власну стелю за десяток тем і діставав 429 на
    рівному місці."""
    registry = _registry(tmp_path)
    gate = RateGate(window=60.0, limit=1)
    assert allow_new_session(gate, registry, "gost-0001", "1.2.3.4")[0] is True
    assert all(allow_new_session(gate, registry, "gost-0001", "1.2.3.4")[0] for _ in range(20))


def test_another_address_has_its_own_new_session_budget(tmp_path):
    registry = _registry(tmp_path)
    gate = RateGate(window=60.0, limit=1)
    assert allow_new_session(gate, registry, "gost-0001", "1.2.3.4")[0] is True
    assert allow_new_session(gate, registry, "gost-0002", "1.2.3.4")[0] is False
    assert allow_new_session(gate, registry, "gost-0003", "5.6.7.8")[0] is True


# ── прибирання ────────────────────────────────────────────────────────────────

def test_a_session_untouched_past_the_ttl_is_swept(tmp_path):
    """Диск скінченний, а гість, що заходив раз, не повертається. Без TTL тека росте назавжди."""
    now = 1_000_000.0
    registry = _registry(tmp_path, ttl_s=100.0, clock=lambda: now)
    registry.ensure("gost-0001")
    _age(registry.path_for("gost-0001"), 101.0, now=now)
    assert registry.sweep() == 1
    assert not registry.path_for("gost-0001").exists()


def test_a_fresh_session_is_left_alone(tmp_path):
    now = 1_000_000.0
    registry = _registry(tmp_path, ttl_s=100.0, clock=lambda: now)
    registry.ensure("gost-0001")
    _age(registry.path_for("gost-0001"), 99.0, now=now)
    assert registry.sweep() == 0
    assert registry.path_for("gost-0001").exists()


def test_watching_the_stream_keeps_a_session_alive(tmp_path):
    """Час дотику — mtime файлу, і перегляд потоку теж дотик. Гість, який просто дивиться на своє
    село, не має втратити його тільки тому, що не кинув жодної теми."""
    now = 1_000_000.0
    registry = _registry(tmp_path, ttl_s=100.0, clock=lambda: now)
    registry.ensure("gost-0001")
    _age(registry.path_for("gost-0001"), 101.0, now=now)
    registry.touch("gost-0001")
    assert registry.sweep() == 0


def test_a_swept_session_leaves_no_wal_behind(tmp_path):
    """SQLite тримає стан у трьох файлах. Лишити `-wal` означало б і сміття на диску, і часткове
    воскресіння памʼяті, якщо той самий `sid` колись повернеться."""
    now = 1_000_000.0
    registry = _registry(tmp_path, ttl_s=100.0, clock=lambda: now)
    registry.ensure("gost-0001")
    for suffix in ("-wal", "-shm"):
        registry.path_for("gost-0001").with_name(f"gost-0001.db{suffix}").write_bytes(b"x")
    _age(registry.path_for("gost-0001"), 101.0, now=now)
    registry.sweep()
    assert list((tmp_path / "sessions").iterdir()) == []


def test_a_swept_session_is_dropped_from_the_cache(tmp_path):
    """Живий обʼєкт у кеші пережив би видалення файлу й тихо відтворив би базу — тобто прибирання
    виглядало б зробленим, а тека знову росла б."""
    now = 1_000_000.0
    registry = _registry(tmp_path, ttl_s=100.0, clock=lambda: now)
    registry.get("gost-0001")
    _age(registry.path_for("gost-0001"), 101.0, now=now)
    registry.sweep()
    assert "gost-0001" not in registry._cache


def test_the_session_ceiling_holds(tmp_path):
    """TTL сам по собі не рятує: сотня свіжих сесій за годину так само забиває диск, хоч жодна ще
    не протухла."""
    now = 1_000_000.0
    registry = _registry(tmp_path, ttl_s=10_000.0, limit=2, clock=lambda: now)
    for i in range(5):
        registry.ensure(f"gost-{i:04d}")
        _age(registry.path_for(f"gost-{i:04d}"), 100.0 - i, now=now)
    assert registry.sweep() == 3
    assert registry.count() == 2


def test_the_ceiling_drops_the_oldest_first(tmp_path):
    """Гість, що заходив учора, дорожчий за того, кого не бачили місяць."""
    now = 1_000_000.0
    registry = _registry(tmp_path, ttl_s=10_000.0, limit=1, clock=lambda: now)
    registry.ensure("staryi-001")
    _age(registry.path_for("staryi-001"), 5_000.0, now=now)
    registry.ensure("svizhyi-01")
    _age(registry.path_for("svizhyi-01"), 5.0, now=now)
    registry.sweep()
    assert registry.path_for("svizhyi-01").exists()
    assert not registry.path_for("staryi-001").exists()


def test_a_session_in_work_is_never_swept(tmp_path):
    """Видалити базу з-під живого віча означало б обірвати саме той прогін, за який уже заплатили
    токенами. Гонки тут немає за побудовою — і саме тому обіцянку треба закріпити тестом."""
    now = 1_000_000.0
    registry = _registry(tmp_path, ttl_s=100.0, limit=0, clock=lambda: now)
    registry.ensure("gost-0001")
    _age(registry.path_for("gost-0001"), 5_000.0, now=now)
    assert registry.sweep(keep={"gost-0001"}) == 0
    assert registry.path_for("gost-0001").exists()


def test_the_core_sweeps_on_start(tmp_path):
    """Процес міг простояти місяць. Якби прибирання жило лише в циклі, порт відкривався б із
    повною текою й чекав перших пʼяти хвилин."""
    now = time.time()
    bus, queue, registry, runner = _live(tmp_path, ttl_s=100.0)
    registry.ensure("gost-0001")
    _age(registry.path_for("gost-0001"), 101.0, now=now)
    runner.start()
    runner.stop()
    assert runner.sessions_swept == 1
    assert registry.count() == 0


def test_the_core_keeps_sweeping_while_it_idles(tmp_path):
    """Ядро живе тижнями. Прибирання лише на старті означало б, що тека росте весь час аптайму."""
    now = time.time()
    bus, queue, registry, runner = _live(tmp_path, ttl_s=100.0)
    runner.sweep_every_s = 0.0
    runner.resume()
    runner.start()
    registry.ensure("gost-0002")
    _age(registry.path_for("gost-0002"), 101.0, now=now)
    deadline = time.time() + 3.0
    while time.time() < deadline and registry.count():
        time.sleep(0.02)
    runner.stop()
    assert registry.count() == 0, "простій — саме той час, коли є коли прибирати"


def test_health_shows_how_many_sessions_there_are(tmp_path):
    """«Памʼять забилась» мусить бути помітно з `/health`, а не з `du` по теці, коли диск уже
    скінчився."""
    _, _, registry, runner = _live(tmp_path, ttl_s=2 * 86400.0, limit=7)
    registry.ensure("gost-0001")
    health = runner.health()["sessions"]
    assert health["count"] == 1 and health["limit"] == 7
    assert health["ttlDays"] == 2.0


# ── налаштування ──────────────────────────────────────────────────────────────

def test_the_ttl_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("PLOSHCHA_SESSION_TTL_DAYS", "3")
    monkeypatch.setenv("PLOSHCHA_MAX_SESSIONS", "42")
    assert ttl_seconds() == 3 * 86400.0
    assert max_sessions() == 42


def test_a_broken_setting_falls_back_instead_of_killing_the_core(monkeypatch):
    """Порожня або крива змінна в `.env` — звичайна річ. Падіння на старті через неї означало б,
    що ПЛОЩА не піднімається взагалі, і причина видна лише в стеку."""
    for bad in ("", "не число", "0", "-5"):
        monkeypatch.setenv("PLOSHCHA_SESSION_TTL_DAYS", bad)
        monkeypatch.setenv("PLOSHCHA_MAX_SESSIONS", bad)
        assert ttl_seconds() == DEFAULT_TTL_DAYS * 86400.0
        assert max_sessions() == DEFAULT_MAX_SESSIONS


def test_a_run_without_a_session_is_not_broadcast_to_guests():
    """★ Прогін без сесії (консоль, соак, curl) НЕ належить усім.

    Мітка `None` у шині означає «стан ядра», а не «чужа розмова». Доти будь-яка тема, кинута повз
    браузер, летіла в потік КОЖНОГО гостя — і чужі теми зʼявлялись у нього на Дошці, хоч він їх не
    кидав. Тепер такий прогін має власну мітку, а інспектор без `sid` бачить усе, як і доти.
    """
    bus = EventBus()
    bus.publish({"n": "чужий прогін"}, LiveRunner._own(None))
    bus.publish({"n": "своє"}, LiveRunner._own(SID_A))
    assert [e["n"] for e in bus.since(0, SID_B)[0]] == [], "гість не бачить нічийного прогону"
    assert [e["n"] for e in bus.since(0, SID_A)[0]] == ["своє"]
    assert len(bus.since(0, None)[0]) == 2, "інспектор без sid бачить усе"
