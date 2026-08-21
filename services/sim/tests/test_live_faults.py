"""Шляхи ПОМИЛОК живого ядра: чи чути поламку, а не лише її наслідок.

Скарга, з якої це почалось, звучала як «ядро не відповідає» або «щось вилітає, а причини не видно».
Обидва симптоми — не про сам механізм, а про шлях спостереження: ядро працювало, просто мовчало.
Тут стережуться саме ті чотири місця, де мовчання коштувало теми або сеансу:

  • команда, що впала, мусить дати ВІДПОВІДЬ, а не обрив зʼєднання;
  • айтем черги не зникає в аренді, хоч би де стався збій після лізу;
  • покинута аренда вбитого процесу повертається сама;
  • мертвий робітник не має права звітувати «running».
"""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from ploshcha_sim.adapters import FakeLlm, FakeToolbox, PresetEffort
from ploshcha_sim.adapters.queue_sqlite import SqliteQueue
from ploshcha_sim.adapters.router_profile import single_model_router
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.domain.governor import Governor
from ploshcha_sim.live import EventBus, LiveRunner, handle_command, serve
from ploshcha_sim.live.runner import LEASE_TTL_S

FINAL = json.dumps({"tool": "final_answer", "text": "Готово."}, ensure_ascii=False)


def _orch(trace, run_id, place=None):
    return Orchestrator(single_model_router(FakeLlm([FINAL], model="fake")), PresetEffort(),
                        FakeToolbox(), verifier=False, trace=trace, run_id=run_id)


def _runner(tmp_path, *, make=_orch, tokens: int = 10_000, clock=time.time, **kw):
    bus = EventBus()
    queue = SqliteQueue(str(tmp_path / "q.db"), clock=clock)
    return bus, queue, LiveRunner(bus, queue, make, governor=Governor(max_tokens=tokens), **kw)


def _until(check, seconds: float = 3.0):
    deadline = time.time() + seconds
    while time.time() < deadline and not check():
        time.sleep(0.02)
    return check()


# ── команда, що впала ─────────────────────────────────────────────────────────

class _BrokenRunner:
    """Рівно те, що ламається насправді: зіпсована залежність усередині команди."""

    queue = None
    current = None
    state = "paused"
    stopped_reason = None
    last_error = None

    def health(self) -> dict:
        return {"state": self.state}

    def pause(self) -> None:
        raise RuntimeError("сховище стану недоступне")


def _post(port: int, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/command",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def broken_server():
    runner = _BrokenRunner()
    httpd = serve(EventBus(), runner, port=0)
    yield httpd.server_address[1], runner
    httpd.shutdown()


def test_a_command_that_blew_up_still_answers_the_client(broken_server):
    """Доти виняток у команді летів повз хендлер, і клієнт не діставав НІЧОГО — обрив замість
    відповіді. Саме так і виглядає «ядро не відповідає»: воно живе, просто мовчить."""
    port, _ = broken_server
    code, body = _post(port, {"kind": "pause"})
    assert code == 500
    assert "сховище стану недоступне" in body["error"], "причина мусить доїхати до клієнта"


def test_a_command_that_blew_up_leaves_its_reason_in_health(broken_server):
    """Інспектор не має доступу до логів процесу: якщо причини немає в `/health`, її немає ніде."""
    port, runner = broken_server
    _post(port, {"kind": "pause"})
    assert runner.last_error and "RuntimeError" in runner.last_error


def test_a_healthy_command_is_not_touched_by_the_error_boundary(broken_server):
    """Межа помилки не має ковтати нормальні коди: 400 лишається 400, а не стає 500."""
    port, _ = broken_server
    assert _post(port, {"kind": "невідома"})[0] == 400


def test_a_topic_without_a_queue_is_refused_with_a_reason(tmp_path):
    """`requeue` цю перевірку мав, `topic` — ні, тож той самий стан ядра давав `AttributeError`."""
    _, _, runner = _runner(tmp_path)
    runner.queue = None
    code, body = handle_command({"kind": "topic", "text": "Гребля"}, runner)
    assert code == 400 and body["error"] == "черги немає"


# ── айтем не зникає в аренді ──────────────────────────────────────────────────

class _BrokenDecisions:
    """Чинні ухвали читаються з SQLite ПІСЛЯ лізу — і саме там база буває зайнята."""

    def standing(self):
        raise RuntimeError("база зайнята")


def test_a_crash_before_the_agent_starts_does_not_swallow_the_item(tmp_path):
    """Збій між лізом і агентом летів у зовнішній `except` циклу, який про айтем не знає, — і той
    лишався `leased` НАЗАВЖДИ: `lease` бере лише `pending`, `requeue_dead` бачить лише `dead`."""
    bus, queue, runner = _runner(tmp_path, decisions=_BrokenDecisions())
    queue.put("k", {"task": "тема"})
    runner._run_one(queue.lease("w"))

    assert not queue.stats().get("leased"), "аренда мусить бути знята, а не зависнути"
    assert queue.stats().get("pending") == 1, "тема повертається в чергу, а не зникає"
    assert "база зайнята" in (runner.health()["lastError"] or "")


def test_a_crash_after_the_run_does_not_swallow_the_item_either(tmp_path):
    """Запис у бази чуток і ухвал теж стоїть після лізу: він так само може впасти."""
    class _BrokenRumours:
        def add(self, *a, **kw):
            raise RuntimeError("диск повний")

    bus, queue, runner = _runner(tmp_path, rumours=_BrokenRumours())
    queue.put("k", {"task": "тема"})

    def make(trace, run_id, place=None):
        from ploshcha_sim.ports.trace import StepRecord

        trace.emit(StepRecord(run_id=run_id, tick=1, agent="rumour", stage="judge", model="m",
                              lane="none", prompt="", raw_output="",
                              parsed={"who": "koval", "claim": "кажуть, вовк"}))
        return _orch(trace, run_id)

    runner.make_orchestrator = make
    runner._run_one(queue.lease("w"))
    assert not queue.stats().get("leased")


def test_an_item_that_keeps_crashing_dies_instead_of_looping_forever(tmp_path):
    """Повернення в чергу не має стати вічним колом: інакше одна зламана тема зʼїдає весь бюджет."""
    bus, queue, runner = _runner(tmp_path, decisions=_BrokenDecisions())
    queue.put("k", {"task": "тема"})
    for _ in range(queue.max_attempts):
        runner._run_one(queue.lease("w"))
    assert queue.stats() == {"dead": 1}


def test_a_crashed_run_gets_a_terminal_outcome_not_just_a_complaint(tmp_path):
    """`task.outcome` — єдине місце потоку, де видно `incidents`. Доти при падінні летів самий
    `run.error`, тобто прогін не мав термінального стану взагалі, а сцена лишалась відкритою."""
    bus, queue, runner = _runner(tmp_path)
    queue.put("k", {"task": "тема"})

    def boom(trace, run_id, place=None):
        raise RuntimeError("двигун не піднявся")

    runner.make_orchestrator = boom
    runner.resume()
    runner.start()
    assert _until(lambda: any(e["type"] == "run.error" for e in bus.since(0)[0]))
    runner.stop()

    types = [e["type"] for e in bus.since(0)[0]]
    assert "task.outcome" in types
    outcome = next(e for e in bus.since(0)[0] if e["type"] == "task.outcome")
    assert outcome["payload"]["outcome"] == "failure"
    assert any("двигун не піднявся" in i for i in outcome["payload"]["incidents"])
    assert types.index("task.outcome") < types.index("run.error"), \
        "статусом прогону лишається помилка, тому вона йде ОСТАННЬОЮ"


def test_a_crashed_run_is_not_reported_as_done(tmp_path):
    """`run.done` після `run.error` затер би сам факт падіння: фронт тримає один статус прогону."""
    bus, queue, runner = _runner(tmp_path)
    queue.put("k", {"task": "тема"})
    runner.make_orchestrator = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("нема"))
    runner.resume()
    runner.start()
    assert _until(lambda: any(e["type"] == "run.error" for e in bus.since(0)[0]))
    runner.stop()
    assert not [e for e in bus.since(0)[0] if e["type"] == "run.done"]


# ── покинута аренда ───────────────────────────────────────────────────────────

def _frozen_clock(start: float = 1000.0):
    now = [start]
    return now, (lambda: now[0])


def test_an_abandoned_lease_comes_back_when_the_core_starts(tmp_path):
    """Убитий процес (SIGKILL, перезавантаження) не встигає нічого повернути. Ніхто не кликав
    `recover_stale`, тож тема лежала `leased` вічно — а виглядало це як «Дошка приймає, віче ні»."""
    now, clock = _frozen_clock()
    bus, queue, runner = _runner(tmp_path, clock=clock)
    queue.put("k", {"task": "тема"})
    queue.lease("померлий")
    assert queue.stats() == {"leased": 1}

    now[0] += LEASE_TTL_S + 1
    runner.start()
    runner.stop()
    assert runner.recovered == 1
    assert queue.stats().get("pending") == 1
    assert runner.health()["recovered"] == 1, "число мусить бути видно в /health"


def test_a_live_lease_is_never_taken_from_the_worker(tmp_path):
    """Зворотний бік: якщо відбирати аренду одразу, два робітники візьмуть ту саму тему."""
    now, clock = _frozen_clock()
    bus, queue, runner = _runner(tmp_path, clock=clock)
    queue.put("k", {"task": "тема"})
    queue.lease("живий")
    runner.start()
    runner.stop()
    assert runner.recovered == 0 and queue.stats() == {"leased": 1}


def test_requeue_also_picks_up_an_abandoned_lease(tmp_path):
    """Команду кличуть саме тоді, коли «тема зникла» — а вона рятувала лише `dead`."""
    now, clock = _frozen_clock()
    bus, queue, runner = _runner(tmp_path, clock=clock)
    queue.put("k", {"task": "тема"})
    queue.lease("померлий")
    now[0] += LEASE_TTL_S + 1

    code, body = handle_command({"kind": "requeue"}, runner)
    assert code == 200 and body["recovered"] == 1
    assert body["queue"].get("pending") == 1


def test_a_queue_that_cannot_be_recovered_says_why_instead_of_refusing_to_start(tmp_path):
    """Зламана черга не має ховати ядро цілком: причина йде в `lastError`, а сервер піднімається."""
    bus, queue, runner = _runner(tmp_path)

    def boom(_older):
        raise RuntimeError("схема стара")

    runner.queue.recover_stale = boom
    runner.start()
    runner.stop()
    assert "схема стара" in (runner.health()["lastError"] or "")


# ── мертвий робітник ──────────────────────────────────────────────────────────

class _Doom(BaseException):
    """`except Exception` цього не ловить — саме так робітник і вмирав нечутно."""


class _DoomedGovernor(Governor):
    def stop_reason(self, **kw):
        raise _Doom()


# Смерть робітника тут НАВМИСНА, тому попередження pytest про неї — очікуваний шум, а не сигнал.
doomed = pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")


def _doomed(tmp_path):
    bus, queue, runner = _runner(tmp_path)
    runner.governor = _DoomedGovernor()
    runner.resume()
    runner.start()
    # Смерть мусить статись УСЕРЕДИНІ цього тесту: інакше виняток потоку спливе в наступному.
    runner.join()
    return bus, runner


@doomed
def test_a_dead_worker_is_not_reported_as_running(tmp_path):
    """Фронт чекає розмову, доки `state == "running"`. Мертвий потік із цим написом вішав
    «Село думу думає…» назавжди — це і є «ядро не відповідає»."""
    _, runner = _doomed(tmp_path)
    assert _until(lambda: not runner.health()["alive"])
    assert runner.health()["state"] == "stopped"
    assert runner.stopped_reason, "смерть робітника мусить мати названу причину"


@doomed
def test_a_dead_worker_says_so_in_the_stream_too(tmp_path):
    bus, runner = _doomed(tmp_path)
    assert _until(lambda: any(e["type"] == "run.degraded" for e in bus.since(0)[0]))
    degraded = next(e for e in bus.since(0)[0] if e["type"] == "run.degraded")
    assert degraded["payload"]["stage"] == "worker"


def test_a_worker_stopped_on_purpose_is_not_called_dead(tmp_path):
    """Інакше кожна штатна зупинка писала б у потік хибну тривогу."""
    bus, queue, runner = _runner(tmp_path)
    runner.start()
    runner.stop()
    runner.join()
    assert not [e for e in bus.since(0)[0]
                if e["type"] == "run.degraded" and e["payload"]["stage"] == "worker"]
    assert runner.stopped_reason == "зупинено вручну"


def test_health_shows_liveness_before_the_worker_is_even_started(tmp_path):
    """`alive` мусить бути фактом, а не наміром: до старту робітника немає."""
    _, _, runner = _runner(tmp_path)
    assert runner.health()["alive"] is False


def test_printing_a_traceback_can_never_kill_the_worker(tmp_path, monkeypatch):
    """Реальний випадок відвʼязаного процесу: stderr закритий, і `print_exc` перетворював
    оброблену помилку на смерть потоку — тобто на тишу замість діагностики."""
    import traceback as tb

    monkeypatch.setattr(tb, "print_exc", lambda *a, **kw: (_ for _ in ()).throw(
        ValueError("I/O operation on closed file")))
    bus, queue, runner = _runner(tmp_path)
    queue.put("k", {"task": "тема"})
    runner.make_orchestrator = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("впало"))
    runner.resume()
    runner.start()
    assert _until(lambda: any(e["type"] == "run.error" for e in bus.since(0)[0]))
    assert runner.health()["alive"], "робітник мусить пережити збій друку стека"
    runner.stop()


# ── стеля витрат ──────────────────────────────────────────────────────────────

def test_a_cap_hit_between_items_leaves_the_queue_whole(tmp_path):
    """Стеля перевіряється ПЕРЕД лізом, тож незапущена тема мусить лишитись `pending`, а не
    згоріти разом із циклом."""
    bus, queue, runner = _runner(tmp_path, tokens=1)
    queue.put("k", {"task": "тема"})
    runner.governor.record(tokens=10_000)
    runner.resume()
    runner.start()
    assert _until(lambda: runner.stopped_reason is not None)
    runner.stop()

    assert queue.stats() == {"pending": 1}, "черга мусить лишитись цілою"
    degraded = [e for e in bus.since(0)[0] if e["type"] == "run.degraded"]
    assert degraded and degraded[0]["payload"]["stage"] == "governor"


def test_a_cap_is_announced_before_the_state_says_stopped(tmp_path):
    """Порядок: спершу ДОКАЗ у потоці, потім стан. Інакше спостерігач бачить «спинилось», іде
    читати причину — а її ще не опублікували."""
    bus, queue, runner = _runner(tmp_path, tokens=1)
    runner.governor.record(tokens=10_000)
    runner.resume()
    runner.start()
    assert _until(lambda: runner.stopped_reason is not None)
    runner.stop()
    assert any(e["type"] == "run.degraded" for e in bus.since(0)[0])


def test_a_refused_resume_does_not_answer_ok(tmp_path):
    """Кнопка, що нічого не робить, але відповідає «ok», — той самий клас брехні шляху
    спостереження: механізм стоїть, а прилад каже, що поїхали."""
    _, _, runner = _runner(tmp_path)
    runner.stopped_reason = "межа токенів"
    code, body = handle_command({"kind": "resume"}, runner)
    assert code == 200
    assert body["ok"] is False and body["stoppedReason"] == "межа токенів"


def test_a_normal_resume_still_answers_ok(tmp_path):
    _, _, runner = _runner(tmp_path)
    assert handle_command({"kind": "resume"}, runner)[1]["ok"] is True


# ── шина під час зупинки ──────────────────────────────────────────────────────

def test_a_viewer_waiting_on_a_closed_bus_is_released_not_hung(tmp_path):
    """Закрита шина мусить будити тих, хто чекає: інакше вкладка спостерігача висить до таймауту."""
    bus = EventBus()
    done = threading.Event()

    def waiter():
        bus.wait(0, timeout=5.0)
        done.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.05)
    bus.close()
    assert done.wait(2.0), "close() мусить розбудити глядача"
    assert bus.closed
