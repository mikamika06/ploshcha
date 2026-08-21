"""Живий цикл: черга → оркестратор → потік подій. Окремий потік, бо виклики моделі блокують.

Порядок увімкнення тут навмисний і зворотний до інтуїції: **губернатор і пауза — перші**, робота —
остання. Живий цикл смикає себе без людини, і забутий процес витрачає гроші доти, доки хтось не
помітить. Тому стан за замовчуванням — ПАУЗА, а стеля перевіряється ПЕРЕД кроком, не після.
"""

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from ..adapters.projector import StreamProjector
from ..domain.governor import Governor
from ..ports.trace import StepRecord, TracePort
from .bus import EventBus
from .sessions import SWEEP_EVERY_S, clean_sid

IDLE_SLEEP_S = 0.4
# Скільки аренда вважається живою. Убитий процес (SIGKILL, перезавантаження) не встигає нічого
# повернути, і айтем лишається `leased` НАЗАВЖДИ: `lease` бере лише `pending`, а `requeue_dead`
# бачить лише `dead`. Тобто задача зникала з черги, не потрапивши в жодну статистику провалів.
LEASE_TTL_S = 900.0


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BusTrace(TracePort):
    """Траса, що одразу проєктується в події. Той самий мапінг, що в пакетному режимі.

    `sid` їде поруч із подією, а не в ній: контракт подій фронт валідує строго, тож зайве поле в
    конверті коштувало б відкинутої події, а не просто зайвого байта.
    """

    def __init__(self, bus: EventBus, projector: StreamProjector, sid: str | None = None):
        self.bus = bus
        self.projector = projector
        self.sid = sid
        self.records: list[StepRecord] = []

    def emit(self, record: StepRecord) -> None:
        record.seq = len(self.records)
        self.records.append(record)
        self.bus.publish(self.projector.feed(record), self.sid)


class _Crash:
    """Результат прогону, що не відбувся. Форма та сама, яку читає `StreamProjector.close`."""

    outcome = "failure"
    evidence = None
    verdict_kind = None
    scratch: list = []
    notes: list = []
    tokens = 0
    aux_tokens = 0

    def __init__(self, message: str):
        self.incidents = [f"live_crash:{message[:180]}"]


class LiveRunner:
    """`make_orchestrator(trace, run_id)` віддає готовий оркестратор; черга дає задачі."""

    def __init__(self, bus: EventBus, queue, make_orchestrator: Callable,
                 *, governor: Governor | None = None, scene: dict | None = None,
                 worker: str = "ploshcha", paused: bool = True,
                 estimate_tokens: int = 2000, cast: list[dict] | None = None,
                 decisions=None, rumours=None, sessions=None, latency=None,
                 sweep_every_s: float = SWEEP_EVERY_S):
        self.bus = bus
        self.queue = queue
        # Прилад тривалості: «чому так довго» мусить мати число, а не здогад.
        self.latency = latency
        self.make_orchestrator = make_orchestrator
        self.governor = governor or Governor()
        self.scene = scene
        self.cast = cast
        self.decisions = decisions
        self.rumours = rumours
        # Реєстр сесій. Порожньо — старий режим: одне спільне село на всіх (CLI, соак, тести).
        self.sessions = sessions
        self.sweep_every_s = sweep_every_s
        self.worker = worker
        self.estimate_tokens = estimate_tokens
        self._paused = threading.Event()
        if paused:
            self._paused.set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.state = "paused" if paused else "running"
        self.tick = 0
        self.last_error: str | None = None
        self.stopped_reason: str | None = None
        self.runs_done = 0
        # Скільки покинутих аренд повернуто в чергу на старті. Нуль — теж відповідь, тому число
        # видиме завжди, а не лише коли щось сталось.
        self.recovered = 0
        # Поточний агент: слово гостя має потрапити в те віче, що ЙДЕ ЗАРАЗ, а не в наступне.
        self.current = None
        # ★ Чиє саме віче зараз іде. Без цього гість вкидав би слово в чужу розмову — свого віча
        # він не бачить, чуже чує, і виглядає це як «моє слово пішло невідомо куди».
        self.current_sid = None
        self.sessions_swept = 0

        self._last_sweep = 0.0

    # ── керування ────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        # Перед першим лізом підбираємо те, що лишилось від УБИТОГО процесу. Без цього кожен
        # SIGKILL списував по темі: у базі вона лежала `leased`, а виглядало це як «Дошка
        # приймає теми, а віче не починається».
        self.recovered += self._recover_stale()
        # (а) Прибирання на СТАРТІ. Процес міг простояти місяць — і тоді періодичний обхід
        # усередині циклу вперше спрацював би лише через пʼять хвилин після підняття порту.
        self._sweep()
        self._thread = threading.Thread(target=self._loop, name="live-runner", daemon=True)
        self._thread.start()

    def _recover_stale(self, older_than_s: float = LEASE_TTL_S) -> int:
        if self.queue is None:
            return 0
        try:
            return int(self.queue.recover_stale(older_than_s) or 0)
        except Exception as exc:
            # Черга може не піднятись (файл зайнятий, схема стара) — але це не привід не стартувати
            # зовсім. Причина лягає в `lastError`, тобто в `/health`, а не в нікуди.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return 0

    def pause(self) -> None:
        self._paused.set()
        with self._lock:
            self.state = "paused"

    def resume(self) -> None:
        if self.stopped_reason is not None:
            return
        self._paused.clear()
        with self._lock:
            self.state = "running"

    def stop(self) -> None:
        self._stop.set()
        self._paused.set()
        with self._lock:
            self.state = "stopped"
            self.stopped_reason = self.stopped_reason or "зупинено вручну"
        self.bus.close()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # ── стан ─────────────────────────────────────────────────────────────────
    def health(self) -> dict:
        with self._lock:
            state, reason, err = self.state, self.stopped_reason, self.last_error
        spend = self.governor.spend
        return {
            "state": state,
            # ★ Головне поле цього звіту. Фронт чекає розмову доти, доки `state == "running"`, і
            # мертвий потік із написом «running» вішав «Село думу думає…» назавжди — це і є
            # «ядро не відповідає». Живість потоку — факт, а не намір, тож її і питаємо.
            "alive": self._thread is not None and self._thread.is_alive(),
            "stoppedReason": reason,
            "lastError": err,
            "tick": self.tick,
            "runsDone": self.runs_done,
            "recovered": self.recovered,
            # Скільки СПРАВДІ триває виклик кожного ярусу: «чому так довго» має вимірюватись,
            # а не вгадуватись. Тримаємо останні заміри в памʼяті, без окремого сховища.
            "latency": self.latency.summary() if self.latency is not None else {},
            "spend": {"items": spend.items_done, "tokens": spend.tokens,
                      "usd": round(spend.usd, 6)},
            "caps": {"maxItems": self.governor.max_items, "maxTokens": self.governor.max_tokens,
                     "maxUsd": self.governor.max_usd},
            "queue": self.queue.stats() if self.queue is not None else {},
            "events": {"nextSeq": self.bus.next_seq, "dropped": self.bus.dropped},
            # Сесії видимі як число: «памʼять забилась» мусить бути помітно з `/health`, а не з
            # `du` по теці, коли диск уже скінчився.
            "sessions": ({"count": self.sessions.count(), "swept": self.sessions_swept,
                          "limit": self.sessions.limit,
                          "ttlDays": round(self.sessions.ttl_s / 86400.0, 3)}
                         if self.sessions is not None else {}),
        }

    # ── цикл ─────────────────────────────────────────────────────────────────
    def _degrade(self, reason: str, *, stage: str = "governor") -> None:
        # Спершу ДОКАЗ у потоці, потім оголошення стану: інакше спостерігач бачить «зупинено»,
        # іде читати причину — а її ще не опублікували. Той самий порядок, що `ack` після роботи.
        # ★ Без `sid`, тобто ВСІМ. Стеля витрат і смерть робітника — стан самого ядра: воно
        # спинилось для кожного, і мовчати про це перед іншими гостями означало б лишити їх із
        # написом «Село думу думає…» назавжди. Розмова належить сесії, ядро — нікому.
        self._paused.set()
        proj = StreamProjector(f"guard-{uuid.uuid4().hex[:8]}", _now())
        self.bus.publish(proj._envelope("run.degraded",
                                        {"stage": stage, "reason": reason}, self.tick))
        with self._lock:
            self.stopped_reason = reason
            self.state = "paused"

    def _sweep(self) -> int:
        """(б) Періодичне прибирання — у ТОМУ САМОМУ потоці, що веде прогони.

        Окремий потік-прибиральник довелось би синхронізувати з прогоном, щоб не видалити базу
        з-під живого віча. Тут така гонка неможлива за побудовою, а ціна — обхід теки раз на
        `sweep_every_s` між айтемами.
        """
        if self.sessions is None:
            return 0
        now = time.time()
        if now - self._last_sweep < self.sweep_every_s:
            return 0
        self._last_sweep = now
        try:
            gone = int(self.sessions.sweep(keep={self.current_sid} if self.current_sid else None))
        except Exception as exc:
            # Тека може бути недоступна — але це не привід зупиняти село. Причина лягає в
            # `/health`, а не в нікуди.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return 0
        self.sessions_swept += gone
        return gone

    def _loop(self) -> None:
        # ★ `finally` тут не косметика. Потік може вмерти й повз `except Exception`: досить
        # `BaseException` або збою в самому обробнику (`print_exc` на закритому stderr у
        # відвʼязаному процесі — саме такий випадок). Тоді робітника вже немає, а `state` далі
        # каже «running», і кожна нова тема просто лягає в чергу назавжди.
        try:
            while not self._stop.is_set():
                try:
                    if self._paused.is_set():
                        time.sleep(IDLE_SLEEP_S)
                        continue
                    reason = self.governor.stop_reason(next_tokens=self.estimate_tokens)
                    if reason is not None:
                        self._degrade(reason)
                        continue
                    item = self.queue.lease(self.worker) if self.queue is not None else None
                    if item is None:
                        self._sweep()
                        time.sleep(IDLE_SLEEP_S)
                        continue
                    self._run_one(item)
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._print_exc()
                    proj = StreamProjector(f"loop-{uuid.uuid4().hex[:8]}", _now())
                    # Падіння самого циклу — теж стан ядра, тому теж усім.
                    self.bus.publish(proj._envelope("run.error",
                                                    {"message": self.last_error}, self.tick))
                    self._degrade(f"цикл упав: {self.last_error}", stage="loop")
        finally:
            if not self._stop.is_set():
                self._died()

    @staticmethod
    def _print_exc() -> None:
        """Друк стека не має бути причиною смерті потоку: у відвʼязаному процесі stderr закритий."""
        try:
            traceback.print_exc()
        except Exception:
            pass

    def _died(self) -> None:
        """Робітник помер сам. Це кажемо ВГОЛОС — і в потоці, і в `/health`."""
        reason = self.last_error or "невідома причина"
        try:
            proj = StreamProjector(f"dead-{uuid.uuid4().hex[:8]}", _now())
            self.bus.publish(proj._envelope(
                "run.degraded", {"stage": "worker", "reason": f"робітник помер: {reason}"},
                self.tick))
        except Exception:
            pass
        self._paused.set()
        with self._lock:
            self.stopped_reason = self.stopped_reason or f"робітник помер: {reason}"
            self.state = "stopped"

    def _restore_decisions(self, proj: StreamProjector, decisions=None) -> list[dict]:
        decisions = self.decisions if decisions is None else decisions
        if decisions is None:
            return []
        out: list[dict] = []
        for d in decisions.standing():
            out.append(proj._envelope("event.happened", {"event": {
                "id": f"standing-{d['who']}", "kind": "decision", "label": d["label"],
                "description": "чинна ухвала минулого віча",
                "place": {"poi": d["poi"]}, "involves": [d["who"]]}}, 0))
            out += proj._walk(d["who"], d["poi"], 0)
        return out

    def _run_one(self, item) -> None:
        # ★ Сховища беруться ПІД АЙТЕМ, а не раз на старті: `sid` лежить у самій задачі, тож те
        # саме ядро й той самий цикл ведуть різні села — по одному на гостя.
        sid = clean_sid((item.payload or {}).get("sid"))
        self.current_sid = sid
        run_id = f"live-{item.key[:8]}-{uuid.uuid4().hex[:6]}"
        proj = StreamProjector(run_id, _now(), scene=self.scene, max_ticks=64, cast=self.cast)
        # ★ Аренду тримає ВЕСЬ метод, а не лише виклик агента. Доти `proj.start()`, чинні ухвали й
        # післяпрогінний запис у бази стояли ПОЗА `try`: збій там (зайнятий SQLite, повна шина)
        # летів у зовнішній `except` циклу, який про айтем нічого не знає, — і той лишався `leased`
        # назавжди. Тобто тема зникала мовчки, не потрапивши навіть у `dead`.
        try:
            session = self.sessions.get(sid) if self.sessions is not None else None
            if session is not None and session.cast:
                # Склад оголошується всередині `proj.start()`, тобто ПІСЛЯ цього рядка, — тож
                # підміна безпечна, і на сцені гостя стоять люди з ЙОГО села.
                proj.cast = session.cast
            self._work(item, proj, run_id, sid, session)
        except Exception as exc:
            self._crashed(item, proj, exc, sid)
        finally:
            self.current_sid = None

    def _work(self, item, proj: StreamProjector, run_id: str, sid: str | None = None,
              session=None) -> None:
        task = str((item.payload or {}).get("task") or item.key)
        decisions = self.decisions if session is None else session.decisions
        rumours = self.rumours if session is None else session.rumours
        make = self.make_orchestrator if session is None else session.make_agent
        self.bus.publish(proj.start(), sid)
        # ★ Чинні ухвали повертаються на сцену ДО розмови: інакше «поставили сторожа» жило б рівно
        # один прогін, і рішення знову не мало б наслідку.
        self.bus.publish(self._restore_decisions(proj, decisions), sid)
        trace = BusTrace(self.bus, proj, sid)
        orch = make(trace, run_id, (item.payload or {}).get("place"))
        self.current = orch
        template = getattr(orch, "budget_template", None)
        result = orch.run(task, seed=1,
                          budget=template.model_copy(deep=True) if template else None)
        self.current = None
        if rumours is not None:
            for rec in trace.records:
                if rec.agent == "rumour" and (rec.parsed or {}).get("claim"):
                    rumours.add(task, str(rec.parsed.get("who") or ""),
                                str(rec.parsed["claim"]))
        if decisions is not None:
            for rec in trace.records:
                if rec.agent == "council" and (rec.parsed or {}).get("poi"):
                    d = rec.parsed
                    decisions.add(task, str(d["label"]), str(d.get("who") or ""),
                                  str(d["poi"]))
        self.bus.publish(proj.close(result, done=True), sid)
        tokens = int(getattr(result, "tokens", 0) or 0) + int(getattr(result, "aux_tokens", 0) or 0)
        self.governor.record(tokens=tokens)
        self.tick = max(self.tick, getattr(result, "steps", 0) or 0)
        self.runs_done += 1
        if self.queue is not None:
            self.queue.ack(item.key, {"tokens": tokens,
                                      "outcome": getattr(result, "outcome", "answer")})

    def _crashed(self, item, proj: StreamProjector, exc: Exception, sid: str | None = None) -> None:
        """Падіння прогону мусить ЗАКРИТИ прогін, а не лише поскаржитись.

        Доти летів самий `run.error`, тож термінального стану задачі не було взагалі: у сцени
        прогін лишався відкритим, а в потоці бракувало `task.outcome` — того єдиного місця, де
        видно `incidents`. Порядок навмисний: спершу підсумок, `run.error` останній, бо саме він
        лишається як статус прогону.
        """
        self.last_error = f"{type(exc).__name__}: {exc}"
        self._print_exc()
        self.current = None
        try:
            self.bus.publish(proj.close(_Crash(self.last_error), done=False), sid)
        except Exception:
            pass
        self.bus.publish(proj._envelope("run.error", {"message": self.last_error}, self.tick), sid)
        if self.queue is not None:
            try:
                self.queue.fail(item.key, self.last_error)
            except Exception as inner:
                # Черга не прийняла провал — тоді айтем висить в аренді, і про це мусить бути
                # видно з `/health`, інакше «зникла тема» знову не має пояснення.
                self.last_error = f"{self.last_error} | черга не прийняла провал: {inner}"
