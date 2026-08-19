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

IDLE_SLEEP_S = 0.4


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class BusTrace(TracePort):
    """Траса, що одразу проєктується в події. Той самий мапінг, що в пакетному режимі."""

    def __init__(self, bus: EventBus, projector: StreamProjector):
        self.bus = bus
        self.projector = projector
        self.records: list[StepRecord] = []

    def emit(self, record: StepRecord) -> None:
        record.seq = len(self.records)
        self.records.append(record)
        self.bus.publish(self.projector.feed(record))


class LiveRunner:
    """`make_orchestrator(trace, run_id)` віддає готовий оркестратор; черга дає задачі."""

    def __init__(self, bus: EventBus, queue, make_orchestrator: Callable,
                 *, governor: Governor | None = None, scene: dict | None = None,
                 worker: str = "ploshcha", paused: bool = True,
                 estimate_tokens: int = 2000, cast: list[dict] | None = None,
                 decisions=None, rumours=None):
        self.bus = bus
        self.queue = queue
        self.make_orchestrator = make_orchestrator
        self.governor = governor or Governor()
        self.scene = scene
        self.cast = cast
        self.decisions = decisions
        self.rumours = rumours
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
        # Поточний агент: слово гостя має потрапити в те віче, що ЙДЕ ЗАРАЗ, а не в наступне.
        self.current = None

    # ── керування ────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="live-runner", daemon=True)
        self._thread.start()

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
            "stoppedReason": reason,
            "lastError": err,
            "tick": self.tick,
            "runsDone": self.runs_done,
            "spend": {"items": spend.items_done, "tokens": spend.tokens,
                      "usd": round(spend.usd, 6)},
            "caps": {"maxItems": self.governor.max_items, "maxTokens": self.governor.max_tokens,
                     "maxUsd": self.governor.max_usd},
            "queue": self.queue.stats() if self.queue is not None else {},
            "events": {"nextSeq": self.bus.next_seq, "dropped": self.bus.dropped},
        }

    # ── цикл ─────────────────────────────────────────────────────────────────
    def _degrade(self, reason: str, *, stage: str = "governor") -> None:
        # Спершу ДОКАЗ у потоці, потім оголошення стану: інакше спостерігач бачить «зупинено»,
        # іде читати причину — а її ще не опублікували. Той самий порядок, що `ack` після роботи.
        self._paused.set()
        proj = StreamProjector(f"guard-{uuid.uuid4().hex[:8]}", _now())
        self.bus.publish(proj._envelope("run.degraded",
                                        {"stage": stage, "reason": reason}, self.tick))
        with self._lock:
            self.stopped_reason = reason
            self.state = "paused"

    def _loop(self) -> None:
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
                    time.sleep(IDLE_SLEEP_S)
                    continue
                self._run_one(item)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                proj = StreamProjector(f"loop-{uuid.uuid4().hex[:8]}", _now())
                self.bus.publish(proj._envelope("run.error",
                                                {"message": self.last_error}, self.tick))
                self._degrade(f"цикл упав: {self.last_error}", stage="loop")

    def _restore_decisions(self, proj: StreamProjector) -> list[dict]:
        if self.decisions is None:
            return []
        out: list[dict] = []
        for d in self.decisions.standing():
            out.append(proj._envelope("event.happened", {"event": {
                "id": f"standing-{d['who']}", "kind": "decision", "label": d["label"],
                "place": {"poi": d["poi"]}, "involves": [d["who"]]}}, 0))
            out += proj._walk(d["who"], d["poi"], 0)
        return out

    def _run_one(self, item) -> None:
        task = str((item.payload or {}).get("task") or item.key)
        run_id = f"live-{item.key[:8]}-{uuid.uuid4().hex[:6]}"
        ts = _now()
        proj = StreamProjector(run_id, ts, scene=self.scene, max_ticks=64, cast=self.cast)
        self.bus.publish(proj.start())
        # ★ Чинні ухвали повертаються на сцену ДО розмови: інакше «поставили сторожа» жило б рівно
        # один прогін, і рішення знову не мало б наслідку.
        self.bus.publish(self._restore_decisions(proj))
        trace = BusTrace(self.bus, proj)
        try:
            orch = self.make_orchestrator(trace, run_id)
            self.current = orch
            template = getattr(orch, "budget_template", None)
            result = orch.run(task, seed=1,
                              budget=template.model_copy(deep=True) if template else None)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.bus.publish(proj._envelope("run.error", {"message": self.last_error}, self.tick))
            self.current = None
            if self.queue is not None:
                self.queue.fail(item.key, self.last_error)
            return
        self.current = None
        if self.rumours is not None:
            for rec in trace.records:
                if rec.agent == "rumour" and (rec.parsed or {}).get("claim"):
                    self.rumours.add(task, str(rec.parsed.get("who") or ""),
                                     str(rec.parsed["claim"]))
        if self.decisions is not None:
            for rec in trace.records:
                if rec.agent == "council" and (rec.parsed or {}).get("poi"):
                    d = rec.parsed
                    self.decisions.add(task, str(d["label"]), str(d.get("who") or ""),
                                       str(d["poi"]))
        self.bus.publish(proj.close(result, done=True))
        tokens = int(getattr(result, "tokens", 0) or 0) + int(getattr(result, "aux_tokens", 0) or 0)
        self.governor.record(tokens=tokens)
        self.tick = max(self.tick, getattr(result, "steps", 0) or 0)
        self.runs_done += 1
        if self.queue is not None:
            self.queue.ack(item.key, {"tokens": tokens,
                                      "outcome": getattr(result, "outcome", "answer")})
