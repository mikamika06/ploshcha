"""Живий цикл: черга → оркестратор → потік подій. Окремий потік, бо виклики моделі блокують.

Порядок увімкнення тут навмисний і зворотний до інтуїції: **губернатор і пауза — перші**, робота —
остання. Живий цикл смикає себе без людини, і забутий процес витрачає гроші доти, доки хтось не
помітить. Тому стан за замовчуванням — ПАУЗА, а стеля перевіряється ПЕРЕД кроком, не після.
"""

import threading
import zlib
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone

from ..adapters.projector import StreamProjector
from ..domain.governor import Governor
from ..ports.trace import StepRecord, TracePort
from .bus import EventBus
from .sessions import SWEEP_EVERY_S, clean_sid

IDLE_SLEEP_S = 0.4
# Як часто наглядач дивиться на чергу і як довго порожній робітник живе до згасання.
SUPERVISE_EVERY_S = 2.0
WORKER_IDLE_EXIT_S = 120.0
# Скільки аренда вважається живою. Убитий процес (SIGKILL, перезавантаження) не встигає нічого
# повернути, і айтем лишається `leased` НАЗАВЖДИ: `lease` бере лише `pending`, а `requeue_dead`
# бачить лише `dead`. Тобто задача зникала з черги, не потрапивши в жодну статистику провалів.
LEASE_TTL_S = 900.0
# ★ Скільки віче говорить після того, як його перестали слухати.
#
# Пільга, а не миттєвий обрив: закрита вкладка й перезавантажена сторінка з боку сервера
# виглядають ОДНАКОВО — обривом SSE. Людина, що натиснула F5, вертається з тим самим `sid` і
# мусить застати своє віче на місці. Число зроблено з двох, які вже є в коді: обрив помічається
# не раніше першого удару серця в мертвий сокет (`HEARTBEAT_S` = 15 с), а клієнт після помилки
# перепідключається через секунду (`LiveDriver`). Три удари серця дають запас на перезавантаження
# сторінки з холодним кешем і на одну невдалу спробу підключення.
#
# ★ І ОСЬ ЧОГО ЦЯ ПІЛЬГА КОШТУЄ. Заміряно наживо 2026-08-28 (умова `viche`): гість вийшов після
# другої репліки, тиша настала через 51.3 с — і прогін устиг витратити 19 935 токенів із 24 761,
# тобто пільга віддала назад лише 4 826. Ручне «завершити» на пʼятій репліці для порівняння
# лишало 5 859. Причина в тому, що віче йде хвилину-півтори, а не чотири, як гадалось, коли тут
# писалось «чверть віча»: пільга — це не чверть розмови, а більша її половина. Число лишається
# 45 с свідомо (F5 коштував би гостю всієї розмови, а закрита вкладка й перезавантаження звідси
# нерозрізненні), але ціна названа: подешевшати воно може лише коротшою пільгою.
#
# ★ І ОСЬ ЧОМУ РОЗУМНІШОГО ЧИСЛА ТУТ НЕМАЄ — три перевірені способи його здобути, усі відкинуті:
#
#   • «хай фронт скаже, що вкладку закрито». Браузер такого сигналу не має: `pagehide`,
#     `beforeunload` і `visibilitychange` спрацьовують на ПЕРЕЗАВАНТАЖЕННІ так само, як на
#     закритті, тож маячок звідти гасив би віче на кожному F5 — рівно те, від чого пільга й
#     стереже. Команда `finish` для цього вже є і нічого нового не потребує;
#   • «коротша пільга тому, хто жодного разу не був у потоці довше за секунду». Такий гість не має
#     віча: щоб кинути тему, треба стояти на сторінці, а стояння на сторінці і є довгий потік.
#     Правило вмикалось би для роботів, у яких і так нічого спиняти, тобто віддавало б нуль;
#   • «менше 45 с». Саме перезавантаження цю пільгу майже не витрачає: нове зʼєднання встигає
#     стати на облік РАНІШЕ, ніж помирає старе (`watching` рахує зʼєднання, а не людей), тож при
#     F5 запис у `_left` найчастіше не заводиться взагалі. Пільга лишається запасом на повільне
#     перезавантаження — і саме запас тут різати нема за чим.
#
# Заміряне дешевшання лежить не в цьому числі, а поруч: наглядач тепер знімає ще й ТЕМУ, яка
# лежить у черзі неорендованою (`_hush_abandoned` → `finish` → `_drop_pending`). Пільга віддає
# хвіст розмови — 4 826 токенів, — а знята тема віддає цілий прогін, медіана якого на 121
# записаному прод-айтемі 19 093 токени.
#
# Обидва боки перевірені наживо 2026-08-28 (локальне ядро, умова `viche`, справжній шлюз, чотири
# прогони на 44 270 токенів): повне віче — 26 109 токенів і 84.1 с на 21 репліку; те саме віче,
# спинене на третій секунді, — 1 456; віче, згорнуте ЦИМ наглядачем після пільги, — 14 855, тобто
# пільга віддала 11 254; а тема, яка лежала в черзі покинутої сесії, знялась за 66 с після обриву
# потоку й не витратила НІЧОГО. Ті 66 с — це 45 с пільги плюс два запізнення, обидва вже описані:
# обрив мовчазного потоку помічається аж на ударі серця (заміряно 10.2 с), а наглядач ходить раз
# на `WATCH_EVERY_S`.
ABANDON_S = 45.0
# Як часто наглядач питає, чи лишився хтось на тому боці. Дрібніше немає сенсу: пільга рахується
# десятками секунд.
WATCH_EVERY_S = 5.0


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


# Місце в мапі живих віч між орендою теми й появою оркестратора: сесія вже зайнята, агента ще нема.
_WAITING = object()
# ★ Місце в тій самій мапі між кінцем розмови й останньою її подією.
#
# Доти сесія звільнялась ОДРАЗУ після `orch.run`, а закриття (`task.outcome`, хроніка) летіло в
# шину вже після цього. Вільний робітник встигав узяти наступну тему того самого гостя й почати
# публікувати `run.started`, поки минулий прогін ще договорював, — і в потоці однієї сесії дві
# розмови перепліталися. Мітка тримає сесію зайнятою до останньої опублікованої події, але агентом
# не є: слово гостя в неї вже не потрапляє (`agent_for` віддає її, а `tell` на ній немає), тож
# «зараз віча немає» лишається чесною відповіддю.
_CLOSING = object()


class LiveRunner:
    """`make_orchestrator(trace, run_id)` віддає готовий оркестратор; черга дає задачі."""

    def __init__(self, bus: EventBus, queue, make_orchestrator: Callable,
                 *, governor: Governor | None = None, scene: dict | None = None,
                 worker: str = "ploshcha", paused: bool = True,
                 estimate_tokens: int = 2000, cast: list[dict] | None = None,
                 decisions=None, rumours=None, sessions=None, latency=None,
                 sweep_every_s: float = SWEEP_EVERY_S, workers: int = 1,
                 max_workers: int | None = None, abandon_s: float = ABANDON_S):
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
        # ★ Скільки віч ідуть ОДНОЧАСНО.
        #
        # Один робітник тримав усе село в один рядок: прогін ≈2 хвилини (шлюз, не процесор —
        # завантаження сервера 0.07), тож восьмеро гостей означали чергу з десяти тем і двадцять
        # хвилин очікування під написом «Село думу думає». Робота тут — чекання на мережу, тому
        # паралельні потоки не б'ються за процесор; черга вже безпечна для них (WAL і
        # `BEGIN IMMEDIATE` в оренді).
        self.workers = max(1, int(workers))
        # ★ Стеля робітників — і зростаємо до неї САМІ, за чергою.
        #
        # Фіксоване число завжди або замало (черга з десяти тем і двадцять хвилин очікування), або
        # забагато (шість голодних потоків на порожньому селі). Робота тут — чекання на шлюз, тож
        # ціна зайвого потоку майже нульова, а ціна браку — людина дивиться на «Село думу думає».
        self.max_workers = max(self.workers, int(max_workers if max_workers is not None
                                                 else self.workers * 4))
        self.estimate_tokens = estimate_tokens
        self._paused = threading.Event()
        if paused:
            self._paused.set()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        # Оренда теми — під власним замком: два вільні робітники не мають узяти дві теми одного гостя.
        self._lease_lock = threading.Lock()
        self.state = "paused" if paused else "running"
        self.tick = 0
        self.last_error: str | None = None
        self.stopped_reason: str | None = None
        self.runs_done = 0
        # Скільки покинутих аренд повернуто в чергу на старті. Нуль — теж відповідь, тому число
        # видиме завжди, а не лише коли щось сталось.
        self.recovered = 0
        # Хто саме зараз говорить: `sid` гостя → його оркестратор. Мапа, а не одне поле, бо віч
        # тепер кілька водночас — і слово гостя мусить потрапити в СВОЄ, а не в те, що почалось
        # останнім. Ключ `""` — прогін без сесії (CLI, соак).
        self._active: dict[str, object] = {}
        self.sessions_swept = 0
        # Хто СЛУХАЄ: `sid` → скільки відкритих потоків. Рахуємо саме зʼєднання, а не людей: одна
        # вкладка може перепідключитись раніше, ніж стара помітить обрив, і тоді нуля не буває.
        self._listeners: dict[str, int] = {}
        # Коли останній слухач цієї сесії відпав. Сесія, якої тут немає, слухачів не мала ніколи —
        # це CLI, соак і тести, і їх пільга не стосується. Запис живе рівно до пільги: далі його
        # викреслює `_hush_abandoned`, інакше набір ріс би на сто байтів з кожним гостем, який
        # бодай раз відкрив потік, і жодного разу не меншав.
        self._left: dict[str, float] = {}
        self.abandon_s = float(abandon_s)
        # Скільки віч згорнуто без слухача — разом із темами, які встигли лише полежати в черзі.
        # Нуль — теж відповідь, тому число видиме завжди.
        self.hushed = 0
        # Про кого попросили тиші, поки віче ще не почалось (тема орендована, оркестратора нема).
        # Без цього прохання губилось саме в тому вікні, де гість найчастіше й тисне: одразу
        # після того, як кинув тему.
        self._hushing: set[str] = set()

        self._last_sweep = 0.0
        # Робітник, що гасне через простій, не має рахуватись за смерть циклу.
        self._retiring = False

    # ── керування ────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._threads:
            return
        # Перед першим лізом підбираємо те, що лишилось від УБИТОГО процесу. Без цього кожен
        # SIGKILL списував по темі: у базі вона лежала `leased`, а виглядало це як «Дошка
        # приймає теми, а віче не починається».
        self.recovered += self._recover_stale()
        # (а) Прибирання на СТАРТІ. Процес міг простояти місяць — і тоді періодичний обхід
        # усередині циклу вперше спрацював би лише через пʼять хвилин після підняття порту.
        self._sweep()
        for _ in range(self.workers):
            self._spawn()
        if self.max_workers > self.workers:
            sup = threading.Thread(target=self._supervise, name="live-supervisor", daemon=True)
            sup.start()
        # Наглядач за покинутими вічами — окремим потоком, бо робітники стоять у виклику моделі
        # хвилинами й спитати їх ні про що.
        threading.Thread(target=self._watch, name="live-watch", daemon=True).start()

    def _lease(self, worker: str):
        """Беремо тему — але не ту, чия сесія вже говорить.

        Резервуємо `sid` ТУТ-таки, під тим самим замком: інакше два вільні робітники, побачивши
        порожньо, візьмуть дві теми одного гостя одночасно — рівно те, що ми й лікуємо.
        """
        if self.queue is None:
            return None
        with self._lease_lock:
            with self._lock:
                busy = tuple(sid for sid in self._active if sid)
            try:
                item = self.queue.lease(worker, busy) if busy else self.queue.lease(worker)
            except TypeError:
                # Черга без підтримки фільтра (фейк у тестах, стара реалізація) — беремо як є.
                item = self.queue.lease(worker)
            if item is not None:
                sid = clean_sid((item.payload or {}).get("sid"))
                if sid:
                    with self._lock:
                        self._active.setdefault(sid, _WAITING)
            return item

    def _spawn(self) -> None:
        """Ще один робітник. Імʼя унікальне, бо воно ж іде в оренду черги."""
        with self._lock:
            n = self._spawned = getattr(self, "_spawned", 0) + 1
            name = self.worker if (self.max_workers == 1) else f"{self.worker}-{n}"
            t = threading.Thread(target=self._loop, args=(name,), name=f"live-runner-{n}",
                                 daemon=True)
            self._threads = [x for x in self._threads if x.is_alive()] + [t]
        t.start()

    def _supervise(self) -> None:
        """Додає робітників, поки черга не порожня. Зайві гаснуть самі (див. `_loop`)."""
        while not self._stop.is_set():
            time.sleep(SUPERVISE_EVERY_S)
            if self._paused.is_set() or self.queue is None:
                continue
            try:
                pending = int(self.queue.stats().get("pending", 0))
            except Exception:
                continue
            alive = self._alive_workers()
            want = min(self.max_workers, alive + pending)
            for _ in range(max(0, want - alive)):
                self._spawn()

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
        for t in self._threads:
            t.join(timeout)

    # ── доступ до живих віч ──────────────────────────────────────────────────
    def agent_for(self, sid: str | None):
        """Оркестратор ЦЬОГО гостя, якщо його віче зараз іде.

        Прогін без сесії чутний усім, тому гість без `sid` дістає будь-яке, що йде.
        """
        with self._lock:
            if sid:
                return self._active.get(sid) or self._active.get("")
            return next(iter(self._active.values()), None)

    def finish(self, sid: str | None) -> bool:
        """Згорнути віче ЦІЄЇ сесії. Повертає, чи було що згортати.

        ★ Тільки своє. Ключ шукається ТОЧНО, без запасного варіанта на спільний прогін, — на
        відміну від `agent_for`, де запасний варіант законний: чути спільне віче може кожен, а
        спиняти чуже — ніхто. Гість без сесії (консоль, соак) так само дістає рівно свій прогін
        під ключем `""`, і на віче гостей не дотягується.

        Прогін гине не одразу: агент домовляє такт і закривається сам (`Viche.hush`). Робітник при
        цьому лишається живий — він доведе прогін до `task.outcome`, поверне айтем черзі й візьме
        наступний. Убити потік означало б лишити тему навіки в аренді.

        ★ І та сама команда знімає тему, яка ще ЛЕЖИТЬ У ЧЕРЗІ.

        Доти «завершити» доходило рівно до прогону, вже орендованого робітником, а неорендована
        тема того самого гостя спокійно дочікувалась свого. Вікно тут не теоретичне: наглядач
        дивиться на чергу раз на `SUPERVISE_EVERY_S` = 2 с і добирає робітника саме за наявністю
        `pending`. Тобто гість натискав «завершити» — і за дві секунди село починало гомоніти про
        тему, від якої він щойно відмовився, за повну ціну прогону (медіана прод-прогону 19 093
        токени на 121 записаному айтемі).

        Порядок двох дій навмисний: спершу черга, потім жива розмова. У зворотному лишається дірка
        рівно в один ліз — робітник устигне орендувати тему між нашим поглядом на `_active` і
        нашим `DELETE`, і вона піде говорити вже після «завершити». Так само орендована мить тому
        тема потрапляє в `_active` як `_WAITING` і дістає прохання про тишу, тобто обидва шляхи
        закриті.
        """
        key = clean_sid(sid) or ""
        dropped = self._drop_pending(key)
        with self._lock:
            agent = self._active.get(key)
            if agent is None:
                return dropped > 0
            if agent is _WAITING:
                # Тема вже орендована, оркестратор ще будується. Прохання чекає на нього в
                # `_work`: інакше воно згинуло б рівно у вікні між орендою і першим словом.
                self._hushing.add(key)
                return True
        hush = getattr(agent, "hush", None)
        if hush is None:
            return dropped > 0
        hush()
        return True

    def _drop_pending(self, key: str) -> int:
        """Теми ЦІЄЇ сесії, які ще лежать у черзі неорендованими. Повертає, скільки знято.

        Спільного кошика (`key == ""`) це не стосується навмисно: туди падають теми без сесії —
        консоль, соак, старий клієнт, — і вони не належать нікому окремо. Одна команда з `curl`
        стирала б там чужу роботу, тоді як своє віче вона однаково спиняє.

        Черга може не вміти скасовувати (фейк у тестах, стара реалізація) — тоді це просто нуль, а
        не `AttributeError` посеред команди гостя. Збій самої черги теж не має валити команду:
        причина лягає в `lastError`, тобто в `/health`, а не в нікуди.
        """
        if not key or self.queue is None:
            return 0
        cancel = getattr(self.queue, "cancel_pending", None)
        if cancel is None:
            return 0
        try:
            return int(cancel(sid=key) or 0)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return 0

    # ── хто слухає ───────────────────────────────────────────────────────────
    @contextmanager
    def watching(self, sid: str | None):
        """Один відкритий потік подій. Вихід із блоку = глядач відпав, хай там що його зняло.

        Лічильник, а не прапорець: перезавантажена сторінка встигає відкрити НОВЕ зʼєднання
        раніше, ніж помирає старе, і на прапорці це виглядало б як «пішов, потім прийшов».
        """
        key = clean_sid(sid)
        if not key:
            # Інспектор без `sid` бачить усе й нічиїм слухачем не є: рахувати його означало б
            # тримати живим будь-яке покинуте віче, поки відкритий один службовий потік.
            yield
            return
        with self._lock:
            self._listeners[key] = self._listeners.get(key, 0) + 1
            self._left.pop(key, None)
        try:
            yield
        finally:
            with self._lock:
                left = self._listeners.get(key, 1) - 1
                if left > 0:
                    self._listeners[key] = left
                else:
                    self._listeners.pop(key, None)
                    self._left[key] = time.time()

    def _hush_abandoned(self) -> int:
        """Віче, яке нікому слухати довше за пільгу, згортається саме.

        Причина в числах — у `Viche.hush`: покинутий прогін догомонює 18 902 токени з 24 761,
        тобто 76% віча. Пільга (`ABANDON_S`) існує тому, що обрив потоку і закрита вкладка з цього боку
        нерозрізненні; чому саме 45 с — там-таки, біля константи.

        ★ Разом із розмовою знімається й ТЕМА, ЩО ЩЕ ЛЕЖИТЬ У ЧЕРЗІ (`finish` → `_drop_pending`), і
        саме тут це важить найбільше. Пільга віддає лише хвіст розмови — заміряно 4 826 токенів із
        24 761, — а покинута тема, яка ще не почалась, віддає ЦІЛИЙ прогін: медіана 19 093 токени
        на 121 записаному прод-айтемі. Тобто найдорожче в цьому наглядачі не те, що він спиняє, а
        те, чого він не дає початись.

        ★ Запис про того, хто пішов, тут-таки й ВМИРАЄ, і не лише коли було що спиняти.

        Доти зі списку викреслювались тільки сесії, чиє віче зараз ішло (`sid in self._active`), а
        решта лишалась у `_left` назавжди — приблизно сто байтів на кожного гостя, який бодай раз
        відкрив потік і закрив вкладку. На довгограючому проді це витік, тим прикріший, що росте
        він рівно з відвідуваністю. Після пільги запис однаково вже нічого не тримає: розмову
        згорнуто, тему з черги знято, а нове зʼєднання завело б свій запис із нуля (`watching`).
        """
        now = time.time()
        with self._lock:
            # Межа включна: пільга в нуль секунд мусить означати нуль. Із суворим «більше»
            # рішення впиралось у зернистість годинника — два виклики `time.time()` підряд тут
            # дають те саме число, — тож `abandon_s = 0.0` спрацьовував через раз. На 45 с це не
            # міняє нічого, а на нулі — міняє все.
            gone = [sid for sid, left in self._left.items()
                    if now - left >= self.abandon_s and not self._listeners.get(sid)]
        stopped = 0
        for sid in gone:
            # Прохання одне на прогін: далі сесія лишається без слухача, і повторювати його
            # щопʼять секунд означало б лічильник, який рахує наглядача, а не покинуті віча.
            with self._lock:
                self._left.pop(sid, None)
            if self.finish(sid):
                stopped += 1
        self.hushed += stopped
        return stopped

    def _watch(self) -> None:
        while not self._stop.is_set():
            time.sleep(WATCH_EVERY_S)
            try:
                self._hush_abandoned()
            except Exception as exc:
                # Наглядач не має права завалити ядро: причина лягає в `/health`, а не в нікуди.
                self.last_error = f"{type(exc).__name__}: {exc}"

    @property
    def current(self):
        """Сумісність зі старим полем: будь-яке віче, що зараз іде."""
        with self._lock:
            return next(iter(self._active.values()), None)

    @property
    def current_sid(self) -> str | None:
        with self._lock:
            for key in self._active:
                if key:
                    return key
            return None

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
            "alive": any(t.is_alive() for t in self._threads),
            "stoppedReason": reason,
            "lastError": err,
            "tick": self.tick,
            "runsDone": self.runs_done,
            "workers": {"alive": self._alive_workers(), "busy": len(self._active),
                        "min": self.workers, "max": self.max_workers},
            "recovered": self.recovered,
            # Скільки віч згорнуто без слухача і скільки потоків відкрито зараз. Мовчазна економія
            # нічим не відрізняється від мовчазної поламки, поки її не видно з `/health`.
            "hushed": self.hushed,
            "listeners": sum(self._listeners.values()),
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
        with self._lock:
            if now - self._last_sweep < self.sweep_every_s:
                return 0
            self._last_sweep = now
            # Тримаємо бази ВСІХ, чиї віча зараз ідуть, а не лише одного: з кількома робітниками
            # «поточний» більше не один, і прибиральник міг би видалити базу з-під сусіднього віча.
            keep = {sid for sid in self._active if sid}
        try:
            gone = int(self.sessions.sweep(keep=keep or None))
        except Exception as exc:
            # Тека може бути недоступна — але це не привід зупиняти село. Причина лягає в
            # `/health`, а не в нікуди.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return 0
        self.sessions_swept += gone
        return gone

    def _loop(self, worker: str | None = None) -> None:
        idle_since = time.time()
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
                    item = self._lease(worker or self.worker)
                    if item is None:
                        # Зайвий робітник гасне сам: тримати десяток порожніх потоків після
                        # напливу немає сенсу, а базовий склад лишається завжди.
                        if (time.time() - idle_since > WORKER_IDLE_EXIT_S
                                and self._alive_workers() > self.workers):
                            self._retiring = True   # гасне добровільно, це не поламка
                            return
                        self._sweep()
                        time.sleep(IDLE_SLEEP_S)
                        continue
                    idle_since = time.time()
                    self._run_one(item)
                    idle_since = time.time()
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._print_exc()
                    proj = StreamProjector(f"loop-{uuid.uuid4().hex[:8]}", _now())
                    # Падіння самого циклу — теж стан ядра, тому теж усім.
                    self.bus.publish(proj._envelope("run.error",
                                                    {"message": self.last_error}, self.tick))
                    self._degrade(f"цикл упав: {self.last_error}", stage="loop")
        finally:
            # Смерть ОДНОГО робітника з кількох — ще не смерть села: решта тягне чергу далі.
            # Оголошуємо зупинку лише коли живих не лишилось.
            # Смерть ОДНОГО робітника з кількох — ще не смерть села, але останній, хто гасне
            # НЕ добровільно, мусить сказати це вголос.
            if not self._stop.is_set() and self._alive_workers() <= 1 and not self._retiring:
                self._died()

    def _alive_workers(self) -> int:
        return sum(1 for t in self._threads if t.is_alive())

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
            # ★ Напис збираємо з ТЕМИ, а не з того, що лежить у `label`.
            #
            # У базі є ухвали, записані ще старим кодом: там у назві службова лічба разом з
            # огризком від літописця («відхилили: за 2, проти 4, утримались 0 · Все страш…»). Вони
            # спливають на Дошці щоразу, коли починається нове віче, бо чинні ухвали відновлюються.
            # Тема ж лежить поруч, у тому самому рядку, і з неї напис виходить читабельний завжди.
            head = "відхилили" if str(d.get("label", "")).startswith("відхилили") else "ухвалили"
            topic = " ".join(str(d.get("topic") or "").split())
            label = f"{head}: {topic}"[:120] if topic else str(d["label"])[:120]
            out.append(proj._envelope("event.happened", {"event": {
                "id": f"standing-{d['who']}", "kind": "decision", "label": label,
                "description": "чинна ухвала минулого віча",
                "place": {"poi": d["poi"]}, "involves": [d["who"]]}}, 0))
            out += proj._walk(d["who"], d["poi"], 0)
        return out

    @staticmethod
    def _own(sid: str | None) -> str:
        """★ Чия це розмова для ШИНИ.

        Мітка `None` означає «спільне», і саме її дістають прогони без сесії — з консолі, з соаку,
        з `curl`. Такий прогін бачили ВСІ гості одночасно: чужі теми лізли кожному на Дошку, хоч
        він їх не кидав. Спільними лишаються тільки події стану ядра (стеля витрат, смерть
        робітника) — вони справді стосуються всіх. Прогін без сесії дістає власну мітку `cli`:
        інспектор без `sid` усе одно бачить усе (правило шини), а гість — лише своє.
        """
        return sid or "cli"

    def _run_one(self, item) -> None:
        # ★ Сховища беруться ПІД АЙТЕМ, а не раз на старті: `sid` лежить у самій задачі, тож те
        # саме ядро й той самий цикл ведуть різні села — по одному на гостя.
        sid = clean_sid((item.payload or {}).get("sid"))
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
            with self._lock:
                self._active.pop(sid or "", None)
                # ★ Прохання про тишу вмирає РАЗОМ із прогоном, якого воно стосувалось.
                #
                # Знімає його `_work` — рядком, до якого прогін може й не доїхати: чинні ухвали й
                # побудова агента ходять у SQLite, і «база зайнята» описана тут-таки, вище, як
                # звичайна річ. Тоді `sid` лишався в наборі назавжди, і НАСТУПНЕ віче цієї сесії
                # гинуло на першому такті, скільки б тем гість не кидав, — тобто збій одного
                # прогону мовчки забирав у гостя всі дальші.
                self._hushing.discard(sid or "")

    def _work(self, item, proj: StreamProjector, run_id: str, sid: str | None = None,
              session=None) -> None:
        task = str((item.payload or {}).get("task") or item.key)
        decisions = self.decisions if session is None else session.decisions
        rumours = self.rumours if session is None else session.rumours
        make = self.make_orchestrator if session is None else session.make_agent
        self.bus.publish(proj.start(), self._own(sid))
        # ★ Чинні ухвали повертаються на сцену ДО розмови: інакше «поставили сторожа» жило б рівно
        # один прогін, і рішення знову не мало б наслідку.
        self.bus.publish(self._restore_decisions(proj, decisions), self._own(sid))
        trace = BusTrace(self.bus, proj, self._own(sid))
        orch = make(trace, run_id, (item.payload or {}).get("place"))
        with self._lock:
            self._active[sid or ""] = orch
            asked = (sid or "") in self._hushing
            self._hushing.discard(sid or "")
        # Прохання, що прийшло, поки віче ще будувалось, доганяє його тут: інакше гість, який
        # передумав одразу після теми, платив би за цілу розмову.
        if asked and hasattr(orch, "hush"):
            orch.hush()
        template = getattr(orch, "budget_template", None)
        # ★ Сід — від САМОГО ПРОГОНУ, а не константа.
        #
        # Заміряно на живій сесії: гість сім разів кинув «Хто я» за дві хвилини й отримав сім
        # прогонів по 21 771 токена з байт-у-байт однаковою хронікою — бо сід стояв одиницею, склад
        # людей є функцією теми, а шлюз на однакових входах детермінований. Для гостя це «зламалось»,
        # для гаманця — 152 тисячі токенів за одну розмову. `run_id` унікальний на кожну постановку
        # теми, тож із нього виходить і відтворюваність (той самий прогін відтворюється за ним), і
        # різність (нова спроба — інша розмова).
        result = orch.run(task, seed=zlib.crc32(run_id.encode()) & 0x7FFFFFFF,
                          budget=template.model_copy(deep=True) if template else None)
        # Розмови вже немає, але події її ще нема в шині. Сесія лишається зайнятою до останньої з
        # них (`_CLOSING`), інакше наступна тема того самого гостя починається впереміш із цією.
        with self._lock:
            self._active[sid or ""] = _CLOSING
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
        # ★ Згорнуте віче каже про себе ВГОЛОС, окремою подією.
        #
        # `task.outcome` несе `viche_hushed` в інцидентах, але інциденти — це журнал прогону, а не
        # звістка глядачеві: сцена їх не читає. Без явної події «розмова скінчилась, бо ти попросив»
        # виглядало б як обрив — та сама давня поламка, коли механізм спрацював, а шлях
        # спостереження мовчить. Мітка сесії тут обовʼязкова: це чиясь розмова, а не стан ядра.
        if "viche_hushed" in (getattr(result, "incidents", None) or []):
            self.bus.publish(proj._envelope("run.degraded",
                                            {"stage": "viche", "reason": "віче завершено"},
                                            self.tick), self._own(sid))
        self.bus.publish(proj.close(result, done=True), self._own(sid))
        tokens = int(getattr(result, "tokens", 0) or 0) + int(getattr(result, "aux_tokens", 0) or 0)
        # Лічильники спільні для всіх робітників, тож рухаємо їх під замком.
        with self._lock:
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
        try:
            self.bus.publish(proj.close(_Crash(self.last_error), done=False), self._own(sid))
        except Exception:
            pass
        self.bus.publish(proj._envelope("run.error", {"message": self.last_error}, self.tick), self._own(sid))
        if self.queue is not None:
            try:
                self.queue.fail(item.key, self.last_error)
            except Exception as inner:
                # Черга не прийняла провал — тоді айтем висить в аренді, і про це мусить бути
                # видно з `/health`, інакше «зникла тема» знову не має пояснення.
                self.last_error = f"{self.last_error} | черга не прийняла провал: {inner}"
