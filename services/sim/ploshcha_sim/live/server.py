"""HTTP+SSE на stdlib: нуль нових залежностей.

Бінд ЛИШЕ на 127.0.0.1 і без прапорця для іншого інтерфейсу. Назовні це дивиться через зворотний
проксі, який тримає TLS і пароль — саме тому тут немає власної автентифікації й немає способу
випадково відкрити порт у світ.

`POST /command` витрачає гроші, тому в самому ядрі стоїть ДРУГИЙ замок, незалежний від проксі:
частота запитів на адресу. Пароль можуть передати далі, проксі можуть налаштувати криво — а стеля
викликів лишається на місці.

Хендлери нічого не рахують — вони лише читають ring-буфер. Уся робота живе в окремому потоці
(`LiveRunner`), інакше довге зʼєднання SSE блокувало б цикл.
"""

import json
import mimetypes
import pathlib
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .sessions import clean_sid

HOST = "127.0.0.1"
HEARTBEAT_S = 15.0
MAX_BODY = 64 * 1024
INDEX = "index.html"
# Скільки команд з однієї адреси за вікно. Тема коштує тисячі токенів, тож це не про навантаження
# на сервер, а про гаманець.
RATE_WINDOW_S = 60.0
RATE_MAX = 6
# Друга стеля, іншої природи: скільки НОВИХ сесій дозволено завести з однієї адреси. Стеля команд
# каже, як часто можна просити; ця — скільки слідів на диску можна лишити. Без неї цикл із новим
# `sid` на кожен запит наплодив би сотні файлів, не перевищивши жодної старої межі.
SESSION_WINDOW_S = 3600.0
SESSION_MAX = 12


class RateGate:
    """Стеля команд на адресу. Без неї один цикл `curl` вичерпує денну стелю токенів за хвилину."""

    def __init__(self, window: float = RATE_WINDOW_S, limit: int = RATE_MAX):
        self.window = window
        self.limit = limit
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, who: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            hits = [t for t in self._hits.get(who, []) if now - t < self.window]
            if len(hits) >= self.limit:
                self._hits[who] = hits
                return False, int(self.window - (now - hits[0])) + 1
            hits.append(now)
            self._hits[who] = hits
            if len(self._hits) > 4096:  # не даємо мапі рости від сканерів
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < self.window}
            return True, 0


def allow_new_session(gate: RateGate, sessions, sid: str | None, who: str) -> tuple[bool, int]:
    """Чи можна цій адресі завести ЩЕ ОДНУ сесію. Відома сесія нічого не витрачає.

    Окрема функція, а не гілка в хендлері, бо це єдине правило, яке коштує диска, — і його треба
    перевіряти тестом, не піднімаючи HTTP.
    """
    if not sid or sessions is None or sessions.known(sid):
        return True, 0
    ok, wait = gate.allow(who)
    if ok:
        # Застовплюємо файлом ОДРАЗУ: інакше гість, який лише клацає команди, лишався б для
        # стелі кількості невидимим, і вона тримала б менше, ніж обіцяє.
        sessions.ensure(sid)
    return ok, wait


def make_handler(bus, runner, static: Path | None = None, origins: tuple[str, ...] = (),
                 feedback: Path | None = None):
    gate = RateGate()
    # Куди лягають скарги гостей. Поруч із базою, бо це стан цього ж села, а не глобальний журнал.
    feedback_path = Path(feedback) if feedback else Path("data/ploshcha/skargy.jsonl")
    new_sessions = RateGate(window=SESSION_WINDOW_S, limit=SESSION_MAX)
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _cors(self) -> None:
            # ★ `*` і пароль несумісні: з `credentials` браузер відкидає зірочку, і вітрина на
            # чужому домені мовчки лишалась би без потоку. Тому віддзеркалюємо ДОЗВОЛЕНЕ джерело.
            origin = self.headers.get("Origin") or ""
            if origins and origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Credentials", "true")
                self.send_header("Vary", "Origin")
            elif not origins:
                self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Last-Event-ID")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self._body(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):
            """Те саме, що GET, але без тіла.

            Базовий обробник відповідав 501, і будь-який зовнішній монітор доступності бачив
            ПЛОЩУ як зламану, хоч вона працювала: перевірки життя роблять саме HEAD.
            """
            self._head_only = True
            try:
                self.do_GET()
            finally:
                self._head_only = False

        def _body(self, payload: bytes) -> None:
            if not getattr(self, "_head_only", False):
                self.wfile.write(payload)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/health":
                return self._json(200, runner.health())
            if path == "/stream":
                return self._stream()
            if static is not None:
                return self._static(path)
            return self._json(404, {"error": "не знайдено"})

        def _static(self, path: str) -> None:
            """Ядро роздає збірку фронта саме.

            Доти фронт жив на девелоперському Vite, тобто прод вимагав ДВА процеси й дозволи між
            ними. Vite до того ж падав, і кожне падіння виглядало як «ПЛОЩА зламалась». Один процес
            прибирає і те, і CORS як клас.
            """
            rel = path.lstrip("/") or INDEX
            target = (static / rel).resolve()
            # Обхід дерева вгору неможливий: усе поза коренем збірки — 404, а не читання диска.
            if not str(target).startswith(str(static.resolve())) or target.is_dir():
                target = static / INDEX
            # ★ Файл, якого немає, — це 404, а НЕ index.html.
            #
            # Запасний шлях на index потрібен лише маршрутам застосунку. Коли він ловив і шляхи з
            # розширенням, застарілий `/assets/index-СТАРИЙХЕШ.js` віддавав HTML із кодом 200 —
            # браузер отримував `text/html` замість скрипта, мовчки його не виконував, і сторінка
            # лишалась білою. Кожна перезбірка міняє хеш, тож на будь-якому кешованому index.html
            # гість отримував саме це.
            if not target.is_file() and pathlib.PurePosixPath(rel).suffix:
                return self._json(404, {"error": "немає такого файлу"})
            if not target.is_file():
                target = static / INDEX
            if not target.is_file():
                return self._json(404, {"error": "збірки немає; зроби pnpm build"})
            body = target.read_bytes()
            kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", kind + ("; charset=utf-8" if "text" in kind
                                                     or "javascript" in kind or "json" in kind
                                                     else ""))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self._body(body)

        def _feedback(self, payload: dict, who: str) -> None:
            """Скарга гостя лягає у файл поруч із базою.

            Окремий шлях, а не `/command`: скарга нічого не запускає й не витрачає ані токена, тож
            і стеля команд її не має різати. Пишемо рядком JSON — читати можна `tail`, і жоден збій
            запису не має права завалити відповідь гостеві.
            """
            text = str(payload.get("text") or "").strip()[:2000]
            if not text:
                raise ValueError("порожня скарга")
            row = {"коли": time.strftime("%Y-%m-%d %H:%M:%S"), "текст": text,
                   "сесія": clean_sid(payload.get("sid")),
                   "звідки": str(payload.get("where") or "")[:80],
                   "адреса": who,
                   "браузер": (self.headers.get("User-Agent") or "")[:200]}
            path = feedback_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        def do_POST(self):
            if self.path.split("?", 1)[0] not in ("/command", "/feedback"):
                return self._json(404, {"error": "не знайдено"})
            who = (self.headers.get("X-Forwarded-For") or self.client_address[0] or "?").split(",")[0].strip()
            ok, wait = gate.allow(who)
            if not ok:
                return self._json(429, {"error": f"занадто часто — спробуй за {wait} с"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._json(413, {"error": "завелике тіло"})
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:
                return self._json(400, {"error": f"не JSON: {exc}"})
            if not isinstance(payload, dict):
                return self._json(400, {"error": "тіло має бути обʼєктом"})
            sessions = getattr(runner, "sessions", None)
            allowed, wait = allow_new_session(new_sessions, sessions,
                                              clean_sid(payload.get("sid")), who)
            if not allowed:
                return self._json(429, {"error": f"забагато нових сесій — спробуй за {wait} с"})
            if self.path.split("?", 1)[0] == "/feedback":
                try:
                    self._feedback(payload, who)
                except ValueError as exc:
                    return self._json(400, {"error": str(exc)})
                except Exception as exc:
                    traceback.print_exc()
                    return self._json(500, {"error": f"скарга не записалась: {exc}"})
                return self._json(200, {"ok": True})
            # ★ Межа помилки. Доти будь-який виняток у розборі команди летів у `handle_one_request`,
            # і клієнт не діставав ВЗАГАЛІ НІЧОГО — обрив зʼєднання замість відповіді. Саме так
            # виглядає «ядро не відповідає»: воно живе, просто мовчить у відповідь на команду.
            try:
                code, body = handle_command(payload, runner)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                # Причина лягає і в `/health`: інспектор мусить знати про це без доступу до логів.
                runner.last_error = message
                return self._json(500, {"error": f"команда впала: {message}"})
            return self._json(code, body)

        def _stream(self):
            query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            since = (query.get("since") or [self.headers.get("Last-Event-ID")])[0]
            sid = clean_sid((query.get("sid") or [None])[0])
            sessions = getattr(runner, "sessions", None)
            # Перегляд — теж дотик. Гість, який лише дивиться на своє село, не має втратити його
            # через тиждень тільки тому, що не кинув жодної теми.
            if sid and sessions is not None and sessions.known(sid):
                sessions.touch(sid)
            cursor = bus.tail_cursor()
            # Звідси глядач слухає наживо; усе раніше — історія, і з неї йому належить лише своє.
            shared_from = cursor
            if since:
                try:
                    cursor = max(0, int(since))
                except ValueError:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self._cors()
            self.end_headers()
            try:
                while True:
                    events, cursor = bus.wait_ids(cursor, HEARTBEAT_S, sid, shared_from)
                    if not events:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        if bus.closed:
                            return
                        continue
                    # `id` — АБСОЛЮТНА позиція події, бо після фільтрації пачка коротша за крок
                    # курсора: рахувати її від довжини пачки означало б віддати клієнтові позицію
                    # з минулого, і реконект тягнув би чуже наново.
                    for index, ev in events:
                        line = json.dumps(ev, ensure_ascii=False)
                        chunk = f"id: {index + 1}\ndata: {line}\n\n"
                        self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def handle_command(payload_in: dict, runner) -> tuple[int, dict]:
    payload = payload_in
    kind = str(payload.get("kind") or "").strip()
    # Ідентифікатор гостя. Невалідний — це `None`, тобто «спільне село», а не відмова: старий
    # клієнт і CLI мусять і далі працювати без жодного `sid`.
    sid = clean_sid(payload.get("sid"))
    if kind == "pause":
        runner.pause()
        return 200, {"ok": True, "state": runner.state}
    if kind == "resume":
        runner.resume()
        # `ok` мусить казати, чи команда СПРАЦЮВАЛА. Після стелі `resume` тихо відмовляє, а
        # відповідь усе одно казала «ok», тож кнопка виглядала натиснутою й нічого не робила.
        started = runner.state == "running"
        return 200, {"ok": started, "state": runner.state,
                     **({"stoppedReason": runner.stopped_reason}
                        if runner.stopped_reason else {})}
    if kind == "stop":
        runner.stop()
        return 200, {"ok": True, "state": runner.state}
    if kind == "requeue":
        if runner.queue is None:
            return 400, {"error": "черги немає"}
        key = payload.get("key")
        moved = runner.queue.requeue_dead(str(key) if key else None)
        # Разом із мертвими підбираємо покинуті аренди: без цього «requeue» не рятував саме той
        # випадок, задля якого його кличуть, — тему, що зависла в `leased` після вбитого процесу.
        recovered = runner._recover_stale() if key is None else 0
        return 200, {"ok": True, "requeued": moved, "recovered": recovered,
                     "queue": runner.queue.stats()}
    if kind in ("say", "whisper"):
        # Слово в ЖИВЕ віче. Якщо саме зараз ніхто не гомонить, чесно кажемо це, а не мовчимо в
        # порожнечу: інакше «я написав, і нічого» знову виглядало б як поламка.
        agent = runner.agent_for(sid) if hasattr(runner, "agent_for") else getattr(runner, "current", None)
        # ★ І віче мусить бути СВОЄ. Гість не бачить чужого прогону в потоці, тож слово, кинуте в
        # нього, зникло б без сліду — а з боку іншого села прилетів би голос нізвідки.
        if agent is None or not hasattr(agent, "tell"):
            return 409, {"error": "зараз віча немає — кинь тему на Дошку"}
        text = str(payload.get("text") or "").strip()
        if not text:
            return 400, {"error": "порожнє слово"}
        agent.tell({"kind": kind, "text": text,
                    **({"to": str(payload["to"])} if payload.get("to") else {})})
        return 200, {"ok": True, "kind": kind}
    if kind == "topic":
        # Черги може не бути (збірка без неї) — тоді це відмова з причиною, як у `requeue`, а не
        # `AttributeError` глибше по коду: він летів повз обробник і обривав зʼєднання
        # без жодної відповіді.
        if runner.queue is None:
            return 400, {"error": "черги немає"}
        text = str(payload.get("text") or "").strip()
        if not text:
            return 400, {"error": "порожня тема"}
        key = payload.get("key") or f"topic-{abs(hash(text)) % 10**10}"
        # Місце їде РАЗОМ із темою: розмова в шинку й розмова на площі — різні процеси, тож
        # місце має бути частиною задачі, а не станом сервера.
        payload = {"task": text, "source": "board"}
        if payload_place := str(payload_in.get("place") or "").strip():
            payload["place"] = payload_place
        # `sid` їде В САМІЙ ЗАДАЧІ: цикл-виконавець один, тож інакше він не знав би, чиє село
        # міняти, поки тема лежить у черзі.
        if sid:
            payload["sid"] = sid
        fresh = runner.queue.put(str(key), payload)
        return 200, {"ok": True, "key": str(key), "fresh": bool(fresh),
                     "queue": runner.queue.stats()}
    return 400, {"error": f"невідома команда {kind!r}"}


class QuietServer(ThreadingHTTPServer):
    """Закрита вкладка спостерігача — не інцидент. Стек `ConnectionResetError` на кожне
    відʼєднання забивав лог і маскував справжні помилки циклу."""

    def handle_error(self, request, client_address):
        import sys

        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def serve(bus, runner, port: int = 8765, static: Path | None = None,
          origins: tuple[str, ...] = (), feedback: Path | None = None) -> ThreadingHTTPServer:
    """`origins` — звідки вітрині дозволено ходити по потік. Порожньо = лише свій же домен."""
    httpd = QuietServer((HOST, port), make_handler(bus, runner, static, origins, feedback))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="live-http", daemon=True).start()
    return httpd
