"""HTTP+SSE на stdlib: нуль нових залежностей.

Бінд ЛИШЕ на 127.0.0.1 і без прапорця для іншого інтерфейсу. Автентифікації тут немає, тому
виставляти це в мережу не можна: `POST /command` кладе роботу в чергу, тобто витрачає гроші.

Хендлери нічого не рахують — вони лише читають ring-буфер. Уся робота живе в окремому потоці
(`LiveRunner`), інакше довге зʼєднання SSE блокувало б цикл.
"""

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
HEARTBEAT_S = 15.0
MAX_BODY = 64 * 1024
INDEX = "index.html"


def make_handler(bus, runner, static: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _cors(self) -> None:
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
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

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
            self.wfile.write(body)

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/command":
                return self._json(404, {"error": "не знайдено"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._json(413, {"error": "завелике тіло"})
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:
                return self._json(400, {"error": f"не JSON: {exc}"})
            return self._json(*handle_command(payload, runner))

        def _stream(self):
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            since = self.headers.get("Last-Event-ID")
            for part in query.split("&"):
                if part.startswith("since="):
                    since = part[len("since="):]
            cursor = bus.tail_cursor()
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
                    events, cursor = bus.wait(cursor, HEARTBEAT_S)
                    if not events:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        if bus.closed:
                            return
                        continue
                    for i, ev in enumerate(events):
                        line = json.dumps(ev, ensure_ascii=False)
                        chunk = f"id: {cursor - len(events) + i + 1}\ndata: {line}\n\n"
                        self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def handle_command(payload_in: dict, runner) -> tuple[int, dict]:
    payload = payload_in
    kind = str(payload.get("kind") or "").strip()
    if kind == "pause":
        runner.pause()
        return 200, {"ok": True, "state": runner.state}
    if kind == "resume":
        runner.resume()
        return 200, {"ok": True, "state": runner.state,
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
        return 200, {"ok": True, "requeued": moved, "queue": runner.queue.stats()}
    if kind in ("say", "whisper"):
        # Слово в ЖИВЕ віче. Якщо саме зараз ніхто не гомонить, чесно кажемо це, а не мовчимо в
        # порожнечу: інакше «я написав, і нічого» знову виглядало б як поламка.
        agent = getattr(runner, "current", None)
        if agent is None or not hasattr(agent, "tell"):
            return 409, {"error": "зараз віча немає — кинь тему на Дошку"}
        text = str(payload.get("text") or "").strip()
        if not text:
            return 400, {"error": "порожнє слово"}
        agent.tell({"kind": kind, "text": text,
                    **({"to": str(payload["to"])} if payload.get("to") else {})})
        return 200, {"ok": True, "kind": kind}
    if kind == "topic":
        text = str(payload.get("text") or "").strip()
        if not text:
            return 400, {"error": "порожня тема"}
        key = payload.get("key") or f"topic-{abs(hash(text)) % 10**10}"
        # Місце їде РАЗОМ із темою: розмова в шинку й розмова на площі — різні процеси, тож
        # місце має бути частиною задачі, а не станом сервера.
        payload = {"task": text, "source": "board"}
        if payload_place := str(payload_in.get("place") or "").strip():
            payload["place"] = payload_place
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


def serve(bus, runner, port: int = 8765, static: Path | None = None) -> ThreadingHTTPServer:
    httpd = QuietServer((HOST, port), make_handler(bus, runner, static))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="live-http", daemon=True).start()
    return httpd
