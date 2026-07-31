"""Інгест документів з uk.wikisource у чергу — ідемпотентно, з провенансом і ліцензією.

Дисципліна та сама, що в `fetch_lexis.py`: джерело, ліцензія й дата лежать у даних, а не в чиїйсь
памʼяті. Різниця в тому, що тут одиниця роботи — **документ**, і вона ставиться в чергу, тому інгест
можна перезапускати скільки завгодно: повторна постановка того самого ключа нічого не робить.

Ліцензія: тексти на Вікіджерелах — переважно **public domain** (автори до 1929), вікірозмітка й
редакторський внесок — **CC BY-SA 4.0**. Для релізу потрібна атрибуція; це записано в манифест.

Запуск:
  uv run python scripts/ingest_wikisource.py --search літопис --limit 20 --fetched 2026-07-31
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ploshcha_sim.adapters.queue_sqlite import SqliteQueue  # noqa: E402

API = "https://uk.wikisource.org/w/api.php"
AGENT = "ploshcha-research/0.1 (oleksandrrsavkov@gmail.com) corpus-ingest"
LICENSE = "тексти переважно PD; вікірозмітка CC BY-SA 4.0"
DELAY = 1.2
MIN_CHARS = 250
CHUNK_CHARS = 2000

DATA = Path(__file__).resolve().parents[1] / "evalkit" / "data"
FOOTER = re.compile(r"\n==\s*(Джерела|Примітки|Посилання)\s*==.*", re.S)

# Текст беремо через `action=parse`, бо він РОЗКРИВАЄ трансклюзію: на Вікіджерелах сторінка часто
# лише навігаційний шаблон, а зміст лежить у сканах (`Сторінка:…djvu/NNN`). Спроба чистити вікітекст
# самотужки давала 12 символів зі 458 — інгест «працював» і приносив порожнечу.
HTML_TAG = re.compile(r"<[^>]+>")
ENTITY = re.compile(r"&#?\w+;")
NAV = re.compile(r"^[◀▶\s]*|[◀▶]")
FOOTER = re.compile(r"(Джерела|Примітки|Посилання|Вікістаття|Вікідані)\b.*", re.S)
SUBLINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


def _plain(html: str) -> str:
    text = HTML_TAG.sub(" ", html)
    text = ENTITY.sub(" ", text)
    text = NAV.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _subpages(title: str, wikitext: str) -> list[str]:
    """Сторінка-версія містить лише посилання на частини — саме там і лежить текст."""
    prefix = title.split("/")[0]
    found = [t.strip() for t in SUBLINK.findall(wikitext or "")]
    return list(dict.fromkeys(t for t in found if t.startswith(prefix + "/")))


def _api(params: dict) -> dict | None:
    cmd = ["curl", "-s", "--max-time", "25", "-H", f"User-Agent: {AGENT}", "-G", API]
    for key, value in params.items():
        cmd += ["--data-urlencode", f"{key}={value}"]
    raw = subprocess.run(cmd, capture_output=True, text=True).stdout
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _titles(query: str, limit: int) -> list[str]:
    data = _api({"action": "query", "list": "search", "format": "json", "srnamespace": "0",
                 "srlimit": str(limit), "srsearch": query})
    return [hit["title"] for hit in data["query"]["search"]] if data else []


def _page(title: str) -> tuple[str, str]:
    """Віддає (чистий текст, вікітекст). Другий потрібен лише щоб знайти підсторінки."""
    data = _api({"action": "parse", "format": "json", "prop": "text|wikitext", "page": title})
    parse = (data or {}).get("parse", {})
    return _plain(parse.get("text", {}).get("*", "")), parse.get("wikitext", {}).get("*", "")


def _chunks(text: str) -> list[str]:
    """Ріжемо по абзацах, а не по символах: розрізаний посередині абзац — зіпсований документ."""
    body = FOOTER.sub("", text).strip()
    out, current = [], ""
    for para in (p.strip() for p in body.split("\n") if p.strip()):
        if len(current) + len(para) > CHUNK_CHARS and current:
            out.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        out.append(current)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--fetched", required=True)
    ap.add_argument("--queue", default="evalkit/data/corpus.db")
    args = ap.parse_args()

    queue = SqliteQueue(args.queue)
    titles = _titles(args.search, args.limit)
    if not titles:
        print("пошук нічого не дав або API недосяжне")
        return 1

    manifest = {"джерело": "uk.wikisource.org", "api": API, "ліцензія": LICENSE,
                "здобуто": args.fetched, "запит": args.search, "документи": [], "відкинуто": []}
    added = 0
    pending = list(titles)
    seen: set[str] = set()
    while pending:
        title = pending.pop(0)
        if title in seen:
            continue
        seen.add(title)
        time.sleep(DELAY)
        text, raw = _page(title)
        if not text and not raw:
            manifest["відкинуто"].append({"назва": title, "причина": "сторінка недосяжна"})
            continue
        if len(text) < MIN_CHARS:
            subs = _subpages(title, raw)[:12]
            manifest["відкинуто"].append({
                "назва": title,
                "причина": f"сторінка-версія, {len(subs)} підсторінок у чергу на обхід"
                           if subs else "порожня або занадто коротка"})
            pending.extend(subs)
            continue
        parts = _chunks(text)
        for index, part in enumerate(parts):
            key = f"{title}#{index}"
            if queue.put(key, {"назва": title, "частина": index, "текст": part}):
                added += 1
        manifest["документи"].append({"назва": title, "частин": len(parts), "символів": len(text)})

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"corpus-{args.search}.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"документів {len(manifest['документи'])}, нових частин у черзі {added}, "
          f"відкинуто {len(manifest['відкинуто'])}")
    print(f"черга: {queue.stats()} | манифест: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
