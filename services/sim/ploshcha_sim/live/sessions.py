"""Своє село кожному гостю: стан сесії живе у власному файлі, старі файли прибираються.

Доти всі сховища сиділи на ОДНОМУ SQLite, тож будь-хто, хто кинув тему, змінював село всім: чужі
ухвали, чужі чутки, чужий літопис. Це не косметика — памʼять і є те, заради чого сюди вертаються,
а спільна памʼять на публічному порті означає, що вертатись немає куди.

Три рішення, які тримають цю річ у межах:

  • **Ідентифікатор — імʼя файлу.** Отже, він приходить із браузера й мусить бути перевірений, бо
    `../../` у `sid` писало б поза теку. Абетка вузька навмисно.
  • **Час дотику — mtime файлу.** Окрема таблиця «коли востаннє заходив» вимагала б своєї схеми,
    своєї міграції і свого запису на кожен перегляд; mtime дає те саме безкоштовно, переживає
    перезапуск ядра (це властивість файлової системи, не процесу) і читається одним `stat`.
    Мінус чесний: копіювання теки чужим інструментом може мітки збити — але ціна помилки тут
    «сесію прибрали на тиждень раніше», а не втрата грошей чи доступу.
  • **Дві стелі, а не одна.** TTL прибирає забуте, стеля кількості прибирає навалу: без другої
    сотня свіжих сесій за годину так само забиває диск, хоч жодна ще не протухла.
"""

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ідентифікатор із браузера стає ІМʼЯМ ФАЙЛУ, тому абетка безпечна за побудовою: ні крапок, ні
# слешів, ні пробілів. Нижня межа довжини — щоб «a» не вважалось сесією; верхня — щоб імʼя файлу
# лишалось у межах будь-якої ФС.
SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

DEFAULT_TTL_DAYS = 7.0
DEFAULT_MAX_SESSIONS = 200
# Як часто цикл витрачає час на обхід теки. Прибирання не термінове: доба лишку нікому не шкодить,
# а `stat` на кожній ітерації циклу — шкодить.
SWEEP_EVERY_S = 300.0


def clean_sid(sid: Any) -> str | None:
    """Перевірений ідентифікатор або `None`. `None` — це «спільне село», а не помилка."""
    if not isinstance(sid, str):
        return None
    sid = sid.strip()
    return sid if SID_RE.match(sid) else None


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


def ttl_seconds() -> float:
    return _env_float("PLOSHCHA_SESSION_TTL_DAYS", DEFAULT_TTL_DAYS) * 86400.0


def max_sessions() -> int:
    return _env_int("PLOSHCHA_MAX_SESSIONS", DEFAULT_MAX_SESSIONS)


@dataclass
class Session:
    """Усе, що прогін бере від сесії. Порожні поля — законний стан для не-віче умов."""

    sid: str | None
    path: str
    make_agent: Callable
    cast: list[dict] | None = None
    decisions: Any = None
    rumours: Any = None
    memory: Any = None
    extra: dict = field(default_factory=dict)


class SessionRegistry:
    """Сховища під `sid`: будуються на вимогу, кешуються, прибираються за віком.

    `build(path, sid)` віддає `Session` — реєстр НЕ знає, з чого складається село, бо це знає
    composition root. Тут лише життєвий цикл файлу.
    """

    def __init__(self, root: str | Path, build: Callable[[str, str | None], Session], *,
                 base: Session | None = None, ttl_s: float | None = None,
                 limit: int | None = None, clock=time.time):
        self.root = Path(root)
        self.build = build
        self.base = base
        self.ttl_s = ttl_seconds() if ttl_s is None else float(ttl_s)
        self.limit = max_sessions() if limit is None else int(limit)
        self._clock = clock
        self._cache: dict[str, Session] = {}
        self.root.mkdir(parents=True, exist_ok=True)

    # ── файли ────────────────────────────────────────────────────────────────
    def path_for(self, sid: str) -> Path:
        return self.root / f"{sid}.db"

    def known(self, sid: str | None) -> bool:
        sid = clean_sid(sid)
        return bool(sid) and (sid in self._cache or self.path_for(sid).exists())

    def ensure(self, sid: str) -> bool:
        """Застовпити сесію ФАЙЛОМ, ще до першої теми. Повертає `True`, якщо створили щойно.

        Порожній файл — законна база SQLite, тож нічого не ламає, зате робить нову сесію одразу
        видимою і для `known`, і для прибирання. Без цього гість, який лише клацає команди й не
        доводить до прогону, лишався б для стелі невидимим.
        """
        path = self.path_for(sid)
        if path.exists():
            self.touch(sid)
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return True

    def touch(self, sid: str | None) -> None:
        """«Гість заходив». Дотик — не лише тема: перегляд потоку теж тримає село живим."""
        sid = clean_sid(sid)
        if not sid:
            return
        path = self.path_for(sid)
        try:
            os.utime(path, None)
        except OSError:
            pass

    def count(self) -> int:
        return sum(1 for _ in self.root.glob("*.db"))

    # ── доступ ───────────────────────────────────────────────────────────────
    def get(self, sid: str | None) -> Session | None:
        """Сесія за ідентифікатором. Без ідентифікатора — спільне село (CLI, старий клієнт)."""
        sid = clean_sid(sid)
        if sid is None:
            return self.base
        cached = self._cache.get(sid)
        if cached is None:
            self.ensure(sid)
            cached = self.build(str(self.path_for(sid)), sid)
            # Кеш не має рости без межі: обʼєкти дешеві, але тримати їх більше, ніж дозволено
            # самих сесій, немає сенсу — це вже витік, а не кеш.
            if len(self._cache) >= max(1, self.limit):
                self._cache.clear()
            self._cache[sid] = cached
        self.touch(sid)
        return cached

    # ── прибирання ───────────────────────────────────────────────────────────
    def sweep(self, keep: set[str] | None = None) -> int:
        """Видалити протухлі й найдавніші понад стелю. Повертає, скільки сесій прибрано.

        `keep` — те, що ЗАРАЗ у роботі. Реєстр прибирає з того ж потоку, що й веде прогони, тож
        насправді збігу бути не може; параметр існує, щоб ця обіцянка не трималась випадково.
        """
        keep = keep or set()
        now = self._clock()
        rows: list[tuple[float, Path]] = []
        for path in self.root.glob("*.db"):
            try:
                rows.append((path.stat().st_mtime, path))
            except OSError:
                continue
        rows.sort()

        gone = 0
        held = 0
        alive: list[tuple[float, Path]] = []
        for mtime, path in rows:
            if path.stem in keep:
                # Те, що в роботі, не чіпаємо ЖОДНОЮ стелею — ні віковою, ні кількісною; але з
                # ліміту воно місце займає, інакше стеля була б на одиницю більшою за обіцяну.
                held += 1
                continue
            if now - mtime > self.ttl_s:
                gone += self._drop(path)
            else:
                alive.append((mtime, path))
        # Найдавніші вилітають першими: гість, який заходив учора, дорожчий за того, кого не
        # бачили місяць, — навіть якщо обидва ще в межах TTL.
        for mtime, path in alive[:max(0, len(alive) + held - self.limit)]:
            gone += self._drop(path)
        return gone

    def _drop(self, path: Path) -> int:
        sid = path.stem
        self._cache.pop(sid, None)
        removed = 0
        # WAL і SHM — окремі файли поруч. Лишити їх означало б і сміття на диску, і «воскресіння»
        # частини стану, якщо той самий `sid` колись повернеться.
        for target in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            try:
                target.unlink()
                removed += 1 if target == path else 0
            except OSError:
                pass
        return removed
