"""Розділити «культурне знання» від «позиційної залипухи».

Кожну пару питаємо ДВІЧІ: автентичне як A, потім автентичне як B. Тоді:
  - модель іде за ЗМІСТОМ -> обирає автентичне обидва рази (знання є);
  - модель іде за ПОЗИЦІЄЮ -> обирає ту саму букву обидва рази (залипуха, суддівство мертве).

Це відрізняє «Lapa не знає культури» від «Lapa знає, але генеративно залипає» — різні діагнози
з різними наслідками для того, як її застосувати.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")

from openai import OpenAI  # noqa: E402

# (автентичне, неавтентичне)
PAIRS = [
    ("Ходи вечеряти, синку.", "Іди кушати, синку."),
    ("Бог у поміч, куме!", "Привіт, друже, як справи?"),
    ("Твоя правда, дядьку Свириде.", "Ти прав, дядя Свирид."),
    ("Треба до млина зерно везти.", "Надо на мельницю зерно везти."),
    ("Що ти, доню, зажурилася?", "Чо грустиш, дочка?"),
    ("Позич сокиру, дровець нарубати.", "Одолжи топор, дров нарубить."),
    ("На толоку вся громада збереться.", "На субботник все соберутся."),
    ("Свят-вечір без куті — не свят-вечір.", "Різдво без салату — не Різдво."),
]


def pick(client, model, first, second) -> str:
    q = (f"Який варіант звучить автентичніше українською сільською мовою? "
         f"Відповідай рівно однією буквою A або B.\nA: {first}\nB: {second}")
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": q}], temperature=0.0, max_tokens=6
    )
    out = (r.choices[0].message.content or "").strip().upper()
    return "A" if out.startswith("A") else ("B" if out.startswith("B") else "?")


def run(client, label, model):
    print(f"\n{'='*80}\n{label}   {model}\n{'='*80}")
    content_ok = 0       # обрало автентичне обидва рази -> знання
    position_stuck = 0   # обрало ту саму букву обидва рази -> залипуха
    for auth, fake in PAIRS:
        p1 = pick(client, model, auth, fake)   # автентичне = A
        p2 = pick(client, model, fake, auth)   # автентичне = B
        by_content = (p1 == "A") and (p2 == "B")
        by_position = (p1 == p2)
        content_ok += by_content
        position_stuck += by_position
        tag = "ЗНАННЯ" if by_content else ("ЗАЛИПЛО-НА-" + p1 if by_position else "шум")
        print(f"  {tag:<16} A-first={p1} B-first={p2}  «{auth[:34]}»")
    n = len(PAIRS)
    print(f"  --- за змістом: {content_ok}/{n}   залипло на позиції: {position_stuck}/{n}")


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        return 1
    client = OpenAI(base_url=url, api_key=key, timeout=180)
    for label, var in [("Lapa-12B", "LAPA_MODEL"), ("Mamay-27B", "MAMAY_MODEL")]:
        if os.environ.get(var):
            run(client, label, os.environ[var])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
