"""Набір, який ВИМАГАЄ фан-ауту: одна задача — чотири рідкісні слова.

Сенс: один цикл мусить зробити чотири пошуки в межах однієї стелі кроків, а граф дає кожному слову
власну дитину. Це той самий важіль, що K7e ловив як «стелю обходу», але тепер перевіряється
розгалуженням, а не памʼяттю про залишок.

Слова беруться з уже грунтованих даних (`lexis-uk.json`), ключі — ті самі, що в `lexis`, тож нічого
нового не вигадується. Кожне слово дає ОКРЕМИЙ чек: чотири чеки на айтем, і всі мусять пройти.

Запуск: uv run python scripts/build_fanout_items.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_lexis_items import derive_keys  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "evalkit" / "data" / "lexis-uk.json"
ITEMS = Path(__file__).resolve().parents[1] / "evalkit" / "items"
TASK = ("Поясни, що означає кожне з цих рідкісних українських слів: {words}.\n"
        "Дай по одному короткому реченню на кожне слово, без JSON.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4)
    args = ap.parse_args()
    WIDTH = args.width
    OUT = ITEMS / ("fanout.jsonl" if WIDTH == 4 else f"fanout{WIDTH}.jsonl")
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    pool = []
    for entry in payload["статті"]:
        if entry["страт"] != "rare":
            continue
        keys, _ = derive_keys(entry["слово"], entry["значення"])
        if len(keys) >= 2:
            pool.append((entry["слово"], keys))
    pool.sort()

    items = []
    for start in range(0, len(pool) - WIDTH + 1, WIDTH):
        group = pool[start:start + WIDTH]
        words = ", ".join(f"«{w}»" for w, _ in group)
        task = TASK.format(words=words)
        # Ключ, що вже є в тексті задачі, проходив би копіюванням: слова названі в задачі, тому
        # відкидаємо збіги з нею так само, як у `lexis`.
        checks = [{"kind": "answered"}, {"kind": "answer_no_json"}]
        clean_group = []
        for word, keys in group:
            clean = [k for k in keys if k.casefold() not in task.casefold()]
            if not clean:
                break
            clean_group.append((word, clean))
            checks.append({"kind": "answer_stem_any", "values": clean})
        if len(clean_group) != WIDTH:
            continue
        items.append({
            "id": "fan-" + "-".join(w for w, _ in clean_group),
            "category": f"fanout_{WIDTH}",
            "task": task,
            "checks": checks,
            "solvable_by": ["словник", "final_answer"],
            "gold": ["; ".join(f"{w} — {ks[0]}" for w, ks in clean_group)],
            "foil": ["; ".join(f"{w} — не знаю" for w, _ in clean_group)],
            "gold_tools": [],
            "toolsets": ["lexis"],
            "chain_len": WIDTH,
        })

    OUT.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                   encoding="utf-8")
    print(f"{len(items)} айтемів по {WIDTH} слів (пул {len(pool)}) -> {OUT}")


if __name__ == "__main__":
    main()
