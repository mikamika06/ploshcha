"""Переоцінює ЗБЕРЕЖЕНІ відповіді набору `lexis` поточними чеками — без нових викликів шлюзу.

Навіщо окремий скрипт: чеки — чиста функція від тексту відповіді, тому виправлення ключа не вимагає
переплачувати за прогін. Але це й дисциплінарний слід: кожна корекція оцінювання видна як різниця
«було/стало» по клітинках, а не тихо переписане число.

Запуск: uv run python scripts/rescore_lexis.py <звіт.json> [<звіт.json> ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.checks import split_checks
from ploshcha_sim.domain.task import TaskResult

ITEMS = Path(__file__).resolve().parents[1] / "evalkit" / "items" / "lexis.jsonl"
STRATA = ("lexis_rare", "lexis_common", "lexis_absent")


def _items() -> dict[str, dict]:
    rows = [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r for r in rows}


def _result(row: dict) -> TaskResult:
    return TaskResult(answer=row["answer"], accepted=row["accepted"], steps=row["steps"],
                      scratch=[{"call": {"tool": t}, "result": {}} for t in row["tools"]],
                      degraded=row["degraded"], partial=row["partial"])


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 1
    items = _items()

    scored: dict[str, dict[str, bool]] = defaultdict(dict)
    verdict: dict[str, list[bool]] = defaultdict(list)
    tokens: dict[str, list[int]] = defaultdict(list)
    old: dict[str, dict[str, bool]] = defaultdict(dict)
    flips = []

    for path in paths:
        for row in json.loads(path.read_text(encoding="utf-8"))["results"]:
            outcome, _ = split_checks(items[row["item_id"]]["checks"], _result(row))
            ok = all(outcome.values())
            scored[row["condition"]][row["item_id"]] = ok
            old[row["condition"]][row["item_id"]] = bool(row["success"])
            tokens[row["condition"]].append(row["tokens"])
            if items[row["item_id"]]["category"] == "lexis_absent":
                verdict[row["condition"]].append(not row["degraded"])
            if ok != bool(row["success"]):
                flips.append((row["condition"], row["item_id"], bool(row["success"]), ok))

    cat = {i: items[i]["category"] for i in items}
    print(f"{'умова':<20}{'усе':>14}{'рідкісні':>12}{'широковідомі':>14}"
          f"{'поза довідн.':>14}{'токени':>9}")
    for name, cells in scored.items():
        parts = [f"{sum(cells.values())}/{len(cells)}={sum(cells.values()) / len(cells):.3f}"]
        for stratum in STRATA:
            sub = [v for i, v in cells.items() if cat[i] == stratum]
            parts.append(f"{sum(sub)}/{len(sub)}" if sub else "—")
        avg = sum(tokens[name]) / len(tokens[name])
        print(f"{name:<20}{parts[0]:>14}{parts[1]:>12}{parts[2]:>14}{parts[3]:>14}{avg:>9.0f}")

    print("\nвердикт верифікатора на страті «поза довідником» (окрема вісь: чи прийняв ВІН відмову):")
    for name, votes in verdict.items():
        print(f"  {name:<20} прийнято {sum(votes)}/{len(votes)}")

    if flips:
        print("\nкорекція оцінювання (було -> стало):")
        for name, item, was, now in flips:
            print(f"  {name:<18}{item:<20}{was} -> {now}")

    print("\nпарування (та сама модель, різниця лише в довіднику):")
    for base, treat in (("lex-loop", "lex-ref@8"), ("lex-plain", "lex-ref@8"),
                        ("lex-plain-lapa", "lex-ref-lapa@8"),
                        ("lex-ref@8", "lex-ref-rec@8"),
                        ("lex-ref-lapa@8", "lex-ref-lapa-rec@8")):
        if base not in scored or treat not in scored:
            continue
        cured = sorted(i for i in scored[base] if not scored[base][i] and scored[treat][i])
        broke = sorted(i for i in scored[base] if scored[base][i] and not scored[treat][i])
        print(f"  {base} -> {treat}: вилікувано {len(cured)}, зламано {len(broke)}, "
              f"net {len(cured) - len(broke)}")
        for i in cured:
            print(f"      + {i}")
        for i in broke:
            print(f"      - {i}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
