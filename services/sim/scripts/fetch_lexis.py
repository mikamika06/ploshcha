"""Збирає словникові статті рідкісної української лексики з uk.wiktionary у заморожений JSON.

Джерело обране не з міркувань зручності: клас скіла «довідка» перевіряється на фактах, яких модель
НЕ знає, тому еталон не можна писати з голови — це означало б вигадати правильну відповідь. Тут
кожне значення і кожен приклад слововжитку приходить з зовнішнього джерела дослівно, з атрибуцією.

Ліцензія джерела: CC BY-SA 4.0 (uk.wiktionary), значення переважно з «Словника УЛІФ» через нього.
Це фіксується в метаданих файлу: релізи мусять зберігати атрибуцію й share-alike.

Запуск: uv run python scripts/fetch_lexis.py --fetched 2026-07-30
"""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

API = "https://uk.wiktionary.org/w/api.php"
AGENT = "ploshcha-research/0.1 (oleksandrrsavkov@gmail.com) grounded-lexicon-collection"
LICENSE = "CC BY-SA 4.0"
DELAY = 1.2
STEM = 3

RARE = [
    "мешти", "дараба", "кептар", "верета", "бантина", "цямрина", "ватра", "ватралка", "будз",
    "ґазда", "ботей", "бартка", "банувати", "царина", "божник", "бердо", "ліжник", "колиба",
    "ґражда", "царинка", "стая", "ватрак", "ваторник", "сволок", "розсоха", "глота", "кичера",
]
COMMON = [
    "бринза", "веретено", "бивень", "макогін", "ослін", "бандура", "кобза", "криниця", "ярмо",
    "рушник", "макітра", "клуня", "стодола", "серп", "діжка", "глечик", "рогач", "ступа",
]
# Свідомо ПОЗА довідником (`lexicon_kb` відкидає цей страт): потрібні, щоб перевірити, що модель не
# починає довіряти інструментові наосліп, коли слова там немає. Слова взяті з категорії
# «Діалектні вирази/uk» механічно, за незрозумілістю, а не за очікуваним результатом.
ABSENT = [
    "абахта", "алькир", "байбара", "бакай", "бардина", "баришівник", "бабешки", "банно",
]

TEMPLATE = re.compile(r"Шаблон:\S+\s*")
SENSES = re.compile(r"====\s*Значення\s*====\n(.*?)(?=\n(?:Синоніми|====|===))", re.S)
SOURCES = re.compile(r"===\s*Джерела\s*===\n(.*?)(?=\n===|\Z)", re.S)


def _get(word: str) -> str | None:
    cmd = ["curl", "-s", "--max-time", "25", "-H", f"User-Agent: {AGENT}", "-G", API]
    for key, value in {
        "action": "query", "prop": "extracts", "explaintext": "1", "format": "json",
        "exlimit": "1", "titles": word,
    }.items():
        cmd += ["--data-urlencode", f"{key}={value}"]
    raw = subprocess.run(cmd, capture_output=True, text=True).stdout
    try:
        page = next(iter(json.loads(raw)["query"]["pages"].values()))
    except (json.JSONDecodeError, KeyError, StopIteration):
        return None
    return page.get("extract")


def _parse(word: str, extract: str | None) -> dict | None:
    if not extract or not (block := SENSES.search(extract)):
        return None
    senses, examples = [], []
    for line in (l.strip() for l in block.group(1).splitlines()):
        if not line:
            continue
        parts = line.split("◆")
        if sense := TEMPLATE.sub("", parts[0]).strip(" .;"):
            senses.append(sense)
        for tail in parts[1:]:
            tail = re.sub(r"\s+", " ", tail).strip(" .;")
            if tail and "Немає прикладів" not in tail:
                examples.append(tail)
    sources = SOURCES.search(extract)
    return {
        "слово": word,
        "значення": senses,
        "приклади": examples,
        "джерело": [l.strip() for l in (sources.group(1).splitlines() if sources else []) if l.strip()],
    }


def _usable(entry: dict) -> str | None:
    """Приклад мусить МІСТИТИ саме слово, інакше він не годиться як вхід задачі."""
    if not entry["значення"]:
        return "порожнє значення"
    if not entry["приклади"]:
        return "немає прикладу слововжитку"
    stem = entry["слово"][:STEM].casefold()
    if not any(stem in ex.casefold() for ex in entry["приклади"]):
        return "приклад не містить слова"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetched", required=True)
    ap.add_argument("--out", default="evalkit/data/lexis-uk.json")
    args = ap.parse_args()

    entries, rejected = [], []
    for stratum, words in (("rare", RARE), ("common", COMMON), ("absent", ABSENT)):
        for word in words:
            time.sleep(DELAY)
            entry = _parse(word, _get(word))
            if entry is None:
                rejected.append({"слово": word, "страт": stratum, "причина": "немає статті"})
                continue
            if reason := _usable(entry):
                rejected.append({"слово": word, "страт": stratum, "причина": reason})
                continue
            entries.append(entry | {"страт": stratum})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "джерело": "uk.wiktionary.org (значення переважно зі «Словника УЛІФ»)",
        "api": API,
        "ліцензія": LICENSE,
        "здобуто": args.fetched,
        "правило_придатності": f"значення непорожнє, є приклад, приклад містить перші {STEM} літери слова",
        "статті": entries,
        "відкинуто": rejected,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    kept = {s: sum(1 for e in entries if e["страт"] == s)
            for s in ("rare", "common", "absent")}
    print(f"придатних {len(entries)} (rare {kept['rare']}, common {kept['common']}, "
          f"absent {kept['absent']}), відкинуто {len(rejected)} -> {out}")
    for r in rejected:
        print(f"  ✘ {r['слово']:<10} {r['страт']:<7} {r['причина']}")


if __name__ == "__main__":
    main()
