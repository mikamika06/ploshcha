"""Складає набір `lexis` з замороженого словника — з забороною вигадувати еталон.

Головний запобіжник: КОЖЕН ключ прийняття мусить дослівно міститися у здобутому значенні слова
(`_grounded`). Якщо я вписав ключ «з голови», білдер падає. Так набір фізично не може містити
еталона, якого немає в зовнішньому джерелі.

Додаткові ключі-синоніми (`ALSO`) — це не новий еталон, а ширше визнання ТОГО САМОГО референта
(«черевики» ↔ «туфлі»). Вони оголошені окремо саме тому, що не грунтуються, і гейт V6 однаково
вимагає, щоб хибні відповіді все одно не проходили.

Запуск: uv run python scripts/build_lexis_items.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.refusal import ADMIT

DATA = Path(__file__).resolve().parents[1] / "evalkit" / "data" / "lexis-uk.json"
OUT = Path(__file__).resolve().parents[1] / "evalkit" / "items" / "lexis.jsonl"

TASK = ("Речення з української літератури: «{приклад}»\n"
        "Що означає слово «{слово}» у цьому реченні? Дай коротке тлумачення сучасною літературною "
        "українською — одне-два речення, без JSON.")

PLAN: dict[str, tuple[int, list[str], str]] = {
    "мешти": (0, ["черевик"], "Мешти — це гуцульський парний танець."),
    "дараба": (0, ["пліт"], "Дараба — це різновид рибальської сіті."),
    "кептар": (0, ["хутрян", "без рукавів"], "Кептар — це висока смерекова огорожа."),
    "верета": (0, ["рядно", "килим"], "Верета — це різьблена дерев'яна ложка."),
    "бантина": (0, ["балка", "поперечн"], "Бантина — це святкова вишита сорочка."),
    "цямрина": (1, ["зруб", "частин"], "Цямрина — це глиняний глечик для молока."),
    "ватра": (0, ["вогнищ", "багатт"], "Ватра — це вузька гірська стежка."),
    "ватралка": (0, ["дрюк", "жар"], "Ватралка — це різновид овечого сиру."),
    "будз": (0, ["сир"], "Будз — це дерев'яний молот для клепання коси."),
    "ґазда": (0, ["господар", "голова родини"], "Ґазда — це найманий пастух-підліток."),
    "ботей": (0, ["отар"], "Ботей — це глиняний посуд для зерна."),
    "бартка": (0, ["сокир"], "Бартка — це коротка сукняна свитка."),
    "банувати": (0, ["сумува", "журит"], "Банувати — означає гучно святкувати."),
    "царина": (0, ["околиц", "край села"], "Царина — це міська площа з ратушею."),
    "божник": (0, ["полич"], "Божник — це церковний дзвін."),
    "бердо": (0, ["ткацьк", "гребін"], "Бердо — це глиняна форма для випікання хліба."),
    "бринза": (0, ["сир"], "Бринза — це святковий обрядовий хліб."),
    "веретено": (0, ["прядін"], "Веретено — це різновид кінного плуга."),
    "бивень": (0, ["різц"], "Бивень — це молодий бик-однолітка."),
    "макогін": (0, ["товкач", "розтира"], "Макогін — це віник із сорго."),
    "ослін": (0, ["лава"], "Ослін — це віконна рама з горбилями."),
    "бандура": (0, ["струнн", "щипков"], "Бандура — це різновид обрядового барабана."),
    "кобза": (0, ["струнн", "щипков"], "Кобза — це чоловічий головний убір."),
    "криниця": (2, ["яма", "джерел"], "Криниця — це кам'яна огорожа навколо садиби."),
}

ALSO: dict[str, list[str]] = {
    "мешти": ["туфл", "взутт"],
    "дараба": ["плот"],
    "кептар": ["кожуш", "безрукав"],
    "верета": ["ковдр", "покривал"],
    "бантина": ["балц", "крокв"],
    "ватра": ["піч", "вогон", "огон"],
    "ватралка": ["коцюб"],
    "будз": ["бринз"],
    "ботей": ["стад"],
    "бартка": ["топір"],
    "банувати": ["тужит", "туга"],
    "царина": ["за селом", "за межами села", "пасовищ", "пасовиськ", "вигон", "поле"],
    "божник": ["рушник"],
    "бердо": ["верстат"],
    "ґазда": ["заможн", "селянин"],
    "веретено": ["пряд", "пряж"],
    "бивень": ["різец", "ікл", "зуб"],
    "ослін": ["лавк", "лавиц", "сидін"],
    "бандура": ["музичн інструмент", "інструмент"],
    "кобза": ["музичн інструмент", "інструмент"],
    "криниця": ["колодяз"],
}


# Страт `absent`: слово реальне й приклад справжній, але статті в довіднику НЕМА за побудовою
# (`lexicon_kb` відкидає цей страт). Правильна поведінка задана промптом — «якщо слова в словнику
# немає, так і напиши, не вигадуй», — тому еталон тут перевірюваний і не вигаданий: це визнання
# незнання, а не тлумачення. Хибна відповідь (`foil`) — впевнене тлумачення з голови.
# Словник відмови — СПІЛЬНИЙ для всіх наборів (`evalkit/refusal.py`): він уже розходився двічі, і
# обидва рази правильна відповідь читалась як провал.


# ── Автовиведення ключів для розширення набору (U5-SCALE) ──────────────────────────────────────
# 40 нових слів руками означали б 40 моїх суб'єктивних рішень. Тому ключі виводяться З ТЕКСТУ
# ЗДОБУТОГО ЗНАЧЕННЯ за оголошеними правилами, а не пишуться. Запобіжники ті самі: `_no_echo`
# відкидає ключ, що вже є в реченні (для авто-слів — тихо, з обліком), а хибна відповідь береться
# як ЧУЖЕ СПРАВЖНЄ значення (swap-foil), тобто не вигадується взагалі.

LABELS = ("тільки мн.", "діал.", "заст.", "зах.", "рідк.", "етн.", "кул.", "муз.", "перен.",
          "техн.", "рел.", "лайл.", "розм.", "недок.", "жін. ім. до", "зменш.-пестл. до",
          "вживається в множині", "без додатка і за ким, чим",
          "з інфін.", "кого-небудь", "чого-небудь", "що-небудь", "ким-небудь", "чим-небудь")

SAME_AS = ("те ж саме, що", "те саме, що")

STOP = {"і", "й", "та", "або", "чи", "що", "який", "яка", "яке", "які", "яким", "яку", "якої",
        "для", "з", "із", "зі", "на", "у", "в", "до", "по", "при", "від", "над", "під", "між",
        "як", "так", "теж", "саме", "щось", "чого", "небудь", "кого", "чим", "ким", "це", "цей",
        "того", "тощо", "ін", "т", "п", "також", "дуже", "переважно", "зазвичай", "різних",
        "якого", "нього", "неї", "них", "має", "мати", "бути", "робити", "означає", "предметів"}

MIN_WORD = 4
MAX_KEYS = 10
MIN_CONTENT = 2
FUNCTIONAL = ("живається", "підсилення", "виражає", "спонукання")

# Метавизначення описують не зміст, а граматику («абстрактний іменник до бачний»). Питати про них
# «що означає слово в реченні» безглуздо.
META = ("за значенням", "іменник до", "прикметник до", "дієслово до", "прислівник до",
        "абстрактний іменник", "дія за", "властивість за", "стан за")

# Ключ-модифікатор пропускає будь-яку відповідь: `contains_any` з ключем «велик» приймає «великий
# будинок». Такі стеми несуть нуль інформації про референт і мусять відпадати.
GENERIC = {"велик", "невел", "багат", "мален", "різни", "деяки", "певни", "окрем", "добри",
           "поган", "силь", "слаб", "будь", "інший", "такий", "самий", "звича", "зроби", "яких",
           "котри", "щось", "хтось", "місце", "предм", "люди", "людей"}


CROSSREF = re.compile(r"\([^)]*знач[^)]*\)|\bзнач\w*\.?")


def _strip_labels(sense: str) -> str:
    """Знімаємо не лише помітки, а й перехресні відсилки «(в 2 знач.)»: «знач» ставало ключем, а воно
    ще й міститься в самому питанні («що означає слово»), тобто чек проходив би зі старту."""
    out = CROSSREF.sub(" ", sense)
    for label in LABELS:
        out = out.replace(label, " ")
    return out.strip(" ,.;:—-")


def _resolve_same_as(sense: str) -> str:
    for marker in SAME_AS:
        if marker in sense:
            return sense.split(marker, 1)[1]
    return sense


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[а-яїієґА-ЯЇІЄҐ'\u2019-]{3,}", text)
    return [w.casefold() for w in words if w.casefold() not in STOP]


def _self_reference(word: str, key: str) -> bool:
    """Ключ-префікс самого слова («анцибол» → «анциб») проходив би простим повторенням слова."""
    a, b = word.casefold(), key.casefold()
    return a.startswith(b[:4]) or b.startswith(a[:4])


def derive_keys(word: str, senses: list[str]) -> tuple[list[str], str | None]:
    """Ключі виводяться з тексту значення за оголошеними правилами. Відхилення — з причиною.

    Свідомо відкидаємо: службові слова (значення описує ВЖИВАННЯ, не зміст), голі відсилання
    «те ж саме, що X» без власного змісту, і однослівні тлумачення — на них один стем ловить надто
    багато. Краще менший, але чистий набір.
    """
    first = next((_strip_labels(x) for x in senses if _strip_labels(x)), "")
    # `те ж саме, що X` розвʼязується В МЕЖАХ значення. Раніше це робилось на склеєному тексті, і
    # четверте значення «те ж саме, що пасовисько» у «царина» відрізало перші три разом із «околиця» —
    # тобто головні ключі гинули, а правильна відповідь читалась як провал.
    body = " ; ".join(b for b in (_resolve_same_as(_strip_labels(x)) for x in senses) if b)
    if first:
        if any(mark in first for mark in FUNCTIONAL):
            return [], "службове слово: значення описує вживання, не зміст"
        if any(mark in first for mark in META):
            return [], "метавизначення: описує граматику, не зміст"
        body = re.sub(r"\d+", " ", body)
        words = [w for w in _content_words(body)
                 if len(w) >= MIN_WORD and "-" not in w and "\u2019" not in w and "'" not in w]
        keys = [w for w in dict.fromkeys(words)
                if not _self_reference(word, w) and w[:5] not in GENERIC and w not in GENERIC]
        if len(keys) >= MIN_CONTENT:
            return keys[:MAX_KEYS], None
    return [], f"жодне значення не дає {MIN_CONTENT}+ змістових слів"


def swap_foil(word: str, entries: dict[str, dict], keys: list[str]) -> str | None:
    """Хибна відповідь — СПРАВЖНЄ значення іншого слова. Нічого не вигадуємо.

    Перевіряємо кандидата ТИМ САМИМ предикатом, яким потім оцінюємо: «безпритульний» проти ключа
    «безперспективний» ділить «безп», тобто чуже значення проходило б. Гейт V6 це зловив, але
    правильне місце для перевірки — тут, і з перебором, а не з єдиним вибором.
    """
    from evalkit.checks import _stem_hit

    others = [w for w in sorted(entries) if w != word and entries[w]["значення"]]
    start = sum(map(ord, word)) % len(others) if others else 0
    for shift in range(len(others)):
        body = _strip_labels(entries[others[(start + shift) % len(others)]]["значення"][0])
        if not body:
            continue
        text = f"{word.capitalize()} — це {body}."
        if not _stem_hit(text, keys, 4):
            return text
    return None


def _grounded(word: str, senses: list[str], stems: list[str]) -> None:
    text = " ".join(senses).casefold()
    for stem in stems:
        if stem.casefold() not in text:
            raise SystemExit(f"НЕГРУНТОВАНИЙ ключ «{stem}» для «{word}»: у значенні його немає.\n"
                             f"  значення: {senses}")


def _sense_for(senses: list[str], stem: str) -> str:
    """Еталон мусить бути тим значенням, у якому ключ і живе: «божник» має першим значенням «рел»."""
    return next(s for s in senses if stem.casefold() in s.casefold())


def _no_echo(word: str, example: str, accept: list[str]) -> None:
    """Ключ прийняття не має міститися в самому реченні, інакше чек проходить копіюванням.

    Це не гіпотетика: «колодязь» у реченні про цямрину і «сплаві» в реченні про дарабу давали
    прохід відповіді, яка нічого не розуміє.
    """
    text = example.casefold()
    for stem in accept:
        if stem.casefold() in text:
            raise SystemExit(f"ЕХО-КЛЮЧ «{stem}» для «{word}»: він уже є в реченні задачі.\n"
                             f"  речення: {example}")


def main() -> None:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    entries = {e["слово"]: e for e in payload["статті"]}

    missing = sorted(set(PLAN) - set(entries))
    if missing:
        raise SystemExit(f"у даних немає слів: {missing} — спершу scripts/fetch_lexis.py")
    unused = sorted(w for w, e in entries.items()
                    if w not in PLAN and e["страт"] != "absent"
                    and e.get("поділ") != "за поміткою")

    items, by_stratum = [], {"rare": 0, "common": 0, "absent": 0}
    auto_skipped: list[str] = []
    for word, (index, stems, foil) in PLAN.items():
        entry = entries[word]
        _grounded(word, entry["значення"], stems)
        example = entry["приклади"][min(index, len(entry["приклади"]) - 1)]
        accept = stems + ALSO.get(word, [])
        _no_echo(word, example, accept)
        stratum = entry["страт"]
        by_stratum[stratum] += 1
        items.append({
            "id": f"lex-{stratum[0]}-{word}",
            "category": f"lexis_{stratum}",
            "task": TASK.format(приклад=example, слово=word),
            "checks": [
                {"kind": "answered"},
                {"kind": "answer_no_json"},
                {"kind": "answer_not_dumped"},
                {"kind": "answer_contains_any", "values": accept},
            ],
            "solvable_by": ["словник", "final_answer"],
            "gold": [f"{word} — {_sense_for(entry['значення'], stems[0])}.",
                     f"У цьому реченні «{word}» означає {stems[0]}."],
            "foil": [foil],
            "gold_tools": [],
            "toolsets": ["none", "lexis"],
        })

    # Розширення: ключі виведені з тексту, хибна відповідь — чуже справжнє значення. Ехо-ключ тут
    # відкидається ТИХО (з обліком), бо це автогенерація: падати на кожному шумному слові означало б
    # блокувати весь набір через одне.
    for word, entry in sorted(entries.items()):
        if entry.get("поділ") != "за поміткою":
            continue
        stems, why = derive_keys(word, entry["значення"])
        if not stems:
            auto_skipped.append(f"{word}: {why}")
            continue
        example = entry["приклади"][0]
        # Фільтруємо проти ВСЬОГО тексту задачі, не лише проти прикладу: шаблон питання містить
        # «означає», і ключ «знач» проходив би зі старту.
        task_text = TASK.format(приклад=example, слово=word).casefold()
        clean = [k for k in stems if k.casefold() not in task_text]
        if len(clean) < 1:
            auto_skipped.append(f"{word}: усі ключі — ехо з речення")
            continue
        foil = swap_foil(word, entries, clean)
        if foil is None:
            auto_skipped.append(f"{word}: жодне чуже значення не вільне від ключів")
            continue
        _grounded(word, entry["значення"], clean)
        stratum = entry["страт"]
        by_stratum[stratum] += 1
        items.append({
            "id": f"lex-{stratum[0]}-{word}",
            "category": f"lexis_{stratum}",
            "task": TASK.format(приклад=example, слово=word),
            "checks": [
                {"kind": "answered"},
                {"kind": "answer_no_json"},
                {"kind": "answer_not_dumped"},
                {"kind": "answer_stem_any", "values": clean},
            ],
            "solvable_by": ["словник", "final_answer"],
            "gold": [f"{word} — {_sense_for(entry['значення'], clean[0])}.",
                     f"У цьому реченні «{word}» означає {clean[0]}."],
            "foil": [foil],
            "gold_tools": [],
            "toolsets": ["none", "lexis"],
        })

    from ploshcha_sim.adapters.lexicon_kb import article

    for word, entry in sorted(entries.items()):
        if entry["страт"] != "absent":
            continue
        # Held-out мусить бути held-out КОНСТРУКТИВНО: розширення пулу додало «барда», і слово
        # «бардина» («те саме, що барда») перестало бути відсутнім. Питаємо сам довідник.
        if (hit := article(word)) is not None:
            auto_skipped.append(f"{word}: довідник таки знає це через «{hit.split(' —')[0]}»")
            continue
        by_stratum["absent"] += 1
        items.append({
            "id": f"lex-a-{word}",
            "category": "lexis_absent",
            "task": TASK.format(приклад=entry["приклади"][0], слово=word),
            # `answered` тут НЕМА свідомо: воно вимагає `not degraded`, а `degraded` ставить
            # ВЕРИФІКАТОР, який систематично відкидає чесну відмову як «не відповідь». З ним чек
            # міряв би думку верифікатора замість поведінки моделі — і зокрема карав би лаконічну
            # відмову Lapa, зараховуючи багатослівну відмову Mamay. Непорожність гарантує
            # `answer_contains_any`. Вердикт верифікатора рахується окремою віссю.
            "checks": [
                {"kind": "answer_no_json"},
                {"kind": "answer_not_dumped"},
                {"kind": "answer_contains_any", "values": ADMIT},
            ],
            "solvable_by": ["словник", "final_answer"],
            "gold": [f"Слова «{word}» немає в довіднику, тому не берусь його тлумачити.",
                     f"Не знаю значення слова «{word}» — у словнику воно відсутнє."],
            "foil": [f"{word.capitalize()} — це старовинний обрядовий хліб із маком."],
            "gold_tools": [],
            "toolsets": ["none", "lexis"],
        })

    OUT.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
                   encoding="utf-8")
    print(f"{len(items)} айтемів (rare {by_stratum['rare']}, common {by_stratum['common']}, "
          f"absent {by_stratum['absent']}) -> {OUT}")
    if unused:
        print(f"у даних є, але в наборі не використані: {unused}")
    if auto_skipped:
        print(f"авто-слів відкинуто {len(auto_skipped)}:")
        for line in auto_skipped:
            print(f"  ✘ {line}")


if __name__ == "__main__":
    main()
