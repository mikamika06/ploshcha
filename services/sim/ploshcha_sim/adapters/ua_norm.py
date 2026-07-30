"""Український мовний скіл: правила й словники, БЕЗ LLM.

Перший скіл проєкту, що є українським за змістом, а не за назвами тестових даних. Покриває осі
таксономії U2 (норма/правопис), U4 (калька) і частину U6 (кличний відмінок імен).

Межа оголошена: правила покривають те, що є в наборі `ua-lang`, і не претендують на повноту
української морфології. Кожне правило має джерело в назві, щоб його можна було оскаржити.
"""

CALQUES: dict[str, str] = {
    "приймати участь": "брати участь",
    "приймаю участь": "беру участь",
    "приймаємо участь": "беремо участь",
    "приймати міри": "вживати заходів",
    "на протязі": "протягом",
    "являється": "є",
    "являюсь": "є",
    "дякую вас": "дякую вам",
    "відноситись до людей": "ставитися до людей",
    "відноситися до людей": "ставитися до людей",
    "в залежності від": "залежно від",
    "з метою": "щоб",
    "являє собою": "є",
    "у якості": "як",
    "не дивлячись на": "незважаючи на",
    "співпадати": "збігатися",
    "приходити до висновку": "доходити висновку",
    "по крайній мірі": "принаймні",
    "мати місце": "відбуватися",
    "на сьогоднішній день": "сьогодні",
}

SPELLING_2019: dict[str, tuple[str, ...]] = {
    "проект": ("проєкт",),
    "проектний": ("проєктний",),
    "аудиторія": ("авдиторія", "аудиторія"),
    "аудієнція": ("авдієнція", "аудієнція"),
    "ефір": ("етер", "ефір"),
    "кафедра": ("катедра", "кафедра"),
    "фауна": ("фавна", "фауна"),
}

FEMININE: dict[str, str] = {
    "викладач": "викладачка",
    "продавець": "продавчиня",
    "лікар": "лікарка",
    "філолог": "філологиня",
    "директор": "директорка",
    "вчитель": "вчителька",
    "автор": "авторка",
    "член": "членкиня",
    "борець": "борчиня",
    "фотограф": "фотографка",
}

VOCATIVE_EXCEPTIONS: dict[str, str] = {
    "ігор": "Ігоре",
    "лев": "Леве",
    "любов": "Любове",
}

HARD_SOFT_STEMS = "жчшщ"
VOWELS = "аеєиіїоуюя"


def fix_calques(text: str) -> list[dict]:
    """Кальки шукаються ЛІВИМ найдовшим збігом, щоб «приймати участь у» не ловилось двічі."""
    low = text.casefold()
    found = []
    for wrong in sorted(CALQUES, key=len, reverse=True):
        at = low.find(wrong)
        if at >= 0 and not any(f["at"] <= at < f["at"] + len(f["хибне"]) for f in found):
            found.append({"хибне": wrong, "правильне": CALQUES[wrong], "at": at})
    found.sort(key=lambda f: f["at"])
    fixed = text
    for f in found:
        fixed = _replace_ci(fixed, f["хибне"], f["правильне"])
    return [{"хибне": f["хибне"], "правильне": f["правильне"]} for f in found], fixed


def _replace_ci(text: str, needle: str, repl: str) -> str:
    low, nlow = text.casefold(), needle.casefold()
    at = low.find(nlow)
    if at < 0:
        return text
    return text[:at] + repl + text[at + len(needle):]


def vocative(word: str, gender: str | None = None) -> str:
    """Кличний відмінок за правописом 2019 §87.

    Тонкість, на якій я спершу помилився: прізвища на **-ко** беруть **-у** (Кузьменко → Кузьменку),
    а імена на **-о** беруть **-е** (Петро → Петре). Обидва — друга відміна, але різні підтипи.
    Жіночі прізвища на приголосний не відмінюються (Ткач → Ткач), тому потрібен рід; без нього рід
    визначається за закінченням (-а/-я = жіночий).
    """
    if not word:
        return word
    low = word.casefold()
    if low in VOCATIVE_EXCEPTIONS:
        return VOCATIVE_EXCEPTIONS[low]
    if gender is None:
        gender = "f" if low.endswith(("а", "я")) else "m"
    if low.endswith("ко"):
        return word[:-1] + "у"
    if low.endswith("о"):
        return word[:-1] + "е"
    if low.endswith("а"):
        stem = word[:-1]
        return stem + ("е" if stem[-1:].casefold() in HARD_SOFT_STEMS else "о")
    if low.endswith("я"):
        # 1-ша відміна, мʼяка група: -ія → -іє (Марія→Маріє); після приголосного -ю (Оля→Олю);
        # інакше -е (земля→земле). Правопис 2019 §87.
        if low.endswith("ія"):
            return word[:-1] + "є"
        if len(low) > 1 and low[-2] not in VOWELS:
            return word[:-1] + "ю"
        return word[:-1] + "е"
    if gender == "f":
        return word
    if low.endswith("ь"):
        return word[:-1] + "ю"
    if low.endswith("й"):
        return word[:-1] + "ю"
    if low[-1] in VOWELS:
        return word
    if low.endswith(("к", "г", "х", "ч", "ж", "ш", "щ", "р", "ц")):
        return word + "у"
    return word + "е"


def euphony(word_after: str) -> str:
    """Милозвучність: «у» перед приголосним, «в» перед голосним (правопис 2019 §12)."""
    if not word_after:
        return "у"
    return "в" if word_after[0].casefold() in VOWELS else "у"


def feminitive(word: str) -> str | None:
    """Фемінітиви за правописом 2019 §32: -ка / -иня / -иця; словник має перевагу над правилом."""
    low = word.casefold()
    if low in FEMININE:
        return FEMININE[low]
    if low.endswith("ець"):
        return word[:-3] + "чиня"
    if low.endswith("ач") or low.endswith("яч"):
        return word + "ка"
    if low.endswith("ар") or low.endswith("ор") or low.endswith("ир"):
        return word + "ка"
    if low.endswith("ик") or low.endswith("ік"):
        return word + "иня"
    if low.endswith("ог"):
        return word + "иня"
    return None


def spelling_2019(word: str) -> list[str]:
    """Варіанти за правописом 2019; порожній список означає «слово поза цим правилом»."""
    return list(SPELLING_2019.get(word.casefold(), ()))
