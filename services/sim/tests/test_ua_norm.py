import pytest

from ploshcha_sim.adapters.ua_norm import (
    euphony,
    feminitive,
    fix_calques,
    spelling_2019,
    vocative,
)

VOCATIVE = [
    ("Петро", None, "Петре"),
    ("Павло", None, "Павле"),
    ("Кузьменко", None, "Кузьменку"),
    ("Шевченко", None, "Шевченку"),
    ("Ольга", None, "Ольго"),
    ("Микола", None, "Миколо"),
    ("Марія", None, "Маріє"),
    ("Оля", None, "Олю"),
    ("Ігор", None, "Ігоре"),
    ("Василь", None, "Василю"),
    ("добродій", None, "добродію"),
    ("пан", None, "пане"),
    ("Ткач", "f", "Ткач"),
    ("Ткач", "m", "Ткачу"),
]


@pytest.mark.parametrize("word,gender,want", VOCATIVE)
def test_vocative_matches_the_2019_rules(word, gender, want):
    assert vocative(word, gender) == want


def test_the_two_endings_i_got_wrong_first():
    """Прізвища на -ко беруть -у, імена на -о беруть -е. Спершу я віддавав обидва без змін."""
    assert vocative("Кузьменко") == "Кузьменку" and vocative("Петро") == "Петре"
    assert vocative("Марія") == "Маріє", "-ія → -іє, а не -іе"


def test_the_ua_lang_gold_forms_are_reproduced():
    """Скіл мусить давати рівно ті форми, які набір `ua-lang` вважає правильними."""
    assert f"{vocative('Петро')} {vocative('Кузьменко')}" == "Петре Кузьменку"
    assert vocative("Ольга") == "Ольго"
    assert f"{vocative('пан')} {vocative('Ігор')}" == "пане Ігоре"
    assert f"{vocative('добродій')} {vocative('Василь')}" == "добродію Василю"


def test_common_nouns_are_out_of_scope_and_that_is_declared():
    """Межа оголошена: правило для ІМЕН. «земля → земле» потребує лексикону, і ми його не маємо."""
    assert vocative("земля") == "землю", "здрібнілі імена й загальні іменники не розрізняються"


@pytest.mark.parametrize("text,wrong,right", [
    ("приймати участь у конференції", "приймати участь", "брати участь"),
    ("на протязі тижня", "на протязі", "протягом"),
    ("Він являється керівником.", "являється", "є"),
    ("дякую вас за допомогу", "дякую вас", "дякую вам"),
    ("приймати міри для вирішення", "приймати міри", "вживати заходів"),
    ("відноситись до людей з повагою", "відноситись до людей", "ставитися до людей"),
])
def test_calques_are_found_and_fixed(text, wrong, right):
    found, fixed = fix_calques(text)
    assert [f["хибне"] for f in found] == [wrong]
    assert right in fixed and wrong not in fixed.casefold()


def test_a_clean_sentence_is_left_alone():
    found, fixed = fix_calques("Ми беремо участь у змаганнях.")
    assert found == [] and fixed == "Ми беремо участь у змаганнях."


def test_the_longest_calque_wins():
    """«приймати участь» і «приймати міри» не мають ловитись як одна помилка двічі."""
    found, _ = fix_calques("приймати участь та приймати міри")
    assert [f["хибне"] for f in found] == ["приймати участь", "приймати міри"]


@pytest.mark.parametrize("word,prep", [("Львові", "у"), ("Одесі", "в"), ("Києві", "у"),
                                       ("Ужгороді", "в"), ("Черкасах", "у")])
def test_euphony_picks_the_preposition(word, prep):
    assert euphony(word) == prep


@pytest.mark.parametrize("word,want", [("викладач", "викладачка"), ("продавець", "продавчиня"),
                                       ("лікар", "лікарка"), ("філолог", "філологиня")])
def test_feminitives_follow_2019(word, want):
    assert feminitive(word) == want


def test_an_unknown_word_returns_none_not_a_guess():
    assert feminitive("стіл") is None, "порожня відповідь краща за вигадану форму"


def test_spelling_2019_returns_both_permitted_variants():
    assert spelling_2019("проект") == ["проєкт"]
    assert spelling_2019("аудиторія") == ["авдиторія", "аудиторія"]
    assert spelling_2019("хата") == [], "слово поза цим правилом — не вигадуємо варіантів"


def test_the_skill_uses_no_model():
    """Це правило й словник; поява LLM у цьому модулі означала б втрату сенсу скіла."""
    import inspect

    from ploshcha_sim.adapters import ua_norm
    source = inspect.getsource(ua_norm)
    for forbidden in ("Llm", "openai", "generate(", "prompt"):
        assert forbidden not in source, forbidden
