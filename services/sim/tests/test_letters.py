"""Зведення письма: те, з чим звіряються списки, — і те, чого воно НЕ має права зачепити.

Гейти змісту тут не судяться, для них є `test_zmist_guards.py`. Судиться сам нормалізатор і одна
його межа, від якої залежить усе інше: він робить копію ДЛЯ ПЕРЕВІРОК. Оригінал гостя лишається
оригіналом — інакше «GitHub» приїхав би на сцену кирилицею, а село перебрехало б людині її слово.
"""

import pytest

import ploshcha_sim.domain.viche as viche
from ploshcha_sim.agents.viche import _SPEAKERS
from ploshcha_sim.domain.letters import (
    APOSTROPHES,
    TWINS_LOWER,
    TWINS_UPPER,
    fold,
    for_checks,
    has_cyrillic,
    transliterate,
)
from ploshcha_sim.domain.viche import (
    about_accusation,
    about_violence,
    suspect,
    vows_violence,
)

# Списки, які звіряються зі зведеним письмом. Якщо запис написано інакше, ніж `for_checks` зводить
# текст, він не спрацює НІКОЛИ — і мовчки.
CLOSED_LISTS = (
    "HARM_ROOTS", "HARM_VOWS", "HARM_SELF", "HARM_HINTS",
    "VIOLENCE_VERBS", "VIOLENCE_TARGETS", "VIOLENCE_FORCES", "VIOLENCE_HINTS",
    "VIOLENCE_VOWS", "VOW_TARGETS", "VIOLENCE_PLACE_PREPS", "VIOLENCE_PLACE_PRONOUNS",
    "CRIME_ROOTS", "CRIME_HINTS", "ACCUSE_ROOTS", "ACCUSE_FILLERS",
    "SLUR_ROOTS", "SLUR_DOUBTFUL",
)

# Звичайна сільська мова: те, чого зведення не має права зачепити нічим, крім регістру й пробілів.
PLAIN_VILLAGE = (
    "Гребля протікає, а дощі обіцяють на тому тижні.",
    "Кум із Липʼянки бачив таке ж під осінь.",
    "Мʼясо на ярмарок повеземо в понеділок, як підсохне.",
    "Буряк цього року дрібний, зоря пізня, порядок на току сякий-такий.",
    "Сьогодні льон вибирали, а завтра толока коло криниці.",
    "Редьюсер тут ні до чого, а от млин таки треба лагодити.",
    "Пʼятеро дітей у хаті, і всі здорові, слава Богу.",
)


def test_the_apostrophe_is_typed_with_whatever_is_at_hand_and_all_of_it_is_one_word():
    """Апостроф у Дошці набирають пʼятьма різними знаками, а слово в списку одне.

    Заміряно в цьому репозиторії: «`» 2 508 разів, «'» 1 094, «ʼ» 491, «’» 13, «´» 1. Списки
    написані одним із них, гість набирає будь-яким — отже звіряти треба вигляд без апострофа
    взагалі, і саме це `for_checks` і робить.
    """
    forms = [f"мене б{ch}є чоловік" for ch in APOSTROPHES]
    forms += ["мене бє чоловік", "мене бье чоловік", "МЕНЕ БЄ ЧОЛОВІК", "  мене   бє  чоловік  "]

    keys = {for_checks(t) for t in forms}
    assert keys == {"мене бє чоловік"}, "усі написання зводяться до одного"
    assert all(about_violence(t) for t in forms), "і всі до одного лишаються тим самим гейтом"


def test_a_soft_sign_before_a_yotted_vowel_is_the_apostrophe_only_after_a_labial():
    """«бье» — не українське письмо, а перенесене чуже, і воно однозначне. «редьюсер» — слово.

    Заміряно по 240 354 словах цього репозиторію: «ья», «ьє», «ьї» не трапились жодного разу,
    «ью» — чотири (усі «редьюсер»), «ье» — шість (усі сама ця одрука). Тому правило звужене до
    губних і «р», тобто до тих приголосних, при яких апостроф і стоїть.
    """
    assert fold("бье") == "б'є" and fold("вья") == "в'я" and fold("мью") == "м'ю"
    assert fold("редьюсер") == "редьюсер", "«дь» не губний — слово лишається цілим"
    assert fold("сьогодні льон") == "сьогодні льон", "«ьо» — звичайне українське письмо"


@pytest.mark.parametrize("text", PLAIN_VILLAGE)
def test_ordinary_village_speech_is_not_damaged(text):
    """Звичайна мова міняється рівно на регістр і пробіли — більше нормалізатор її не чіпає.

    Це не окраса тесту, а межа: кожне зайве правило тут коштувало б хибного спрацювання гейта на
    мирній темі, тобто відмови села гомоніти про власну біду.
    """
    assert fold(text) == " ".join(text.lower().replace("ʼ", "'").split())
    assert for_checks(text) == fold(text).replace("'", "")
    assert suspect(text, _SPEAKERS) is None, "мирний рядок лишається мирним"


def test_a_latin_twin_is_replaced_only_inside_a_word_that_already_has_cyrillic():
    """Двійник — це літера з однаковою ФОРМОЮ, і міняється вона лише там, де кирилиця вже є.

    «чоловiк» із латинською «i» — те саме слово, тільки набране двома розкладками; «GitHub» —
    назва, і кирилиці в ній нема чого шукати. Розрізняє їх не словник, а склад самого слова.
    """
    assert for_checks("Мене бье чоловiк") == "мене бє чоловік"
    assert for_checks("На GitHub виклали мапу") == "на github виклали мапу"
    assert not has_cyrillic("GitHub") and has_cyrillic("чоловiк")


@pytest.mark.parametrize("lat, cyr", sorted(TWINS_LOWER.items()) + sorted(TWINS_UPPER.items()))
def test_every_twin_survives_a_whole_word_and_a_bare_pair(lat, cyr):
    """Кожна пара перевіряється окремо, бо великі двійники ширші за малі.

    Мала латинська «b» на «б» не схожа, а велика «B» від «В» не відрізняється зовсім — тому заміна
    робиться ДО зведення регістру, і саме це тут і судиться.
    """
    assert for_checks(f"село{lat}хата") == f"село{cyr.lower()}хата"
    assert for_checks(f"{lat}село") == f"{cyr.lower()}село"


def test_transliteration_takes_a_wholly_latin_line_and_leaves_a_mixed_one_alone():
    """Латиниця читається кирилицею лише тоді, коли кирилиці немає ЖОДНОЇ.

    Мішаний рядок має український кістяк: латинське слово в ньому — назва, а не переписане
    речення, і кирилиця з нього була б вигадкою нормалізатора.
    """
    assert transliterate("vbyu susida, yak shche raz pustyt khudobu") \
        == "вбю сусіда, як ще раз пустит худобу"
    assert transliterate("Odarka vkrala hroshi") == "одарка вкрала гроші"
    assert transliterate("На GitHub виклали мапу") == "На GitHub виклали мапу"
    assert vows_violence("vbyu susida, yak shche raz pustyt khudobu"), \
        "після зведення це та сама обіцянка, що й «вбʼю сусіда»"


@pytest.mark.parametrize("name", ("Meow", "GitHub", "Hello world", "Lorem ipsum dolor sit amet"))
def test_a_peaceful_latin_line_costs_nothing_after_transliteration(name):
    """Транслітерація мирного рядка нічого не ловить — інакше вона купувала б дірки хибними тривогами.

    Це друга половина ціни: зведення письма має право повертати списками пропущене, але не має
    права вигадувати небезпеку там, де її не було.
    """
    assert suspect(name, _SPEAKERS) is None
    assert suspect(f"{name} — от і вся новина, а гребля тече", _SPEAKERS) is None


@pytest.mark.parametrize("name", CLOSED_LISTS)
def test_every_closed_list_is_written_in_the_checked_alphabet(name):
    """Запис, написаний не тим виглядом, не спрацює НІКОЛИ — і мовчки. Тому це судить тест.

    Списків вісімнадцять, записів у них 350. Апостроф у трьох із них («бʼє» у `VIOLENCE_VERBS`,
    «вбʼю» у `VIOLENCE_VOWS` і `HARM_VOWS`) пишеться так, як у словнику, а зводиться `for_checks`
    при самому визначенні — тобто читає це людина в одному вигляді, а звіряє машина в іншому, і
    саме цей шов і перевіряється.
    """
    entries = getattr(viche, name)
    assert entries, "порожній список нічого не ловить"
    assert [x for x in entries if for_checks(x) != x] == []


def test_dropping_the_apostrophe_collides_with_nothing_measured_in_this_repository():
    """Викидання апострофа — замір, а не смак: по 240 354 словах репозиторію зіткнення рівно одне.

    Це «вбю» і «вбʼю», обидва з ядра самопошкодження, де вони й лежали дублікатом саме через брак
    цього зведення. Тепер дублікат зайвий: одного запису досить на всі три написання. Живе він
    тепер у `HARM_VOWS`, бо мішень із запису виїхала в окремий список (`HARM_SELF`): «вбʼю» без
    неї — це й погроза іншому, і забитий цвях.
    """
    assert for_checks("вб'ю себе") == for_checks("вбю себе") == "вбю себе"
    assert "вбю" in viche.HARM_VOWS and "вб'ю" not in viche.HARM_VOWS
    assert viche.about_self_harm("вб'ю себе") and viche.about_self_harm("ВБЮ СЕБЕ")


def test_folding_is_idempotent_because_the_lists_are_folded_at_definition():
    """Зведення другий раз нічого не міняє — інакше список, зведений при визначенні, поїхав би."""
    for text in PLAIN_VILLAGE + ("МЕНЕ БЄ ЧОЛОВІК", "vbyu susida", "Мене бье чоловiк"):
        assert for_checks(for_checks(text)) == for_checks(text)


def test_a_village_role_typed_in_latin_now_names_a_person_of_this_village():
    """Ролі села записані латиницею («divchyna», «parubok», «shynkar»), а означають вони людину.

    `_people_forms` проганяє список людей тим самим `for_checks`, тож роль читається тим словом,
    яким її кличуть у селі. Це не побічний ефект зведення, а те, чого бракувало: `VOW_TARGETS` уже
    тримає «коваля», «попа» й «старосту» дослівно — роль у цьому селі така сама людина, як Одарка.

    Ціна заміряна на тому самому корпусі з 74 тем: через це смуга заговорила рівно на одній темі
    («…там же дівчина Гриця отруїла, а ми про толоку гомонимо») і на жодній із 24 мирних.
    """
    assert about_accusation("парубок украв коня", _SPEAKERS)
    assert about_accusation("шинкар підпалив клуню", _SPEAKERS)
    assert suspect("коваль отруїв криницю", _SPEAKERS) == "accusation", \
        "непряма основа платить за присуд, а вироку сама не ухвалює"
    assert suspect("парубок косить отаву", _SPEAKERS) is None
    assert suspect("дівчина принесла води", _SPEAKERS) is None
