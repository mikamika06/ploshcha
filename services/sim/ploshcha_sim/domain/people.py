"""Породження селян: роль стала, людина — ні.

Роль — це когнітивний слот системи (лінза, вміння, тяжіння до локації), і поведінка віча залежить
саме від нього, тому набір ролей фіксований. Породжується ЛЮДИНА: норов, імʼя, мова, вигляд.

Поділ праці тут навмисний і той самий, що всюди в цьому ядрі:

  код   — які ролі є, які осі норову, який вигляд      (визначено даними, відтворювано за сідом)
  модель— імʼя, коротка історія, примовка              (це вигадка, і лише вона)
  код   — ознака → вага в партитурі й фільтр вигляду    (щоб ознака ВПЛИВАЛА, а не була підписом)

★ Головне правило: ознака мусить щось міняти в поведінці. Ознака, яка лише написана в промпті, —
наліпка; такі ми вже бачили, коли лінза протікала в репліку замість того, щоб її формувати.
"""

import hashlib
import random

from pydantic import BaseModel, Field

# Осі норову. Значення 0..1; крайнощі рідкісні, бо село з самих диваків не читається як село.
AXES: tuple[str, ...] = ("гарячий", "довірливий", "прийшлий", "старий", "заможний")

# Короткі ключі полюсів — саме вони їдуть у контракт і по них фронт рахує вигляд.
# ★ Назву ОСІ віддавати не можна: вісь «старий» описує і старого, і молодого, тож молода дівчина
# приїжджала на сцену з ознакою «старий» і фарбувалась сивиною. Ознака мусить називати ПОЛЮС.
AXIS_KEYS: dict[str, tuple[str, str]] = {
    "гарячий": ("тихий", "гарячий"),
    "довірливий": ("підозріливий", "довірливий"),
    "прийшлий": ("свій", "прийшлий"),
    "старий": ("молодий", "старий"),
    "заможний": ("бідний", "заможний"),
}

AXIS_POLES: dict[str, tuple[str, str]] = {
    "гарячий": ("тихий і повільний", "гарячий, лізе першим"),
    "довірливий": ("підозріливий, усе перепитує", "довірливий, вірить на слово"),
    "прийшлий": ("свій, тутешній зроду", "прийшлий, у селі недавно"),
    "старий": ("молодий", "старий, багато пам'ятає"),
    "заможний": ("бідний, рахує кожен гріш", "заможний, має що втрачати"),
}

# Скільки ролей виходить на віче з усіх наявних.
VILLAGE_SIZE = 8

# Зміщення осей за РОЛЛЮ. Без нього кубик видавав молодого діда й гарячу шептуху — варіативність
# є, а село не читається. Зміщуємо, але не фіксуємо: діапазон лишається, просто зсунутий.
ROLE_BIAS: dict[str, dict[str, float]] = {
    "did": {"старий": 0.34, "гарячий": -0.18},
    "sheptu": {"старий": 0.28, "гарячий": -0.14, "довірливий": -0.20},
    "parubok": {"старий": -0.36, "гарячий": 0.30},
    "divchyna": {"старий": -0.30},
    "mati": {"старий": -0.06, "гарячий": -0.16},
    "shynkar": {"заможний": 0.24, "довірливий": -0.16},
    "koval": {"гарячий": 0.10},
    "mirosh": {"заможний": 0.12},
    "starosta": {"старий": 0.20, "гарячий": -0.16},
    "pip": {"старий": 0.18, "гарячий": -0.24, "довірливий": -0.12},
    "chumak": {"прийшлий": 0.34},
    "diak": {"довірливий": -0.10},
}


class Trait(BaseModel):
    """Одна вісь норову з числом. Число, а не слово, бо з нього рахуються ваги й вигляд."""

    axis: str
    value: float

    @property
    def pole(self) -> str:
        low, high = AXIS_POLES[self.axis]
        return high if self.value >= 0.5 else low

    @property
    def key(self) -> str:
        """Короткий ключ полюса — для контракту й для вигляду."""
        low, high = AXIS_KEYS[self.axis]
        return high if self.value >= 0.5 else low

    @property
    def strength(self) -> float:
        """Наскільки виражена: 0 — посередині, 1 — на краю."""
        return abs(self.value - 0.5) * 2


class Person(BaseModel):
    role: str
    name: str = ""
    bio: str = ""
    saying: str = ""
    traits: dict[str, float] = Field(default_factory=dict)

    def trait(self, axis: str) -> Trait:
        return Trait(axis=axis, value=self.traits.get(axis, 0.5))

    @property
    def marked(self) -> list[Trait]:
        """Лише виражені ознаки — решта шум, який не варто ні згадувати, ні рахувати."""
        out = [self.trait(a) for a in AXES]
        return sorted([t for t in out if t.strength >= 0.34], key=lambda t: -t.strength)


def _rng(seed: int, salt: str) -> random.Random:
    return random.Random(f"{seed}:{salt}")


def roll_traits(seed: int, role: str) -> dict[str, float]:
    """Норов кидає КОД. Трикутний розподіл: середина частіша за краї, тож село з людей, а не з
    карикатур; але хвости лишаються, інакше всі виходять однаково сірі.

    Роль ЗМІЩУЄ осі, не задає їх: дід схильний бути старим, парубок — гарячим, чумак — прийшлим.
    Без цього перший же живий прогін дав молодого діда, і село перестало читатись.
    """
    rng = _rng(seed, f"traits:{role}")
    bias = ROLE_BIAS.get(role, {})
    out: dict[str, float] = {}
    for axis in AXES:
        base = (rng.random() + rng.random()) / 2
        out[axis] = round(min(1.0, max(0.0, base + bias.get(axis, 0.0))), 3)
    return out


def village_roles(seed: int, roles: list[str], size: int = VILLAGE_SIZE) -> list[str]:
    """Хто взагалі живе в цьому селі. Той самий сід — те саме село."""
    picked = sorted(roles, key=lambda r: hashlib.sha256(f"{seed}:{r}".encode()).digest())
    return sorted(picked[:max(2, min(size, len(roles)))], key=roles.index)


def look(person: Person) -> str:
    """Ознаки → CSS-фільтр спрайта.

    Нових малюнків не треба: вигляд рахується з норову. Старший сивіший, заможніший насиченіший,
    прийшлий іншого відтінку. Значення стримані навмисно — село має лишатись цілісним, а не
    перетворитись на набір різнокольорових фігурок.
    """
    old = person.trait("старий").value
    rich = person.trait("заможний").value
    alien = person.trait("прийшлий").value
    sat = round(0.72 + rich * 0.34 - old * 0.16, 3)
    bright = round(1.02 - old * 0.12 + rich * 0.05, 3)
    hue = round((alien - 0.5) * 26)
    sepia = round(0.06 + old * 0.16, 3)
    return f"saturate({sat}) sepia({sepia}) brightness({bright}) hue-rotate({hue}deg)"


def pace(person: Person) -> float:
    """Швидкість ходи: гарячий поспішає, старий бреде. Теж наслідок норову, а не окреме поле."""
    return round(1.0 + person.trait("гарячий").value * 0.45 - person.trait("старий").value * 0.35, 3)


def beat_weights(person: Person) -> dict[str, float]:
    """★ Заради чого все це: ознака міняє ПАРТИТУРУ, а не лише підпис.

    Гарячий частіше перебиває й каже коротше; підозріливий частіше сумнівається; довірливий
    швидше несе чутку; старий частіше згадує.
    """
    return {
        "перебити": round(0.6 + person.trait("гарячий").value * 1.4, 2),
        "засумніватись": round(0.6 + (1 - person.trait("довірливий").value) * 1.4, 2),
        "згадати": round(0.5 + person.trait("старий").value * 1.5, 2),
        "піддакнути": round(0.5 + person.trait("довірливий").value * 1.2, 2),
    }


def remembers(person: Person) -> bool:
    """★ Прийшлий НЕ має доступу до памʼяті села — і тому бачить те, чого свої вже не помічають.

    Це не флейвор: у пакет такої людини не кладуть минулих віче, отже вона справді міркує з нуля.
    """
    return person.trait("прийшлий").value < 0.62


NAME_MAX = 24
BIO_MAX = 120
SAYING_MAX = 60


def people_schema(roles: list[str]) -> dict:
    """Схема для однієї ланки, що робить із осей норову живих людей.

    Енуми й `maxLength` тут не обережність, а наслідок замірів: числове поле вішає генерацію
    (`finish=stop` і нескінченні пробіли), а необмежене текстове зриває стелю (`finish=length`,
    4 успіхи з 8). Тому жодного числа в схемі й межа на кожен рядок.
    """
    return {
        "type": "object",
        "properties": {
            "люди": {
                "type": "array",
                "minItems": len(roles),
                "maxItems": len(roles),
                "items": {
                    "type": "object",
                    "properties": {
                        "роль": {"type": "string", "enum": list(roles)},
                        "імʼя": {"type": "string", "maxLength": NAME_MAX},
                        "про_себе": {"type": "string", "maxLength": BIO_MAX},
                        "примовка": {"type": "string", "maxLength": SAYING_MAX},
                    },
                    "required": ["роль", "імʼя", "про_себе", "примовка"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["люди"],
        "additionalProperties": False,
    }


def describe(role: str, traits: dict[str, float], lens: str) -> str:
    """Замовлення на людину: роль, лінза й лише ВИРАЖЕНІ ознаки.

    Невиражені не згадуємо взагалі — інакше модель починає їх відпрацьовувати, і всі виходять
    однаково «врівноважені», тобто нецікаві.
    """
    person = Person(role=role, traits=traits)
    marked = ", ".join(t.pole for t in person.marked) or "нічим особливо не вирізняється"
    return f"- {role}: {lens}. Норов: {marked}"


# ★ Стать ролі — властивість СПРАЙТА, а не тексту.
#
# Малюнок для кожної ролі фіксований: `did` — дід, `mati` — молодиця. Імена ж вигадує модель, і
# вона регулярно давала жіноче імʼя чоловічій фігурі («дід Свирид: як я ще дівкою була»). Це
# видно на екрані як розсинхрон малюнка й підпису, тому вирішує КОД, а не модель.
GENDER_BY_ROLE = {
    "did": "ч", "koval": "ч", "mirosh": "ч", "parubok": "ч", "pip": "ч",
    "starosta": "ч", "chumak": "ч", "diak": "ч",
    "sheptu": "ж", "shynkar": "ж", "mati": "ж", "divchyna": "ж",
}

# Слова-титули перед імʼям: рід визначає саме імʼя, а не «дід» чи «баба».
_TITLES = {"дід", "баба", "бабка", "пан", "пані", "отець", "тітка", "кум", "кума", "дядько",
           "молодиця", "старий", "стара"}
# Чоловічі імена на -а/-я — виняток із правила, тож перелічені явно.
_MALE_A = {"микола", "ілля", "сава", "кузьма", "хома", "лука", "юхим", "гаврило", "гаврила",
           "данило", "михайло", "марко", "петро", "павло", "дмитро", "богдан", "левко", "василь"}
# Запасні імена, коли модель промахнулась статтю. Беруться за роллю, тобто стабільно.
_FALLBACK = {
    "did": "дід Свирид", "koval": "Остап", "mirosh": "Панас", "parubok": "Іван",
    "pip": "отець Тарас", "starosta": "Гнат", "chumak": "Мирон", "diak": "Юхим",
    "sheptu": "баба Горпина", "shynkar": "Одарка", "mati": "Марія", "divchyna": "Оксана",
}


def name_gender(name: str) -> str | None:
    """`ч`/`ж` за самим імʼям; `None` — не беремось судити.

    Правило просте й українське: імʼя на -а/-я жіноче, крім переліку чоловічих винятків
    (Микола, Ілля, Сава…). Титул попереду відкидаємо — «дід Марія» має ловитись саме як помилка.
    """
    parts = [w for w in name.replace("’", "'").split() if w]
    while parts and parts[0].lower().strip(".,") in _TITLES:
        parts.pop(0)
    if not parts:
        return None
    first = parts[0].lower().strip(".,'\"")
    if first in _MALE_A:
        return "ч"
    if first.endswith(("а", "я")):
        return "ж"
    return "ч"


def fit_gender(role: str, name: str) -> str:
    """Імʼя, що не свариться з малюнком. Промах моделі міняємо на сталу заміну за роллю."""
    want = GENDER_BY_ROLE.get(role)
    if want is None or not name:
        return name
    got = name_gender(name)
    if got is None or got == want:
        return name
    return _FALLBACK.get(role, name)


def repair_people(raw: dict | None, roles: list[str],
                  traits: dict[str, dict[str, float]]) -> list[Person]:
    """Лагодить КОД: чужа роль, порожнє імʼя, дублікат — усе відкидається, норов лишається наш.

    Норов моделі не віддаємо взагалі: він визначений кубиком, і дозволити його переписати означало б
    віддати моделі те, що вже вирішено даними.
    """
    out: list[Person] = []
    seen: set[str] = set()
    for item in (raw or {}).get("люди") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("роль") or "")
        name = " ".join(str(item.get("імʼя") or "").split())
        if role not in roles or role in seen or not name:
            continue
        seen.add(role)
        out.append(Person(
            role=role, name=fit_gender(role, name)[:NAME_MAX],
            bio=" ".join(str(item.get("про_себе") or "").split())[:BIO_MAX],
            saying=" ".join(str(item.get("примовка") or "").split())[:SAYING_MAX],
            traits=traits.get(role, {}),
        ))
    return out
