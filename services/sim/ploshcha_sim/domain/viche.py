"""Віче — розмова села, а не задача з відповіддю.

Чим це принципово відрізняється від решти домену: тут **немає правильної відповіді**, отже немає й
`abstain`. Гейт `outcome_of` («шукав і не знайшов → відмова») — інструмент дослідницького режиму, і в
розмові він дає рівно те, що ми й побачили живцем: на кожну тему «нема в довіднику».

Три речі вирішені тут, у коді, а не моделлю:

1. **Склад учасників** — від хешу теми. Дані визначають крок повністю, отже це не робота моделі; та
   сама новина завжди збирає тих самих людей, інакше повтор виглядав би як інша подія.
2. **Збурення** (перебивки) — кубик за сідом. Спонтанність НЕ береться з виконавця: Lapa вибирати не
   вміє, це виміряно, і ставити її туди означало б свідомо ламати систему.
3. **Валідність партитури** — схемою, а не вірою: `хто` і `хід` це енуми з дозволених значень, тож
   такт із неіснуючим селянином неможливо навіть висловити. Той самий прийом, що в `E-locked`.
"""

import hashlib
import random

from pydantic import BaseModel, Field

# Довжина розмови тримається СХЕМОЮ, а не проханням у промпті. Перший живий замір: `minItems=4`
# давав рівно 4-5 тактів — модель виконує мінімум, бо мінімум це і є вимога, а `maxItems` лише дозвіл.
# Те саме правило, що з `tool` у `E-locked`: звʼязує схема, не проза.
MAX_BEATS = 20
MIN_BEATS = 12
INTERRUPT_P = 0.28
MAX_INTERRUPTS = 4
# Частка одного голосу. Живий замір: партитура віддала Іванові 6 тактів із 15 — він же дав і 5 з 5
# повторів. Балансує КОД: скільки разів людині говорити — визначено складом, не судженням.
MAX_SHARE = 0.34


class Persona(BaseModel):
    """Селянин = когнітивна конфігурація, а не стиль тексту: роль + лінза, крізь яку він дивиться."""

    role: str
    name: str
    lens: str


# Імена мусять збігатися з тим, що ПІДПИСАНО на екрані (каст фронта), інакше ядро пише в репліку
# «та що ти, Гнате», а глядач бачить над тим самим спрайтом «Панас» — вигадка ламається на очах.
PERSONAS: tuple[Persona, ...] = (
    Persona(role="did", name="дід Свирид", lens="памʼять: чи бувало таке раніше, як тоді обійшлось"),
    Persona(role="sheptu", name="баба Горпина", lens="прикмета й пересторога: до чого воно йдеться"),
    Persona(role="koval", name="Остап", lens="діло: що робити руками вже завтра"),
    Persona(role="mirosh", name="Панас", lens="погода, врожай і лічба наслідків"),
    Persona(role="shynkar", name="Одарка", lens="гроші й поголос: кому це вигідно"),
    Persona(role="mati", name="Марія", lens="родина: як це вдарить по дітях і хаті"),
    Persona(role="parubok", name="Іван", lens="гаряче й коротко, лізе перший, без розважань"),
    Persona(role="divchyna", name="Оксана", lens="люди й чутки: хто що казав і кому вірити"),
)

BY_ROLE = {p.role: p for p in PERSONAS}


def public_cast(people: list | None = None) -> list[dict]:
    """Склад, який ядро ОГОЛОШУЄ сцені.

    Доти вісім селян приходили з фікстури `quiet-day.jsonl`, а не від ядра — і це був корінь цілого
    класу дефектів: староста з попом для сцени не існували (їхні слова чути, бульбашки нема), а імена
    розходились (ядро «мірошник Гнат», екран «Панас»), тож я латав це вирівнюванням імен руками.
    Коли склад оголошує ядро, розійтись неможливо за побудовою.
    """
    generated = {x.role: x for x in (people or [])}
    folk = [*(personas_from(people) if people else PERSONAS), STAROSTA, PIP, GUEST]
    out = []
    for p in folk:
        person = generated.get(p.role)
        entry = {"id": p.role, "name": p.name,
                 "role": "chumak" if p is GUEST else p.role,
                 "bio": (person.bio if person and person.bio else p.lens)}
        if person is not None:
            # Ознаки їдуть у контракт РЯДКАМИ, а вигляд із них рахує фронт. Так ядро не вигадує
            # кольорів, а сцена не вигадує норову — кожен робить своє.
            entry["traits"] = [t.key for t in person.marked]
        out.append(entry)
    return out

# Ти — не бог над селом, а сусід у гурті: маєш роль, спрайт і голос, як усі. Роль `chumak` («той,
# хто прийшов»), бо гість справді дивиться на село свіжим оком.
GUEST = Persona(role="hist", name="ти",
                lens="сторонній голос із гурту: питає й заперечує, коли має що сказати")

STAROSTA = Persona(role="starosta", name="староста",
                   lens="громада: звести сказане й назвати, до чого дійшли")
PIP = Persona(role="pip", name="піп",
              lens="сумнів: чи не сказав хтось зайвого без підстави")

ANSWER_GUEST = "відповісти_гостю"
# Скільки людей відгукуються на твоє слово. Двоє, а не всі: інакше кожна твоя репліка спиняла б
# розмову й перетворювала віче на розмову з тобою.
GUEST_REPLIES = 2

# ★ «піти_питати» ПРИБРАНО. Заміряно на 57 живих вічах: `viche_scout_empty` 288 разів — тобто
# посланий майже завжди вертався ні з чим, а хід усе одно коштував дитину-агента з власним
# циклом. Хід, який нічого не приносить, — не хід, а витрата.
MOVES: tuple[str, ...] = (
    "згадати", "засумніватись", "спитати_діло", "порахувати",
    "пожалітись", "пожартувати", "піддакнути", "заперечити",
)

# Дія тіла на такті. Це ПОСТАНОВКА, а не рішення: вибір лишається закритим списком, тож його
# можна довірити моделі, не даючи їй жодної влади над змістом розмови.
DEEDS: tuple[str, ...] = (
    "стоїть", "підходить", "відступає", "ходить", "розводить_руками", "відвертається",
)
# Якщо Мамай дії не дав — беремо ту, що випливає з самого ходу. Крок, визначений даними,
# належить коду, а не моделі.
DEED_OF_MOVE: dict[str, str] = {
    "згадати": "стоїть",
    "засумніватись": "відступає",
    "спитати_діло": "підходить",
    "порахувати": "стоїть",
    "пожалітись": "розводить_руками",
    "пожартувати": "стоїть",
    "піддакнути": "підходить",
    "заперечити": "підходить",
}

MOVE_HINT: dict[str, str] = {
    "згадати": "пригадай схожий випадок з минулого села",
    "засумніватись": "усумнись у сказаному, але не грубо",
    "спитати_діло": "спитай, що робити практично",
    "порахувати": "прикинь наслідки в числах або в мірах",
    "пожалітись": "поскаржся, як це вдарить по тобі",
    "пожартувати": "збий напругу коротким жартом",
    "піддакнути": "погодься й додай своє",
    "заперечити": "не погодься й скажи чому",
    ANSWER_GUEST: "відгукнись на щойно сказане — погодься або зваж",
}

INTERRUPT_MOVE = "перебити"
SUMMARY_MOVE = "підсумувати"
DOUBT_MOVE = "засумніватись_вголос"


class Beat(BaseModel):
    """Один такт партитури. Слова тут НЕМАЄ — слова породжує виконавець."""

    хто: str
    хід: str
    дія: str = ""
    у_відповідь: int | None = None
    інструмент: str | None = None
    запит: str | None = None


class Score(BaseModel):
    такти: list[Beat] = Field(default_factory=list)


def personas_from(people: list) -> tuple[Persona, ...]:
    """Породжені люди → персони віча: імʼя й примовка від моделі, ЛІНЗА лишається від ролі.

    Лінза не породжується: вона визначає, як людина дивиться на новину, тобто це слот системи.
    Породжується той, хто крізь неї дивиться.
    """
    out = []
    for person in people:
        base = BY_ROLE.get(person.role)
        out.append(Persona(role=person.role, name=person.name or person.role,
                           lens=base.lens if base else person.bio or person.role))
    return tuple(out)


# Хід → що він робить зі стосунками. Виводиться з ПАРТИТУРИ: питати про це модель означало б
# платити за відповідь, яку ми вже маємо.
BOND_OF_MOVE: dict[str, float] = {
    "піддакнути": 1.0, "згадати": 0.2, "пожартувати": 0.3,
    "заперечити": -1.0, "засумніватись": -0.6, INTERRUPT_MOVE: -0.4,
}


def bonds_from(beats: list[Beat]) -> list[tuple[str, str, float]]:
    """Хто кому піддакнув — зблизились; хто заперечив — розійшлись."""
    out: list[tuple[str, str, float]] = []
    for i, beat in enumerate(beats):
        target = beat.у_відповідь
        if not target or not (1 <= target <= i):
            continue
        other = beats[target - 1].хто
        delta = BOND_OF_MOVE.get(beat.хід, 0.0)
        if delta and other != beat.хто:
            out.append((beat.хто, other, delta))
    return out


def interrupt_chance(person) -> float:
    """Ймовірність перебивки для конкретної людини.

    ★ Ось заради чого норов узагалі існує: гарячий перебиває частіше, тихий майже ніколи. Якби
    ознака лишалась написом у промпті, вона не міняла б нічого.
    """
    from .people import beat_weights

    return min(0.55, INTERRUPT_P * beat_weights(person)["перебити"])


def cast_for(topic: str, size: int) -> list[Persona]:
    """Склад від хешу теми: детерміновано й без моделі. Та сама новина — ті самі люди."""
    size = max(2, min(size, len(PERSONAS)))
    digest = hashlib.sha256(topic.strip().encode("utf-8")).digest()
    order = sorted(PERSONAS, key=lambda p: hashlib.sha256(digest + p.role.encode()).digest())
    return sorted(order[:size], key=lambda p: PERSONAS.index(p))


MOODS = ("радість", "тривога", "спокій", "туга", "піднесення")
# ★ Сила — ЕНУМ, а не число. Замір на живому шлюзі: поле `{"type": "number"}` вішає генерацію —
# модель віддає `"сила": ` і далі нескінченні пробіли, `finish=stop` на 113 токенах, JSON не
# розбирається. Підняття стелі не допомагає, бо це не обрізання. Той самий наш закон: звʼязує енум.
FORCES = ("ледь", "помірно", "дуже")
# Куди ухвала ставить людину. Лише POI, які СПРАВДІ є на сцені: місце, якого нема, дало б рішення
# без наслідку — тобто знову «намальовану» механіку.
DECISION_POIS = ("ploshcha", "kuznya", "mlyn", "shynok", "tserkva", "doshka", "dzvin", "stavok")
FORCE_VALUE = {"ледь": 0.35, "помірно": 0.7, "дуже": 1.0}


def chronicle_schema(roles: list[str]) -> dict:
    """Хроніка дня: заголовок, оповідь, настрій, ухвала, чутка.

    Думок тут БІЛЬШЕ НЕМАЄ — вони пішли в окремий маленький виклик (`thoughts_schema`), бо в
    спільній схемі їх зрізало разом із хвостом відповіді: 19 доїздів із 57.
    """
    return {
        "type": "object",
        "properties": {
            "заголовок": {"type": "string"},
            "оповідь": {"type": "string"},
            "настрій": {"type": "string", "enum": list(MOODS)},
            "сила": {"type": "string", "enum": list(FORCES)},
            # Чутка — твердження БЕЗ підстави, сказане вголос. Їде тією ж ланкою, що хроніка й
            # ухвала: усе це судження про ту саму розмову, і платити за нього тричі нема за що.
            "чутка": {
                "type": "object",
                "properties": {
                    "є": {"type": "string", "enum": ["так", "ні"]},
                    "хто": {"type": "string", "enum": list(roles)},
                    "що": {"type": "string", "maxLength": 90},
                    "підстава": {"type": "string", "enum": ["була", "не було"]},
                },
                "required": ["є", "хто", "що", "підстава"],
                "additionalProperties": False,
            },
            # Ухвала їде в ТІЙ САМІЙ ланці, що хроніка: окремий виклик коштував би стільки ж, а
            # рішення — це судження про ту саму розмову. `хто`/`де` енумами, текст із межею: обидва
            # обмеження — з заміру, не з обережності.
            "ухвала": {
                "type": "object",
                "properties": {
                    "ухвалено": {"type": "string", "enum": ["так", "ні"]},
                    "що": {"type": "string", "maxLength": 90},
                    "хто": {"type": "string", "enum": list(roles)},
                    "де": {"type": "string", "enum": list(DECISION_POIS)},
                },
                "required": ["ухвалено", "що", "хто", "де"],
                "additionalProperties": False,
            },
        },
        "required": ["заголовок", "оповідь", "настрій", "сила", "ухвала", "чутка"],
        "additionalProperties": False,
    }


def thoughts_schema(roles: list[str]) -> dict:
    """Думки — ОКРЕМИМ, маленьким викликом.

    ★ Заміряно на 57 живих вічах: літопис доїжджав 57 разів, а думки лише 19. Вони стояли останнім
    полем найбільшої відповіді прогону — і саме їх зрізало першими, щойно вивід уривався. Схема на
    два поля ламається значно рідше за схему на сім, і коштує вона копійки проти самої розмови.
    """
    return {
        "type": "object",
        "properties": {
            "думки": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(roles),
                "items": {
                    "type": "object",
                    "properties": {"хто": {"type": "string", "enum": list(roles)},
                                   "думка": {"type": "string", "maxLength": 160}},
                    "required": ["хто", "думка"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["думки"],
        "additionalProperties": False,
    }


VALENCE = {"радість": 0.7, "піднесення": 0.5, "спокій": 0.1, "тривога": -0.5, "туга": -0.8}


def mood_view(label: str, force: str | float) -> dict:
    """Настрій у формі контракту. Знак дає ЯРЛИК, силу — слово: інакше модель віддавала б
    «тривога» з додатною валентністю, і погода на сцені суперечила б тексту."""
    base = VALENCE.get(label, 0.0)
    if isinstance(force, str):
        strength = FORCE_VALUE.get(force, 0.7)
    else:
        strength = min(1.0, max(0.0, abs(float(force or 0.7))))
    value = base if strength == 0 else (abs(base) * strength) * (1 if base >= 0 else -1)
    return {"valence": round(max(-1.0, min(1.0, value)), 2), "label": label}


def plan_steps(beats: list["Beat"]) -> list[str]:
    """Партитура людською мовою — щоб її можна було ПОВІСИТИ на Дошці, а не читати в логу."""
    out: list[str] = []
    for i, b in enumerate(beats, start=1):
        who = BY_ROLE.get(b.хто)
        name = who.name if who else b.хто
        move = MOVE_HINT.get(b.хід, b.хід).split(",")[0]
        reply = f" (у відповідь на {b.у_відповідь})" if b.у_відповідь else ""
        out.append(f"{i}. {name} — {move}{reply}")
    return out


def score_schema(roles: list[str], tools: list[str],
                 beats: tuple[int, int] | None = None) -> dict:
    """`хто`/`хід` — енуми. Такт із чужим селянином або вигаданим ходом висловити неможливо."""
    beat = {
        "type": "object",
        "properties": {
            "хто": {"type": "string", "enum": list(roles)},
            "хід": {"type": "string", "enum": list(MOVES)},
            # Дія тіла на цьому такті: закритий енум, тож «вигадати» жест неможливо.
            "дія": {"type": "string", "enum": list(DEEDS)},
            "у_відповідь": {"type": ["integer", "null"]},
            "інструмент": {"type": ["string", "null"], "enum": [*tools, None]},
            # ★ `maxLength` тут не косметика. Замір на 8 сідах: без нього партитура зривалась у
            # 4/8 випадків із `finish=length` — модель писала в «запит» цілі речення й вибивала
            # стелю. Це ОБРІЗАННЯ, не зависання на типі (хроніка зависала інакше — `finish=stop`
            # і нескінченні пробіли на числовому полі). Два різні дефекти, і плутати їх не можна.
            "запит": {"type": ["string", "null"], "maxLength": 60},
        },
        "required": ["хто", "хід", "дія", "у_відповідь", "інструмент", "запит"],
        "additionalProperties": False,
    }
    low, high = beats or (MIN_BEATS, MAX_BEATS)
    return {
        "type": "object",
        "properties": {"такти": {"type": "array", "minItems": low,
                                 "maxItems": high, "items": beat}},
        "required": ["такти"],
        "additionalProperties": False,
    }


# ── позиції: віче мусить ЩОСЬ вирішувати ────────────────────────────────────────────────────
#
# Доти розмова нічого не міняла: ухвала наприкінці була переказом стенограми одним викликом, тобто
# резюме, а не результатом. Тепер у кожного є ПОЗИЦІЯ як дані, і її рухає код за тим, що сталось у
# такті. Рішення випадає з підрахунку, а не з судження моделі про власну ж розмову.
STANCE_MIN, STANCE_MAX = -1.0, 1.0
VOTES: tuple[str, ...] = ("за", "проти", "утримуюсь")


def stance_label(v: float) -> str:
    if v >= 0.34:
        return "за"
    if v <= -0.34:
        return "проти"
    return "вагається"


def stance_start(roles: list[str]) -> dict[str, float]:
    return {r: 0.0 for r in roles}


def stance_after(beat: Beat, stances: dict[str, float], standing: dict[str, float] | None,
                 fact_found: bool | None) -> dict[str, float]:
    """Як такт зрушив позиції. Правила ДЕТЕРМІНОВАНІ: це визначено даними, тож належить коду.

    Вага мовця — його репутація: кому спростували чутку, того й слухають менше. Той самий принцип,
    що вже ріже йому час слова, тут ріже і вплив на чужу думку.
    """
    out = dict(stances)
    who = beat.хто
    if who not in out:
        return out
    w = max(0.4, min(1.4, (standing or {}).get(who, 1.0)))

    def move(role: str, delta: float) -> None:
        out[role] = max(STANCE_MIN, min(STANCE_MAX, out.get(role, 0.0) + delta))

    move_kind = beat.хід
    if move_kind == "заперечити":
        move(who, -0.34 * w)
    elif move_kind == "піддакнути":
        move(who, +0.30 * w)
    elif move_kind == "засумніватись":
        move(who, -0.16 * w if out.get(who, 0.0) > 0 else +0.10 * w)
    elif move_kind == "пожалітись":
        move(who, -0.26 * w)
    elif move_kind == "спитати_діло":
        move(who, +0.08 * w)
    elif move_kind in ("згадати", "порахувати"):
        # Факт важить більше за слово: знайдена довідка тягне мовця до «за», ненайдена — назад.
        move(who, (+0.28 if fact_found else -0.10) * w)
    return out


def stance_view(stances: dict[str, float], cast_names: dict[str, str] | None = None) -> str:
    """Позиції словами — саме це бачить Мамай, плануючи наступну хвилю."""
    if not stances:
        return ""
    parts = [f"{(cast_names or {}).get(r, r)}: {stance_label(v)}" for r, v in stances.items()]
    return "ПОЗИЦІЇ ЗАРАЗ: " + " | ".join(parts)


def vote_schema() -> dict:
    """Голос — закритий енум плюс короткий рядок причини: рівно те, що виконавець тягне."""
    return {
        "type": "object",
        "properties": {
            "голос": {"type": "string", "enum": list(VOTES)},
            "чому": {"type": "string", "maxLength": 90},
        },
        "required": ["голос", "чому"],
        "additionalProperties": False,
    }


def tally(votes: list[tuple[str, str]]) -> dict:
    """Підрахунок. Ухвала — це число, а не переказ."""
    counts = {v: 0 for v in VOTES}
    for _, v in votes:
        if v in counts:
            counts[v] += 1
    total = sum(counts.values())
    if not total:
        return {"ухвалено": False, "лічба": counts, "підсумок": "віче не дійшло голосу"}
    passed = counts["за"] > counts["проти"]
    return {
        "ухвалено": passed,
        "лічба": counts,
        "підсумок": (f"{'ухвалили' if passed else 'відхилили'}: "
                     f"за {counts['за']}, проти {counts['проти']}, "
                     f"утримались {counts['утримуюсь']}"),
    }


def line_schema() -> dict:
    """★ Жодного поля вибору: ні «хто наступний», ні «чи брати інструмент», ні «чи я впорався».

    Виконавець не вирішує — він перетворює пакет даних на пряму мову. Саме тому це працює з Lapa.
    """
    return {
        "type": "object",
        "properties": {"репліка": {"type": "string"}},
        "required": ["репліка"],
        "additionalProperties": False,
    }


def repair_score(raw: dict | None, roles: list[str], tools: list[str],
                 standing: dict[str, float] | None = None) -> list[Beat]:
    """Лагодить КОД, не модель: чужа роль, невідомий хід, посилання в майбутнє — усе відкидається.

    Схема це вже гарантує на строгому ярусі, але шлюз не завжди строгий, а мовчазний дефект
    інструмента дорожчий за зайву перевірку.
    """
    beats: list[Beat] = []
    quota: dict[str, int] = {}
    base_cap = max(2, int(MAX_BEATS * MAX_SHARE))
    # ★ Репутація = ЧАС СЛОВА. Кому спростували чутку, того слухають менше — не метафорично, а
    # буквально меншою квотою тактів. Рахує код: це визначено даними, а не судженням моделі.
    caps = {r: max(1, round(base_cap * (standing or {}).get(r, 1.0))) for r in roles}
    for item in (raw or {}).get("такти") or []:
        if not isinstance(item, dict):
            continue
        who, move = str(item.get("хто") or ""), str(item.get("хід") or "")
        if who not in roles or move not in MOVES:
            continue
        if quota.get(who, 0) >= caps.get(who, base_cap):
            continue
        quota[who] = quota.get(who, 0) + 1
        reply = item.get("у_відповідь")
        index = int(reply) if isinstance(reply, int) and 1 <= reply <= len(beats) else None
        tool = item.get("інструмент")
        tool = str(tool) if tool in tools else None
        deed = str(item.get("дія") or "")
        if deed not in DEEDS:
            deed = DEED_OF_MOVE.get(move, "стоїть")  # дію, визначену ходом, ставить код
        beats.append(Beat(хто=who, хід=move, дія=deed, у_відповідь=index, інструмент=tool,
                          запит=str(item.get("запит")) if item.get("запит") else None))
        if len(beats) >= MAX_BEATS:
            break
    return beats


def guest_beats(said_index: int, roles: list[str], recent: list[str], seed: int,
                text: str) -> list[Beat]:
    """Хто відгукнеться на слово гостя — вирішує КОД.

    Спокуса була перепланувати всю партитуру Mamay'єм на кожну твою репліку. Це виклик дорогого
    слота на кожне натискання, а вибір тут визначений даними: беремо тих, хто щойно НЕ говорив, щоб
    відповідали не ті самі двоє.
    """
    rng = random.Random(f"{seed}:guest:{said_index}:{text[:40]}")
    fresh = [r for r in roles if r not in recent[-2:]] or list(roles)
    rng.shuffle(fresh)
    return [Beat(хто=r, хід=ANSWER_GUEST, дія="підходить", у_відповідь=said_index)
            for r in fresh[:GUEST_REPLIES]]


def scatter(beats: list[Beat], roles: list[str], seed: int, topic: str,
            people: dict | None = None, heat: float = 1.0,
            bonds: dict | None = None) -> list[Beat]:
    """Збурення партитури кубиком за сідом — джерело спонтанності, яке НЕ вимагає вибору моделі.

    Той самий сід і та сама тема дають ті самі перебивки: несподіванка для глядача, відтворюваність
    для заміру.
    """
    if len(beats) < 2:
        return beats
    rng = random.Random(f"{seed}:{topic}")
    out: list[Beat] = []
    added = 0
    for beat in beats:
        out.append(beat)
        speaker = people.get(beat.хто) if people else None
        chance = (interrupt_chance(speaker) if speaker else INTERRUPT_P) * heat
        if added >= MAX_INTERRUPTS or rng.random() > chance:
            continue
        others = [r for r in roles if r != beat.хто]
        if not others:
            continue
        # Перебиває не хто попало: вага норову вирішує, кому дістанеться слово поперек черги.
        if people:
            pool = [r for r in others if r in people]
            # ★ Стосунки міняють, ХТО влізе поперек: хто вже сварився з мовцем, тягнеться
            # перебити його. Без цього сварка лишалась би записом у базі, а не поведінкою.
            weights = []
            for r in pool:
                w = max(0.05, interrupt_chance(people[r]))
                if bonds is not None:
                    w *= 1.0 + max(0.0, -bonds.get(tuple(sorted((r, beat.хто))), 0.0)) * 0.3
                weights.append(w)
            who = rng.choices(pool, weights=weights)[0] if pool else rng.choice(others)
        else:
            who = rng.choice(others)
        added += 1
        out.append(Beat(хто=who, хід=INTERRUPT_MOVE, дія="підходить", у_відповідь=len(out)))
    return out
