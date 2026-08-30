"""Ярус промовляння: кому дістається слот `speak` — і чому в проді він лишається дешевим.

Минулий круг переграв збережені пакети по одному виклику на такт і назвав дорогий ярус ДЕШЕВШИМ
(16 813 токенів проти 17 539 на плечі `mark`), бо всі репліки проходять із першої спроби. Оцінку
перевірено вісьмома ПОВНИМИ живими вічами в прод-умові (сіди 1 і 2, теми «вовк» і «мито», кеш
розбито однаково в обох плечах; сирі дані — `tier-live-{lapa,mamay}-{1,2}.json` і лічильники
`tier-live-count.py` у `docs/research/eval-runs/`), і вона не справдилась: ремонт справді зникає (спроб
мовця на такт 1.19 → 1.00, ремонтів 15 → 0, ескалацій 6 → 0, дефектних причин голосу 9 із 24 → 0),
але прогін дорожчає — 60 147 → 73 628 токенів (+22.4%), 223.2 → 342.9 с, — а головна метрика
падає: пар без жодної ознаки звʼязку 53.5% → 61.3%. Підхоплення «як є» зростає (9.9% → 14.7%), та
на вирівняній довжині обертається програшем (зріз обох реплік до 12 слів: 7.0% → 2.7%), бо метрика
є функцією довжини, а дорогий ярус говорить удвічі довше (23.1 слова проти 12.7).

Тому важіль є, а дефолт вимкнений — і стережеться тут саме це: вимкнений лишає прод байт-у-байт
таким, як був; ввімкнений переставляє РІВНО один слот; назва яруса їде разом із моделлю, інакше
звіт підписав би дорогий виклик дешевим ярусом.
"""

from evalkit.conditions import CONDITIONS
from ploshcha_sim.adapters import FakeLlm
from ploshcha_sim.compose import build_viche, build_viche_router
from ploshcha_sim.domain.spec import AppSpec
from ploshcha_sim.domain.task import Budget
from ploshcha_sim.domain.viche import cast_for

from test_viche import NEWS, WaveLlm, beat, lines, score

CHEAP, RICH = "cheap", "rich"


def _talk(**changes):
    """Віче на двох різних фейках: у кожного своя назва моделі, тож видно, хто саме говорив."""
    trio = [p.role for p in cast_for(NEWS, 3)]
    sc = score(beat(trio[0], "згадати"), beat(trio[1], "порахувати", reply=1),
               beat(trio[2], "пожалітись", reply=1))
    script = [sc] * 4 + lines(14)
    lapa, mamay = WaveLlm(list(script), model=CHEAP), WaveLlm(list(script), model=RICH)
    spec = CONDITIONS["viche"].with_(max_width=3, viche_sense=False, **changes)
    agent = build_viche(spec, lapa=lapa, mamay=mamay)
    result = agent.run(NEWS, seed=1, budget=Budget(max_steps=30, max_tokens=99_999))
    return lapa, mamay, result


def _router(**changes):
    lapa, mamay = FakeLlm([""], model=CHEAP), FakeLlm([""], model=RICH)
    return build_viche_router(CONDITIONS["viche"].with_(**changes), lapa=lapa, mamay=mamay)


def test_the_speaking_stays_on_the_cheap_tier_while_the_lever_is_off():
    """Дефолт зберігає теперішню поведінку — інакше жоден порахований прогін не порівнюваний.

    Це не формальність: рівно цим важелем знято плече заміру, тож якби він стояв ввімкненим сам
    собою, прод поїхав би на ярус, який на живих вічах коштує +22.4% і дає гіршу головну метрику.
    """
    router = _router()
    assert router.route("speak").model == CHEAP
    assert router.lane("speak") == "lapa"

    lapa, _, result = _talk()
    assert result.tokens_by_stage_lane.get("speak|lapa"), result.tokens_by_stage_lane
    assert "speak|mamay" not in result.tokens_by_stage_lane
    assert lapa.calls, "репліки мусять іти дешевим ярусом"


def test_the_lever_moves_the_speaking_slot_to_the_expensive_tier():
    """Ввімкнений важіль — це те плече, на якому знято числа; без нього його нічим зібрати."""
    router = _router(viche_reply_lane="mamay")
    assert router.route("speak").model == RICH
    assert router.lane("speak") == "mamay"


def test_the_lane_name_travels_with_the_model_into_the_report():
    """★ Мапа й назва яруса переставляються РАЗОМ, бо звіт читають за назвою, а не за моделлю.

    `lane` іде в гаманець (`Budget.spend`) і в трасу. Переставивши саму мапу, ми дістали б прогін,
    у якому дорогий виклик оплачений і підписаний як дешевий, — тобто ціна яруса (408 токенів на
    виклик проти 273 на живих вічах) не була б видима у звіті взагалі.
    """
    lapa, mamay, result = _talk(viche_reply_lane="mamay")

    assert result.tokens_by_stage_lane.get("speak|mamay"), result.tokens_by_stage_lane
    assert "speak|lapa" not in result.tokens_by_stage_lane
    assert not lapa.calls, "дешевому ярусу не лишається жодного виклику віча"
    assert mamay.calls


def test_the_lever_touches_the_speaking_slot_and_nothing_else():
    """Плечі мусять різнитись РІВНО одним: інакше замір міряв би дві осі одразу.

    Суддя тут окремо названий не для симетрії: інваріант «ніколи Lapa-судить-Lapa» тримається саме
    на тому, що ярус судді — власна вісь (`build_router`, `judge_lane`), і зсув промовляння не має
    права його зачепити.
    """
    router = _router(viche_reply_lane="mamay")
    for kind in ("judge", "decide", "generate", "synthesize"):
        assert router.route(kind).model == RICH, kind
    for kind in ("parse", "classify", "select", "ground", "gate"):
        assert router.route(kind).model == CHEAP, kind
        assert router.lane(kind) == "lapa", kind

    back = _router(viche_reply_lane="lapa")
    assert back.route("speak").model == CHEAP and back.lane("speak") == "lapa"


def test_the_lever_is_an_axis_of_the_run_not_an_ornament():
    """Рядок усередині агента був би невидимий ні в `sha256` умови, ні у звіті.

    Той самий припис, що вже стоїть на штрафі повторення, згасанні ланцюга, суміжній парі й голому
    пакеті: важіль, який не входить у відбиток умови, робить два різні прогони однойменними.
    """
    assert AppSpec().viche_reply_lane == "", "дефолт зберігає теперішню поведінку"
    assert AppSpec().sha256 != AppSpec().with_(viche_reply_lane="mamay").sha256
    assert (CONDITIONS["viche"].sha256
            != CONDITIONS["viche"].with_(viche_reply_lane="mamay").sha256)


def test_the_prod_condition_keeps_speaking_cheap_and_the_reason_is_measured():
    """★ Прод лишається на дешевому ярусі — і це запис заміру, а не мовчазний дефолт.

    Числа підстави — вісім живих віч у цій самій умові: 60 147 → 73 628 токенів (+22.4%, і жодне
    дороге віче не дешевше за жодне дешеве: 17 272-21 231 проти 14 376-15 290), пар без ознаки
    звʼязку 53.5% → 61.3%, підхоплення на зрізі обох реплік до 12 слів 7.0% → 2.7%. Виграш
    дорогого яруса реальний, але лежить в іншому місці (ремонтів 15 → 0, ескалацій 6 → 0,
    дефектних причин голосу 9 із 24 → 0, огризків ≤ 6 слів 17 → 0), і коштує він 13 481 токен на
    вісім віч, тоді як увесь ремонт дешевого яруса важить ≈ 7 700.
    """
    for name in ("viche", "viche-notools"):
        assert CONDITIONS[name].viche_reply_lane == "", name
