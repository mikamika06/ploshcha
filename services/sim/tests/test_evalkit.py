from evalkit.checks import check, run_checks
from evalkit.dialogue import (
    LENGTH_CEILING,
    PLAY_PICKUP,
    aligned_pickup,
    coherence,
    distinctness,
    leakage,
    numbers,
    overlap,
    retelling,
)
from evalkit.harness import (
    EvalItem,
    load_items,
    orchestrator_runner,
    run_eval,
    single_call_runner,
)
from evalkit.report import aggregate, format_report
from ploshcha_sim.adapters.llm_fake import FakeLlm
from ploshcha_sim.adapters.router_profile import PresetEffort, single_model_router
from ploshcha_sim.adapters.tools_fake import FakeToolbox
from ploshcha_sim.agents.orchestrator import Orchestrator
from ploshcha_sim.domain.task import TaskResult


def _res(answer=None, scratch=None, accepted=False, degraded=False):
    return TaskResult(answer=answer, scratch=scratch or [], accepted=accepted, degraded=degraded)


def test_answer_contains_casefold():
    r = _res(answer="Результат: 309524.")
    assert check({"kind": "answer_contains", "value": "309524"}, r)
    assert not check({"kind": "answer_contains", "value": "999"}, r)


def test_used_tool_and_multi_hop():
    r = _res(scratch=[{"call": {"tool": "calc"}}, {"call": {"tool": "check_date"}}])
    assert check({"kind": "used_tool", "tool": "calc"}, r)
    assert check({"kind": "multi_hop", "n": 2}, r)
    assert not check({"kind": "multi_hop", "n": 3}, r)


def test_abstain_and_no_data_tool():
    r = _res(answer="Привіт!", scratch=[])
    assert check({"kind": "abstain"}, r)
    assert check({"kind": "no_data_tool"}, r)
    r2 = _res(answer="x", scratch=[{"call": {"tool": "lookup_fact"}}])
    assert not check({"kind": "abstain"}, r2)


def test_unknown_check_raises():
    try:
        check({"kind": "nope"}, _res())
        assert False
    except ValueError:
        pass


def test_run_checks_all_keys():
    r = _res(answer="600", scratch=[{"call": {"tool": "calc"}}])
    out = run_checks([{"kind": "used_tool", "tool": "calc"}, {"kind": "answer_contains", "value": "600"}], r)
    assert all(out.values())
    assert len(out) == 2


def test_run_eval_and_report_math():
    items = [
        EvalItem(id="a", category="x", task="t", checks=[{"kind": "answer_contains", "value": "ok"}]),
        EvalItem(id="b", category="y", task="t", checks=[{"kind": "answer_contains", "value": "ok"}]),
    ]
    good = lambda task, seed: _res(answer="ok")  # noqa: E731
    flaky = lambda task, seed: _res(answer="ok" if seed == 0 else "no")  # noqa: E731
    results = run_eval(items, {"good": good, "flaky": flaky}, seeds=[0, 1])
    reps = {r["condition"]: r for r in aggregate(results)}
    assert reps["good"]["success_rate"] == 1.0
    assert reps["good"]["pass_k"] == 1.0
    assert reps["flaky"]["success_rate"] == 0.5
    assert reps["flaky"]["pass_k"] == 0.0
    assert "condition" in format_report(results)


def test_load_starter_items():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "evalkit", "items", "starter.jsonl")
    items = load_items(path)
    assert len(items) >= 12
    assert {i.category for i in items} >= {"tool_needed", "multi_hop", "abstain", "grounding"}


def test_single_call_runner():
    llm = FakeLlm(["309524"])
    run = single_call_runner(llm)
    r = run("скільки 347*892", 0)
    assert r.answer == "309524"
    assert r.scratch == []


def test_orchestrator_runner_integration():
    def make_orch():
        llm = FakeLlm([
            '{"tool": "calc", "expr": "347*892"}',
            '{"tool": "final_answer", "text": "309524"}',
        ])
        return Orchestrator(single_model_router(llm), PresetEffort(), FakeToolbox(), verifier=False)

    run = orchestrator_runner(make_orch)
    r = run("скільки 347*892", 0)
    assert check({"kind": "used_tool", "tool": "calc"}, r)
    assert check({"kind": "answer_contains", "value": "309524"}, r)


# ── зчеплення реплік ──────────────────────────────────────────────────────────
# Прилад був сліпий до головної скарги: `distinct2` 0.93-0.97 на прогонах, де 60% пар реплік не
# мали між собою звʼязку. Ці тести стережуть саме те, чого не міряла різність.

ВОВК = "Кажуть, за річкою бачили вовка, і він унадився до кошари."

ВІДПОВІДІ = [
    "Іван: Сторожу треба ставити при кошарі, і смолоскипи палити цілу ніч.",
    "Панас: Смолоскипи горітимуть, доки смоли стане, а сторожа проситиме хліба.",
    "Одарка: Панасе, а хто ту сторожу годуватиме — ти чи громада?",
    "Панас: Та ні, Одарко, не я один, бо й вівці не мої одні.",
]

МОНОЛОГИ = [
    "Іван: Пам'ятаю, торік дощі позаливали ярину, і батько казав орати глибше.",
    "Одарка: Ковалеві треба нових обценьків, бо старі геть розбиті.",
    "Іван: На ярмарку в Гнівані ціни на полотно знову підскочили.",
]

# Той самий мовець, та сама історія двічі — справжня пара зі збереженого прогону
# `viche-1788026697.json`, спільних змістових основ 0.40.
ПЕРЕКАЗИ = [
    "Іван: Пам'ятаю, колись теж вовк до кошари забрався, та люди його прогнали.",
    "Одарка: Та годі згадувати, треба сторожу при кошарі ставити.",
    "Іван: Та вже було таке, як я ще хлопцем був, — вовк до кошари забрався."
    " Але ми його прогнали, і більше не приходив.",
]


def test_coherence_reply_pairs_are_high():
    bond = coherence(ВІДПОВІДІ, ВОВК)
    assert bond["пар"] == 3
    assert bond["зчеплення"] == 1.0
    assert (bond["підхоплення"], bond["звертання"], bond["реакція"]) == (1, 2, 1)


def test_coherence_monologues_are_low():
    bond = coherence(МОНОЛОГИ, ВОВК)
    assert bond["пар"] == 2
    assert bond["зчеплення"] == 0.0
    assert bond["підхоплення"] == bond["звертання"] == bond["реакція"] == 0


def test_coherence_is_deterministic():
    """Число мусить рахуватись само на кожному прогоні, отже воно не має права хитатись."""
    first = coherence(ВІДПОВІДІ, ВОВК)
    assert all(coherence(list(ВІДПОВІДІ), ВОВК) == first for _ in range(5))
    assert coherence(МОНОЛОГИ + ВІДПОВІДІ, ВОВК) == coherence(МОНОЛОГИ + ВІДПОВІДІ, ВОВК)


def test_coherence_news_words_are_not_a_bond():
    """Половина «звʼязку» в аудиті була тим, що всі кажуть слово «вовк»: 48.5% проти 13.6%."""
    news_only = [
        "Іван: Кажуть, вовк унадився до кошари, і за річкою його бачили.",
        "Панас: Той вовк унадився до кошари, бо за річкою йому голодно.",
    ]
    assert coherence(news_only, ВОВК)["підхоплення"] == 0
    assert coherence(news_only, "")["підхоплення"] == 1


def test_coherence_skips_closing_and_votes():
    """Староста, піп і голоси говорять ПРО розмову, а не в ній: пар вони не утворюють."""
    closed = coherence(ВІДПОВІДІ + [
        "староста: На віче зійшлися на тому, що треба сторожа.",
        "піп: Чи справді то вовк, а не пес?",
        "Іван: проти. бо сторожа дорого стане",
        "Одарка: за",
    ], ВОВК)
    assert closed == coherence(ВІДПОВІДІ, ВОВК)


def test_coherence_ignores_own_continuation():
    """Той самий мовець двічі поспіль — це не пара: сам до себе не звертаються."""
    solo = ["Іван: Сторожу треба ставити при кошарі.",
            "Іван: І смолоскипи палити цілу ніч, бо кошара при лісі."]
    assert coherence(solo, ВОВК)["пар"] == 0
    assert coherence(solo, ВОВК)["зчеплення"] == 0.0


def test_coherence_sees_what_distinctness_misses():
    """★ Той самий прогін, на якому прилад показував зелене: `viche-1787496811`, перші рядки.

    Заміряно на всьому файлі: `distinct2` 0.975, `overlap2` 0.002 — і зчеплення 0.059, тобто
    найгірше з 53 збережених прогонів. Тут ті самі рядки в мініатюрі: різність під стелею, звʼязку
    немає жодного.
    """
    мито = "Пан прислав писаря: із наступного тижня мито на переправі вдвічі більше."
    lines = [
        "Остап: Ти, пане писарю, не жартуй! Ми не маємо стільки грошей, щоб платити вдвічі більше!",
        "Одарка: Дивиться на світ так: гроші й поголос: кому це вигідно.",
        "Оксана: Не погоджуюсь, бо це несправедливо.",
        "Марія: Я йду дізнатись про пан писар",
    ]
    texts = [ln.split(": ", 1)[1] for ln in lines]
    assert distinctness(texts, 2) > 0.9
    assert overlap(texts, 2) < 0.05
    assert coherence(lines, мито)["зчеплення"] == 0.0


def test_numbers_carry_bond_next_to_distinctness():
    """Блок чисел прогону мусить нести зчеплення ПОРУЧ із різністю.

    Доти в звіті стояли самі `distinct2` й `overlap2`, і на обох цих наборах вони однакові —
    різність не відрізняє розмову від двох монологів узагалі.
    """
    talk, solo = numbers(ВІДПОВІДІ, ВОВК), numbers(МОНОЛОГИ, ВОВК)
    assert set(talk) == {"distinct2", "overlap2", "зчеплення", "ознаки_звʼязку",
                         "переказ", "перекази", "протік", "протіки",
                         # Вирівняне підхоплення — теж частина того самого шматка: голе число
                         # всередині `ознаки_звʼязку` є функцією довжини репліки, і без стелі,
                         # довжини й еталона поруч читати його не можна.
                         "вирівняне", "вирівнювання"}
    assert talk["distinct2"] > 0.9 and solo["distinct2"] > 0.9
    assert talk["зчеплення"] == 1.0 and solo["зчеплення"] == 0.0
    assert set(solo["ознаки_звʼязку"]) == {"пар", "підхоплення", "звертання", "реакція"}
    # Переказ власної думки їде тим самим шматком: різність його не бачить так само, як зчеплення.
    assert numbers(ПЕРЕКАЗИ, ВОВК)["переказ"] == 0.4 and talk["переказ"] == 0.0
    # Довжина, стеля й еталон їдуть у тому самому звіті: без них 0.333 не відрізнити від 0.333,
    # купленого довшим рядком.
    assert talk["вирівняне"] == 0.333
    assert talk["вирівнювання"] == {"як_є": 0.333, "пар": 3, "підхоплень": 1,
                                    "пар_у_стелі": 3, "підхоплень_у_стелі": 1, "стеля": 16,
                                    "слів": 10.2, "еталон": 0.03, "до_еталона": 0.303}


# ── підхоплення з вирівнюванням на довжину ────────────────────────────────────
# Та сама розмова двома довжинами. `ДОВГІ` й `КОРОТКІ` кажуть ОДНЕ Й ТЕ САМЕ — сторожу треба
# ставити, а годувати її нема кому, — але в довгому варіанті випадкових спільних основ набирається
# три (`сторож`, `смолоскип`, `громад`), а в короткому одна. Голе підхоплення через це стрибає з
# 0.0 до 1.0 без жодної зміни звʼязності: рівно так попередні круги діставали «регрес» за кожну
# правку, що вкоротила репліку (`docs/research/dialogue-tier-vs-content.md`, дослід 3).

ДОВГІ = [
    "Іван: Сторожу треба ставити при кошарі, і смолоскипи палити цілу ніч, бо інакше вівці"
    " пропадуть, а громада потім питатиме з нас усіх.",
    "Панас: Сторожа проситиме хліба щодня, смолоскипи горітимуть, доки смоли стане, і хто те"
    " все громаді полічить, як не ти, старий скупердяю.",
]

КОРОТКІ = [
    "Іван: Сторожу треба ставити при кошарі.",
    "Панас: А хліба тій сторожі хто дасть?",
]


def test_aligned_pickup_does_not_pay_for_a_longer_line():
    """★ Голе підхоплення міряє ДОВЖИНУ репліки, і стеля це вимикає.

    Заміряно на двох українських пʼєсах (`docs/research/dialogue-tier-vs-content.md`, дослід 3):
    «Мина Мазайло» дає 6.7% без стелі, 4.1% при рядках до 20 слів, 2.8% при 16 і 1.0% при 12;
    «Бондарівна» — 11.4%, 2.5%, 3.7%, 0.0%. Тут те саме в мініатюрі: та сама думка, сказана на
    21 і 20 слів, дає підхоплення 1.0, а сказана на 5 і 6 слів — 0.0.
    """
    assert aligned_pickup(ДОВГІ, ВОВК)["як_є"] == 1.0
    assert aligned_pickup(КОРОТКІ, ВОВК)["як_є"] == 0.0
    # А з вирівнюванням довгий варіант не рахується взагалі: пара за стелею, міряти нема чого.
    assert aligned_pickup(ДОВГІ, ВОВК)["пар_у_стелі"] == 0
    assert aligned_pickup(КОРОТКІ, ВОВК)["вирівняне"] == 0.0


def test_aligned_pickup_moves_with_the_ceiling_on_the_very_same_text():
    """Зміст закріплено, рухається лише стеля — і число рухається разом із нею.

    Це найкоротший доказ того, що ворота на голому підхопленні міряють не діалог: той самий
    байт-у-байт текст дає `None` при стелі 16 і 1.0 при стелі 60.
    """
    assert aligned_pickup(ДОВГІ, ВОВК, ceiling=16)["вирівняне"] is None
    assert aligned_pickup(ДОВГІ, ВОВК, ceiling=60)["вирівняне"] == 1.0
    assert aligned_pickup(ДОВГІ, ВОВК, ceiling=60)["пар_у_стелі"] == 1


def test_aligned_pickup_is_empty_and_not_zero_when_no_pair_fits():
    """★ Нуль і «не заміряно» — різні речі, і плутати їх дорого.

    Нуль читався б як «підхоплення немає», хоч насправді немає ПАР потрібної довжини; саме така
    підміна й записувала в регрес правки, що вкоротили репліку. Тому `None`, і відстань до
    еталона теж `None` — від невідомого не віднімають.
    """
    empty = aligned_pickup(ДОВГІ, ВОВК)
    assert empty["вирівняне"] is None and empty["до_еталона"] is None
    assert empty["пар"] == 1, "пара є, просто вона за стелею"


def test_aligned_pickup_reports_the_distance_to_the_human_play():
    """Ворота — це відстань до людської норми, а не до вигаданих 13%.

    3.0% — це 6 пар із 199 на репліках до 16 слів у «Бондарівні» й «Мині Мазайлі» разом, той самий
    `_stems` і той самий поріг. Ціль 13% походила з ручної розмітки НАШОГО корпусу
    (`docs/research/dialogue-audit.md`) і при нашій довжині рядка недосяжна.
    """
    row = aligned_pickup(ВІДПОВІДІ, ВОВК)
    assert (row["еталон"], row["стеля"]) == (PLAY_PICKUP, LENGTH_CEILING)
    assert (PLAY_PICKUP, LENGTH_CEILING) == (0.030, 16)
    assert row["до_еталона"] == round(row["вирівняне"] - PLAY_PICKUP, 3)
    # Село, яке говорить коротко й повз тему, стоїть НИЖЧЕ за пʼєсу — і це видно знаком.
    assert aligned_pickup(МОНОЛОГИ, ВОВК)["до_еталона"] == -PLAY_PICKUP


def test_aligned_pickup_carries_the_length_that_explains_the_number():
    """Середня довжина репліки їде поруч, бо без неї два однакові числа непорівнянні."""
    assert aligned_pickup(ДОВГІ, ВОВК)["слів"] == 20.5
    assert aligned_pickup(КОРОТКІ, ВОВК)["слів"] == 5.5
    assert aligned_pickup(ВІДПОВІДІ, ВОВК)["слів"] == 10.2


def test_aligned_pickup_counts_the_same_pairs_as_coherence():
    """Той самий `_talk` і той самий поріг, що й у зчеплення: інакше два числа звіту розійшлись би.

    Староста, піп і голоси пар не утворюють, свій за своїм — теж, а `як_є` мусить збігатися з
    підхопленням із `ознаки_звʼязку`, поділеним на пари.
    """
    закрито = ВІДПОВІДІ + ["староста: На віче зійшлися на тому, що треба сторожа.",
                           "Одарка: за"]
    bond = coherence(закрито, ВОВК)
    row = aligned_pickup(закрито, ВОВК)
    assert row["пар"] == bond["пар"] == 3
    assert row["як_є"] == round(bond["підхоплення"] / bond["пар"], 3)
    assert row["пар_у_стелі"] <= row["пар"]
    # Лічильники, а не самі частки: зведення багатьох прогонів складається саме з них.
    assert row["підхоплень"] == bond["підхоплення"] == 1
    assert row["підхоплень_у_стелі"] <= row["підхоплень"]


def test_aligned_pickup_is_deterministic():
    """Число мусить рахуватись само на кожному прогоні, отже воно не має права хитатись."""
    first = aligned_pickup(ВІДПОВІДІ, ВОВК)
    assert all(aligned_pickup(list(ВІДПОВІДІ), ВОВК) == first for _ in range(5))
    assert aligned_pickup(МОНОЛОГИ + ВІДПОВІДІ, ВОВК) == aligned_pickup(МОНОЛОГИ + ВІДПОВІДІ, ВОВК)


ПРОТІК = [
    "Іван: Кажуть, за річкою бачили вовка, і він унадився до кошари. Що робити?",
    "Одарка: Та треба сторожу ставити, а не язиком плескати.",
]


def test_leakage_is_the_number_the_audit_asked_for_three_rounds():
    """★ Протік пакета доти не мав ЖОДНОГО постійного числа — його щоразу ловив саморобний шпигун.

    Це головний борг приладу за визнанням аудиту (`docs/research/dialogue-audit.md`, розділ 21
    п. 2 і розділ 22 п. 4): флагманська правка другого круга обіцяла «підказка не тече», і
    підтвердити це можна було лише разовим шпигуном, який потім викидали.

    Ловить він рівно те, що просив аудит: найдовший ланцюжок слів репліки, спільний із промптом,
    від чотирьох слів. Справжня репліка живого прогону («Кажуть, за річкою бачили вовка, і він
    унадився до кошари. Що робити?») дає 10.
    """
    dirty = leakage(ПРОТІК, ВОВК)
    assert dirty["протік"] == 10 and dirty["протіків"] == 1
    assert dirty["найдовший"][0] == "Іван"
    clean = leakage(ВІДПОВІДІ, ВОВК)
    assert clean["протіків"] == 0, "жива розмова про вовка новини дослівно не вертає"
    assert clean["найдовший"] == [], "найдовше називаємо лише тоді, коли є що назвати"


def test_leakage_uses_the_same_threshold_as_the_guard_in_the_core():
    """Прилад і сторож мусять рахувати ОДНІЄЮ функцією й одним порогом.

    Розійшовшись, число перестало б означати «сторож мовчить дарма» — рівно та вада, через яку
    переказ власної думки міряють тим самим мішком основ, що й `_same_meaning`.
    """
    from ploshcha_sim.agents.viche import LEAK_GRAM
    import evalkit.dialogue as dialogue
    from ploshcha_sim.agents.viche import _leak_len

    assert LEAK_GRAM == 4 and dialogue.LEAK_GRAM is LEAK_GRAM
    assert dialogue._leak_len is _leak_len


def test_leakage_ignores_the_closers_and_the_votes_like_the_other_two():
    """Староста й голоси відсіяні тим самим `_talk`: вони говорять ПРО розмову, а не в ній, і
    зведення старости переказує новину за обовʼязком, а не через протік."""
    with_closer = ["Староста: Кажуть, за річкою бачили вовка, і він унадився до кошари."]
    assert leakage(with_closer, ВОВК)["протіків"] == 0


def test_retelling_names_the_worst_pair_by_text():
    """Прилад мусить давати НАЙГІРШУ ПАРУ текстом, а не бінарне «пар понад поріг — нуль».

    Саме бінарне число три круги показувало зелене: пороги стояли вищі за реальні повтори, тож
    «нуль пар» означав лише те, що ніхто не переказав себе дослівно.
    """
    told = retelling(ПЕРЕКАЗИ)
    assert (told["переказ"], told["переказів"]) == (0.4, 1)
    who, earlier, text = told["найгірша_пара"]
    assert who == "Іван" and earlier.startswith("Пам'ятаю") and text.startswith("Та вже було")
    assert retelling(МОНОЛОГИ)["переказів"] == 0, "різні думки одного мовця — не переказ"


def test_retelling_skips_votes_and_closers():
    """Голос переказує ВЛАСНУ репліку за побудовою — на нього стоїть окремий сторож у `_vote_why`,
    і рахувати те саме двічі означало б показувати ваду там, де її вже полагоджено."""
    assert retelling(ПЕРЕКАЗИ + [
        "Іван: за. бо вовк до кошари забрався, а люди його прогнали",
        "староста: На віче зійшлися на тому, що вовк до кошари забрався.",
    ]) == retelling(ПЕРЕКАЗИ)
