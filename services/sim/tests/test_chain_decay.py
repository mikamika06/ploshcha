"""Згасання ланцюга відповідей за глибиною: розмова замикає тему, а не тягне її нескінченно.

Механізм узятий із «bounded autonomy» (arXiv:2604.04703) — єдиного з огляду поля, що заміряний у
живій багатокористувацькій грі (20 природних завершень із 20 проти 0 із 20) і не кладе в пакет ані
слова: рішення тут НЕТЕКСТОВЕ, такт лунає чи ні.

★ Важіль вимкнений за замовчуванням, і це рішення за ЗАМІРОМ. Офлайн на восьми збережених живих
прогонах зі шпигуном по тактах (147 зіграних тактів, 76 ланцюгів): гасити є що — ланцюгів
завглибшки 3 і більше 16 із 76 (21.1%) — але зчеплення сусідньої пари за глибиною такту-відповіді
дає 20.0% на корені (65 пар), 30.8% на першій ланці (39), 64.7% на другій (17) і 28.6% на третій і
глибше (14). Тобто згасання обміняло б найслабші ланки (28.6%) на почини наступної хвилі (20.0%).
Тому тут стережеться саме проводка й детермінізм, а не обіцяний виграш.
"""

import pytest

from evalkit.conditions import CONDITIONS
from ploshcha_sim.agents import viche as V
from ploshcha_sim.compose import VICHE_KWARGS, build_viche
from ploshcha_sim.domain.task import Budget
from ploshcha_sim.domain.spec import AppSpec
from ploshcha_sim.domain.viche import (
    CHAIN_FREE,
    Beat,
    chain_alive,
    chain_depths,
    damp_chain,
    scatter,
)
from rule_llm import RuleLlm

NEWS = "Кажуть, за річкою бачили вовка."


def chain(length: int, tag: str = "т1") -> list[Beat]:
    """Партитура одним ланцюгом: кожен наступний такт відповідає попередньому."""
    return [Beat(хто="did" if i % 2 else "koval", хід="заперечити", мітка=f"{tag}:{i + 1}",
                 у_відповідь=f"{tag}:{i}" if i else None)
            for i in range(length)]


def test_the_depth_is_counted_along_the_marks():
    """Глибина рахується по МІТЦІ, тій самій, що вже несе адресу: іншого стану для цього немає."""
    assert chain_depths(chain(5)) == [0, 1, 2, 3, 4]


def test_a_second_root_starts_its_own_chain():
    """Такт без цілі — почин, і глибина починається з нуля навіть посеред хвилі."""
    beats = chain(3) + chain(2, tag="т2")
    assert chain_depths(beats) == [0, 1, 2, 0, 1]


def test_a_link_that_leads_nowhere_is_a_root_again():
    """Посилання на мітку, якої в хвилі немає, не робить такт відповіддю: глибина 0, не 1."""
    beats = [Beat(хто="did", хід="згадати", мітка="т1:1", у_відповідь="кудись:9")]
    assert chain_depths(beats) == [0]


def test_branches_of_one_beat_share_its_depth():
    """Дві відповіді на один такт — обидві на глибині 1, а не 1 і 2: ланцюг не список."""
    beats = chain(2) + [Beat(хто="mirosh", хід="піддакнути", мітка="т1:3", у_відповідь="т1:1")]
    assert chain_depths(beats) == [0, 1, 1]


def test_the_curve_is_full_then_falls_to_zero():
    """Крива взята числом: повна до `CHAIN_FREE`, далі рівними частками вниз, нуль на `zero`."""
    assert CHAIN_FREE == 2
    assert [chain_alive(d, zero=4) for d in range(6)] == [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]


def test_the_first_exchange_never_fades():
    """★ Перший обмін дешевий: почин і відповідь на нього не гаснуть за жодного сіда й глибини.

    Це не смак, а умова живучості хвилі: перший такт партитури почин за побудовою
    (`repair_score` не дає йому цілі), і якби гаснув він, хвиля виходила б порожньою, а порожня
    хвиля обриває віче на наступному ж такті.
    """
    beats = chain(2)
    for seed in range(50):
        assert damp_chain(beats, seed, NEWS, zero=3) == beats


def test_the_deep_links_do_fade():
    """Важіль мусить ДОЇЖДЖАТИ: на жорсткій кривій третя ланка не лунає ніколи."""
    beats = chain(5)
    for seed in range(20):
        out = damp_chain(beats, seed, NEWS, zero=3)
        assert [b.мітка for b in out] == ["т1:1", "т1:2", "т1:3"]


def test_the_same_seed_gives_the_same_silence():
    """Той самий сід і та сама тема — ті самі згаслі такти: інакше прогони не порівнюються."""
    beats = chain(6)
    for seed in range(10):
        assert (damp_chain(beats, seed, NEWS, zero=5)
                == damp_chain(beats, seed, NEWS, zero=5))


def test_another_seed_gives_another_silence():
    """Кубик справді кидається: на драбині сідів вижили не однакові хвости."""
    beats = chain(6)
    seen = {tuple(b.мітка for b in damp_chain(beats, s, NEWS, zero=6)) for s in range(30)}
    assert len(seen) > 1


def test_a_faded_beat_leaves_no_orphans():
    """Гасне не лише такт, а й усе, що на нього посилалось: ланцюг обривається, а не сиротіє.

    Сирота — це такт, який показує на мітку, якої в хвилі вже немає: `_aim` її не розвʼяже, і
    замість адресата мовець дістане порожнє місце. Крива тут `zero=5` навмисно: на жорсткішій
    сирота неможливий за побудовою (нащадок глибший, отже гасне й сам), тож сторож нічого не
    стеріг би.
    """
    beats = chain(6)
    for seed in range(60):
        out = damp_chain(beats, seed, NEWS, zero=5)
        marks = {b.мітка for b in out}
        for b in out:
            assert b.у_відповідь is None or b.у_відповідь in marks, f"сирота на сіді {seed}"


def test_a_wave_is_never_left_empty():
    """Порожня хвиля обриває віче на наступному ж такті, тож ланцюг лишає принаймні почин."""
    for seed in range(30):
        assert damp_chain(chain(4), seed, NEWS, zero=1, free=0)


def test_the_lever_is_off_until_a_depth_is_named():
    """`None` означає «партитура недоторкана»: жоден такт не гасне, прогони лишаються тими самими.

    Дефолт стережеться саме тут, бо це єдине, що робить пораховані прогони порівнюваними з
    новими: важіль, який тихо вмикається сам, зсунув би всі числа, ні в чому не зізнавшись.
    """
    beats = chain(6)
    for seed in range(20):
        out = scatter(beats, ["did", "koval"], seed, NEWS)
        assert {b.мітка for b in beats} <= {b.мітка for b in out}


def test_the_scatter_damps_the_score_before_the_interrupts():
    """Згасання судить ПАРТИТУРУ: перебивка — одна ланка вбік і власний кубик, а не ланка ланцюга.

    Тому глибокі такти партитури зникають, а перебивки лишаються перебивками — інакше
    спонтанність із коду лічилась би кроком суперечки.
    """
    beats = chain(6)
    out = scatter(beats, ["did", "koval", "mirosh"], 3, NEWS, chain_zero=3)
    planned = [b for b in out if not b.мітка.endswith("+п")]
    assert [b.мітка for b in planned] == ["т1:1", "т1:2", "т1:3"]


def test_the_agent_hands_the_depth_to_the_score_of_every_wave():
    """Проводка мусить доїжджати до ЖИВОГО циклу, а не лише до поля агента.

    Стережеться саме виклик `scatter`: параметр, який осів у `self` і нікуди не пішов, — це той
    самий клас мовчазної втрати, проти якого написаний `VICHE_KWARGS`.
    """
    from test_viche import NEWS as TASK, beat, build, lines, score

    seen = []
    plain = V.scatter

    def spy(*args, **kw):
        seen.append(args[7] if len(args) > 7 else kw.get("chain_zero"))
        return plain(*args, **kw)

    for decay in (None, 4):
        agent, _ = build([score(beat("did"), beat("koval", "заперечити", 1))] + lines(20), width=2)
        agent.chain_decay = decay
        seen.clear()
        V.scatter = spy
        try:
            agent.run(TASK, seed=1, budget=Budget(max_steps=12, max_tokens=99_999))
        finally:
            V.scatter = plain
        assert seen and set(seen) == {decay}, f"хвилі дістали {set(seen)}, а не {decay}"


def test_the_decay_is_an_axis_of_the_run_not_an_ornament():
    """Поле мусить рухати `sha256` умови: інакше два різні прогони звітують під одним іменем."""
    spec = AppSpec(mode="viche")
    assert spec.viche_chain_decay is None, "дефолт зберігає теперішню поведінку"
    assert spec.sha256 != spec.with_(viche_chain_decay=4).sha256


def test_the_composition_root_carries_the_decay_into_the_viche():
    """Магічної глибини всередині агента бути не може: важіль приходить лише зі специфікації."""
    lapa, mamay = RuleLlm("lapa"), RuleLlm("mamay")
    spec = AppSpec(mode="viche")
    assert "chain_decay" in VICHE_KWARGS
    assert build_viche(spec, lapa=lapa, mamay=mamay).chain_decay is None
    tuned = build_viche(spec.with_(viche_chain_decay=4), lapa=lapa, mamay=mamay)
    assert tuned.chain_decay == 4


@pytest.mark.parametrize("name", ["viche", "viche-notools"])
def test_the_production_condition_leaves_the_lever_off(name):
    """Вимкнено ЗА ЗАМІРОМ: згасання обміняло б ланки зі зчепленням 28.6% на почини з 20.0%, а
    природного завершення не дає — цикл добирає такти до `mode.beats[1]`, і на 57 збережених
    прогонах віча розмову обриває стеля кроків (26 із 43), а не довгий ланцюг."""
    assert CONDITIONS[name].viche_chain_decay is None
