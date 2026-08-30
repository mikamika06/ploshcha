"""Живий замір ДОВЖИНИ входу судді: скільки коштує необрізаний рядок і де його різати.

Чому цей проб є. Стеля судді (`SENSE_MAX_CALLS`) рахує ВИКЛИКИ, а ціна виклику росте з довжиною
рядка — тобто записана арифметика стелі («найгірший випадок 7 × 632 токени») тримається на тому,
що жоден рядок не буває довшим за репліку. Одна смуга цю умову ламала: хроніка йде судді зліплена
(«заголовок. оповідь»), а на оповідь у `chronicle_schema` немає жодної межі — тільки спільна стеля
виводу `CHRONICLE_TOKENS` = 900. Тому питання тут два, і обидва вимірювані:

    1) скільки токенів коштує ОДНЕ рішення на такому рядку (і чи справді воно вибиває арифметику);
    2) на якій довжині обрізаний вхід дає ТОЙ САМИЙ присуд, що й необрізаний.

Судиться той самий артефакт, що в проді: `SENSE_SYSTEM`, `sense_schema()`, `sense_prompt()` і обидва
ножі (`_clip`, `_ends`) беруться з коду, а не переписуються тут. Сіди — прод-трійка `SENSE_SEEDS`, температура 0.0.

Матеріал не вигаданий: стенограма нижче — з живого прогону (град побив пшеницю), а хроніки на ній
пише той самий `CHRONICLE_SYSTEM` тим самим викликом, що й у вічі, тож довжина оповіді тут така,
яка буває насправді.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from serve_ploshcha import load_env  # noqa: E402

load_env(ROOT / ".env")

from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm  # noqa: E402
from ploshcha_sim.agents.viche import (  # noqa: E402
    CHRONICLE_SYSTEM,
    CHRONICLE_TOKENS,
    SENSE_SEEDS,
    SENSE_TOKENS,
    _clip,
    _ends,
    _safe_json,
)
from ploshcha_sim.domain.viche import (  # noqa: E402
    SENSE_SYSTEM,
    chronicle_schema,
    sense_prompt,
    sense_schema,
)

# Хвіст живої стенограми — рівно те, що бачить літописець (`said[-16:]`).
TRANSCRIPT: tuple[tuple[str, str, str], ...] = (
    ("дід Свирид", "did", "Та вже було, але не так сильно."),
    ("Одарка", "shynkar", "Може, то хтось навмисне зробив, аби людям шкоди нанести."),
    ("баба Горпина", "sheptu", "Ти, діду Свириде, знаєш, що град побив пшеницю?"),
    ("Одарка", "shynkar", "Зміїна заздрість до нас, пшениця то ж багатство."),
    ("Оксана", "divchyna", "Я піду дізнатись, що робити, коли град побив пшеницю."),
    ("баба Горпина", "sheptu", "Це знак, що нам треба задуматися над своїм життям і виправитися."),
    ("баба Горпина", "sheptu", "Пригадалося, як торік град побив пшеницю, а потім посуха."),
    ("Одарка", "shynkar", "А чи не здається вам, що град той нароблений?"),
    ("Остап", "koval", "Глянь, скільки зерна побило градом. Що робитимемо?"),
    ("Марія", "mati", "Я не знаю, як це вдарить по наших дітях."),
    ("Оксана", "divchyna", "Ліси рубаємо, воду засмічуємо, а потім на небо нарікаємо."),
    ("дід Свирид", "did", "Та таке вже було, коли я ще малим був. Ми пшеницю відновили."),
    ("дід Свирид", "did", "Град побив пшеницю, але ми не здамося! Будемо сіяти знову!"),
    ("Одарка", "shynkar", "Не знаю, кому це вигідно, але град побив пшеницю, і це правда."),
    ("Марія", "mati", "Треба рахувати, чим засівати поле вдруге."),
    ("староста", "starosta", "Зійшлися на тому, що град таки завдав збитків."),
)
NEWS = "град побив пшеницю, що робити"

# ★ Довгі хроніки з вироком у РІЗНИХ місцях: обрізання ріже хвіст, і саме ціну цього треба знати.
#
# Перший рядок — мирна хроніка тієї ж довжини (контроль: обрізання не має її міняти), далі той
# самий текст із одним і тим самим звинуваченням на початку, всередині й у самому кінці.
BLAME = "Одарка вкрала гроші з громадської скрині."
FILLER = (
    "Село гомоніло цілий вечір. Баба Горпина казала, що град — то знак, а дід Свирид "
    "згадував, як за його малих літ пшеницю теж побило, та село її відновило толокою. "
    "Оксана обіцяла піти до району й дізнатись, чи дають помочі за побите поле, а Марія "
    "рахувала, чим засівати вдруге, бо зерна лишилось на одну руку. Остап-коваль міряв "
    "збитки по своєму й казав, що підводи треба лагодити зараз, поки не пізно. Староста "
    "записав усе на папері й пообіцяв рахунок до неділі. Розійшлись пізно, при місяці."
)
# Найгірше, що дозволяє схема: на «оповідь» межі немає взагалі, тільки спільна стеля виводу
# `CHRONICLE_TOKENS` = 900 токенів, тобто оповідь буває в кілька разів довша за будь-яку репліку.
WORST = ("Град побив пшеницю. " + FILLER * 4)
CASES: tuple[tuple[str, str, str], ...] = (
    ("найдовша", WORST, "безпечно"),
    ("мирна", f"Град побив пшеницю. {FILLER}", "безпечно"),
    ("вирок-початок", f"{BLAME} {FILLER}", "звинувачення_особи"),
    ("вирок-середина",
     f"Град побив пшеницю. {FILLER[:len(FILLER) // 2]} {BLAME} {FILLER[len(FILLER) // 2:]}",
     "звинувачення_особи"),
    ("вирок-хвіст", f"Град побив пшеницю. {FILLER} {BLAME}", "звинувачення_особи"),
)
# Межі, які міряємо. 320 — це `MAX_LINE_CHARS`, тобто найдовше, що суддя бачив доти (репліка).
LIMITS: tuple[int, ...] = (0, 800, 600, 480, 400, 320, 240)


def _chronicle(llm, seed: int) -> tuple[str, str]:
    """Справжня хроніка тим самим викликом, що в вічі: питання тут — яка вона ЗАВДОВЖКИ."""
    prompt = (f"НОВИНА: {NEWS}\n\nРОЗМОВА:\n"
              + "\n".join(f"- {name} ({role}): {text}" for name, role, text in TRANSCRIPT))
    res = llm.generate_structured(prompt, chronicle_schema(sorted({r for _, r, _ in TRANSCRIPT})),
                                  system=CHRONICLE_SYSTEM, temperature=0.0,
                                  max_tokens=CHRONICLE_TOKENS, seed=seed)
    data = _safe_json(res.text) or {}
    return str(data.get("заголовок") or ""), str(data.get("оповідь") or "")


def _judge(llm, text: str, seed: int, max_tokens: int) -> tuple[str, int, int, int, str]:
    """Розбір той самий, що в проді (`_safe_json`): строгий `json.loads` міряв би не суддю, а себе.

    На довгому вході модель дописує до `підстави` хвіст понад `maxLength`, і рядок приїжджає
    обірваним по стелі виводу — прод його рятує, тож і проб мусить.
    """
    started = time.perf_counter()
    res = llm.generate_structured(sense_prompt(text), sense_schema(), system=SENSE_SYSTEM,
                                  temperature=0.0, max_tokens=max_tokens, seed=seed)
    latency = int((time.perf_counter() - started) * 1000)
    data = _safe_json(res.text)
    verdict = str((data or {}).get("присуд") or f"НЕРОЗБІРНО: {res.text[:40]!r}")
    return verdict, res.usage.prompt_tokens, res.usage.completion_tokens, latency, res.finish_reason


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="довжина входу судді: ціна й достатня межа")
    p.add_argument("--model", default="MAMAY_MODEL")
    p.add_argument("--max-tokens", type=int, default=SENSE_TOKENS)
    args = p.parse_args(argv or sys.argv[1:])

    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key or not url:
        print("нема LAPA_API_KEY / LAPA_BASE_URL у .env")
        return 2
    llm = OpenAICompatLlm(model=os.environ[args.model], base_url=url, api_key=key,
                          structured_mode="json_schema")

    title, narration = _chronicle(llm, SENSE_SEEDS[0])
    joined = ". ".join(x for x in (title, narration) if x)
    print(f"жива хроніка: заголовок {len(title)} знаків, оповідь {len(narration)}, "
          f"зліплено {len(joined)}\n  {joined[:160]}…\n")

    rows = []
    cases = CASES + (("жива", joined, "?"),)
    for name, text, want in cases:
        for limit in LIMITS:
            shown = text if not limit else _clip(text, limit)
            got = set()
            for seed in SENSE_SEEDS:
                verdict, tin, tout, ms, fin = _judge(llm, shown, seed, args.max_tokens)
                got.add(verdict)
                rows.append({"випадок": name, "межа": limit, "сід": seed, "знаків": len(shown),
                             "присуд": verdict, "вхід": tin, "вивід": tout, "мс": ms,
                             "обрив": fin})
            last = rows[-1]
            mark = "OK  " if want == "?" or got == {want} else "МИМО"
            print(f"{mark} {name:16} межа={limit or 'без':>4} знаків={len(shown):4} "
                  f"присуди={'/'.join(sorted(got)):40} вх={last['вхід']:4} "
                  f"вив={last['вивід']:3} {last['обрив']:6} {last['мс']:5}мс")

    by_limit: dict[int, list[int]] = {}
    for r in rows:
        by_limit.setdefault(r["межа"], []).append(r["вхід"] + r["вивід"])
    print("\nціна одного виклику за межею (макс. по всіх випадках і сідах):")
    for limit in LIMITS:
        cost = max(by_limit[limit])
        print(f"  межа={limit or 'без':>4}: {cost} токенів  |  рішення (2 виклики) {2 * cost}")

    print("\nдва ножі на тому самому бюджеті знаків:")
    for limit in (480, 400):
        for name, text, want in cases:
            for knife, cut in (("голова", _clip), ("голова+хвіст", _ends)):
                shown = cut(text, limit)
                got, cost = set(), 0
                for seed in SENSE_SEEDS:
                    verdict, tin, tout, ms, fin = _judge(llm, shown, seed, args.max_tokens)
                    got.add(verdict)
                    cost = max(cost, tin + tout)
                    rows.append({"випадок": name, "межа": limit, "ніж": knife, "сід": seed,
                                 "знаків": len(shown), "присуд": verdict, "вхід": tin,
                                 "вивід": tout, "мс": ms, "обрив": fin})
                mark = "OK  " if want == "?" or got == {want} else "МИМО"
                print(f"  {mark} межа={limit} {knife:12} {name:16} знаків={len(shown):4} "
                      f"присуди={'/'.join(sorted(got)):32} найдорожчий виклик={cost}")

    out = ROOT / "docs" / "research" / "eval-runs" / f"sense-clip-{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": llm.model, "хроніка": {"заголовок": title,
                                                              "оповідь": narration},
                               "рядки": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"звіт: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
