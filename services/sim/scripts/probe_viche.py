import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from serve_ploshcha import load_env  # noqa: E402

load_env(ROOT / ".env")

from evalkit.conditions import CONDITIONS  # noqa: E402
from evalkit.cost import role_of  # noqa: E402
from evalkit.dialogue import numbers  # noqa: E402
from evalkit.prompts import resolve  # noqa: E402
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm  # noqa: E402
from ploshcha_sim.compose import build_budget, build_viche  # noqa: E402
from ploshcha_sim.domain.viche import stance_match  # noqa: E402

NEWS = [
    "Кажуть, за річкою бачили вовка, і він унадився до кошари.",
    "Пан прислав писаря: із наступного тижня мито на переправі вдвічі більше.",
    "Молодиця з крайньої хати не вийшла на толоку вже третій раз.",
    "Гребля протікає, а дощі обіцяють на тому тижні.",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Я7-В10: віче на справжніх новинах + заміри")
    p.add_argument("--condition", default="viche")
    p.add_argument("--items", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv or sys.argv[1:])

    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key or not url:
        print("нема LAPA_API_KEY / LAPA_BASE_URL у .env")
        return 2
    spec = CONDITIONS[args.condition]
    lapa = OpenAICompatLlm(model=os.environ["LAPA_MODEL"], base_url=url, api_key=key,
                           structured_mode="json_schema")
    mamay = OpenAICompatLlm(model=os.environ["MAMAY_MODEL"], base_url=url, api_key=key,
                            structured_mode="json_schema")
    variant = resolve(spec.prompt_id)

    reports = []
    for index, news in enumerate(NEWS[:args.items]):
        agent = build_viche(spec, lapa=lapa, mamay=mamay, run_id=f"probe{index}",
                            prompt_id=variant.id, prompt_sha=variant.sha256,
                            line_system=variant.render_system(),
                            score_system=resolve("viche/score").render_system(),
                            summary_system=resolve("viche/summary").render_system(),
                            doubt_system=resolve("viche/doubt").render_system(),
                            chronicle_system=resolve("viche/chronicle").render_system())
        started = time.time()
        result = agent.run(news, seed=args.seed, budget=build_budget(spec))
        lines = [ln for ln in (result.answer or "").splitlines() if ln.strip()]
        texts = [ln.split(": ", 1)[1] for ln in lines if ": " in ln]
        speakers = [ln.split(": ", 1)[0] for ln in lines if ": " in ln]
        roles = Counter(role_of(s) for s in spec_stages(result))

        report = {
            "новина": news, "outcome": result.outcome, "реплік": len(texts),
            "кроків": result.steps, "стеля_кроків": build_budget(spec).max_steps,
            "голосів": len(set(speakers)), "токенів": result.tokens + result.aux_tokens,
            "вхідних": sum(result.prompt_by_lane.values()),
            "згенерованих": result.tokens + result.aux_tokens - sum(result.prompt_by_lane.values()),
            "секунд": round(time.time() - started, 1),
            "по_ярусах": dict(result.tokens_by_lane),
            "вхідних_по_ярусах": dict(result.prompt_by_lane),
            "вхідних_по_стадіях": dict(result.prompt_by_stage),
            "по_стадіях": dict(result.tokens_by_stage),
            "по_ролях": dict(roles),
            # Числа про саму розмову — одним шматком з `evalkit.dialogue`, щоб зчеплення не
            # лишилось у разовому аудиті: різність поруч із ним показувала 0.975 там, де звʼязку
            # між репліками не було зовсім.
            **numbers(lines, news),
            "інциденти": result.incidents, "нотатки": result.notes,
            # ★ Партитура й позиції — у ЗВІТ, бо без них наступний круг знову міряв би розмову
            # тимчасовим шпигуном: у 155 попередніх звітах тактів немає жодного, і «хто кого
            # підтримав» з них не відновлюється взагалі.
            "такти": result.beats,
            "позиції": result.stances,
            # Чи міряють код і модель одне й те саме: знак позиції проти голосу того ж селянина.
            "позиції_проти_голосів": stance_match(result.stances),
            "розмова": lines,
        }
        reports.append(report)

        print(f"\n{'=' * 78}\nНОВИНА: {news}")
        print(f"  {result.outcome} · {len(texts)} реплік · {result.steps}/{build_budget(spec).max_steps} кроків · {len(set(speakers))} голосів · "
              f"{report['токенів']} ток (вх {report['вхідних']} / ген {report['згенерованих']}) · {report['секунд']}s · "
              f"яруси {report['по_ярусах']} · distinct2 {report['distinct2']} · "
              f"overlap2 {report['overlap2']} · зчеплення {report['зчеплення']} "
              f"({report['ознаки_звʼязку']}) · переказ {report['переказ']} "
              f"({report['перекази']['переказів']}) · "
              # Протік у рядок проби, а не лише в JSON: три круги поспіль його ловили саморобним
              # шпигуном саме тому, що очима його ніде не було видно.
              f"протік {report['протік']} ({report['протіки']['протіків']})")
        # Підхоплення — функція довжини репліки (6.7% без стелі проти 1.0% при ≤12 словах на
        # одній і тій самій пʼєсі), тож у рядок проби воно йде ЛИШЕ разом зі стелею, довжиною й
        # відстанню до людського еталона: без цих трьох сусідів попередні круги записували в
        # регрес звʼязності будь-яку правку, що вкоротила репліку.
        align = report["вирівнювання"]
        print(f"  підхоплення: як є {align['як_є']} на {align['пар']} парах · "
              f"до {align['стеля']} слів {report['вирівняне']} на {align['пар_у_стелі']} · "
              f"довжина {align['слів']} слів · "
              f"до еталона пʼєс ({align['еталон']}) {align['до_еталона']}")
        match = report["позиції_проти_голосів"]
        print(f"  позиції: рухомих {match['рухомих']}/{match['усіх']} · "
              f"знаком {match['за_знаком']}/{match['звірено']} ({match['частка_знака']}) · "
              f"ярликом {match['збіглось']}/{match['звірено']} ({match['частка']})")
        if result.incidents:
            print(f"  інциденти: {result.incidents}")
        if not args.quiet:
            for ln in lines:
                print(f"    {ln}")

    out = ROOT / "docs" / "research" / "eval-runs" / f"viche-{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"condition": args.condition, "spec": spec.sha256,
                               "seed": args.seed, "runs": reports},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nзвіт: {out}")
    return 0


def spec_stages(result) -> list[str]:
    return [pair.split("|", 1)[0] for pair in result.tokens_by_stage_lane]


if __name__ == "__main__":
    sys.exit(main())
