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
from evalkit.prompts import resolve  # noqa: E402
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm  # noqa: E402
from ploshcha_sim.compose import build_budget, build_viche  # noqa: E402

NEWS = [
    "Кажуть, за річкою бачили вовка, і він унадився до кошари.",
    "Пан прислав писаря: із наступного тижня мито на переправі вдвічі більше.",
    "Молодиця з крайньої хати не вийшла на толоку вже третій раз.",
    "Гребля протікає, а дощі обіцяють на тому тижні.",
]


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def distinctness(lines: list[str], n: int = 2) -> float:
    """Частка унікальних n-грам. Низька = всі говорять однаково (ризик обвалу ентропії Lapa)."""
    total, uniq = 0, set()
    for line in lines:
        words = line.lower().split()
        grams = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
        total += len(grams)
        uniq |= set(grams)
    return len(uniq) / total if total else 0.0


def overlap(lines: list[str], n: int = 2) -> float:
    """Середнє попарне перекриття n-грам між репліками — пряма міра «всі однакові»."""
    grams = [ngrams(t, n) for t in lines]
    pairs = [(i, j) for i in range(len(grams)) for j in range(i + 1, len(grams))]
    if not pairs:
        return 0.0
    scores = []
    for i, j in pairs:
        union = grams[i] | grams[j]
        scores.append(len(grams[i] & grams[j]) / len(union) if union else 0.0)
    return sum(scores) / len(scores)


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
            "distinct2": round(distinctness(texts, 2), 3),
            "overlap2": round(overlap(texts, 2), 3),
            "інциденти": result.incidents, "нотатки": result.notes,
            "розмова": lines,
        }
        reports.append(report)

        print(f"\n{'=' * 78}\nНОВИНА: {news}")
        print(f"  {result.outcome} · {len(texts)} реплік · {result.steps}/{build_budget(spec).max_steps} кроків · {len(set(speakers))} голосів · "
              f"{report['токенів']} ток (вх {report['вхідних']} / ген {report['згенерованих']}) · {report['секунд']}s · "
              f"яруси {report['по_ярусах']} · distinct2 {report['distinct2']} · "
              f"overlap2 {report['overlap2']}")
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
