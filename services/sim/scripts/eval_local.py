import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")

from evalkit.harness import load_items, orchestrator_runner, run_eval, single_call_runner
from evalkit.report import aggregate, format_report
from ploshcha_sim.adapters import FakeToolbox, PresetEffort, profile_router, single_model_router
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.domain.task import Budget

ITEMS = Path(__file__).resolve().parents[1] / "evalkit" / "items" / "starter.jsonl"


def parse_args(argv):
    seeds, limit = [1, 2, 3], None
    for a in argv:
        if a.startswith("--seeds="):
            seeds = [int(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
    return seeds, limit

SYSTEM = (
    "Ти агент з інструментами: check_date(year,event), lookup_fact(entity), calc(expr), final_answer(text). "
    "Аргументи пиши УКРАЇНСЬКОЮ дослівно як у задачі — НЕ перекладай назви подій і людей. "
    "Перевіряй факти інструментом, навіть якщо знаєш відповідь. "
    "НЕ повторюй виклик, який уже зроблено — його результат вище. "
    "Виконай ВСІ частини задачі, і лише тоді заверши через final_answer. "
    "Якщо інструмент не знайшов — не повторюй той самий виклик, спробуй інакше або заверши."
)


def make_llm(model, url, key):
    return OpenAICompatLlm(model=model, base_url=url, api_key=key, structured_mode="json_schema")


def orch_cond(router_factory, verifier, *, recovery=False, max_steps=5):
    def make_orch():
        return Orchestrator(router_factory(), PresetEffort(), FakeToolbox(),
                            verifier=verifier, system=SYSTEM, recovery=recovery)
    return orchestrator_runner(make_orch, budget=Budget(max_steps=max_steps))


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    seeds, limit = parse_args(sys.argv[1:])
    lapa = make_llm(os.environ["LAPA_MODEL"], url, key)
    mamay = make_llm(os.environ["MAMAY_MODEL"], url, key)

    items = load_items(str(ITEMS))
    if limit:
        items = items[:limit]
    runners = {
        "single-mamay": single_call_runner(mamay, system=SYSTEM),
        "single-lapa": single_call_runner(lapa, system=SYSTEM),
        "mamay@5": orch_cond(lambda: single_model_router(mamay), True),
        "mamay@8": orch_cond(lambda: single_model_router(mamay), True, max_steps=8),
        "mamay+rec@8": orch_cond(lambda: single_model_router(mamay), True, recovery=True, max_steps=8),
        "hetero@5": orch_cond(lambda: profile_router(lapa, mamay), True),
        "hetero@8": orch_cond(lambda: profile_router(lapa, mamay), True, max_steps=8),
        "hetero+rec@8": orch_cond(lambda: profile_router(lapa, mamay), True, recovery=True, max_steps=8),
        "hetero-nov@8": orch_cond(lambda: profile_router(lapa, mamay), False, max_steps=8),
    }
    print(f"{len(items)} задач × {len(runners)} умов × {len(seeds)} seed = "
          f"{len(items) * len(runners) * len(seeds)} прогонів\n")
    results = run_eval(items, runners, seeds)
    print(format_report(results))
    out = ROOT / "docs" / "research" / "eval-runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "starter-local.json").write_text(
        json.dumps([r.model_dump() for r in results] + aggregate(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n{len(results)} прогонів × {len(items)} задач; звіт у {out / 'starter-local.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
