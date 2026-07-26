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

from ploshcha_sim.adapters import FakeToolbox, InMemoryTrace, PresetEffort, profile_router, single_model_router
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.domain.task import Budget

TASKS = [
    "Перевір, чи Битва під Крутами відбулася 1918 року.",
    "Дізнайся рік початку Хмельниччини і перевір, чи це 1648.",
    "Хто такий Тарас Шевченко?",
]


def make_llm(model, url, key):
    return OpenAICompatLlm(model=model, base_url=url, api_key=key, structured_mode="json_schema")


def show(label, task, result, trace):
    print(f"\n  [{label}] {task}")
    for r in trace.records:
        p = r.parsed if r.parsed else (r.reject_reason or "-")
        print(f"    {r.stage:<7} {r.model[:22]:<22} {json.dumps(p, ensure_ascii=False)[:70]}")
    print(f"    => answer={result.answer!r} accepted={result.accepted} degraded={result.degraded} "
          f"steps={result.steps}")


def run_config(label, router, task):
    trace = InMemoryTrace()
    orch = Orchestrator(router, PresetEffort(), FakeToolbox(), verifier=True, trace=trace)
    result = orch.run(task, seed=1, budget=Budget(max_steps=5))
    show(label, task, result, trace)


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    lapa = make_llm(os.environ["LAPA_MODEL"], url, key)
    mamay = make_llm(os.environ["MAMAY_MODEL"], url, key)

    print("=" * 80)
    print("КОНФІГ A: усе на Mamay (single_model_router)")
    print("=" * 80)
    for task in TASKS:
        run_config("all-mamay", single_model_router(mamay), task)

    print("\n" + "=" * 80)
    print("КОНФІГ B: гетерогенний (select->Lapa, judge->Mamay)")
    print("=" * 80)
    for task in TASKS:
        run_config("hetero", profile_router(lapa, mamay), task)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
