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

from evalkit.harness import load_items, orchestrator_runner, run_eval
from evalkit.prompts import REGISTRY, load_prompts
from evalkit.report import format_prompt_report, sensitivity
from ploshcha_sim.adapters import FakeToolbox, PresetEffort, profile_router, single_model_router
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.domain.task import Budget

ITEMS = Path(__file__).resolve().parents[1] / "evalkit" / "items" / "starter.jsonl"
DEFAULT_PROMPTS = ["agent/v2", "agent/v2-tail", "agent/v2-lex", "agent/v2-nocaps", "agent/v2-budget"]


def parse_args(argv):
    prompts, seeds, routing, steps, limit = DEFAULT_PROMPTS, [1, 2, 3], "hetero", 8, None
    for a in argv:
        if a.startswith("--prompts="):
            prompts = a.split("=", 1)[1].split(",")
        elif a.startswith("--seeds="):
            seeds = [int(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--routing="):
            routing = a.split("=", 1)[1]
        elif a.startswith("--max-steps="):
            steps = int(a.split("=", 1)[1])
        elif a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
    return prompts, seeds, routing, steps, limit


def make_llm(model, url, key):
    return OpenAICompatLlm(model=model, base_url=url, api_key=key, structured_mode="json_schema")


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    prompt_ids, seeds, routing, max_steps, limit = parse_args(sys.argv[1:])
    registry = load_prompts(REGISTRY)
    missing = [p for p in prompt_ids if p not in registry]
    if missing:
        print(f"немає в реєстрі: {missing}")
        return 1

    lapa = make_llm(os.environ["LAPA_MODEL"], url, key)
    mamay = make_llm(os.environ["MAMAY_MODEL"], url, key)
    router = (lambda: profile_router(lapa, mamay)) if routing == "hetero" \
        else (lambda: single_model_router(mamay))

    items = load_items(str(ITEMS))
    if limit:
        items = items[:limit]

    runners, prompt_map = {}, {}
    for pid in prompt_ids:
        variant = registry[pid]

        def make_orch(v=variant):
            return Orchestrator(router(), PresetEffort(), FakeToolbox(), verifier=True,
                                system=v.render_system(), tail=v.tail or None,
                                prompt_id=v.id, prompt_sha=v.sha256)
        runners[pid] = orchestrator_runner(make_orch, budget=Budget(max_steps=max_steps))
        prompt_map[pid] = pid

    print(f"{len(items)} задач × {len(prompt_ids)} промптів × {len(seeds)} seed = "
          f"{len(items) * len(prompt_ids) * len(seeds)} прогонів; routing={routing} @{max_steps}\n")
    results = run_eval(items, runners, seeds, prompt_ids=prompt_map)
    print(format_prompt_report(results))

    out = ROOT / "docs" / "research" / "eval-runs"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "endpoint": url,
        "routing": routing,
        "max_steps": max_steps,
        "seeds": seeds,
        "prompts": {pid: registry[pid].sha256 for pid in prompt_ids},
        "sensitivity": sensitivity(results),
        "results": [r.model_dump() for r in results],
    }
    (out / "prompt-ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзвіт у {out / 'prompt-ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
