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

from evalkit.harness import (
    gated_runner,
    load_items,
    orchestrator_runner,
    run_eval,
    single_call_runner,
)
from evalkit.prompts import resolve
from evalkit.report import aggregate, format_report, paired
from ploshcha_sim.adapters import FakeToolbox, PresetEffort, profile_router, single_model_router
from ploshcha_sim.adapters.planner_skeleton import SkeletonPlanner
from ploshcha_sim.adapters.tools_fake import DEFAULT_TOOLS
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm
from ploshcha_sim.agents import Orchestrator
from ploshcha_sim.domain.task import Budget

ITEMS_DIR = Path(__file__).resolve().parents[1] / "evalkit" / "items"


def parse_args(argv):
    seeds, limit, items, prompt = [1, 2, 3], None, "starter", PROMPT_ID
    for a in argv:
        if a.startswith("--seeds="):
            seeds = [int(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
        elif a.startswith("--items="):
            items = a.split("=", 1)[1]
        elif a.startswith("--prompt="):
            prompt = a.split("=", 1)[1]
    return seeds, limit, items, prompt

PROMPT_ID = "agent/v2"
PAIRS = (("mamay@8", "mamay+rec@8"), ("hetero@8", "hetero+rec@8"),
         ("hetero@8", "gate-notools-mamay"),
         ("hetero@8", "gate-tools-hetero"),
         ("hetero-plan@8", "hetero-textans@8"),
         ("hetero@8", "hetero-textfull@8"))
NO_TOOLS = [t for t in DEFAULT_TOOLS if t.name == "final_answer"]


def make_llm(model, url, key):
    return OpenAICompatLlm(model=model, base_url=url, api_key=key, structured_mode="json_schema")


def orch_cond(router_factory, verifier, *, recovery=False, max_steps=5, prompt=None,
              answer_channel="schema", planner=None, answer_prompt="answer/plain"):
    variant = prompt or resolve(PROMPT_ID)
    answer_instruction = resolve(answer_prompt).render_system()

    def make_orch():
        return Orchestrator(router_factory(), PresetEffort(), FakeToolbox(),
                            planner=planner() if planner else None,
                            verifier=verifier, system=variant.render_system(),
                            tail=variant.tail or None, prompt_id=variant.id,
                            prompt_sha=variant.sha256, recovery=recovery,
                            answer_channel=answer_channel,
                            answer_instruction=answer_instruction)
    return orchestrator_runner(make_orch, budget=Budget(max_steps=max_steps))


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    seeds, limit, items_name, prompt_id = parse_args(sys.argv[1:])
    variant = resolve(prompt_id)
    print(f"промпт {variant.id} ({variant.slot}, sha={variant.sha256})")
    lapa = make_llm(os.environ["LAPA_MODEL"], url, key)
    mamay = make_llm(os.environ["MAMAY_MODEL"], url, key)

    items = load_items(str(ITEMS_DIR / f"{items_name}.jsonl"))
    if limit:
        items = items[:limit]
    runners = {
        "single-mamay": single_call_runner(mamay, system=variant.render_system()),
        "single-lapa": single_call_runner(lapa, system=variant.render_system()),
        "mamay@5": orch_cond(lambda: single_model_router(mamay), True, prompt=variant),
        "mamay@8": orch_cond(lambda: single_model_router(mamay), True, max_steps=8, prompt=variant),
        "mamay+rec@8": orch_cond(lambda: single_model_router(mamay), True, recovery=True, max_steps=8, prompt=variant),
        "hetero@5": orch_cond(lambda: profile_router(lapa, mamay), True, prompt=variant),
        "hetero@8": orch_cond(lambda: profile_router(lapa, mamay), True, max_steps=8, prompt=variant),
        "hetero+rec@8": orch_cond(lambda: profile_router(lapa, mamay), True, recovery=True, max_steps=8, prompt=variant),
        "hetero-nov@8": orch_cond(lambda: profile_router(lapa, mamay), False, max_steps=8, prompt=variant),
        "gate-notools-mamay": gated_runner(
            mamay, FakeToolbox(tools=NO_TOOLS), system=variant.render_system(),
            loop_runner=orch_cond(lambda: profile_router(lapa, mamay), True, max_steps=8,
                                  prompt=variant)),
        "gate-notools-lapa": gated_runner(
            lapa, FakeToolbox(tools=NO_TOOLS), system=variant.render_system(),
            loop_runner=orch_cond(lambda: profile_router(lapa, mamay), True, max_steps=8,
                                  prompt=variant)),
        "hetero-plan@8": orch_cond(lambda: profile_router(lapa, mamay), True, max_steps=8,
                                   prompt=variant, planner=SkeletonPlanner),
        "hetero-textans@8": orch_cond(lambda: profile_router(lapa, mamay), True, max_steps=8,
                                      prompt=variant, planner=SkeletonPlanner,
                                      answer_channel="text"),
        "hetero-textfull@8": orch_cond(lambda: profile_router(lapa, mamay), True, max_steps=8,
                                       prompt=variant, answer_channel="text",
                                       answer_prompt="answer/full"),
        "gate-tools-hetero": gated_runner(
            mamay, FakeToolbox(), system=variant.render_system(),
            loop_runner=orch_cond(lambda: profile_router(lapa, mamay), True, max_steps=8,
                                  prompt=variant)),
    }
    print(f"{len(items)} задач × {len(runners)} умов × {len(seeds)} seed = "
          f"{len(items) * len(runners) * len(seeds)} прогонів\n")
    results = run_eval(items, runners, seeds,
                       prompt_ids={name: variant.id for name in runners})
    print(format_report(results))
    print()
    for base, treat in PAIRS:
        if base in runners and treat in runners:
            pr = paired(results, base, treat)
            print(f"паровано {base} -> {treat}: клітинок={pr['cells']} "
                  f"вилікувано={pr['fixed']} зламано={pr['broke']} net={pr['net']} "
                  f"(з інцидентами={pr['incident_cells']}, врятовано={pr['rescued_with_incident']})")
    out = ROOT / "docs" / "research" / "eval-runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{items_name}-local.json").write_text(
        json.dumps({"prompt": variant.id, "prompt_sha": variant.sha256, "items": items_name, "seeds": seeds,
                    "aggregate": aggregate(results),
                    "paired": [paired(results, b, t) for b, t in PAIRS
                               if b in runners and t in runners],
                    "results": [r.model_dump() for r in results]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n{len(results)} прогонів × {len(items)} задач; звіт у {out / (items_name + '-local.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
