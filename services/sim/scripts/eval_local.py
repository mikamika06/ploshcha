import hashlib
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

from evalkit.conditions import (
    CONDITIONS,
    PAIRS,
    grid,
    judge_warnings,
    prompt_ids,
    shape_warnings,
    spec_shas,
)
from evalkit.cost import RATIO_RANGE, format_cost, prompt_share, sensitivity
from evalkit.harness import load_items, run_eval
from evalkit.report import aggregate, format_report, paired
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm

ITEMS_DIR = Path(__file__).resolve().parents[1] / "evalkit" / "items"


def parse_args(argv):
    seeds, limit, items, only = [1, 2, 3], None, "starter", None
    for a in argv:
        if a.startswith("--seeds="):
            seeds = [int(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
        elif a.startswith("--items="):
            items = a.split("=", 1)[1]
        elif a.startswith("--conditions="):
            only = a.split("=", 1)[1].split(",")
    return seeds, limit, items, only


def make_llm(model, url, key):
    return OpenAICompatLlm(model=model, base_url=url, api_key=key, structured_mode="json_schema")


def main():
    key, url = os.environ.get("LAPA_API_KEY"), os.environ.get("LAPA_BASE_URL")
    if not key:
        print("нема LAPA_API_KEY")
        return 1
    seeds, limit, items_name, only = parse_args(sys.argv[1:])
    unknown = [c for c in (only or []) if c not in CONDITIONS]
    if unknown:
        print(f"невідомі умови: {unknown}; доступні: {sorted(CONDITIONS)}")
        return 1

    lapa = make_llm(os.environ["LAPA_MODEL"], url, key)
    mamay = make_llm(os.environ["MAMAY_MODEL"], url, key)
    runners = grid(only, lapa=lapa, mamay=mamay)
    specs = {name: CONDITIONS[name] for name in runners}

    items = load_items(str(ITEMS_DIR / f"{items_name}.jsonl"))
    if limit:
        items = items[:limit]

    warnings = shape_warnings(runners)
    judges = judge_warnings(runners)
    for name, why in judges.items():
        print(f"  ⚠ {name}: {why}")
    for name, spec in specs.items():
        print(f"{name:<24} spec={spec.sha256} промпт={spec.prompt_id} "
              f"режим={spec.mode} routing={spec.routing} канал={spec.answer_channel}"
              + (f"  ⚠ {' '.join(warnings[name])}" if name in warnings else ""))
    print(f"\n{len(items)} задач × {len(runners)} умов × {len(seeds)} seed = "
          f"{len(items) * len(runners) * len(seeds)} прогонів\n")

    results = run_eval(items, runners, seeds,
                       prompt_ids=prompt_ids(runners), spec_shas=spec_shas(runners))
    print(format_report(results))
    print()
    print(format_cost(results, specs))
    shares = prompt_share(results)
    if shares:
        print("частка промпту: " + " ".join(f"{c}={v:.0%}" for c, v in sorted(shares.items())))
    for base, treat in PAIRS:
        if base in runners and treat in runners:
            s = sensitivity(results, base, treat)
            if s:
                spans = " ".join(f"×{r:.1f}:{v['treat_cheaper_by']:+.1%}"
                                 for r, v in s["ratios"].items())
                print(f"чутливість {base} -> {treat}: {spans} "
                      f"{'стійко' if s['robust'] else 'ПЕРЕВЕРТАЄТЬСЯ'}")
    print()
    for base, treat in PAIRS:
        if base in runners and treat in runners:
            pr = paired(results, base, treat)
            print(f"паровано {base} -> {treat}: клітинок={pr['cells']} "
                  f"вилікувано={pr['fixed']} зламано={pr['broke']} net={pr['net']} "
                  f"(з інцидентами={pr['incident_cells']}, врятовано={pr['rescued_with_incident']})")

    out = ROOT / "docs" / "research" / "eval-runs"
    out.mkdir(parents=True, exist_ok=True)
    # Ключ від складу умов: інакше наступний прогін перезаписує сирі виводи попереднього,
    # і офлайн-перерахунок після зміни предикатів стає неможливим (урок V6 §7).
    tag = hashlib.sha256("|".join(sorted(runners)).encode("utf-8")).hexdigest()[:8]
    (out / f"{items_name}-local-{tag}.json").write_text(
        json.dumps({"items": items_name, "seeds": seeds,
                    "specs": {n: s.model_dump(mode="json") | {"sha": s.sha256}
                              for n, s in specs.items()},
                    "shape_warnings": warnings,
                    "aggregate": aggregate(results),
                    "paired": [paired(results, b, t) for b, t in PAIRS
                               if b in runners and t in runners],
                    "results": [r.model_dump() for r in results]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n{len(results)} прогонів × {len(items)} задач; звіт у {out / f'{items_name}-local-{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
