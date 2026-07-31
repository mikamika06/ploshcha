"""Переграє СУДДЮ на збережених відповідях — без повторного прогону завдання.

Судівські експерименти інакше коштують три кроки на клітинку замість одного виклику. Тут беруться
збережені `answer` + `scratch` зі звіту, і викликається лише `verify()` з обраним режимом і ярусом.
Це також ідеально паровано: обидва судді дістають БАЙТ-У-БАЙТ той самий вхід.

Запуск:
  uv run python scripts/rejudge.py <звіт.json> --condition lex-am-jm --mode grounded --lane mamay
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_env(ROOT / ".env")

from evalkit.checks import split_checks  # noqa: E402
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm  # noqa: E402
from ploshcha_sim.adapters.router_profile import PresetEffort, single_model_router  # noqa: E402
from ploshcha_sim.agents.verify import verify  # noqa: E402
from ploshcha_sim.domain.evidence import evidence_state  # noqa: E402
from ploshcha_sim.domain.task import TaskResult  # noqa: E402

ITEMS = Path(__file__).resolve().parents[1] / "evalkit" / "items"


def _truth(item: dict, row: dict) -> bool:
    result = TaskResult(answer=row["answer"], accepted=row["accepted"], steps=row["steps"],
                        scratch=row.get("scratch") or [], degraded=row["degraded"],
                        partial=row["partial"])
    return all(split_checks(item["checks"], result)[0].values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--items", default="lexis")
    ap.add_argument("--mode", default="grounded")
    ap.add_argument("--lane", default="mamay", choices=("mamay", "lapa"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    items = {json.loads(l)["id"]: json.loads(l)
             for l in (ITEMS / f"{args.items}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    rows = [r for r in json.loads(Path(args.report).read_text(encoding="utf-8"))["results"]
            if r["condition"] == args.condition and r["seed"] == args.seed]
    if not rows:
        print(f"у звіті немає умови «{args.condition}» з seed={args.seed}")
        return 1
    if not any(r.get("scratch") for r in rows):
        print("у звіті немає `scratch` — переграти суддю не можна, потрібен новий прогін")
        return 1

    model = os.environ["MAMAY_MODEL"] if args.lane == "mamay" else os.environ["LAPA_MODEL"]
    llm = OpenAICompatLlm(model=model, base_url=os.environ.get("LAPA_BASE_URL"),
                          api_key=os.environ["LAPA_API_KEY"], structured_mode="json_schema")
    router, effort = single_model_router(llm, lane=args.lane), PresetEffort()

    kinds, agree, fa, fr, flips = Counter(), 0, 0, 0, []
    for row in rows:
        item = items[row["item_id"]]
        scratch = row.get("scratch") or []
        verdict = verify(item["task"], row["answer"], router, effort, evidence=scratch,
                         seed=args.seed, mode=args.mode, grounding="required",
                         absent=evidence_state(scratch) is False)
        truth = _truth(item, row)
        kinds[verdict.kind] += 1
        agree += verdict.accepted == truth
        fa += verdict.accepted and not truth
        fr += truth and not verdict.accepted
        if bool(row["accepted"]) != verdict.accepted:
            flips.append((row["item_id"], row["verdict_kind"], verdict.kind, truth))

    n = len(rows)
    print(f"{args.condition} | суддя={args.lane} режим={args.mode} | n={n}")
    print(f"  згода з чеком {agree}/{n}  хибно прийняв {fa}  хибно відкинув {fr}")
    print(f"  види: {dict(kinds)}")
    if flips:
        print("  змінилось проти збереженого вердикту:")
        for item_id, was, now, truth in flips:
            print(f"    {item_id:<22}{was} -> {now}   (чек каже: {truth})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
