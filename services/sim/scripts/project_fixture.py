"""Породжує фікстуру подій зі СПРАВЖНЬОГО прогону ядра — без шлюзу, детерміновано.

Це доказ сумісності, а не ілюстрація: фікстуру пише Python-ядро, а перевіряє TS-валідатор контракту
(`pnpm --filter @ploshcha/contract-ts test`). Доки такого файлу не було, контракт 1.1 лишався
гіпотезою — ніхто не знав, чи його поля складаються з того, що ядро реально віддає.

Прогін іде на `FakeLlm` за скриптом, тому нуль токенів і байт-у-байт відтворюваність.

Запуск: uv run python scripts/project_fixture.py --ts 2026-07-31T09:00:00Z
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ploshcha_sim.adapters import FakeLlm, FakeToolbox, InMemoryTrace, PresetEffort
from ploshcha_sim.adapters.projector import project_run
from ploshcha_sim.adapters.router_profile import profile_router
from ploshcha_sim.adapters.tools_lexis import LEXIS_TOOLS
from ploshcha_sim.agents import Orchestrator

OUT = Path(__file__).resolve().parents[3] / "packages" / "fixtures" / "runs" / "projected-run.jsonl"
RUN_ID = "projected-lexis"
SCENE = {"id": "ploshcha", "name": "Площа"}

TASK = ("Речення з української літератури: «Гірськими плаями пливуть ботеї овець».\n"
        "Що означає слово «ботей» у цьому реченні?")


def _call(tool: str, **args) -> str:
    return json.dumps({"tool": tool, **args}, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True)
    args = ap.parse_args()

    script = [
        _call("словник", слово="ботей"),
        _call("final_answer", text="У цьому реченні «ботей» означає отару овець."),
        "У цьому реченні «ботей» означає отару овець.",
        json.dumps({"kind": "supported", "reason": "стаття підтверджує"}, ensure_ascii=False),
    ]
    llm = FakeLlm(script, model="fixture")
    trace = InMemoryTrace()
    orch = Orchestrator(profile_router(llm, llm), PresetEffort(),
                        FakeToolbox(tools=LEXIS_TOOLS), trace=trace, run_id=RUN_ID,
                        verify_mode="grounded", answer_channel="text")
    result = orch.run(TASK, seed=1)

    events = project_run(list(trace.records), result, run_id=RUN_ID, ts=args.ts, scene=SCENE,
                         started_at=args.ts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
                   encoding="utf-8")

    kinds = {}
    for e in events:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    print(f"{len(events)} подій -> {OUT}")
    print(f"типи: {kinds}")
    print(f"outcome={result.outcome} evidence={result.evidence} verdict={result.verdict_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
