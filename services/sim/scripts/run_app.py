"""K11 — один запускач для будь-якого застосунку: конфіг + джерело задач + продовження після зупинки.

Капстон ядра: тут сходяться всі шари, і жоден із них не має власного входу.
  • **що робити** — `AppSpec` (умова з реєстру або JSON): інструменти, routing, суддя, охорона, граф;
  • **над чим** — набір айтемів або корпусна черга;
  • **як довго** — `Governor` (айтеми/токени/долари + kill-switch);
  • **як продовжити** — черга: задачі стають одиницями роботи, тому зупинка лишає решту в ній.

Resume не окремий механізм: він **наслідок** того, що робота живе в черзі, а `ack` іде після роботи.

Запуск:
  uv run python scripts/run_app.py --spec lex-ref-auto@8 --items inject --max-items 2
  uv run python scripts/run_app.py --spec lex-ref-auto@8 --items inject       # добере решту
"""

import argparse
import json
import os
import sys
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

from evalkit.batch import aggregate_results, run_queue  # noqa: E402
from evalkit.checks import split_checks  # noqa: E402
from evalkit.conditions import CONDITIONS, judge_warnings, runner_for  # noqa: E402
from evalkit.harness import load_items  # noqa: E402
from ploshcha_sim.adapters.llm_openai import OpenAICompatLlm  # noqa: E402
from ploshcha_sim.adapters.queue_sqlite import SqliteQueue  # noqa: E402
from ploshcha_sim.domain.governor import Governor  # noqa: E402
from ploshcha_sim.domain.spec import AppSpec  # noqa: E402
from ploshcha_sim.ports.queue import WorkItem  # noqa: E402

ITEMS = Path(__file__).resolve().parents[1] / "evalkit" / "items"
RUNS = ROOT / "docs" / "research" / "eval-runs"


def _spec(name: str) -> AppSpec:
    if name in CONDITIONS:
        return CONDITIONS[name]
    path = Path(name)
    if path.exists():
        return AppSpec(**json.loads(path.read_text(encoding="utf-8")))
    raise SystemExit(f"невідома умова «{name}»; доступні: {sorted(CONDITIONS)[:8]}…")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="назва умови або шлях до JSON з AppSpec")
    ap.add_argument("--items", help="набір із evalkit/items")
    ap.add_argument("--queue", default=None, help="корпусна черга замість набору")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=0.0)
    ap.add_argument("--kill-file", default="evalkit/data/STOP")
    ap.add_argument("--work", default=None, help="файл черги прогону (за замовчуванням із назви)")
    args = ap.parse_args()

    if not args.items and not args.queue:
        raise SystemExit("потрібен --items або --queue")
    if not os.environ.get("LAPA_API_KEY"):
        raise SystemExit("нема LAPA_API_KEY")

    spec = _spec(args.spec)
    lapa = OpenAICompatLlm(model=os.environ["LAPA_MODEL"], base_url=os.environ.get("LAPA_BASE_URL"),
                           api_key=os.environ["LAPA_API_KEY"], structured_mode="json_schema")
    mamay = OpenAICompatLlm(model=os.environ["MAMAY_MODEL"], base_url=os.environ.get("LAPA_BASE_URL"),
                            api_key=os.environ["LAPA_API_KEY"], structured_mode="json_schema")
    runner = runner_for(spec, lapa=lapa, mamay=mamay)

    source = args.items or Path(args.queue).stem
    work_path = args.work or f"evalkit/data/app-{source}-{spec.sha256[:8]}.db"
    work = SqliteQueue(work_path)

    checks: dict[str, list[dict]] = {}
    if args.items:
        for item in load_items(str(ITEMS / f"{args.items}.jsonl")):
            checks[item.id] = item.checks
            work.put(item.id, {"task": item.task})
    else:
        for chunk in SqliteQueue(args.queue).results():
            work.put(chunk.key, {"task": chunk.payload.get("текст", "")})

    def handler(item: WorkItem) -> dict:
        result = runner(item.payload["task"], args.seed)
        outcome, hygiene = ({}, {})
        if item.key in checks:
            outcome, hygiene = split_checks(checks[item.key], result)
        return {
            "findings": [{"answer": result.answer, "outcome": result.outcome}],
            "success": bool(outcome) and all(outcome.values()),
            "hygiene_ok": all(hygiene.values()) if hygiene else True,
            "tokens": result.tokens + result.aux_tokens,
            "notes": result.notes,
        }

    governor = Governor(max_items=args.max_items, max_tokens=args.max_tokens,
                        max_usd=args.max_usd, kill_file=args.kill_file)
    warn = judge_warnings([args.spec]) if args.spec in CONDITIONS else {}
    for name, why in warn.items():
        print(f"  ⚠ {name}: {why}")

    print(f"умова {args.spec} (sha={spec.sha256}) · джерело {source} · черга {work_path}")
    print(f"до прогону: {work.stats()}")
    report = run_queue(work, "app", handler, governor=governor)
    results = work.results()
    summary = aggregate_results(results)
    scored = [r for r in results if r.result and "success" in r.result]
    passed = sum(1 for r in scored if r.result["success"])

    print(f"зроблено {report.done}, провалів {report.failed}, повернуто {report.recovered}")
    print(f"зупинка: {report.stopped or 'черга спорожніла'} · після: {report.stats}")
    if scored:
        print(f"успіх {passed}/{len(scored)} = {passed / len(scored):.3f} · токенів {report.tokens}")
    print(f"покриття {summary['coverage']} · мертвих {len(summary['dead'])}")

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"app-{source}-{spec.sha256[:8]}.json"
    out.write_text(json.dumps({
        "spec": spec.model_dump(), "spec_sha": spec.sha256, "source": source, "seed": args.seed,
        "stopped": report.stopped, "done": report.done, "failed": report.failed,
        "tokens": report.tokens, "success": passed, "scored": len(scored),
        "items": [{"key": r.key, "status": r.status, **(r.result or {})} for r in results],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"звіт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
