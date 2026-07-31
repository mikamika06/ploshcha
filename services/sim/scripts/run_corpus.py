"""Прогін по корпусній черзі: узяти документ, обробити, підтвердити — з лімітом і продовженням.

Обробник підключається, а не вшитий: `years` — детермінований (регулярка, нуль токенів), потрібний
щоб перевірити ЗШИВКУ інгест → черга → пакет → агрегація на справжніх даних без витрат. Прив'язка
МОДЕЛІ до цього корпусу — вже застосунок (A3), не машинерія масштабу.

Що тут доводиться:
  • зупинка за лімітом лишає решту В ЧЕРЗІ (не губить);
  • повторний запуск продовжує з того місця;
  • kill-switch спиняє негайно, незалежно від залишку бюджету;
  • агрегація називає `dead` і порожні вголос.

Запуск:
  uv run python scripts/run_corpus.py --handler years --max-items 3
  uv run python scripts/run_corpus.py --handler years          # продовжує решту
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit.batch import aggregate_results, run_queue  # noqa: E402
from ploshcha_sim.adapters.queue_sqlite import SqliteQueue  # noqa: E402
from ploshcha_sim.domain.governor import Governor  # noqa: E402
from ploshcha_sim.ports.queue import WorkItem  # noqa: E402

YEAR = re.compile(r"\b(1[0-9]{3})\b")
REPORT = Path(__file__).resolve().parents[1] / "evalkit" / "data" / "corpus-report.json"


def years_handler(item: WorkItem) -> dict:
    """Детермінований обробник: витягує роки. Нуль токенів, тому зшивку можна ганяти скільки треба."""
    text = item.payload.get("текст", "")
    found = sorted(set(YEAR.findall(text)))
    return {"findings": [{"документ": item.payload.get("назва"), "роки": found}] if found else [],
            "tokens": 0}


HANDLERS = {"years": years_handler}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="evalkit/data/corpus.db")
    ap.add_argument("--handler", default="years", choices=sorted(HANDLERS))
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=0.0)
    ap.add_argument("--kill-file", default="evalkit/data/STOP")
    ap.add_argument("--worker", default="w1")
    args = ap.parse_args()

    queue = SqliteQueue(args.queue)
    governor = Governor(max_items=args.max_items, max_tokens=args.max_tokens,
                        max_usd=args.max_usd, kill_file=args.kill_file)
    before = queue.stats()
    report = run_queue(queue, args.worker, HANDLERS[args.handler], governor=governor)
    summary = aggregate_results(queue.results())

    print(f"черга до: {before}")
    print(f"зроблено {report.done}, провалів {report.failed}, повернуто після краху {report.recovered}")
    print(f"зупинка: {report.stopped or 'черга спорожніла'}")
    print(f"черга після: {report.stats}")
    print(f"покриття {summary['coverage']}, знахідок {len(summary['findings'])}, "
          f"мертвих {len(summary['dead'])}, порожніх {len(summary['empty'])}")
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
