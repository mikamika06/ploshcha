from collections import defaultdict

from .harness import EvalResult


class Aggregate:
    def __init__(self, condition: str):
        self.condition = condition
        self.n = 0
        self.success = 0
        self.steps = 0
        self.tokens = 0
        self.pass_k = 0.0
        self.by_category: dict[str, tuple[int, int]] = {}

    @property
    def success_rate(self) -> float:
        return self.success / self.n if self.n else 0.0

    @property
    def avg_steps(self) -> float:
        return self.steps / self.n if self.n else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.tokens / self.n if self.n else 0.0

    @property
    def quality_per_ktoken(self) -> float:
        return (self.success_rate / (self.avg_tokens / 1000)) if self.avg_tokens else 0.0

    def as_dict(self) -> dict:
        return {
            "condition": self.condition, "n": self.n,
            "success_rate": round(self.success_rate, 3),
            "pass_k": round(self.pass_k, 3),
            "avg_steps": round(self.avg_steps, 2),
            "avg_tokens": round(self.avg_tokens, 1),
            "quality_per_ktoken": round(self.quality_per_ktoken, 4),
            "by_category": {c: round(s / n, 3) for c, (s, n) in self.by_category.items()},
        }


def _pass_k(rows: list[EvalResult]) -> float:
    by_item: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_item[r.item_id].append(r.success)
    if not by_item:
        return 0.0
    return sum(1 for outcomes in by_item.values() if all(outcomes)) / len(by_item)


def aggregate(results: list[EvalResult]) -> list[dict]:
    by_cond: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_cond[r.condition].append(r)

    reports = []
    for condition, rows in by_cond.items():
        agg = Aggregate(condition)
        cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in rows:
            agg.n += 1
            agg.success += int(r.success)
            agg.steps += r.steps
            agg.tokens += r.tokens
            cat[r.category][0] += int(r.success)
            cat[r.category][1] += 1
        agg.pass_k = _pass_k(rows)
        agg.by_category = {c: (s, n) for c, (s, n) in cat.items()}
        reports.append(agg.as_dict())
    return reports


def by_prompt(results: list[EvalResult]) -> dict[str, dict]:
    groups: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        groups[r.prompt_id or r.condition].append(r)
    out = {}
    for key, rows in groups.items():
        n = len(rows)
        success = sum(1 for r in rows if r.success)
        incidents: dict[str, int] = defaultdict(int)
        for r in rows:
            for code in r.incidents:
                incidents[code] += 1
        out[key] = {
            "n": n,
            "success_rate": round(success / n, 3) if n else 0.0,
            "pass_k": round(_pass_k(rows), 3),
            "avg_tokens": round(sum(r.tokens for r in rows) / n, 1) if n else 0.0,
            "incidents": dict(incidents),
            "partial": sum(1 for r in rows if r.partial),
        }
    return out


def sensitivity(results: list[EvalResult]) -> dict:
    groups = by_prompt(results)
    rates = {k: v["success_rate"] for k, v in groups.items()}
    if not rates:
        return {"prompts": 0}
    best = max(rates, key=lambda k: rates[k])
    worst = min(rates, key=lambda k: rates[k])
    cells: dict[tuple[str, int], set[bool]] = defaultdict(set)
    for r in results:
        cells[(r.item_id, r.seed)].add(r.success)
    flipped = sum(1 for outcomes in cells.values() if len(outcomes) > 1)
    return {
        "prompts": len(rates),
        "spread": round(rates[best] - rates[worst], 3),
        "best": best,
        "best_rate": rates[best],
        "worst": worst,
        "worst_rate": rates[worst],
        "flip_rate": round(flipped / len(cells), 3) if cells else 0.0,
    }


def format_prompt_report(results: list[EvalResult]) -> str:
    groups = by_prompt(results)
    lines = [f"{'prompt':<20} {'n':>4} {'success':>8} {'pass^k':>7} {'tokens':>8}  incidents"]
    lines.append("-" * 72)
    for key in sorted(groups, key=lambda k: -groups[k]["success_rate"]):
        g = groups[key]
        lines.append(f"{key:<20} {g['n']:>4} {g['success_rate']:>8.3f} {g['pass_k']:>7.3f} "
                     f"{g['avg_tokens']:>8.1f}  {g['incidents'] or '—'}")
    s = sensitivity(results)
    lines.append("")
    lines.append(f"spread={s['spread']}  flip_rate={s['flip_rate']}  "
                 f"best={s['best']}({s['best_rate']})  worst={s['worst']}({s['worst_rate']})")
    return "\n".join(lines)


def format_report(results: list[EvalResult]) -> str:
    reports = aggregate(results)
    lines = []
    header = f"{'condition':<18} {'n':>4} {'success':>8} {'pass^k':>7} {'steps':>6} {'tokens':>8} {'q/ktok':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for rep in reports:
        lines.append(
            f"{rep['condition']:<18} {rep['n']:>4} {rep['success_rate']:>8.3f} "
            f"{rep['pass_k']:>7.3f} {rep['avg_steps']:>6.2f} {rep['avg_tokens']:>8.1f} "
            f"{rep['quality_per_ktoken']:>7.4f}"
        )
    return "\n".join(lines)
