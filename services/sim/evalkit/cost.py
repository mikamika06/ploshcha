"""Ціна прогону в доларах.

Прайс-мапа живе у вимірювальному шарі, а не в колесі: ціна — властивість постачальника
й моменту часу, а не системи. Зміна цін не інвалідовує замір, лише його економічну інтерпретацію.

Для гетерогенної умови точна ціна невідома: `EvalResult` не зберігає, скільки токенів обробив
кожен ярус. Тому замість вгаданої частки повертаємо ІНТЕРВАЛ [дешевший ярус, дорожчий ярус].
Точкова оцінка з'явиться, коли `TaskResult` понесе `tokens_by_tier` (борг 25).
"""

USD_PER_MTOK = {"lapa": 0.10, "mamay": 0.30}
UNKNOWN_RATE = max(USD_PER_MTOK.values())


def rate_bounds(routing: str) -> tuple[float, float]:
    if routing == "hetero":
        return min(USD_PER_MTOK.values()), max(USD_PER_MTOK.values())
    r = USD_PER_MTOK.get(routing, UNKNOWN_RATE)
    return r, r


def cost_usd(tokens: int, routing: str) -> tuple[float, float]:
    lo, hi = rate_bounds(routing)
    return tokens / 1_000_000 * lo, tokens / 1_000_000 * hi


def tokens_by_condition(results) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        out[r.condition] = out.get(r.condition, 0) + r.tokens + r.aux_tokens
    return out


def cost_of_results(results, specs) -> dict[str, tuple[float, float]]:
    """Ціна на умову як інтервал. `specs` — мапа умова -> AppSpec."""
    totals = tokens_by_condition(results)
    return {c: cost_usd(t, getattr(specs.get(c), "routing", "unknown"))
            for c, t in totals.items()}


def usd_per_success(results, specs) -> dict[str, tuple[float, float]]:
    cost = cost_of_results(results, specs)
    wins: dict[str, int] = {}
    for r in results:
        wins[r.condition] = wins.get(r.condition, 0) + int(r.success)
    return {c: ((lo / wins[c], hi / wins[c]) if wins.get(c) else (float("inf"), float("inf")))
            for c, (lo, hi) in cost.items()}


def format_cost(results, specs) -> str:
    lines = ["умова                     токени      $ (інтервал)        $/успіх"]
    per = usd_per_success(results, specs)
    cost = cost_of_results(results, specs)
    toks = tokens_by_condition(results)
    for c in sorted(cost):
        lo, hi = cost[c]
        plo, phi = per[c]
        span = f"{lo:.5f}-{hi:.5f}" if hi > lo else f"{lo:.5f}"
        pspan = "—" if plo == float("inf") else (f"{plo:.5f}-{phi:.5f}" if phi > plo else f"{plo:.5f}")
        lines.append(f"{c:<24}{toks[c]:>8}   {span:>18}   {pspan}")
    return "\n".join(lines)
