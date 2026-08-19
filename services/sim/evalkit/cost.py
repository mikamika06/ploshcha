"""Ціна прогону в доларах.

Прайс-мапа живе у вимірювальному шарі, а не в колесі: ціна — властивість постачальника
й моменту часу, а не системи. Зміна цін не інвалідовує замір, лише його економічну інтерпретацію.

★ ДЖЕРЕЛО ЦІН (2026-07-29). Lapathoniia **не публікує** прайс (бета, видимість = тижневе
використання токенів), тому власних цін немає. Використано **ринковий проксі за розміром базової
моделі**: Lapa побудована на Gemma-3-12B, Mamay — на Gemma-3-27B, а serverless-ціни цих моделей
відомі: 12B ≈ $0.05/Mtok вхід і $0.15/Mtok вихід, 27B ≈ $0.08 і $0.16
(pricepertoken.com / openrouter, липень 2026).

**Це проксі, не факт.** Головне, що з нього випливає: справжнє співвідношення ярусів ~1.1-1.6×, а не
3×, як я спершу вписав навмання. Різниця критична — під вигаданим 3× гетерогенний routing виглядав
удвічі дешевшим, під реальним проксі виграш падає до одиниць відсотків. Тому будь-яке економічне
твердження супроводжується `sensitivity()`: чи витримує висновок увесь діапазон співвідношень.

Вхід і вихід рахуються ОКРЕМО, бо в наших прогонах промпт домінує (історія кроків), а різниця цін
між ярусами на вході (1.6×) майже вдвічі більша, ніж на виході (1.07×).
"""

PRICES = {
    "lapa": (0.05, 0.15),
    "mamay": (0.08, 0.16),
}
UNKNOWN_PRICE = (0.08, 0.16)
RATIO_RANGE = (1.0, 3.0)


def price(lane: str) -> tuple[float, float]:
    return PRICES.get(lane, UNKNOWN_PRICE)


def blended(lane: str, prompt_share: float = 0.8) -> float:
    """Ставка за Mtok при заданій частці промпту — для інтервалів, де розкладки немає."""
    p_in, p_out = price(lane)
    return p_in * prompt_share + p_out * (1.0 - prompt_share)


def rate_bounds(routing: str, prompt_share: float = 0.8) -> tuple[float, float]:
    if routing == "hetero":
        rates = [blended(lane, prompt_share) for lane in PRICES]
        return min(rates), max(rates)
    r = blended(routing if routing in PRICES else "unknown", prompt_share)
    return r, r


def cost_usd(tokens: int, routing: str) -> tuple[float, float]:
    lo, hi = rate_bounds(routing)
    return tokens / 1_000_000 * lo, tokens / 1_000_000 * hi


def lane_cost(by_lane: dict[str, int], prompt_by_lane: dict[str, int] | None = None) -> float:
    """Точна ціна: вхід і вихід за окремими ставками свого ярусу."""
    prompts = prompt_by_lane or {}
    total = 0.0
    for lane, tokens in by_lane.items():
        p_in, p_out = price(lane)
        prompt = min(prompts.get(lane, 0), tokens)
        total += (prompt * p_in + (tokens - prompt) * p_out) / 1_000_000
    return total


ROLE_OF_STAGE = {
    "parse": "executor", "classify": "executor", "select": "executor",
    "ground": "executor", "gate": "executor", "speak": "executor",
    "decide": "orchestrator", "generate": "orchestrator", "synthesize": "orchestrator",
    "judge": "verifier",
    "mem_read": "memory", "mem_write": "memory", "recall": "memory",
    "importance": "memory", "reflect": "memory",
}
ROLE_ORDER = ("orchestrator", "executor", "verifier", "memory", "other")


def role_of(stage: str) -> str:
    return ROLE_OF_STAGE.get(stage, "other")


def _fold_roles(by_stage: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for stage, tokens in (by_stage or {}).items():
        role = role_of(stage)
        out[role] = out.get(role, 0) + tokens
    return out


def roles_by_condition(results, field: str = "tokens_by_stage") -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        acc = out.setdefault(r.condition, {})
        for role, tokens in _fold_roles(getattr(r, field, None) or {}).items():
            acc[role] = acc.get(role, 0) + tokens
    return out


def stages_by_condition(results, field: str = "tokens_by_stage") -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        acc = out.setdefault(r.condition, {})
        for stage, tokens in (getattr(r, field, None) or {}).items():
            acc[stage] = acc.get(stage, 0) + tokens
    return out


def role_lane_by_condition(results) -> dict[str, dict[tuple[str, str], int]]:
    out: dict[str, dict[tuple[str, str], int]] = {}
    for r in results:
        acc = out.setdefault(r.condition, {})
        for pair, tokens in (getattr(r, "tokens_by_stage_lane", None) or {}).items():
            stage, _, lane = pair.partition("|")
            key = (role_of(stage), lane or "unknown")
            acc[key] = acc.get(key, 0) + tokens
    return out


def role_lane_prompts(results) -> dict[str, dict[tuple[str, str], int]]:
    out: dict[str, dict[tuple[str, str], int]] = {}
    for r in results:
        acc = out.setdefault(r.condition, {})
        for pair, tokens in (getattr(r, "prompt_by_stage_lane", None) or {}).items():
            stage, _, lane = pair.partition("|")
            key = (role_of(stage), lane or "unknown")
            acc[key] = acc.get(key, 0) + tokens
    return out


def role_attribution_gap(results) -> dict[str, int]:
    gaps: dict[str, int] = {}
    for r in results:
        by_stage = sum((getattr(r, "tokens_by_stage", None) or {}).values())
        total = r.tokens + r.aux_tokens
        if by_stage != total:
            gaps[r.condition] = gaps.get(r.condition, 0) + (total - by_stage)
    return gaps


def role_cost(results) -> dict[str, dict[str, float]]:
    cross, prompts = role_lane_by_condition(results), role_lane_prompts(results)
    out: dict[str, dict[str, float]] = {}
    for cond, cells in cross.items():
        acc = out.setdefault(cond, {})
        for (role, lane), tokens in cells.items():
            p_in, p_out = price(lane)
            prompt = min(prompts.get(cond, {}).get((role, lane), 0), tokens)
            acc[role] = acc.get(role, 0.0) + (prompt * p_in + (tokens - prompt) * p_out) / 1_000_000
    return out


def format_roles(results) -> str:
    cross, prompts = role_lane_by_condition(results), role_lane_prompts(results)
    costs, gaps = role_cost(results), role_attribution_gap(results)
    lines = ["умова                     роль          ярус      токени  частка  промпт%        $"]
    for cond in sorted(set(cross) | set(gaps)):
        cells = cross.get(cond, {})
        total = sum(cells.values())
        if not total:
            lines.append(f"{cond:<24}  ★ НЕ ВІДНЕСЕНО {gaps.get(cond, 0)} токенів — "
                         f"розкладки по ролях немає взагалі")
            continue
        ordered = sorted(cells, key=lambda k: (ROLE_ORDER.index(k[0])
                                               if k[0] in ROLE_ORDER else 99, k[1]))
        for role, lane in ordered:
            tokens = cells[(role, lane)]
            pr = prompts.get(cond, {}).get((role, lane), 0)
            p_in, p_out = price(lane)
            prompt = min(pr, tokens)
            usd = (prompt * p_in + (tokens - prompt) * p_out) / 1_000_000
            lines.append(f"{cond:<24}  {role:<12}  {lane:<8}{tokens:>8}  {tokens/total:>6.0%}  "
                         f"{(prompt/tokens if tokens else 0):>6.0%}  {usd:>9.5f}")
        lines.append(f"{cond:<24}  ── усього {total:>8} токенів, "
                     f"${sum(costs.get(cond, {}).values()):.5f}")
        if gaps.get(cond):
            lines.append(f"{cond:<24}  ★ НЕ ВІДНЕСЕНО {gaps[cond]} токенів — дірка в обліку")
    return "\n".join(lines)


def tokens_by_condition(results) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        out[r.condition] = out.get(r.condition, 0) + r.tokens + r.aux_tokens
    return out


def lanes_by_condition(results) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        acc = out.setdefault(r.condition, {})
        for lane, tokens in (getattr(r, "tokens_by_lane", None) or {}).items():
            acc[lane] = acc.get(lane, 0) + tokens
    return out


def prompts_by_condition(results) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        acc = out.setdefault(r.condition, {})
        for lane, tokens in (getattr(r, "prompt_by_lane", None) or {}).items():
            acc[lane] = acc.get(lane, 0) + tokens
    return out


def prompt_share(results) -> dict[str, float]:
    lanes, prompts = lanes_by_condition(results), prompts_by_condition(results)
    out = {}
    for cond, by_lane in lanes.items():
        total = sum(by_lane.values())
        if total:
            out[cond] = sum(prompts.get(cond, {}).values()) / total
    return out


def attributed(by_lane: dict[str, int]) -> bool:
    """Розкладка корисна лише якщо в ній немає «unknown» і вона непорожня."""
    return bool(by_lane) and "unknown" not in by_lane


def cost_of_results(results, specs) -> dict[str, tuple[float, float]]:
    """Ціна на умову: точка (lo == hi), якщо ярус відомий; інакше інтервал."""
    totals = tokens_by_condition(results)
    lanes = lanes_by_condition(results)
    out = {}
    for cond, tokens in totals.items():
        by_lane = lanes.get(cond, {})
        if attributed(by_lane):
            exact = lane_cost(by_lane, prompts_by_condition(results).get(cond, {}))
            out[cond] = (exact, exact)
        else:
            out[cond] = cost_usd(tokens, getattr(specs.get(cond), "routing", "unknown"))
    return out


def lane_share(results) -> dict[str, dict[str, float]]:
    out = {}
    for cond, by_lane in lanes_by_condition(results).items():
        total = sum(by_lane.values())
        if total:
            out[cond] = {lane: t / total for lane, t in sorted(by_lane.items())}
    return out


def usd_per_success(results, specs) -> dict[str, tuple[float, float]]:
    cost = cost_of_results(results, specs)
    wins: dict[str, int] = {}
    for r in results:
        wins[r.condition] = wins.get(r.condition, 0) + int(r.success)
    return {c: ((lo / wins[c], hi / wins[c]) if wins.get(c) else (float("inf"), float("inf")))
            for c, (lo, hi) in cost.items()}


def sensitivity(results, base: str, treat: str, ratios=RATIO_RANGE) -> dict:
    """Чи витримує економічний висновок увесь діапазон співвідношень цін ярусів.

    Ціни Lapathoniia невідомі, тому «дешевше на X%» під одним співвідношенням може перевернутись під
    іншим. Тут фіксуємо межі: множимо ціну дорожчого ярусу так, щоб співвідношення пройшло діапазон.
    """
    lanes, prompts = lanes_by_condition(results), prompts_by_condition(results)
    if base not in lanes or treat not in lanes:
        return {}
    out = {"base": base, "treat": treat, "ratios": {}}
    cheap_in, cheap_out = PRICES["lapa"]
    for ratio in ratios:
        scaled = {"lapa": (cheap_in, cheap_out),
                  "mamay": (cheap_in * ratio, cheap_out * ratio)}
        costs = {}
        for cond in (base, treat):
            total = 0.0
            for lane, tokens in lanes[cond].items():
                p_in, p_out = scaled.get(lane, scaled["mamay"])
                prompt = min(prompts.get(cond, {}).get(lane, 0), tokens)
                total += (prompt * p_in + (tokens - prompt) * p_out) / 1_000_000
            costs[cond] = total
        out["ratios"][ratio] = {
            "base_usd": costs[base], "treat_usd": costs[treat],
            "treat_cheaper_by": (1 - costs[treat] / costs[base]) if costs[base] else 0.0,
        }
    verdicts = [v["treat_cheaper_by"] > 0 for v in out["ratios"].values()]
    out["robust"] = all(verdicts) or not any(verdicts)
    return out


def _span(lo: float, hi: float) -> str:
    if lo == float("inf"):
        return "—"
    return f"{lo:.5f}" if hi <= lo else f"{lo:.5f}-{hi:.5f}"


def format_cost(results, specs) -> str:
    lines = ["умова                     токени      $            $/успіх       ярус (частка)"]
    per = usd_per_success(results, specs)
    cost = cost_of_results(results, specs)
    toks = tokens_by_condition(results)
    share = lane_share(results)
    for c in sorted(cost):
        parts = share.get(c, {})
        mix = " ".join(f"{k}={v:.0%}" for k, v in parts.items()) or "—"
        lines.append(f"{c:<24}{toks[c]:>8}   {_span(*cost[c]):<12} "
                     f"{_span(*per[c]):<13} {mix}")
    return "\n".join(lines)
