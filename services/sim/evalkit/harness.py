import json
from collections.abc import Callable

from pydantic import BaseModel, Field

from ploshcha_sim.domain.gate import gate_reason, needs_loop
from ploshcha_sim.domain.task import Budget, TaskResult

from .checks import outcome_tier, split_checks

Runner = Callable[[str, int], TaskResult]


class EvalItem(BaseModel):
    id: str
    category: str
    task: str
    checks: list[dict]
    solvable_by: list[str] = Field(default_factory=list)
    gold: list[str] = Field(default_factory=list)
    foil: list[str] = Field(default_factory=list)
    gold_tools: list[str] = Field(default_factory=list)
    toolsets: list[str] = Field(default_factory=list)
    chain_len: int = 0


class EvalResult(BaseModel):
    item_id: str
    category: str
    condition: str
    prompt_id: str = ""
    spec_sha: str = ""
    seed: int
    success: bool
    checks: dict[str, bool]
    hygiene: dict[str, bool] = Field(default_factory=dict)
    hygiene_ok: bool = True
    tier: str = "empty"
    steps: int = 0
    tokens: int = 0
    aux_tokens: int = 0
    tokens_by_lane: dict[str, int] = Field(default_factory=dict)
    prompt_by_lane: dict[str, int] = Field(default_factory=dict)
    accepted: bool = False
    verdict_kind: str | None = None
    outcome: str = "answer"
    evidence: bool | None = None
    degraded: bool = False
    partial: bool = False
    incidents: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    answer: str | None = None


def load_items(path: str) -> list[EvalItem]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(EvalItem.model_validate(json.loads(line)))
    return items


def orchestrator_runner(make_orch: Callable[[], object], budget: Budget | None = None) -> Runner:
    def run(task: str, seed: int) -> TaskResult:
        # deep: у Budget є мутабельне поле (`tokens_by_lane`), і поверхнева копія ділила б
        # той самий словник між УСІМА прогонами — токени накопичувались би через задачі.
        return make_orch().run(task, seed, budget=budget.model_copy(deep=True) if budget else None)
    return run


def single_call_runner(llm, *, system: str | None = None, max_tokens: int = 512,
                      lane: str = "unknown") -> Runner:
    def run(task: str, seed: int) -> TaskResult:
        res = llm.generate(task, system=system, max_tokens=max_tokens, seed=seed)
        return TaskResult(answer=res.text, accepted=True, steps=1, tokens=res.usage.total,
                          tokens_by_lane={lane: res.usage.total},
                          prompt_by_lane={lane: res.usage.prompt_tokens})
    return run


def gated_runner(llm, tools, *, system: str | None = None, max_tokens: int = 512,
                 loop_runner: Runner | None = None, lane: str = "unknown") -> Runner:
    """Гейт зі складу інструментів: без інструментів даних цикл не потрібен (M0 §5 — правилами).

    Прогін через оркестратор обовʼязково загортає відповідь у тул-схему з латинськими ключами,
    що псує український вивід (UA-hardness §4.6). Заміряно: прямий виклик 1.000 проти 0.867.
    """
    if needs_loop(tools.specs()) and loop_runner is not None:
        return loop_runner

    def run(task: str, seed: int) -> TaskResult:
        res = llm.generate(task, system=system, max_tokens=max_tokens, seed=seed)
        return TaskResult(answer=res.text, accepted=True, steps=1, tokens=res.usage.total,
                          tokens_by_lane={lane: res.usage.total},
                          prompt_by_lane={lane: res.usage.prompt_tokens},
                          notes=[f"gate:{gate_reason(tools.specs())}"])
    return run


def run_eval(items: list[EvalItem], runners: dict[str, Runner], seeds: list[int],
             prompt_ids: dict[str, str] | None = None,
             spec_shas: dict[str, str] | None = None) -> list[EvalResult]:
    out: list[EvalResult] = []
    for item in items:
        for condition, runner in runners.items():
            for seed in seeds:
                result = runner(item.task, seed)
                checks, hygiene = split_checks(item.checks, result)
                out.append(EvalResult(
                    item_id=item.id, category=item.category, condition=condition,
                    prompt_id=(prompt_ids or {}).get(condition, ""),
                    spec_sha=(spec_shas or {}).get(condition, ""), seed=seed,
                    success=all(checks.values()) if checks else False, checks=checks,
                    hygiene=hygiene, hygiene_ok=all(hygiene.values()),
                    tier=outcome_tier(result),
                    steps=result.steps, tokens=result.tokens, aux_tokens=result.aux_tokens,
                    tokens_by_lane=dict(getattr(result, 'tokens_by_lane', {}) or {}),
                    prompt_by_lane=dict(getattr(result, 'prompt_by_lane', {}) or {}),
                    accepted=result.accepted, degraded=result.degraded, partial=result.partial,
                    verdict_kind=result.verdict_kind, outcome=result.outcome,
                    evidence=result.evidence,
                    incidents=list(result.incidents), notes=list(result.notes),
                    tools=[x["call"]["tool"] for x in result.scratch],
                    answer=result.answer,
                ))
    return out
