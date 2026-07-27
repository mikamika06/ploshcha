import json
from collections.abc import Callable

from pydantic import BaseModel, Field

from ploshcha_sim.domain.task import Budget, TaskResult

from .checks import run_checks

Runner = Callable[[str, int], TaskResult]


class EvalItem(BaseModel):
    id: str
    category: str
    task: str
    checks: list[dict]
    solvable_by: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    item_id: str
    category: str
    condition: str
    seed: int
    success: bool
    checks: dict[str, bool]
    steps: int = 0
    tokens: int = 0
    accepted: bool = False
    degraded: bool = False
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
        return make_orch().run(task, seed, budget=budget.model_copy() if budget else None)
    return run


def single_call_runner(llm, *, system: str | None = None, max_tokens: int = 512) -> Runner:
    def run(task: str, seed: int) -> TaskResult:
        res = llm.generate(task, system=system, max_tokens=max_tokens, seed=seed)
        return TaskResult(answer=res.text, accepted=True, steps=1, tokens=res.usage.total)
    return run


def run_eval(items: list[EvalItem], runners: dict[str, Runner], seeds: list[int]) -> list[EvalResult]:
    out: list[EvalResult] = []
    for item in items:
        for condition, runner in runners.items():
            for seed in seeds:
                result = runner(item.task, seed)
                checks = run_checks(item.checks, result)
                out.append(EvalResult(
                    item_id=item.id, category=item.category, condition=condition, seed=seed,
                    success=all(checks.values()), checks=checks,
                    steps=result.steps, tokens=result.tokens,
                    accepted=result.accepted, degraded=result.degraded, answer=result.answer,
                ))
    return out
