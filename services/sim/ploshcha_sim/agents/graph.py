"""AgentGraph — супервізор: розбити, роздати дітям, злити.

Три речі, які тут вирішені навмисно й проти спокуси:

1. **Розбиття детерміноване** (`domain/graph.py`), не «попроси модель розкласти задачу». K7g/K7h і
   ABSTAIN тричі показали: крок, повністю визначений даними, віддавати моделі — програш.
2. **Стан доказів дитини піднімається в синтез** (борг K9). Якщо дитина сказала «даних немає», а
   батько бачить лише її текст, відсутність зникає — і синтез латає дірку вигадкою.
3. **Паралель детермінована**: результати збираються за ІНДЕКСОМ, не за порядком завершення.
   Інакше той самий прогін із тим самим seed давав би різний вивід, і `pass^k` став би шумом.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from ..domain.finding import Finding, merge_evidence, merge_outcome, missing_tasks
from ..domain.graph import Split, plan_split
from ..domain.task import Budget, TaskResult
from ..ports.agent import AgentPort
from ..ports.trace import StepRecord, TracePort

CHILD_TEMPLATE = "{task}\n\nЗосередься ЛИШЕ на «{item}»: поясни саме його."

SYNTHESIS_HEADER = "Зведення підзадач:"
MISSING_HEADER = "Даних не знайдено для:"

FANOUT_NOTE = "fanout"
NO_FANOUT_NOTE = "fanout_not_applicable"


class AgentGraph(AgentPort):
    """Супервізор над фабрикою дітей. Сам моделі не викликає — уся мова живе в дітях."""

    def __init__(self, child: Callable[[Budget], AgentPort], *, trace: TracePort | None = None,
                 run_id: str = "graph", max_depth: int = 2, max_width: int = 6,
                 workers: int = 4, template: str = CHILD_TEMPLATE):
        self.child = child
        self.trace = trace
        self.run_id = run_id
        self.max_depth = max_depth
        self.max_width = max_width
        self.workers = workers
        self.template = template

    def run(self, task: str, seed: int = 0, budget: Budget | None = None,
            depth: int = 1) -> TaskResult:
        budget = budget or Budget()
        split = plan_split(task, budget, template=self.template, depth=depth,
                           max_depth=self.max_depth)
        if split is None or split.width > self.max_width:
            # Фан-аут не виправданий — це найчастіший випадок, і він мусить бути ДЕШЕВИМ:
            # ніякого зайвого виклику, просто одна дитина на всю задачу.
            result = self.child(budget).run(task, seed=seed, budget=budget)
            result.notes.append(NO_FANOUT_NOTE)
            return result

        findings = self._fan_out(split, seed)
        return self._synthesize(task, findings, budget, split)

    def _fan_out(self, split: Split, seed: int) -> list[Finding]:
        def one(index: int) -> Finding:
            subtask = split.tasks[index]
            child = self.child(split.child_budget.model_copy(deep=True))
            span = f"{self.run_id}/{index}"
            result = child.run(subtask, seed=seed, budget=split.child_budget.model_copy(deep=True))
            self._emit(span, subtask, result, split.depth)
            return Finding(task=subtask, claim=result.answer, kind=result.outcome,
                           evidence=result.evidence, provenance=span,
                           tokens=result.tokens + result.aux_tokens, steps=result.steps,
                           incidents=list(result.incidents))

        if self.workers <= 1 or split.width == 1:
            return [one(i) for i in range(split.width)]
        with ThreadPoolExecutor(max_workers=min(self.workers, split.width)) as pool:
            # `map` зберігає порядок аргументів — саме тому паралель тут детермінована.
            return list(pool.map(one, range(split.width)))

    def _synthesize(self, task: str, findings: list[Finding], budget: Budget,
                    split: Split) -> TaskResult:
        """Зведення робить КОД: зміст повністю визначений дитячими відповідями.

        Дірки називаються вголос (`MISSING_HEADER`) — це і є підйом стану доказів у синтез, без якого
        супервізор мовчки видавав би часткове за повне.
        """
        lines = [SYNTHESIS_HEADER]
        for finding in findings:
            if finding.usable:
                lines.append(f"— {finding.claim.strip()}")
        missing = missing_tasks(findings)
        if missing:
            lines.append(f"{MISSING_HEADER} {len(missing)} з {len(findings)}.")

        merged = Budget(max_steps=budget.max_steps, max_tokens=budget.max_tokens)
        merged.steps_used = sum(f.steps for f in findings)
        merged.tokens_used = sum(f.tokens for f in findings)
        for finding in findings:
            merged.tokens_by_lane["child"] = merged.tokens_by_lane.get("child", 0) + finding.tokens

        outcome = merge_outcome(findings)
        return TaskResult(
            answer="\n".join(lines) if len(lines) > 1 else None,
            accepted=outcome == "answer",
            outcome=outcome,
            evidence=merge_evidence(findings),
            degraded=outcome == "failure",
            steps=merged.steps_used,
            tokens=merged.tokens_used,
            tokens_by_lane=dict(sorted(merged.tokens_by_lane.items())),
            incidents=[i for f in findings for i in f.incidents],
            notes=[FANOUT_NOTE, f"width={split.width}", f"depth={split.depth}"]
            + [f"missing={m}" for m in missing],
            scratch=[{"call": {"tool": "subagent", "task": f.task},
                      "result": {"відомо": f.evidence, "claim": f.claim},
                      "found": f.evidence} for f in findings],
        )

    def _emit(self, span: str, task: str, result: TaskResult, depth: int) -> None:
        if self.trace is None:
            return
        self.trace.emit(StepRecord(
            run_id=span, parent_run_id=self.run_id, span=span, tick=depth, agent="subagent",
            stage="synthesize", model="graph", prompt=task, raw_output=result.answer or "",
            parsed={"outcome": result.outcome, "evidence": result.evidence},
            schema_valid=result.outcome != "failure", world_valid=result.evidence is not False,
        ))
