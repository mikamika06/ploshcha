import json

from ..domain.task import Budget, TaskResult, TaskState
from ..ports.planner import Planner
from ..ports.router import EffortPolicy, ModelRouter, StepKind
from ..ports.tool import ToolPort
from ..ports.trace import StepRecord, TracePort
from .verify import verify as run_verify


def _safe_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _render(state: TaskState) -> str:
    lines = [f"Задача: {state.task}"]
    for item in state.scratch:
        lines.append(f"Виклик: {json.dumps(item['call'], ensure_ascii=False)}")
        lines.append(f"Результат: {json.dumps(item['result'], ensure_ascii=False)}")
    lines.append("Наступний крок — один JSON (виклик інструмента або final_answer):")
    return "\n".join(lines)


class LinearPlanner(Planner):
    def next_kind(self, state) -> StepKind:
        return "select"


class Orchestrator:
    def __init__(self, router: ModelRouter, effort: EffortPolicy, tools: ToolPort,
                 planner: Planner | None = None, verifier: bool = True,
                 memory=None, trace: TracePort | None = None, run_id: str = ""):
        self.router = router
        self.effort = effort
        self.tools = tools
        self.planner = planner or LinearPlanner()
        self.verifier = verifier
        self.memory = memory
        self.trace = trace
        self.run_id = run_id

    def run(self, task: str, seed: int = 0, budget: Budget | None = None) -> TaskResult:
        state = TaskState(task=task, budget=budget or Budget())
        seen: set[str] = set()
        while not state.done and state.budget.can_continue():
            kind = self.planner.next_kind(state)
            llm = self.router.route(kind)
            cfg = self.effort.effort(kind)
            schema = self.tools.strict_schema() if cfg.tier == "strict" else self.tools.wire_schema()
            res = llm.generate_structured(_render(state), schema, max_tokens=cfg.max_tokens, seed=seed)
            state.budget.spend(res.usage.total)
            call, reason = self.tools.parse(_safe_json(res.text))
            self._emit(state, kind, llm.model, res, call, reason, seed)
            if call is None:
                state.degraded = True
                break
            if call.tool == "final_answer":
                state.answer = str(call.args.get("text", ""))
                state.done = True
                break
            sig = json.dumps({"tool": call.tool, **call.args}, sort_keys=True, ensure_ascii=False)
            if sig in seen:
                state.degraded = True
                break
            seen.add(sig)
            result = self.tools.call(call)
            state.scratch.append({
                "call": {"tool": call.tool, **call.args},
                "result": result.value if result.ok else {"error": result.error},
            })

        if not state.done:
            state.degraded = True

        accepted = False
        reason = None
        if state.done and self.verifier:
            verdict = run_verify(task, state.answer, self.router, self.effort,
                                 seed=seed, trace=self.trace, run_id=self.run_id)
            accepted = verdict.accepted
            reason = verdict.reason
            if not accepted:
                state.degraded = True
        elif state.done:
            accepted = True

        return TaskResult(
            answer=state.answer, accepted=accepted, verdict_reason=reason,
            degraded=state.degraded, steps=state.budget.steps_used, scratch=state.scratch,
        )

    def _emit(self, state, kind, model, res, call, reason, seed):
        if self.trace is None:
            return
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=state.budget.steps_used, agent="orchestrator", stage=kind,
            model=model, prompt=_render(state), raw_output=res.text,
            parsed={"tool": call.tool, **call.args} if call else None,
            schema_valid=call is not None, world_valid=call is not None,
            reject_reason=reason, usage=res.usage, latency_ms=res.latency_ms,
            finish_reason=res.finish_reason, seed=seed, ablation={"kind": kind},
        ))
