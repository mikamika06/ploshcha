import json
from collections.abc import Callable

from ..domain.gate import FINAL_TOOL
from ..domain.recovery import (
    CAPS,
    NOT_A_FAILURE,
    attempt_key,
    SOFT_CODES,
    StepOutcome,
    build,
    classify,
    is_near_duplicate,
    partial_answer,
    policy,
    recovery_cap,
    signature,
)
from ..domain.task import Budget, TaskPlan, TaskResult, TaskState
from ..ports.memory import Notebook
from ..ports.planner import Planner
from ..ports.router import EffortPolicy, ModelRouter, StepKind, escalated
from ..ports.tool import ToolPort
from ..ports.trace import StepRecord, TracePort
from .verify import verify as run_verify


def _safe_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


VERBATIM_WINDOW = 2


STEP_INSTRUCTION = "Наступний крок — один JSON (виклик інструмента або final_answer):"


def _render(state: TaskState, recall=None, verbatim: int | None = None,
            tail: str | None = None, instruction: str | None = None) -> str:
    lines = [f"Задача: {state.task}"]
    if recall is not None and recall.hits:
        lines.append("Пригадане:")
        for hit in recall.hits:
            lines.append(f"  {hit.item.id} · {hit.item.text}")
    scratch = state.scratch if verbatim is None else state.scratch[-verbatim:]
    for item in scratch:
        lines.append(f"Виклик: {json.dumps(item['call'], ensure_ascii=False)}")
        lines.append(f"Результат: {json.dumps(item['result'], ensure_ascii=False)}")
    for hint in state.hints:
        lines.append(f"Підказка: {hint}")
    lines.append(instruction if instruction is not None else STEP_INSTRUCTION)
    if tail:
        lines.append(tail)
    return "\n".join(lines)


def _recall_query(state: TaskState) -> str:
    parts = [state.task]
    if state.plan is not None:
        step = state.plan.current()
        if step is not None:
            parts.append(step.goal)
    if state.scratch:
        parts.append(json.dumps(state.scratch[-1]["result"], ensure_ascii=False))
    return " ".join(parts)


def _note_text(call: dict, result) -> tuple[str, str]:
    payload = json.dumps(result.value if result.ok else {"error": result.error},
                         ensure_ascii=False)
    text = f"{json.dumps(call, ensure_ascii=False)} → {payload}"
    if not result.ok:
        return text, "error"
    known = _tool_known(result)
    return text, "empty" if known is False else "fact"


def _tool_known(result) -> bool | None:
    if not result.ok or not isinstance(result.value, dict):
        return None
    known = result.value.get("known")
    return known if isinstance(known, bool) else None


class LinearPlanner(Planner):
    def next_kind(self, state) -> StepKind:
        return "select"


class Orchestrator:
    def __init__(self, router: ModelRouter, effort: EffortPolicy, tools: ToolPort,
                 planner: Planner | None = None, verifier: bool = True,
                 memory=None, trace: TracePort | None = None, run_id: str = "",
                 system: str | None = None, recovery: bool = False,
                 notebook: Callable[[], Notebook] | None = None,
                 tail: str | None = None, prompt_id: str = "", prompt_sha: str = "",
                 answer_channel: str = "schema", answer_instruction: str | None = None):
        self.router = router
        self.effort = effort
        self.tools = tools
        self.planner = planner or LinearPlanner()
        self.verifier = verifier
        self.memory = memory
        self.trace = trace
        self.run_id = run_id
        self.system = system
        self.recovery = recovery
        self.notebook = notebook
        self.tail = tail
        self.prompt_id = prompt_id
        self.prompt_sha = prompt_sha
        self.answer_channel = answer_channel
        self.answer_instruction = answer_instruction

    def run(self, task: str, seed: int = 0, budget: Budget | None = None) -> TaskResult:
        state = TaskState(task=task, budget=budget or Budget())
        state.plan = self.planner.plan(task, self.tools)
        notebook = self.notebook() if self.notebook is not None else None
        seen: list[str] = []
        history: list[tuple[str, dict]] = []
        history_ok: list[bool] = []
        while not state.done and state.budget.can_continue():
            kind = self.planner.next_kind(state)
            if state.route_as:
                kind = state.route_as
                state.route_as = None
            llm = self.router.route(kind)
            cfg = self.effort.effort(kind)
            if state.overrides:
                cfg = cfg.model_copy(update=state.overrides)
                state.overrides = {}
            answering = self._is_answer_step(state)
            schema = self.tools.strict_schema() if cfg.tier == "strict" else self.tools.wire_schema()
            recall = None
            if notebook is not None:
                recall = notebook.recall(_recall_query(state), state.budget.steps_used)
                self._emit_memory(state, "mem_read", recall.as_trace(), seed)
            prompt = _render(state, recall, VERBATIM_WINDOW if notebook is not None else None,
                             self.tail, self.answer_instruction if answering else None)
            if answering:
                res = llm.generate(prompt, system=self.system, temperature=cfg.temperature,
                                   max_tokens=cfg.max_tokens, seed=seed)
                state.budget.spend(res.usage.total)
                self._emit(state, kind, llm.model, res, None, None, seed, prompt, "none")
                state.answer = res.text.strip()
                state.done = True
                break
            res = llm.generate_structured(prompt, schema, system=self.system,
                                          temperature=cfg.temperature,
                                          max_tokens=cfg.max_tokens, seed=seed)
            state.budget.spend(res.usage.total)
            call, reason = self.tools.parse(_safe_json(res.text))
            self._emit(state, kind, llm.model, res, call, reason, seed, prompt, cfg.tier)
            state.hints = []

            if call is None:
                code = classify(StepOutcome(raw_output=res.text, finish_reason=res.finish_reason,
                                            reject_reason=reason))
                self._note_incident(notebook, state, code, seed)
                if self._recover(state, code, cfg, kind):
                    continue
                state.degraded = True
                break

            if call.tool == FINAL_TOOL:
                if self.answer_channel == "text":
                    state.answer = self._answer_freely(state, llm, cfg, seed, notebook)
                else:
                    state.answer = str(call.args.get("text", ""))
                state.done = True
                break

            sig = signature(call.tool, call.args)
            duplicate = sig in seen
            near = not duplicate and self.recovery and is_near_duplicate(
                call.tool, call.args, history, succeeded=history_ok)
            if duplicate or near:
                code = classify(StepOutcome(raw_output=res.text, duplicate=duplicate,
                                            near_duplicate=near))
                self._note_incident(notebook, state, code, seed, detail=sig)
                if self._recover(state, code, cfg, kind):
                    continue
                state.degraded = True
                break

            seen.append(sig)
            history.append((call.tool, dict(call.args)))
            result = self.tools.call(call)
            history_ok.append(result.ok and _tool_known(result) is not False)
            if result.ok and state.plan is not None:
                state.plan.advance()
            entry = {
                "call": {"tool": call.tool, **call.args},
                "result": result.value if result.ok else {"error": result.error},
            }
            state.scratch.append(entry)
            if notebook is not None:
                text, sort = _note_text(entry["call"], result)
                if notebook.note(text, state.budget.steps_used, sort=sort) is not None:
                    self._emit_memory(state, "mem_write", {"sort": sort, "text": text}, seed)
            if self.recovery:
                code = classify(StepOutcome(raw_output=res.text, tool_ok=result.ok,
                                            tool_known=_tool_known(result)))
                if code is not None:
                    self._recover(state, code, cfg, kind, detail=result.error)

        if not state.done:
            state.degraded = True

        accepted = False
        reason = None
        if state.done and not state.partial and self.verifier:
            verdict = run_verify(task, state.answer, self.router, self.effort,
                                 evidence=state.scratch, seed=seed, trace=self.trace, run_id=self.run_id)
            state.budget.spend_aux(verdict.tokens)
            accepted = verdict.accepted
            reason = verdict.reason
            if not accepted:
                state.degraded = True
        elif state.done and not state.partial:
            accepted = True

        return TaskResult(
            answer=state.answer, accepted=accepted, verdict_reason=reason,
            degraded=state.degraded, partial=state.partial, steps=state.budget.steps_used,
            tokens=state.budget.tokens_used, aux_tokens=state.budget.aux_tokens,
            incidents=state.incidents, notes=state.notes, scratch=state.scratch,
        )

    def _answer_freely(self, state: TaskState, llm, cfg, seed, notebook) -> str:
        """Відповідь — не дія, тому не йде через схему дій (K7b)."""
        recall = notebook.recall(_recall_query(state), state.budget.steps_used) \
            if notebook is not None else None
        prompt = _render(state, recall, VERBATIM_WINDOW if notebook is not None else None,
                         self.tail, self.answer_instruction)
        res = llm.generate(prompt, system=self.system, temperature=cfg.temperature,
                           max_tokens=cfg.max_tokens, seed=seed)
        state.budget.spend(res.usage.total)
        self._emit(state, "synthesize", llm.model, res, None, None, seed, prompt, "none")
        return res.text.strip()

    def _is_answer_step(self, state: TaskState) -> bool:
        if self.answer_channel != "text" or state.plan is None:
            return False
        step = state.plan.current()
        return step is not None and step.tool_hint == FINAL_TOOL

    def _routable(self, kind: StepKind) -> bool:
        try:
            self.router.route(kind)
        except KeyError:
            return False
        return True

    def _recover(self, state: TaskState, code, cfg, kind: StepKind,
                 detail: str | None = None) -> bool:
        if not self.recovery or code is None:
            return False
        soft = code in SOFT_CODES
        target: StepKind | None = None
        fresh_plan: TaskPlan | None = None
        while True:
            rung = policy(code, state.attempts, plan_exists=state.plan is not None,
                          spent=0 if soft else state.recoveries,
                          cap_total=None if soft else recovery_cap(state.budget.max_steps))
            if rung is None:
                return False
            if rung == "escalate":
                target = escalated(kind)
                if target is not None and self._routable(target):
                    break
                state.attempts[attempt_key("escalate", soft)] = CAPS["escalate"]
                target = None
                continue
            if rung == "replan":
                fresh_plan = self.planner.replan(state.task, state)
                if fresh_plan is not None:
                    break
                state.attempts[attempt_key("replan", soft)] = CAPS["replan"]
                continue
            break
        key = attempt_key(rung, soft)
        state.attempts[key] = state.attempts.get(key, 0) + 1
        if not soft:
            state.recoveries += 1
        if code in NOT_A_FAILURE:
            state.notes.append(code)
        else:
            state.incidents.append(code)
        plan = build(code, rung, current_max_tokens=cfg.max_tokens, detail=detail)
        if rung == "partial":
            text = partial_answer(state.scratch)
            if text is None:
                return False
            state.answer = text
            state.done = True
            state.partial = True
            state.degraded = True
            return False
        if plan.hint:
            state.hints.append(plan.hint)
        state.overrides = plan.overrides
        if rung == "escalate":
            state.route_as = target
        if rung == "replan":
            state.plan = fresh_plan
        return True

    def _note_incident(self, notebook, state, code, seed, detail: str | None = None) -> None:
        if notebook is None or code is None:
            return
        text = f"збій {code}" + (f": {detail}" if detail else "")
        if notebook.note(text, state.budget.steps_used, sort="incident") is not None:
            self._emit_memory(state, "mem_write", {"sort": "incident", "text": text}, seed)

    def _emit_memory(self, state, stage: str, payload: dict, seed) -> None:
        if self.trace is None:
            return
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=state.budget.steps_used, agent="notebook", stage=stage,
            model="notebook", prompt="", raw_output="", parsed=payload,
            schema_valid=True, world_valid=True, seed=seed,
        ))

    def _emit(self, state, kind, model, res, call, reason, seed, prompt=None, cfg_tier=""):
        if self.trace is None:
            return
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=state.budget.steps_used, agent="orchestrator", stage=kind,
            model=model, prompt=prompt if prompt is not None else _render(state), raw_output=res.text,
            parsed={"tool": call.tool, **call.args} if call else None,
            schema_valid=call is not None, world_valid=call is not None,
            reject_reason=reason, usage=res.usage, latency_ms=res.latency_ms,
            finish_reason=res.finish_reason, seed=seed,
            ablation={"kind": kind, "tier": cfg_tier, "prompt": self.prompt_id,
                      "prompt_sha": self.prompt_sha},
        ))
