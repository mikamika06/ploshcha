from ..domain.task import TaskPlan, TaskStep
from ..ports.planner import Planner
from ..ports.router import StepKind

FINAL_TOOL = "final_answer"
MAX_GATHER = 4


def _data_tools(tools) -> list[str]:
    return [s.name for s in tools.specs() if s.name != FINAL_TOOL]


class SkeletonPlanner(Planner):
    def __init__(self, gather: int = 1, gather_kind: StepKind = "select",
                 answer_kind: StepKind = "decide"):
        self.gather = gather
        self.gather_kind = gather_kind
        self.answer_kind = answer_kind

    def _build(self, gather: int, hint: str | None) -> TaskPlan:
        steps = [
            TaskStep(id=f"g{i}", goal="здобути потрібне інструментом",
                     kind=self.gather_kind, tool_hint=hint)
            for i in range(gather)
        ]
        steps.append(TaskStep(id="a", goal="дати відповідь", kind=self.answer_kind,
                              tool_hint=FINAL_TOOL))
        return TaskPlan(steps=steps)

    def plan(self, task: str, tools) -> TaskPlan | None:
        available = _data_tools(tools)
        if not available:
            return self._build(0, FINAL_TOOL)
        hint = available[0] if len(available) == 1 else None
        return self._build(self.gather, hint)

    def replan(self, task: str, state) -> TaskPlan | None:
        current = state.plan
        if current is None:
            return None
        gathered = sum(1 for s in current.steps if s.kind == self.gather_kind)
        if gathered >= MAX_GATHER:
            return None
        plan = self._build(gathered + 1, None)
        for step, old in zip(plan.steps, current.steps):
            if old.done and step.kind == old.kind:
                step.done = True
        return plan

    def next_kind(self, state) -> StepKind:
        plan = getattr(state, "plan", None)
        if plan is None:
            return self.gather_kind
        step = plan.current()
        return step.kind if step is not None else self.answer_kind
