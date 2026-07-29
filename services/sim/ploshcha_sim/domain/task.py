from pydantic import BaseModel, Field


class Budget(BaseModel):
    max_steps: int = 8
    max_tokens: int = 100_000
    steps_used: int = 0
    tokens_used: int = 0
    aux_tokens: int = 0
    tokens_by_lane: dict[str, int] = Field(default_factory=dict)
    prompt_by_lane: dict[str, int] = Field(default_factory=dict)

    def can_continue(self) -> bool:
        return self.steps_used < self.max_steps and self.tokens_used < self.max_tokens

    def spend(self, tokens: int, lane: str = "unknown", prompt: int = 0) -> None:
        self.steps_used += 1
        self.tokens_used += tokens
        self._attribute(tokens, lane, prompt)

    def spend_aux(self, tokens: int, lane: str = "unknown", prompt: int = 0) -> None:
        self.aux_tokens += tokens
        self._attribute(tokens, lane, prompt)

    def _attribute(self, tokens: int, lane: str, prompt: int = 0) -> None:
        self.tokens_by_lane[lane] = self.tokens_by_lane.get(lane, 0) + tokens
        self.prompt_by_lane[lane] = self.prompt_by_lane.get(lane, 0) + prompt


class TaskStep(BaseModel):
    id: str
    goal: str
    kind: str = "select"
    tool_hint: str | None = None
    done: bool = False


class TaskPlan(BaseModel):
    steps: list[TaskStep] = Field(default_factory=list)

    def current(self) -> TaskStep | None:
        return next((s for s in self.steps if not s.done), None)

    def advance(self) -> None:
        step = self.current()
        if step is not None:
            step.done = True

    def progress(self) -> tuple[int, int]:
        return sum(1 for s in self.steps if s.done), len(self.steps)


class TaskState(BaseModel):
    task: str
    scratch: list[dict] = Field(default_factory=list)
    answer: str | None = None
    done: bool = False
    degraded: bool = False
    partial: bool = False
    budget: Budget = Field(default_factory=Budget)
    hints: list[str] = Field(default_factory=list)
    overrides: dict = Field(default_factory=dict)
    route_as: str | None = None
    attempts: dict[str, int] = Field(default_factory=dict)
    incidents: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    recoveries: int = 0
    plan: TaskPlan | None = None


class TaskResult(BaseModel):
    answer: str | None = None
    accepted: bool = False
    verdict_reason: str | None = None
    degraded: bool = False
    partial: bool = False
    steps: int = 0
    tokens: int = 0
    aux_tokens: int = 0
    incidents: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    tokens_by_lane: dict[str, int] = Field(default_factory=dict)
    prompt_by_lane: dict[str, int] = Field(default_factory=dict)
    scratch: list[dict] = Field(default_factory=list)
