from pydantic import BaseModel, Field
from .evidence import Outcome


class Budget(BaseModel):
    max_steps: int = 8
    max_tokens: int = 100_000
    steps_used: int = 0
    tokens_used: int = 0
    aux_tokens: int = 0
    tokens_by_lane: dict[str, int] = Field(default_factory=dict)
    prompt_by_lane: dict[str, int] = Field(default_factory=dict)
    tokens_by_stage: dict[str, int] = Field(default_factory=dict)
    prompt_by_stage: dict[str, int] = Field(default_factory=dict)
    tokens_by_stage_lane: dict[str, int] = Field(default_factory=dict)
    prompt_by_stage_lane: dict[str, int] = Field(default_factory=dict)

    def can_continue(self) -> bool:
        return self.steps_used < self.max_steps and self.tokens_used < self.max_tokens

    def spend(self, tokens: int, lane: str = "unknown", prompt: int = 0,
              stage: str = "unknown") -> None:
        self.steps_used += 1
        self.tokens_used += tokens
        self._attribute(tokens, lane, prompt, stage)

    def spend_aux(self, tokens: int, lane: str = "unknown", prompt: int = 0,
                  stage: str = "unknown") -> None:
        self.aux_tokens += tokens
        self._attribute(tokens, lane, prompt, stage)

    def _attribute(self, tokens: int, lane: str, prompt: int = 0,
                   stage: str = "unknown") -> None:
        self.tokens_by_lane[lane] = self.tokens_by_lane.get(lane, 0) + tokens
        self.prompt_by_lane[lane] = self.prompt_by_lane.get(lane, 0) + prompt
        self.tokens_by_stage[stage] = self.tokens_by_stage.get(stage, 0) + tokens
        self.prompt_by_stage[stage] = self.prompt_by_stage.get(stage, 0) + prompt
        pair = f"{stage}|{lane}"
        self.tokens_by_stage_lane[pair] = self.tokens_by_stage_lane.get(pair, 0) + tokens
        self.prompt_by_stage_lane[pair] = self.prompt_by_stage_lane.get(pair, 0) + prompt


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
    pending: list[str] = Field(default_factory=list)
    overrides: dict = Field(default_factory=dict)
    route_as: str | None = None
    attempts: dict[str, int] = Field(default_factory=dict)
    failed_tools: list[str] = Field(default_factory=list)
    tool_failures: dict[str, int] = Field(default_factory=dict)
    incidents: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    recoveries: int = 0
    plan: TaskPlan | None = None


class TaskResult(BaseModel):
    answer: str | None = None
    accepted: bool = False
    verdict_reason: str | None = None
    verdict_kind: str | None = None
    outcome: Outcome = "answer"
    evidence: bool | None = None
    degraded: bool = False
    partial: bool = False
    steps: int = 0
    tokens: int = 0
    aux_tokens: int = 0
    incidents: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    tokens_by_lane: dict[str, int] = Field(default_factory=dict)
    prompt_by_lane: dict[str, int] = Field(default_factory=dict)
    tokens_by_stage: dict[str, int] = Field(default_factory=dict)
    prompt_by_stage: dict[str, int] = Field(default_factory=dict)
    tokens_by_stage_lane: dict[str, int] = Field(default_factory=dict)
    prompt_by_stage_lane: dict[str, int] = Field(default_factory=dict)
    scratch: list[dict] = Field(default_factory=list)
    # ★ ПАРТИТУРА Й ПОЗИЦІЇ ЇДУТЬ У ЗВІТ, а не гинуть разом із прогоном.
    #
    # Доти віче віддавало про розмову самі лічильники — `beats=19` нотаткою, і в усіх 43 прогонах
    # із нотатками це було ОДНЕ Й ТЕ САМЕ число. У 155 звітах `docs/research/eval-runs/` немає
    # жодного такту: `grep -l 'у_відповідь'` дає нуль файлів. Тому «хто кого підтримав» не
    # відновлювався з наявних даних узагалі, і кожен круг правок мусив ставити тимчасового
    # шпигуна, щоб побачити те, що система тримала в руках і викидала.
    #
    # `scratch` для цього не годиться: його читають як слід ІНСТРУМЕНТІВ (`evidence_state`,
    # `partial_answer`, проєктор переграє його як виклики), і такт там ламав би доказовий стан.
    beats: list[dict] = Field(default_factory=list)
    stances: list[dict] = Field(default_factory=list)
