from pydantic import BaseModel, Field


class Budget(BaseModel):
    max_steps: int = 8
    max_tokens: int = 100_000
    steps_used: int = 0
    tokens_used: int = 0

    def can_continue(self) -> bool:
        return self.steps_used < self.max_steps and self.tokens_used < self.max_tokens

    def spend(self, tokens: int) -> None:
        self.steps_used += 1
        self.tokens_used += tokens


class TaskState(BaseModel):
    task: str
    scratch: list[dict] = Field(default_factory=list)
    answer: str | None = None
    done: bool = False
    degraded: bool = False
    budget: Budget = Field(default_factory=Budget)


class TaskResult(BaseModel):
    answer: str | None = None
    accepted: bool = False
    verdict_reason: str | None = None
    degraded: bool = False
    steps: int = 0
    scratch: list[dict] = Field(default_factory=list)
