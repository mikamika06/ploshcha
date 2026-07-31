"""Порт трас."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from .llm import LlmUsage


class StepRecord(BaseModel):
    run_id: str
    # Ієрархія траси: без `parent_run_id` вивід N дітей злипається в одну плоску стрічку, і
    # незрозуміло, чий саме крок дав результат. `span` відрізняє дітей одного батька.
    parent_run_id: str | None = None
    span: str | None = None
    tick: int
    agent: str
    stage: str = Field(description="act | plan | reflect")
    model: str
    lane: str = "unknown"
    prompt: str
    raw_output: str
    parsed: dict | None = None
    schema_valid: bool = False
    world_valid: bool = False
    reject_reason: str | None = None
    usage: LlmUsage = Field(default_factory=LlmUsage)
    latency_ms: int = 0
    finish_reason: str | None = None
    seed: int | None = None
    ablation: dict = Field(default_factory=dict)
    seq: int = 0


class TracePort(ABC):
    @abstractmethod
    def emit(self, record: StepRecord) -> None: ...
