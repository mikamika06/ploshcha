import hashlib
import json
from typing import Literal

from pydantic import BaseModel

Mode = Literal["single", "loop", "gated"]
Routing = Literal["hetero", "mamay", "lapa"]
Toolset = Literal["default", "ua", "registry", "registry_agg", "registry_teach", "registry_sum", "registry_reduce",
                   "docs", "docs_agg", "docs_years",
                   "ua_norm", "reference", "lexis", "none"]
PlannerKind = Literal["none", "skeleton"]
MemoryKind = Literal["none", "notebook"]
AnswerChannel = Literal["schema", "text"]
VerifyMode = Literal["basic", "grounded", "auto"]
JudgeLane = Literal["auto", "mamay", "lapa"]


class AppSpec(BaseModel):
    model_config = {"frozen": True}

    mode: Mode = "loop"
    routing: Routing = "hetero"
    gate_direct: Routing = "mamay"
    toolset: Toolset = "default"
    prompt_id: str = "agent/v2"
    answer_prompt_id: str = "answer/plain"
    answer_channel: AnswerChannel = "schema"
    planner: PlannerKind = "none"
    plan_gather: int = 1
    plan_guard: bool = False
    coverage: bool = False
    coverage_guard: bool = False
    memory: MemoryKind = "none"
    verifier: bool = True
    verify_mode: VerifyMode = "basic"
    judge_lane: JudgeLane = "auto"
    absent_answer: bool = False
    recovery: bool = False
    max_steps: int = 5
    max_tokens: int = 512
    temperature: float = 0.0

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def with_(self, **changes) -> "AppSpec":
        return self.model_copy(update=changes)


class ExperimentSpec(BaseModel):
    model_config = {"frozen": True}

    items: str
    seeds: tuple[int, ...] = (1, 2, 3)
    conditions: tuple[str, ...] = ()
    limit: int | None = None
