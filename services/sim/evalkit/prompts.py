import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Slot = Literal["orchestrator", "verify", "act"]
Status = Literal["frozen", "candidate", "rejected"]
Placement = Literal["head", "tail", "both"]

NEW_CALL_ACTIONS = frozenset({"new_call", "other_call", "other_args", "repeat"})
FINISH_ACTIONS = frozenset({"final_answer", "final_if_no_alt"})
ACTIONS = NEW_CALL_ACTIONS | FINISH_ACTIONS

CANONICAL_STATES = ((True, False, False), (False, True, False), (False, False, True),
                    (False, False, False))


class TableRow(BaseModel):
    a: bool
    b: bool = False
    c: bool = False
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)

    @property
    def state(self) -> tuple[bool, bool, bool]:
        return (self.a, self.b, self.c)


class PromptStyle(BaseModel):
    caps: bool = False
    form: Literal["ти", "Ви"] = "ти"
    lexicon: Literal["own", "train", "neutral"] = "own"
    lang: str = "uk"


class PromptVariant(BaseModel):
    id: str
    slot: Slot = "orchestrator"
    family: str = ""
    head: str
    tail: str = ""
    tail_rule: str | None = None
    placement: Placement = "head"
    style: PromptStyle = Field(default_factory=PromptStyle)
    rules: list[str] = Field(default_factory=list)
    table: list[TableRow] = Field(default_factory=list)
    provenance: str = ""
    status: Status = "candidate"
    defect: str | None = None
    defect_class: Literal["structural", "semantic"] | None = None
    sha256: str = ""

    def digest(self) -> str:
        payload = f"{self.head}\x00{self.tail}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def violations(self) -> list[str]:
        found: list[str] = []
        if not self.table:
            found.append("empty_table")
        seen: set[tuple[bool, bool, bool]] = set()
        for row in self.table:
            if row.state in seen:
                found.append(f"duplicate_state:{row.state}")
            seen.add(row.state)
            if not row.allow:
                found.append(f"deadlock:{row.state}")
            overlap = set(row.allow) & set(row.deny)
            if overlap:
                found.append(f"contradiction:{row.state}:{sorted(overlap)}")
            unknown = (set(row.allow) | set(row.deny)) - ACTIONS
            if unknown:
                found.append(f"unknown_action:{sorted(unknown)}")
            if not row.a and "final_answer" in row.allow:
                found.append(f"premature_finish:{row.state}")
            if row.a and NEW_CALL_ACTIONS & set(row.allow):
                found.append(f"call_after_completion:{row.state}")
        for state in CANONICAL_STATES:
            if state not in seen:
                found.append(f"uncovered_state:{state}")
        if self.tail and self.tail_rule is None:
            found.append("tail_without_rule")
        if self.tail_rule is not None and self.tail_rule not in self.rules:
            found.append(f"tail_rule_undeclared:{self.tail_rule}")
        if self.tail and self.placement == "head":
            found.append("tail_set_but_placement_head")
        return found

    def render_system(self) -> str:
        return self.head


def load_prompts(path: str | Path) -> dict[str, PromptVariant]:
    variants: dict[str, PromptVariant] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            variant = PromptVariant.model_validate(json.loads(line))
            if variant.id in variants:
                raise ValueError(f"дубль prompt id: {variant.id}")
            variants[variant.id] = variant
    return variants


REGISTRY = Path(__file__).parent / "prompts" / "agent.jsonl"


def resolve(prompt_id: str, path: str | Path | None = None) -> PromptVariant:
    variants = load_prompts(path or REGISTRY)
    if prompt_id not in variants:
        raise KeyError(f"немає prompt id {prompt_id}; є: {sorted(variants)}")
    return variants[prompt_id]
