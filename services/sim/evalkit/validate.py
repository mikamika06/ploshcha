from pathlib import Path

from pydantic import BaseModel, Field

from ploshcha_sim.domain.task import TaskResult

from .checks import split_checks
from .harness import EvalItem, load_items

ITEMS_DIR = Path(__file__).parent / "items"


class ItemReport(BaseModel):
    item_id: str
    has_gold: bool = False
    gold_failed: list[str] = Field(default_factory=list)
    has_foil: bool = False
    foil_vacuous: bool = False

    @property
    def ok(self) -> bool:
        return not self.gold_failed and not self.foil_vacuous


def synth_result(answer: str, tools: list[str] | None = None) -> TaskResult:
    scratch = [{"call": {"tool": t}, "result": {}} for t in (tools or [])]
    return TaskResult(answer=answer, accepted=True, steps=1 + len(scratch), scratch=scratch)


def validate_item(item: EvalItem) -> ItemReport:
    report = ItemReport(item_id=item.id, has_gold=bool(item.gold), has_foil=bool(item.foil))
    for answer in item.gold:
        outcome, _ = split_checks(item.checks, synth_result(answer, item.gold_tools))
        report.gold_failed.extend(name for name, ok in outcome.items() if not ok)
    if item.foil:
        vacuous = True
        for answer in item.foil:
            outcome, _ = split_checks(item.checks, synth_result(answer, item.gold_tools))
            if any(not ok for ok in outcome.values()):
                vacuous = False
        report.foil_vacuous = vacuous
    return report


def validate_items(items: list[EvalItem]) -> list[ItemReport]:
    return [validate_item(i) for i in items]


def coverage(items: list[EvalItem]) -> float:
    return sum(1 for i in items if i.gold) / len(items) if items else 0.0


def validate_file(name: str) -> list[ItemReport]:
    return validate_items(load_items(str(ITEMS_DIR / f"{name}.jsonl")))


def item_sets() -> list[str]:
    return sorted(p.stem for p in ITEMS_DIR.glob("*.jsonl"))


def format_report(name: str, reports: list[ItemReport]) -> str:
    bad = [r for r in reports if not r.ok]
    covered = sum(1 for r in reports if r.has_gold)
    lines = [f"{name}: {len(reports)} айтемів, з еталоном {covered}, дефектних чеків {len(bad)}"]
    for r in bad:
        if r.gold_failed:
            lines.append(f"  {r.item_id}: еталон НЕ проходить {r.gold_failed}")
        if r.foil_vacuous:
            lines.append(f"  {r.item_id}: хибна відповідь проходить усі чеки — чек порожній")
    return "\n".join(lines)
