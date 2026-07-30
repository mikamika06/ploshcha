from pathlib import Path

from pydantic import BaseModel, Field

from ploshcha_sim.compose import TOOLSETS
from ploshcha_sim.domain.task import TaskResult

from .checks import is_hygiene, split_checks
from .harness import EvalItem, load_items

ITEMS_DIR = Path(__file__).parent / "items"

TOOL_KINDS = {"used_tool": "tool", "used_tool_any": "tools"}

ITEM_SET_TOOLSETS: dict[str, tuple[str, ...]] = {
    "starter": ("default",),
    "recover": ("default",),
    "audit": ("default", "ua", "reference"),
    "ua-lang": ("default", "none", "ua_norm"),
    "ua-extract": ("default", "none"),
    "chain": ("registry", "registry_agg", "registry_teach", "registry_sum", "registry_reduce"),
    "docs": ("docs", "docs_agg", "docs_years"),
}


class ItemReport(BaseModel):
    item_id: str
    has_gold: bool = False
    gold_failed: list[str] = Field(default_factory=list)
    has_foil: bool = False
    foil_passed: list[str] = Field(default_factory=list)
    unsatisfiable: list[str] = Field(default_factory=list)

    @property
    def foil_vacuous(self) -> bool:
        return bool(self.foil_passed)

    @property
    def ok(self) -> bool:
        return not self.gold_failed and not self.foil_passed and not self.unsatisfiable


def synth_result(answer: str, tools: list[str] | None = None) -> TaskResult:
    scratch = [{"call": {"tool": t}, "result": {}} for t in (tools or [])]
    return TaskResult(answer=answer, accepted=True, steps=1 + len(scratch), scratch=scratch)


def wanted_tools(spec: dict) -> list[str]:
    field = TOOL_KINDS.get(spec["kind"])
    if field is None:
        return []
    value = spec.get(field)
    return [value] if isinstance(value, str) else list(value or [])


def toolset_violations(item: EvalItem, toolset: str) -> list[str]:
    """Результатний тул-предикат мусить бути здійсненним під КОЖНИМ набором, під яким ганяється айтем.

    Дефект 13 (= повторення 8): чек вимагав `запис`, а в агрегатному наборі такого інструмента
    немає, тому правильні відповіді отримували нуль. Перевірка статична — синтез тут не потрібен.
    Гігієні бути нездійсненною дозволено: `tool_calls_at_least` саме й міряє, чи був обхід.
    """
    names = tool_names(toolset)
    bad = []
    for spec in item.checks:
        if is_hygiene(spec):
            continue
        wanted = wanted_tools(spec)
        if wanted and not (set(wanted) & names):
            bad.append(f"{spec['kind']}{wanted} нездійсненний під набором «{toolset}»")
    return bad


def tool_names(toolset: str) -> set[str]:
    return {t.name for t in TOOLSETS[toolset]}


def phantom_gold_tools(item: EvalItem, toolsets: tuple[str, ...]) -> list[str]:
    """`gold_tools` монтує фейкову трасу — інструмент з неї мусить існувати хоч в одному наборі."""
    known = set().union(*(tool_names(ts) for ts in toolsets)) if toolsets else set()
    return [f"gold_tools містить «{n}», якого немає в наборах {list(toolsets)}"
            for n in item.gold_tools if n not in known]


def item_toolsets(item: EvalItem) -> tuple[str, ...]:
    return tuple(item.toolsets) if item.toolsets else ("default",)


def validate_item(item: EvalItem, toolsets: tuple[str, ...] = ()) -> ItemReport:
    report = ItemReport(item_id=item.id, has_gold=bool(item.gold), has_foil=bool(item.foil))
    for answer in item.gold:
        outcome, _ = split_checks(item.checks, synth_result(answer, item.gold_tools))
        report.gold_failed.extend(name for name, ok in outcome.items() if not ok)
    for answer in item.foil:
        outcome, _ = split_checks(item.checks, synth_result(answer, item.gold_tools))
        if outcome and all(outcome.values()):
            report.foil_passed.append(answer[:60])
    chosen = toolsets or item_toolsets(item)
    for toolset in chosen:
        report.unsatisfiable.extend(toolset_violations(item, toolset))
    report.unsatisfiable.extend(phantom_gold_tools(item, chosen))
    return report


def validate_items(items: list[EvalItem], toolsets: tuple[str, ...] = ()) -> list[ItemReport]:
    return [validate_item(i, toolsets) for i in items]


def coverage(items: list[EvalItem]) -> float:
    return sum(1 for i in items if i.gold) / len(items) if items else 0.0


def validate_file(name: str) -> list[ItemReport]:
    toolsets = ITEM_SET_TOOLSETS.get(name, ("default",))
    return validate_items(load_items(str(ITEMS_DIR / f"{name}.jsonl")), toolsets)


def item_sets() -> list[str]:
    return sorted(p.stem for p in ITEMS_DIR.glob("*.jsonl"))


def format_report(name: str, reports: list[ItemReport]) -> str:
    bad = [r for r in reports if not r.ok]
    covered = sum(1 for r in reports if r.has_gold)
    sets = ", ".join(ITEM_SET_TOOLSETS.get(name, ("default",)))
    lines = [f"{name}: {len(reports)} айтемів, з еталоном {covered}, "
             f"дефектних чеків {len(bad)} (набори: {sets})"]
    for r in bad:
        if r.gold_failed:
            lines.append(f"  {r.item_id}: еталон НЕ проходить {r.gold_failed}")
        for passed in r.foil_passed:
            lines.append(f"  {r.item_id}: хибна відповідь проходить усі чеки — «{passed}…»")
        for miss in r.unsatisfiable:
            lines.append(f"  {r.item_id}: {miss}")
    return "\n".join(lines)
