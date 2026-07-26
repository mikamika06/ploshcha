import time
from typing import Callable

from pydantic import BaseModel

from ..ports.tool import ToolCall, ToolPort, ToolResult, ToolSpec

EVENT_YEARS = {
    "початок Хмельниччини": 1648,
    "Битва під Крутами": 1918,
    "проголошення незалежності України": 1991,
    "Акт Злуки": 1919,
}
FACTS = {
    "Тарас Шевченко": "Український поет і художник, автор «Кобзаря».",
    "Битва під Крутами": "Бій 29 січня 1918 року між військами УНР і більшовиками.",
    "Іван Мазепа": "Гетьман Війська Запорозького 1687–1709.",
}


class CheckDateArgs(BaseModel):
    year: int
    event: str


class LookupFactArgs(BaseModel):
    entity: str


class CalcArgs(BaseModel):
    expr: str


class FinalAnswerArgs(BaseModel):
    text: str


class Tool:
    def __init__(self, name: str, description: str, params: type[BaseModel], fn: Callable):
        self.name = name
        self.description = description
        self.params = params
        self.fn = fn

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, params=self.params.model_json_schema())


def _check_date(a: CheckDateArgs) -> dict:
    actual = EVENT_YEARS.get(a.event)
    return {"matches": actual == a.year, "actual_year": actual, "known": actual is not None}


def _lookup_fact(a: LookupFactArgs) -> dict:
    return {"fact": FACTS.get(a.entity), "known": a.entity in FACTS}


def _calc(a: CalcArgs) -> dict:
    if not set(a.expr) <= set("0123456789+-*/(). "):
        raise ValueError("disallowed characters")
    return {"result": eval(a.expr, {"__builtins__": {}}, {})}


def _final_answer(a: FinalAnswerArgs) -> dict:
    return {"text": a.text}


DEFAULT_TOOLS = [
    Tool("check_date", "Перевірити, чи рік відповідає історичній події.", CheckDateArgs, _check_date),
    Tool("lookup_fact", "Знайти факт про сутність.", LookupFactArgs, _lookup_fact),
    Tool("calc", "Обчислити арифметичний вираз.", CalcArgs, _calc),
    Tool("final_answer", "Завершити й повернути фінальну відповідь.", FinalAnswerArgs, _final_answer),
]


class FakeToolbox(ToolPort):
    def __init__(self, tools: list[Tool] | None = None):
        self._tools = {t.name: t for t in (tools if tools is not None else DEFAULT_TOOLS)}

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def call(self, request: ToolCall) -> ToolResult:
        t0 = time.perf_counter()
        tool = self._tools.get(request.tool)
        if tool is None:
            return ToolResult(tool=request.tool, ok=False, error="unknown_tool")
        try:
            args = tool.params(**request.args)
            value = tool.fn(args)
            ok, err = True, None
        except Exception as e:
            value, ok, err = None, False, f"{type(e).__name__}: {e}"
        return ToolResult(
            tool=request.tool, ok=ok, value=value, error=err,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
