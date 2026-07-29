from ..domain.gate import FINAL_TOOL
from ..domain.skill import SkillSpec, shape_notes
from ..ports.skill import Skill, SkillRegistry
from ..ports.tool import ToolCall, ToolPort, ToolResult, ToolSpec
from .tools_fake import Tool

ANSWER_CAPABILITY = "core.answer"


class ToolSkill(Skill):
    def __init__(self, tool: Tool, spec: SkillSpec):
        self.tool = tool
        self._spec = spec

    @property
    def spec(self) -> SkillSpec:
        return self._spec


class SkillBox(ToolPort, SkillRegistry):
    """Реєстр скілів, що лишається звичайним ToolPort — оркестратор не змінюється зовсім."""

    def __init__(self, skills: list[ToolSkill]):
        self._skills = list(skills)
        self._by_name = {s.tool.name: s for s in self._skills}

    def skills(self) -> list[Skill]:
        return list(self._skills)

    def scoped(self, capabilities: tuple[str, ...]) -> "SkillBox":
        keep = tuple(capabilities) + (ANSWER_CAPABILITY,)
        return SkillBox([s for s in self._skills if s.spec.capability in keep])

    def specs(self) -> list[ToolSpec]:
        return [s.tool.spec() for s in self._skills]

    def call(self, request: ToolCall) -> ToolResult:
        skill = self._by_name.get(request.tool)
        if skill is None:
            return ToolResult(tool=request.tool, ok=False, error="unknown_tool")
        return _run(skill.tool, request)

    def notes(self) -> list[str]:
        return shape_notes(self.skill_specs())


def _run(tool: Tool, request: ToolCall) -> ToolResult:
    import time
    t0 = time.perf_counter()
    try:
        value = tool.fn(tool.params(**request.args))
        ok, err = True, None
    except Exception as exc:
        value, ok, err = None, False, f"{type(exc).__name__}: {exc}"
    return ToolResult(tool=request.tool, ok=ok, value=value, error=err,
                      latency_ms=int((time.perf_counter() - t0) * 1000))


def declare(tools: list[Tool], declarations: dict[str, SkillSpec]) -> SkillBox:
    """Оголошення на кожен інструмент; `final_answer` отримує стандартне."""
    out = []
    for tool in tools:
        spec = declarations.get(tool.name)
        if spec is None:
            spec = SkillSpec(name=tool.name, capability=ANSWER_CAPABILITY) \
                if tool.name == FINAL_TOOL else SkillSpec(name=tool.name, capability="unknown")
        out.append(ToolSkill(tool, spec))
    return SkillBox(out)
