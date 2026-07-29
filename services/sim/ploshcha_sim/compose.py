from .adapters.memory_notebook import NotebookMemory
from .adapters.planner_skeleton import SkeletonPlanner
from .adapters.router_profile import PresetEffort, profile_router, single_model_router
from .adapters.skills_declared import skillbox
from .adapters.tools_fake import DEFAULT_TOOLS, FakeToolbox
from .adapters.tools_docs import DOCS_AGG_TOOLS, DOCS_TOOLS
from .adapters.tools_registry import AGG_TOOLS, REGISTRY_TOOLS
from .adapters.tools_ua import UA_TOOLS
from .agents import Orchestrator
from .domain.gate import FINAL_TOOL
from .domain.spec import AppSpec
from .domain.task import Budget

NO_DATA_TOOLS = [t for t in DEFAULT_TOOLS if t.name == FINAL_TOOL]
TOOLSETS = {"default": DEFAULT_TOOLS, "ua": UA_TOOLS,
            "registry": REGISTRY_TOOLS, "registry_agg": AGG_TOOLS,
            "docs": DOCS_TOOLS, "docs_agg": DOCS_AGG_TOOLS,
            "none": NO_DATA_TOOLS}


def build_toolbox(spec: AppSpec) -> FakeToolbox:
    return FakeToolbox(tools=TOOLSETS[spec.toolset])


def build_skillbox(spec: AppSpec):
    """Той самий набір, але з декларацією форми даних (K7-SKILLS)."""
    return skillbox(spec.toolset, tools=TOOLSETS[spec.toolset])


def build_router(spec: AppSpec, *, lapa, mamay):
    if spec.routing == "hetero":
        return profile_router(lapa, mamay)
    if spec.routing == "mamay":
        return single_model_router(mamay, lane="mamay")
    return single_model_router(lapa, lane="lapa")


def build_planner(spec: AppSpec):
    return SkeletonPlanner(gather=spec.plan_gather) if spec.planner == "skeleton" else None


def build_notebook(spec: AppSpec):
    return NotebookMemory if spec.memory == "notebook" else None


def build_budget(spec: AppSpec) -> Budget:
    return Budget(max_steps=spec.max_steps)


def build_orchestrator(spec: AppSpec, *, lapa, mamay, system: str | None = None,
                       tail: str | None = None, prompt_id: str = "", prompt_sha: str = "",
                       answer_instruction: str | None = None) -> Orchestrator:
    """Composition root: специфікація -> зібраний оркестратор.

    Промпти приходять уже відрендерені: реєстр промптів живе у вимірювальному шарі,
    і колесо не має від нього залежати (той самий поділ, що й у самому Orchestrator).
    """
    return Orchestrator(
        build_router(spec, lapa=lapa, mamay=mamay),
        PresetEffort(),
        build_toolbox(spec),
        planner=build_planner(spec),
        verifier=spec.verifier,
        system=system,
        tail=tail,
        prompt_id=prompt_id,
        prompt_sha=prompt_sha,
        recovery=spec.recovery,
        notebook=build_notebook(spec),
        answer_channel=spec.answer_channel,
        answer_instruction=answer_instruction,
        plan_guard=spec.plan_guard,
    )
