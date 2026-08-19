from .adapters.guard_rules import RuleGuard
from .ports.guard import Policy
from .adapters.memory_notebook import NotebookMemory
from .adapters.planner_skeleton import SkeletonPlanner
from .adapters.router_profile import (
    PresetEffort,
    profile_router,
    sampling_effort,
    single_model_router,
)
from .adapters.skills_declared import skillbox
from .adapters.tools_fake import DEFAULT_TOOLS, FakeToolbox
from .adapters.tools_lexis import LEXIS_TOOLS
from .adapters.tools_docs import DOCS_AGG_TOOLS, DOCS_TOOLS, DOCS_YEARS_TOOLS
from .adapters.tools_registry import (
    AGG_TOOLS,
    REGISTRY_REDUCE_TOOLS,
    REGISTRY_SUM_TOOLS,
    REGISTRY_TEACH_TOOLS,
    REGISTRY_TOOLS,
)
from .adapters.tools_reference import REFERENCE_TOOLS
from .adapters.tools_ua import UA_TOOLS
from .adapters.tools_ua_norm import UA_NORM_TOOLS
from .agents import Orchestrator
from .agents.graph import AgentGraph
from .agents.viche import Viche
from .domain.gate import FINAL_TOOL
from .domain.spec import AppSpec
from .domain.task import Budget

NO_DATA_TOOLS = [t for t in DEFAULT_TOOLS if t.name == FINAL_TOOL]
TOOLSETS = {"default": DEFAULT_TOOLS, "ua": UA_TOOLS,
            "registry": REGISTRY_TOOLS, "registry_agg": AGG_TOOLS,
            "registry_teach": REGISTRY_TEACH_TOOLS,
            "registry_sum": REGISTRY_SUM_TOOLS,
            "registry_reduce": REGISTRY_REDUCE_TOOLS,
            "docs": DOCS_TOOLS, "docs_agg": DOCS_AGG_TOOLS,
            "docs_years": DOCS_YEARS_TOOLS,
            "ua_norm": UA_NORM_TOOLS,
            "reference": REFERENCE_TOOLS,
            "lexis": LEXIS_TOOLS,
            "none": NO_DATA_TOOLS}


def build_toolbox(spec: AppSpec) -> FakeToolbox:
    return FakeToolbox(tools=TOOLSETS[spec.toolset])


def build_skillbox(spec: AppSpec):
    """Той самий набір, але з декларацією форми даних (K7-SKILLS)."""
    return skillbox(spec.toolset, tools=TOOLSETS[spec.toolset])


def build_router(spec: AppSpec, *, lapa, mamay):
    """Ярус СУДДІ — окрема вісь від яруса відповіді.

    Доти `routing` керував усім одразу, тому `routing="lapa"` означало «Lapa судить Lapa» — прямо
    проти оголошеного інваріанта. Порушення було невидиме, бо осі не існувало.
    """
    if spec.routing == "hetero":
        router = profile_router(lapa, mamay)
    elif spec.routing == "mamay":
        router = single_model_router(mamay, lane="mamay")
    else:
        router = single_model_router(lapa, lane="lapa")
    if spec.judge_lane != "auto":
        router.set_judge(mamay if spec.judge_lane == "mamay" else lapa, spec.judge_lane)
    return router


def build_effort(spec: AppSpec):
    """`pass^k` має сенс лише при temperature > 0: при 0 усі seeds дають ту саму трасу (V0)."""
    return PresetEffort() if spec.temperature == 0.0 else sampling_effort(spec.temperature)


def build_planner(spec: AppSpec):
    return SkeletonPlanner(gather=spec.plan_gather) if spec.planner == "skeleton" else None


def build_notebook(spec: AppSpec):
    return NotebookMemory if spec.memory == "notebook" else None


def build_budget(spec: AppSpec) -> Budget:
    return Budget(max_steps=spec.max_steps)


def build_viche(spec: AppSpec, *, lapa, mamay, **kw):
    """Віче — окремий агент, не гілка графа: у графа інша петля й інший критерій успіху."""
    return Viche(build_router(spec, lapa=lapa, mamay=mamay), build_effort(spec),
                 build_toolbox(spec) if spec.toolset != "none" else None,
                 trace=kw.get("trace"), run_id=kw.get("run_id", "viche"),
                 width=spec.max_width, system=kw.get("system"),
                 prompt_id=kw.get("prompt_id", spec.prompt_id),
                 prompt_sha=kw.get("prompt_sha", ""),
                 **{k: kw[k] for k in ("score_system", "line_system", "summary_system",
                                       "doubt_system", "chronicle_system") if k in kw})


def build_graph(spec: AppSpec, *, lapa, mamay, **kw):
    """Граф — це КОНФІГ, не окремий застосунок: дитина збирається тим самим коренем.

    `budget` дитини приходить від графа (поділений), тому тут його не задаємо.
    """
    # `trace`/`run_id` належать графу, а не дитині: раніше вони форвардились у
    # `build_orchestrator`, який їх не приймає, тому трасований граф падав із TypeError —
    # тобто граф не можна було спостерігати взагалі.
    trace = kw.pop("trace", None)
    run_id = kw.pop("run_id", "graph")

    def child(budget):
        orch = build_orchestrator(spec, lapa=lapa, mamay=mamay, **kw)
        orch.trace = trace
        orch.run_id = run_id
        return orch

    return AgentGraph(child, max_depth=spec.max_depth, max_width=spec.max_width,
                      trace=trace, run_id=run_id)


def build_orchestrator(spec: AppSpec, *, lapa, mamay, system: str | None = None,
                       tail: str | None = None, prompt_id: str = "", prompt_sha: str = "",
                       answer_instruction: str | None = None) -> Orchestrator:
    """Composition root: специфікація -> зібраний оркестратор.

    Промпти приходять уже відрендерені: реєстр промптів живе у вимірювальному шарі,
    і колесо не має від нього залежати (той самий поділ, що й у самому Orchestrator).
    """
    return Orchestrator(
        build_router(spec, lapa=lapa, mamay=mamay),
        build_effort(spec),
        build_toolbox(spec),
        planner=build_planner(spec),
        verifier=spec.verifier,
        verify_mode=spec.verify_mode,
        absent_answer=spec.absent_answer,
        guard=RuleGuard(Policy(on_threat="strip" if spec.guard_strip else "note"))
        if spec.guard else None,
        system=system,
        tail=tail,
        prompt_id=prompt_id,
        prompt_sha=prompt_sha,
        recovery=spec.recovery,
        notebook=build_notebook(spec),
        answer_channel=spec.answer_channel,
        answer_instruction=answer_instruction,
        plan_guard=spec.plan_guard,
        coverage=spec.coverage,
        coverage_guard=spec.coverage_guard,
        executor_mode=spec.executor,
        history_window=spec.history_window,
        history_digest=spec.history_digest,
    )
