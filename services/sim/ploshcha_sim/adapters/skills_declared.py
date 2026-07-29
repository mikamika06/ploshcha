from ..domain.skill import SkillSpec
from .registry_kb import VILLAGES, ids_for
from .skillbox import SkillBox, declare
from .tools_fake import DEFAULT_TOOLS
from .tools_registry import AGG_TOOLS, REGISTRY_TOOLS
from .tools_ua import UA_TOOLS

BIGGEST_VILLAGE = max(len(ids_for(v)) for v in VILLAGES)

DEFAULT_DECLARED = {
    "check_date": SkillSpec(name="check_date", capability="history.date", shape="scalar"),
    "lookup_fact": SkillSpec(name="lookup_fact", capability="history.fact", shape="scalar"),
    "calc": SkillSpec(name="calc", capability="math.eval", shape="scalar"),
}

UA_DECLARED = {
    "перевірити_дату": SkillSpec(name="перевірити_дату", capability="history.date", shape="scalar"),
    "знайти_факт": SkillSpec(name="знайти_факт", capability="history.fact", shape="scalar"),
    "обчислити": SkillSpec(name="обчислити", capability="math.eval", shape="scalar"),
}

REGISTRY_DECLARED = {
    "список_записів": SkillSpec(name="список_записів", capability="registry.index",
                                shape="collection", max_items=BIGGEST_VILLAGE, cost_hint=1),
    "запис": SkillSpec(name="запис", capability="registry.record", shape="scalar", cost_hint=1),
    "обчислити": SkillSpec(name="обчислити", capability="math.eval", shape="scalar"),
}

AGG_DECLARED = {
    "записи_села": SkillSpec(name="записи_села", capability="registry.records",
                             shape="aggregate", cost_hint=3),
    "обчислити": SkillSpec(name="обчислити", capability="math.eval", shape="scalar"),
}

DECLARED = {
    "default": (DEFAULT_TOOLS, DEFAULT_DECLARED),
    "ua": (UA_TOOLS, UA_DECLARED),
    "registry": (REGISTRY_TOOLS, REGISTRY_DECLARED),
    "registry_agg": (AGG_TOOLS, AGG_DECLARED),
}


def skillbox(toolset: str, tools=None) -> SkillBox:
    known, declarations = DECLARED.get(toolset, (None, {}))
    return declare(list(tools if tools is not None else known or []), declarations)
