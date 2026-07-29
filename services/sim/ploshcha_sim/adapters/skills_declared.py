from ..domain.skill import SkillSpec
from .docs_kb import DOCUMENTS
from .docs_kb import ids_for as doc_ids
from .registry_kb import VILLAGES, ids_for
from .skillbox import SkillBox, declare
from .tools_docs import DOCS_AGG_TOOLS, DOCS_TOOLS
from .tools_fake import DEFAULT_TOOLS
from .tools_registry import (
    AGG_TOOLS,
    REGISTRY_REDUCE_TOOLS,
    REGISTRY_SUM_TOOLS,
    REGISTRY_TEACH_TOOLS,
    REGISTRY_TOOLS,
)
from .tools_ua import UA_TOOLS

BIGGEST_VILLAGE = max(len(ids_for(v)) for v in VILLAGES)
LONGEST_DOC = max(len(doc_ids(d)) for d in DOCUMENTS)

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

REGISTRY_SUM_DECLARED = {
    "список_записів": SkillSpec(name="список_записів", capability="registry.index",
                                shape="collection", max_items=BIGGEST_VILLAGE),
    "запис": SkillSpec(name="запис", capability="registry.record", shape="scalar"),
    "підсумувати": SkillSpec(name="підсумувати", capability="math.sum", shape="scalar"),
}

REGISTRY_REDUCE_DECLARED = {
    "список_записів": SkillSpec(name="список_записів", capability="registry.index",
                                shape="collection", max_items=BIGGEST_VILLAGE),
    "звести": SkillSpec(name="звести", capability="registry.reduce", shape="aggregate",
                        cost_hint=2),
}

AGG_DECLARED = {
    "записи_села": SkillSpec(name="записи_села", capability="registry.records",
                             shape="aggregate", cost_hint=3),
    "обчислити": SkillSpec(name="обчислити", capability="math.eval", shape="scalar"),
}

DOCS_DECLARED = {
    "список_абзаців": SkillSpec(name="список_абзаців", capability="docs.index",
                                shape="collection", max_items=LONGEST_DOC),
    "абзац": SkillSpec(name="абзац", capability="docs.paragraph", shape="scalar"),
}

DOCS_AGG_DECLARED = {
    "абзаци_документа": SkillSpec(name="абзаци_документа", capability="docs.paragraphs",
                                  shape="aggregate", cost_hint=3),
}

DECLARED = {
    "default": (DEFAULT_TOOLS, DEFAULT_DECLARED),
    "ua": (UA_TOOLS, UA_DECLARED),
    "registry": (REGISTRY_TOOLS, REGISTRY_DECLARED),
    "registry_teach": (REGISTRY_TEACH_TOOLS, REGISTRY_DECLARED),
    "registry_sum": (REGISTRY_SUM_TOOLS, REGISTRY_SUM_DECLARED),
    "registry_reduce": (REGISTRY_REDUCE_TOOLS, REGISTRY_REDUCE_DECLARED),
    "registry_agg": (AGG_TOOLS, AGG_DECLARED),
    "docs": (DOCS_TOOLS, DOCS_DECLARED),
    "docs_agg": (DOCS_AGG_TOOLS, DOCS_AGG_DECLARED),
}


def skillbox(toolset: str, tools=None) -> SkillBox:
    known, declarations = DECLARED.get(toolset, (None, {}))
    return declare(list(tools if tools is not None else known or []), declarations)
