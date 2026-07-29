from ..ports.llm import LlmPort
from ..ports.router import (
    STEP_KINDS,
    EffortConfig,
    EffortPolicy,
    ModelLane,
    ModelRouter,
    StepKind,
)

LAPA_KINDS: tuple[StepKind, ...] = ("parse", "classify", "select", "ground", "gate")
MAMAY_KINDS: tuple[StepKind, ...] = ("decide", "generate", "synthesize", "judge")

LOW_KINDS: tuple[StepKind, ...] = ("parse", "classify", "select", "ground", "gate")
HIGH_KINDS: tuple[StepKind, ...] = ("decide", "generate", "synthesize", "judge")


class ProfileRouter(ModelRouter):
    def __init__(self, mapping: dict[StepKind, LlmPort], default: LlmPort | None = None,
                 lanes: dict[StepKind, ModelLane] | None = None):
        self._map = dict(mapping)
        self._default = default
        self._lanes = dict(lanes or {})

    def route(self, kind: StepKind) -> LlmPort:
        llm = self._map.get(kind, self._default)
        if llm is None:
            raise KeyError(f"no model routed for kind={kind}")
        return llm

    def lane(self, kind: StepKind) -> ModelLane:
        return self._lanes.get(kind, "unknown")


def profile_router(lapa: LlmPort, mamay: LlmPort) -> ProfileRouter:
    mapping: dict[StepKind, LlmPort] = {}
    lanes: dict[StepKind, ModelLane] = {}
    for k in LAPA_KINDS:
        mapping[k], lanes[k] = lapa, "lapa"
    for k in MAMAY_KINDS:
        mapping[k], lanes[k] = mamay, "mamay"
    return ProfileRouter(mapping, lanes=lanes)


def single_model_router(llm: LlmPort, lane: ModelLane = "unknown") -> ProfileRouter:
    return ProfileRouter({k: llm for k in STEP_KINDS},
                         lanes={k: lane for k in STEP_KINDS})


class PresetEffort(EffortPolicy):
    def __init__(self, low: EffortConfig | None = None, high: EffortConfig | None = None,
                 high_kinds: tuple[StepKind, ...] = HIGH_KINDS):
        self._low = low or EffortConfig(think_tokens=0, max_tokens=256, tier="strict", samples=1, verify=False)
        self._high = high or EffortConfig(think_tokens=768, max_tokens=512, tier="wire", samples=1, verify=True)
        self._high_kinds = set(high_kinds)

    def effort(self, kind: StepKind) -> EffortConfig:
        return self._high if kind in self._high_kinds else self._low


def sampling_effort(temperature: float = 0.7, base: EffortPolicy | None = None,
                    high_kinds: tuple[StepKind, ...] = HIGH_KINDS) -> PresetEffort:
    src = base or PresetEffort(high_kinds=high_kinds)
    return PresetEffort(
        low=src.effort(LOW_KINDS[0]).model_copy(update={"temperature": temperature}),
        high=src.effort(HIGH_KINDS[0]).model_copy(update={"temperature": temperature}),
        high_kinds=high_kinds,
    )
