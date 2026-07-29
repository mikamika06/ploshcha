from ploshcha_sim.compose import (
    build_budget,
    build_orchestrator,
    build_skillbox,
    build_toolbox,
)
from ploshcha_sim.domain.skill import shape_notes
from ploshcha_sim.domain.spec import AppSpec

from .harness import Runner, gated_runner, orchestrator_runner, single_call_runner
from .prompts import resolve

BASE = AppSpec()
UA = "agent/v2-ua"

CONDITIONS: dict[str, AppSpec] = {
    "single-mamay": BASE.with_(mode="single", routing="mamay"),
    "single-lapa": BASE.with_(mode="single", routing="lapa"),
    "mamay@5": BASE.with_(routing="mamay"),
    "mamay@8": BASE.with_(routing="mamay", max_steps=8),
    "mamay+rec@8": BASE.with_(routing="mamay", max_steps=8, recovery=True),
    "hetero@5": BASE,
    "hetero@8": BASE.with_(max_steps=8),
    "hetero+rec@8": BASE.with_(max_steps=8, recovery=True),
    "hetero-nov@8": BASE.with_(max_steps=8, verifier=False),
    "gate-notools-mamay": BASE.with_(mode="gated", toolset="none", gate_direct="mamay", max_steps=8),
    "gate-notools-lapa": BASE.with_(mode="gated", toolset="none", gate_direct="lapa", max_steps=8),
    "gate-tools-hetero": BASE.with_(mode="gated", toolset="default", gate_direct="mamay", max_steps=8),
    "hetero-plan@8": BASE.with_(max_steps=8, planner="skeleton"),
    "hetero-textans@8": BASE.with_(max_steps=8, planner="skeleton", answer_channel="text"),
    "hetero-textfull@8": BASE.with_(max_steps=8, answer_channel="text",
                                    answer_prompt_id="answer/full"),
    "hetero-ua-tools@8": BASE.with_(max_steps=8, toolset="ua", prompt_id=UA),
    "hetero-ua-textans@8": BASE.with_(max_steps=8, toolset="ua", prompt_id=UA,
                                       answer_channel="text"),
}

REG = BASE.with_(toolset="registry", prompt_id="agent/v2-reg")
CHAIN: dict[str, AppSpec] = {
    "chain-schema@8": REG.with_(max_steps=8),
    "chain-text@8": REG.with_(max_steps=8, answer_channel="text"),
    "chain-schema@16": REG.with_(max_steps=16),
    "chain-text@16": REG.with_(max_steps=16, answer_channel="text"),
    "chain-text-mem@16": REG.with_(max_steps=16, answer_channel="text", memory="notebook"),
    "chain-text-plan@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton"),
    "chain-text-rec@16": REG.with_(max_steps=16, answer_channel="text", recovery=True),
    "chain-iter-schema@16": REG.with_(max_steps=16, prompt_id="agent/v2-iter"),
    "chain-iter-text@16": REG.with_(max_steps=16, prompt_id="agent/v2-iter",
                                    answer_channel="text"),
    "chain-agg-schema@16": REG.with_(max_steps=16, toolset="registry_agg",
                                     prompt_id="agent/v2-agg"),
    "chain-agg-text@16": REG.with_(max_steps=16, toolset="registry_agg",
                                   prompt_id="agent/v2-agg", answer_channel="text"),
    "chain-text-plan9@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                     plan_gather=9),
    "chain-text-guard9@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                      plan_gather=9, plan_guard=True),
    "chain-text-guard9rec@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                         plan_gather=9, plan_guard=True, recovery=True),
}
CONDITIONS.update(CHAIN)

PAIRS = (("mamay@8", "mamay+rec@8"), ("hetero@8", "hetero+rec@8"),
         ("hetero@8", "gate-notools-mamay"),
         ("hetero@8", "gate-tools-hetero"),
         ("hetero-plan@8", "hetero-textans@8"),
         ("hetero@8", "hetero-textfull@8"),
         ("hetero-textans@8", "hetero-ua-textans@8"),
         ("hetero@8", "hetero-ua-tools@8"),
         ("chain-schema@16", "chain-text@16"),
         ("chain-schema@8", "chain-text@8"),
         ("chain-text@8", "chain-text@16"),
         ("chain-text@16", "chain-text-mem@16"),
         ("chain-text@16", "chain-text-plan@16"),
         ("chain-text@16", "chain-text-rec@16"),
         ("chain-text@16", "chain-agg-text@16"),
         ("chain-agg-schema@16", "chain-agg-text@16"),
         ("chain-text-plan@16", "chain-text-plan9@16"),
         ("chain-text-plan9@16", "chain-text-guard9@16"),
         ("chain-text-guard9@16", "chain-text-guard9rec@16"))


def _model(routing: str, *, lapa, mamay):
    """Прямий виклик потребує ОДНОЇ моделі; `hetero` тут — помилка конфігурації, і вона гучна."""
    return {"mamay": mamay, "lapa": lapa}[routing]


def runner_for(spec: AppSpec, *, lapa, mamay) -> Runner:
    variant = resolve(spec.prompt_id)
    system = variant.render_system()
    answer_instruction = resolve(spec.answer_prompt_id).render_system()

    if spec.mode == "single":
        return single_call_runner(_model(spec.routing, lapa=lapa, mamay=mamay),
                                  system=system, max_tokens=spec.max_tokens)

    def make_orch():
        return build_orchestrator(spec, lapa=lapa, mamay=mamay, system=system,
                                  tail=variant.tail or None, prompt_id=variant.id,
                                  prompt_sha=variant.sha256,
                                  answer_instruction=answer_instruction)

    loop = orchestrator_runner(make_orch, budget=build_budget(spec))
    if spec.mode == "gated":
        return gated_runner(_model(spec.gate_direct, lapa=lapa, mamay=mamay), build_toolbox(spec),
                            system=system, max_tokens=spec.max_tokens, loop_runner=loop)
    return loop


def grid(names=None, *, lapa, mamay) -> dict[str, Runner]:
    chosen = list(names) if names else list(CONDITIONS)
    return {n: runner_for(CONDITIONS[n], lapa=lapa, mamay=mamay) for n in chosen}


def prompt_ids(names=None) -> dict[str, str]:
    chosen = list(names) if names else list(CONDITIONS)
    return {n: CONDITIONS[n].prompt_id for n in chosen}


def spec_shas(names=None) -> dict[str, str]:
    chosen = list(names) if names else list(CONDITIONS)
    return {n: CONDITIONS[n].sha256 for n in chosen}


def shape_warnings(names=None) -> dict[str, list[str]]:
    """Причина, а не лише бал: умова, що дає плоскому циклу колекцію, позначена в звіті.

    Це властивість КОНФІГУРАЦІЇ, не прогону, тому живе на рівні умови — інакше той самий рядок
    повторювався б у кожній клітинці. Гучно, але не корективно (K7-SKILLS §2).
    """
    chosen = list(names) if names else list(CONDITIONS)
    out = {}
    for name in chosen:
        spec = CONDITIONS[name]
        notes = shape_notes(build_skillbox(spec).skill_specs())
        if notes and spec.mode != "single":
            out[name] = notes
    return out
