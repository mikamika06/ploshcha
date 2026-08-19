from ploshcha_sim.compose import (
    build_graph,
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
    # `lang/plain`, бо в цих умовах інструментів даних НЕМА — тул-агентний промпт перелічував би
    # інструменти, яких не існує. Саме так вони й міряні в K7 (0.846 → 0.923 на `ua-lang`).
    "gate-notools-mamay": BASE.with_(mode="gated", toolset="none", gate_direct="mamay",
                                     max_steps=8, prompt_id="lang/plain"),
    "gate-notools-lapa": BASE.with_(mode="gated", toolset="none", gate_direct="lapa",
                                    max_steps=8, prompt_id="lang/plain"),
    "gate-tools-hetero": BASE.with_(mode="gated", toolset="default", gate_direct="mamay", max_steps=8),
    "hetero-plan@8": BASE.with_(max_steps=8, planner="skeleton"),
    "hetero-plan-locked@8": BASE.with_(max_steps=8, planner="skeleton", executor="locked"),
    "hetero-plan-w2@8": BASE.with_(max_steps=8, planner="skeleton", history_window=2),
    "hetero-plan-w4@8": BASE.with_(max_steps=8, planner="skeleton", history_window=4),
    "hetero-plan-w2d@8": BASE.with_(max_steps=8, planner="skeleton", history_window=2,
                                    history_digest=True),
    "hetero-plan-lock-w2d@8": BASE.with_(max_steps=8, planner="skeleton", executor="locked",
                                         history_window=2, history_digest=True),
    "hetero-textans@8": BASE.with_(max_steps=8, planner="skeleton", answer_channel="text"),
    "hetero-textfull@8": BASE.with_(max_steps=8, answer_channel="text",
                                    answer_prompt_id="answer/full"),
    "hetero-ua-tools@8": BASE.with_(max_steps=8, toolset="ua", prompt_id=UA),
    "hetero-ua-textans@8": BASE.with_(max_steps=8, toolset="ua", prompt_id=UA,
                                       answer_channel="text"),
    "ref@8": BASE.with_(max_steps=8, toolset="reference", prompt_id="agent/v2-ref"),
    "ref-rec@8": BASE.with_(max_steps=8, toolset="reference", prompt_id="agent/v2-ref",
                            recovery=True),
    "ref-locked@8": BASE.with_(max_steps=8, toolset="reference", prompt_id="agent/v2-ref",
                               planner="skeleton", executor="locked"),
    "ref-plan@8": BASE.with_(max_steps=8, toolset="reference", prompt_id="agent/v2-ref",
                             planner="skeleton"),
    # ПЛОЩА: розмова = фан-аут (кожна дитина — голос селянина), довідник у циклі (L6),
    # профіль пари з Я3. Ширина 4 взята НЕ для економії, а щоб розмова мала кілька голосів:
    # K6-WIDTH каже, що граф виправданий від 12-16, і в звіті це названо прямо.
    "ploshcha": BASE.with_(max_steps=8, toolset="reference", prompt_id="agent/v2-ref",
                           graph=True, max_width=4, max_depth=2,
                           history_window=2, history_digest=True),
    "ploshcha-notools": BASE.with_(max_steps=8, prompt_id="agent/v2", graph=True,
                                   max_width=4, max_depth=2,
                                   history_window=2, history_digest=True),
    # Стеля кроків 80 — З ЗАМІРУ, не з голови: живий прогін дав 54 кроки при 13 репліках, бо шість
    # тактів пішли в ремонт по три виклики. Найгірший теоретичний випадок (20 тактів × 3 + партитура
    # + фінал) ≈ 63, тож 80 це запас, а не вгадування.
    # ВІЧЕ — прод-режим ПЛОЩІ: не задача з відповіддю, а розмова. Тому `verifier=False`
    # (сумнів попа тут репліка, не вирок) і toolset="lexis" (словник як «дяк погортав книгу»),
    # а НЕ "reference": у тому шість статей, тож він давав «нема в довіднику» на будь-що.
    "viche": BASE.with_(mode="viche", max_width=6, max_steps=80, max_tokens=220,
                        toolset="lexis", verifier=False, temperature=0.8,
                        prompt_id="viche/v1"),
    "viche-notools": BASE.with_(mode="viche", max_width=6, max_steps=80, max_tokens=220,
                                toolset="none", verifier=False, temperature=0.8,
                                prompt_id="viche/v1"),
    # routing="mamay" — щоб пара з `mamay+rec@8` різнилась ЛИШЕ набором інструментів.
    # Перша версія порівнювала hetero-довідку з mamay-відповідями, тобто дві осі одразу.
    "ref-mamay-rec@8": BASE.with_(max_steps=8, toolset="reference", prompt_id="agent/v2-ref",
                                  routing="mamay", recovery=True),
    "uanorm@8": BASE.with_(max_steps=8, toolset="ua_norm", prompt_id="agent/v2-uanorm",
                           answer_channel="text"),
}

# Мовні набори не мають інструментів даних, тому потребують промпту слота `plain`.
# K4.5 це втратив: решітка успадкувала `agent/v2` (тул-агентний) від starter-подібних умов,
# і прогін `ua-lang` міряв 0.654 замість 0.923 — не регрес моделі, а не той промпт.
PLAIN: dict[str, AppSpec] = {
    "lang-mamay": BASE.with_(mode="single", routing="mamay", toolset="none",
                             prompt_id="lang/plain"),
    "lang-lapa": BASE.with_(mode="single", routing="lapa", toolset="none",
                            prompt_id="lang/plain"),
    "extract-mamay": BASE.with_(mode="single", routing="mamay", toolset="none",
                                prompt_id="extract/plain"),
    "extract-lapa": BASE.with_(mode="single", routing="lapa", toolset="none",
                               prompt_id="extract/plain"),
}
CONDITIONS.update(PLAIN)

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
    "chain-agg-w2d@16": REG.with_(max_steps=16, toolset="registry_agg",
                                  prompt_id="agent/v2-agg", answer_channel="text",
                                  history_window=2, history_digest=True),
    "chain-agg-lock-w2d@16": REG.with_(max_steps=16, toolset="registry_agg",
                                       prompt_id="agent/v2-agg", answer_channel="text",
                                       planner="skeleton", executor="locked",
                                       history_window=2, history_digest=True),
    "chain-text-plan9@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                     plan_gather=9),
    "chain-text-guard9@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                      plan_gather=9, plan_guard=True),
    "chain-guard9-locked@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                       plan_gather=9, plan_guard=True, executor="locked"),
    "chain-guard9-w2@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                    plan_gather=9, plan_guard=True, history_window=2),
    "chain-guard9-w2d@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                     plan_gather=9, plan_guard=True, history_window=2,
                                     history_digest=True),
    "chain-guard9-w4d@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                     plan_gather=9, plan_guard=True, history_window=4,
                                     history_digest=True),
    "chain-guard9-lock-w2d@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                          plan_gather=9, plan_guard=True, executor="locked",
                                          history_window=2, history_digest=True),
    "chain-text-guard9rec@16": REG.with_(max_steps=16, answer_channel="text", planner="skeleton",
                                         plan_gather=9, plan_guard=True, recovery=True),
}
DOC = BASE.with_(toolset="docs", prompt_id="agent/v2-docs")
DOCS: dict[str, AppSpec] = {
    "docs-schema@16": DOC.with_(max_steps=16),
    "docs-text@16": DOC.with_(max_steps=16, answer_channel="text"),
    "docs-agg-schema@16": DOC.with_(max_steps=16, toolset="docs_agg",
                                    prompt_id="agent/v2-docs-agg"),
    "docs-agg-text@16": DOC.with_(max_steps=16, toolset="docs_agg",
                                  prompt_id="agent/v2-docs-agg", answer_channel="text"),
    "docs-years@16": DOC.with_(max_steps=16, toolset="docs_years",
                               prompt_id="agent/v2-docs-years"),
    "docs-years-t7@16": DOC.with_(max_steps=16, toolset="docs_years",
                                  prompt_id="agent/v2-docs-years", temperature=0.7),
}
COVER: dict[str, AppSpec] = {
    "chain-cover-schema@16": REG.with_(max_steps=16, coverage=True),
    "chain-cover-text@16": REG.with_(max_steps=16, coverage=True, answer_channel="text"),
    "docs-cover-schema@16": DOC.with_(max_steps=16, coverage=True),
    "docs-cover-text@16": DOC.with_(max_steps=16, coverage=True, answer_channel="text"),
    "chain-cover-schema@32": REG.with_(max_steps=32, coverage=True),
    "chain-schema@32": REG.with_(max_steps=32),
    "chain-cover-rec@32": REG.with_(max_steps=32, coverage=True, recovery=True),
    "chain-teach@32": REG.with_(max_steps=32, coverage=True, recovery=True,
                                toolset="registry_teach"),
    "chain-coverguard@32": REG.with_(max_steps=32, coverage=True, recovery=True,
                                     coverage_guard=True),
    "chain-both@32": REG.with_(max_steps=32, coverage=True, recovery=True,
                               toolset="registry_teach", coverage_guard=True),
    "chain-sum@32": REG.with_(max_steps=32, coverage=True, recovery=True,
                              toolset="registry_sum", prompt_id="agent/v2-sum"),
    "chain-reduce-t7@32": REG.with_(max_steps=32, coverage=True, recovery=True,
                                    toolset="registry_reduce", prompt_id="agent/v2-reduce",
                                    temperature=0.7),
    "chain-reduce@32": REG.with_(max_steps=32, coverage=True, recovery=True,
                                 toolset="registry_reduce", prompt_id="agent/v2-reduce"),
    "chain-sumguard@32": REG.with_(max_steps=32, coverage=True, recovery=True,
                                   toolset="registry_sum", prompt_id="agent/v2-sum",
                                   coverage_guard=True),
}

CONDITIONS.update(CHAIN)
CONDITIONS.update(DOCS)
CONDITIONS.update(COVER)

# UA3: клас скіла «довідка» на лексиці, якої модель НЕ знає (UA2b показав нуль на загальновідомому).
# Пара `lex-loop` ↔ `lex-ref@8` різниться ЛИШЕ набором інструментів: та сама модель, той самий режим,
# та сама стеля кроків. `lex-plain` існує окремо, щоб перевірити, що сама обгортка циклу нічого не
# змінює, — це емпірична відповідь на питання про конфаунд, а не припущення.
LEXIS: dict[str, AppSpec] = {
    "lex-plain": BASE.with_(mode="single", routing="mamay", toolset="none",
                            prompt_id="lexis/plain"),
    "lex-loop": BASE.with_(max_steps=8, routing="mamay", toolset="none", prompt_id="lexis/plain"),
    "lex-ref@8": BASE.with_(max_steps=8, routing="mamay", toolset="lexis",
                            prompt_id="agent/v2-lexis", answer_channel="text"),
    "lex-plain-lapa": BASE.with_(mode="single", routing="lapa", toolset="none",
                                 prompt_id="lexis/plain"),
    "lex-ref-lapa@8": BASE.with_(max_steps=8, routing="lapa", toolset="lexis",
                                 prompt_id="agent/v2-lexis", answer_channel="text"),
    # Страт «поза довідником» показав, що на відповіді «немає» цикл вмирає з `dup_call` +
    # `no_final_answer` (5 із 8 порожніх). Драбина K5 існує рівно для цього — це її замір.
    "lex-ref-rec@8": BASE.with_(max_steps=8, routing="mamay", toolset="lexis",
                                prompt_id="agent/v2-lexis", answer_channel="text", recovery=True),
    "lex-ref-lapa-rec@8": BASE.with_(max_steps=8, routing="lapa", toolset="lexis",
                                     prompt_id="agent/v2-lexis", answer_channel="text",
                                     recovery=True),
    # K9: той самий прогін із заземленим суддею. Пара різниться ЛИШЕ режимом верифікатора, тому
    # різниця — це саме він, а не інша модель чи інший набір інструментів.
    "lex-ref-vg@8": BASE.with_(max_steps=8, routing="mamay", toolset="lexis",
                               prompt_id="agent/v2-lexis", answer_channel="text",
                               verify_mode="grounded"),
    "lex-ref-lapa-vg@8": BASE.with_(max_steps=8, routing="lapa", toolset="lexis",
                                    prompt_id="agent/v2-lexis", answer_channel="text",
                                    verify_mode="grounded"),
    "lex-loop-vg": BASE.with_(max_steps=8, routing="mamay", toolset="none",
                              prompt_id="lexis/plain", verify_mode="grounded"),
    "lex-ref-auto@8": BASE.with_(max_steps=8, routing="mamay", toolset="lexis",
                                 prompt_id="agent/v2-lexis", answer_channel="text",
                                 verify_mode="auto"),
    # U5-SCALE: t>0 робить `pass^k` справжнім. При t=0 greedy-декодування робить seed-и тождественними,
    # тому всі попередні `pass^k` на цьому наборі були тавтологією.
    "lex-loop-t7": BASE.with_(max_steps=8, routing="mamay", toolset="none",
                              prompt_id="lexis/plain", temperature=0.7, verify_mode="auto"),
    "lex-ref-t7@8": BASE.with_(max_steps=8, routing="mamay", toolset="lexis",
                               prompt_id="agent/v2-lexis", answer_channel="text",
                               temperature=0.7, verify_mode="grounded"),
    "lex-ref-lapa-t7@8": BASE.with_(max_steps=8, routing="lapa", toolset="lexis",
                                    prompt_id="agent/v2-lexis", answer_channel="text",
                                    temperature=0.7, verify_mode="grounded"),
}

# JUDGE-LANE: ярус судді як ОКРЕМА вісь. Доти `routing` керував усім, тому числа K9 отримані в
# режимі самосуду, а `routing="lapa"` ще й порушував інваріант «ніколи Lapa-судить-Lapa» — невидимо,
# бо осі не існувало. t=0 навмисно: відповіді детерміновані, тож дві умови бачать ТІ САМІ відповіді,
# і різниця — це чисто суддя.
JUDGE = BASE.with_(max_steps=8, toolset="lexis", prompt_id="agent/v2-lexis",
                   answer_channel="text", verify_mode="grounded")
JUDGE_LANE: dict[str, AppSpec] = {
    "lex-am-jm": JUDGE.with_(routing="mamay", judge_lane="mamay"),
    "lex-am-jl": JUDGE.with_(routing="mamay", judge_lane="lapa"),
    "lex-al-jl": JUDGE.with_(routing="lapa", judge_lane="lapa"),
    "lex-al-jm": JUDGE.with_(routing="lapa", judge_lane="mamay"),
}
CONDITIONS.update(JUDGE_LANE)

# Регресія поза `lexis`: одне джерело доказів не доводить, що суддя полагоджений загалом.
CONDITIONS["hetero-vg@8"] = BASE.with_(max_steps=8, verify_mode="grounded")
CONDITIONS["hetero-auto@8"] = BASE.with_(max_steps=8, verify_mode="auto")
CONDITIONS.update(LEXIS)
# ABSTAIN-STABILITY: чесна відмова має `pass^k` = 0.000 при t=0.7 — модель уміє її дати, але не
# надійно. Гіпотеза K7g: детермінований крок треба віддати КОДУ. Пара різниться лише цим прапорцем.
# K6: та сама задача одним циклом і графом. Стеля кроків ОДНАКОВА (16), тож граф не має переваги в
# бюджеті — він лише інакше його ділить: 16 на один цикл проти 4 на кожну з чотирьох дітей.
FAN = BASE.with_(max_steps=16, routing="mamay", toolset="lexis", prompt_id="agent/v2-lexis",
                 answer_channel="text", verify_mode="auto")
CONDITIONS["fan-single@16"] = FAN
CONDITIONS["fan-graph@16"] = FAN.with_(graph=True)
# K6-WIDTH: стеля кроків масштабується як 4×ширина, тому кожна дитина має однакові 4 кроки на будь-якій
# ширині, а один цикл дістає ТУ САМУ загальну стелю. Інакше при ширині 12 діти отримали б 1 крок і
# «граф гірший» означало б лише «граф голодний».
for _w in (8, 12, 16):
    # 4d: чи лікується рання термінація циклу підказкою залишку (K7e). Якщо так — межа для
    # суб-агентів зсувається далі, і K6 потрібен ще пізніше, ніж показав K6-WIDTH.
    # Різана рекурсія: ширина вузла 6 при будь-якому N, глибина росте. Це конфіг, який працює для
    # НЕОБМЕЖЕНОГО N, тоді як плоский фан-аут вимагає max_width ≥ N.
    CONDITIONS[f"fan-tree@w{_w}"] = FAN.with_(max_steps=4 * _w, graph=True, max_width=6, max_depth=3)
    CONDITIONS[f"fan-cov@w{_w}"] = FAN.with_(max_steps=4 * _w, coverage=True)
    CONDITIONS[f"fan-single@w{_w}"] = FAN.with_(max_steps=4 * _w)
    CONDITIONS[f"fan-graph@w{_w}"] = FAN.with_(max_steps=4 * _w, graph=True, max_width=_w)
# K10: розділення каналів як ОКРЕМА вісь. Пара різниться лише тим, обгорнуто чужий текст у блок
# даних чи склеєно з завданням, тому різниця — це саме захист, а не інша модель чи промпт.
# Контролем служить НАЯВНА умова `lex-ref-auto@8`: окремий «inj-raw» був би побайтово тією самою
# специфікацією, а тест паритету цього не дозволяє — і правильно, бо два імені на одну умову
# означають два різні числа для одного й того самого прогону.
CONDITIONS["inj-guard@8"] = CONDITIONS["lex-ref-auto@8"].with_(guard=True)
CONDITIONS["inj-strip@8"] = CONDITIONS["lex-ref-auto@8"].with_(guard=True, guard_strip=True)
CONDITIONS["lex-am-jm-contrast"] = CONDITIONS["lex-am-jm"].with_(verify_mode="contrast")
CONDITIONS["lex-ref-abs-t7@8"] = CONDITIONS["lex-ref-t7@8"].with_(absent_answer=True)


PAIRS = (("mamay@8", "mamay+rec@8"), ("hetero@8", "hetero+rec@8"),
         ("hetero@8", "gate-notools-mamay"),
         ("hetero@8", "gate-tools-hetero"),
         ("hetero-plan@8", "hetero-textans@8"),
         ("hetero@8", "hetero-textfull@8"),
         ("hetero-textans@8", "hetero-ua-textans@8"),
         ("hetero@8", "hetero-ua-tools@8"),
         ("lang-mamay", "uanorm@8"),
         ("hetero@8", "ref@8"),
         ("mamay+rec@8", "ref-mamay-rec@8"),
         ("lex-loop", "lex-ref@8"),
         ("lex-plain-lapa", "lex-ref-lapa@8"),
         ("lex-ref@8", "lex-ref-rec@8"),
         ("lex-ref-lapa@8", "lex-ref-lapa-rec@8"),
         ("lex-ref@8", "lex-ref-vg@8"),
         ("lex-ref-lapa@8", "lex-ref-lapa-vg@8"),
         ("lex-loop", "lex-loop-vg"),
         ("lex-loop-t7", "lex-ref-t7@8"),
         ("lex-ref-t7@8", "lex-ref-abs-t7@8"),
         ("lex-ref-auto@8", "inj-guard@8"),
         ("inj-guard@8", "inj-strip@8"),
         ("fan-single@16", "fan-graph@16"),
         ("fan-single@w8", "fan-graph@w8"),
         ("fan-single@w12", "fan-graph@w12"),
         ("lex-am-jm", "lex-am-jl"),
         ("lex-al-jl", "lex-al-jm"),
         ("hetero@8", "hetero-vg@8"),
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
         ("chain-text-guard9@16", "chain-text-guard9rec@16"),
         ("docs-schema@16", "docs-text@16"),
         ("docs-text@16", "docs-agg-text@16"),
         ("docs-agg-schema@16", "docs-agg-text@16"),
         ("docs-agg-schema@16", "docs-years@16"),
         ("chain-schema@16", "chain-cover-schema@16"),
         ("chain-text@16", "chain-cover-text@16"),
         ("docs-schema@16", "docs-cover-schema@16"),
         ("docs-text@16", "docs-cover-text@16"),
         ("chain-cover-schema@32", "chain-cover-rec@32"),
         ("chain-cover-rec@32", "chain-teach@32"),
         ("chain-cover-rec@32", "chain-coverguard@32"),
         ("chain-teach@32", "chain-both@32"),
         ("chain-cover-rec@32", "chain-sum@32"),
         ("chain-sum@32", "chain-sumguard@32"),
         ("chain-cover-rec@32", "chain-reduce@32"),
         ("chain-reduce@32", "chain-reduce-t7@32"),
         ("docs-years@16", "docs-years-t7@16"))


def _model(routing: str, *, lapa, mamay):
    """Прямий виклик потребує ОДНОЇ моделі; `hetero` тут — помилка конфігурації, і вона гучна."""
    return {"mamay": mamay, "lapa": lapa}[routing]


def runner_for(spec: AppSpec, *, lapa, mamay) -> Runner:
    variant = resolve(spec.prompt_id)
    system = variant.render_system()
    answer_instruction = resolve(spec.answer_prompt_id).render_system()

    if spec.mode == "single":
        return single_call_runner(_model(spec.routing, lapa=lapa, mamay=mamay),
                                  system=system, max_tokens=spec.max_tokens, lane=spec.routing)

    def make_orch():
        return build_orchestrator(spec, lapa=lapa, mamay=mamay, system=system,
                                  tail=variant.tail or None, prompt_id=variant.id,
                                  prompt_sha=variant.sha256,
                                  answer_instruction=answer_instruction)

    if spec.graph:
        # Граф іде тим самим шляхом, що й цикл: дитина — звичайний оркестратор, зібраний тим самим
        # коренем. Інакше «суб-агенти» були б окремим застосунком, а не конфігурацією ядра.
        def make_graph():
            return build_graph(spec, lapa=lapa, mamay=mamay, system=system,
                               tail=variant.tail or None, prompt_id=variant.id,
                               prompt_sha=variant.sha256,
                               answer_instruction=answer_instruction)

        return orchestrator_runner(make_graph, budget=build_budget(spec))

    loop = orchestrator_runner(make_orch, budget=build_budget(spec))
    if spec.mode == "gated":
        return gated_runner(_model(spec.gate_direct, lapa=lapa, mamay=mamay), build_toolbox(spec),
                            system=system, max_tokens=spec.max_tokens, loop_runner=loop,
                            lane=spec.gate_direct)
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
        notes = shape_notes(build_skillbox(spec).skill_specs(), coverage=spec.coverage)
        if notes and spec.mode != "single":
            out[name] = notes
    return out


def judge_warnings(names=None) -> dict[str, str]:
    """Самосуд — властивість КОНФІГУРАЦІЇ, тож видно її має бути в звіті, а не в чиїйсь памʼяті.

    Інваріант проєкту забороняє «Lapa судить Lapa»; решта самосуду дозволена, але мусить бути
    підписана, бо модель, що оцінює власний вивід, — це відома системна упередженість.
    """
    from ploshcha_sim.compose import build_router

    class _Stub:
        def __init__(self, name):
            self.model = name

    chosen = list(names) if names else list(CONDITIONS)
    out = {}
    for name in chosen:
        spec = CONDITIONS[name]
        router = build_router(spec, lapa=_Stub("lapa"), mamay=_Stub("mamay"))
        if not spec.verifier or not router.self_judging():
            continue
        lane = router.lane("judge")
        out[name] = ("ПОРУШЕННЯ ІНВАРІАНТА: Lapa судить Lapa" if lane == "lapa"
                     else f"самосуд: {lane} оцінює власний вивід")
    return out
