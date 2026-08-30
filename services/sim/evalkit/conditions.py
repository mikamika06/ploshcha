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
    # тактів пішли в ремонт по три виклики. Найгірший випадок 20 тактів × 3 + партитура + фінал ≈ 63
    # — це ОЦІНКА, порахована на папері, а не замір.
    #
    # ★ І замір 2026-08-27 каже, що обидва числа читати треба обережно: мирне віче про греблю (20
    # реплік) зробило 46 викликів шлюзу, а `budget.steps_used` показав 92 — рівно вдвічі, бо крок
    # лічиться двічі (`Budget.spend` додає одиницю, і `Viche._call` додає ще одну). Тобто 80 тут
    # означає 40 викликів, і жива розмова їх перебирає: `can_continue()` стає False ще до кінця
    # прогону, і єдине, на що це впливає, — замовлення нових хвиль наперед; репліки, хроніка й
    # ухвала доїжджають. Число лишається тим, що заміряне, а не підганяється під подвоєну лічбу:
    # правити тут треба лічбу кроків, а не стелю.
    # ВІЧЕ — прод-режим ПЛОЩІ: не задача з відповіддю, а розмова. Тому `verifier=False`
    # (сумнів попа тут репліка, не вирок) і toolset="lexis" (словник як «дяк погортав книгу»),
    # а НЕ "reference": у тому шість статей, тож він давав «нема в довіднику» на будь-що.
    # `viche_guard=True` — бо ці дві умови і є прод: `deploy.sh` запускає `serve_ploshcha.py
    # --condition viche`, а Дошка — єдине місце, куди пише жива людина з вулиці. Поки поле стояло
    # в дефолтному `False`, уся охорона інʼєкцій була написана, підключена, покрита тестами й
    # МЕРТВА: `Viche.guard is None`, гілка `ORDER_ANSWER` недосяжна. Ціна названа й прийнята:
    # поле входить у `sha256` умови, тож звіти по ній до й після непорівнянні.
    # `viche_sense=True` — з тієї самої причини, що й охорона: ці дві умови і є прод. Заміряно
    # живими викликами, що закритий список тут не рятує в принципі: «Сусідська корова побила мені
    # весь город» діставала телефон 1547 при нулі викликів моделі, а `about_accusation('Марія
    # злодіїв не бачила', _SPEAKERS)` вертав True, тобто село відмовлялось гомоніти про власну
    # крадіжку. Ціна названа й заміряна 2026-08-27 у цій самій умові: 689 токенів на виклик у
    # середньому (670-731 по корпусу з 74 живих тем), стеля 18 викликів на прогін, а заміряне живе
    # віче з шістьма словами гостя платить 8 викликів і 25.9% прогону (5506 із 21 222), а мирне
    # без гостя — 2 виклики й 6.9% (1431 із 20 851). Поле
    # входить у `sha256` умови — звіти по ній до й після непорівнянні.
    # ★ `viche_repetition_penalty` тут НЕ виставлений, і це рішення за заміром, а не за
    # замовчуванням. Важіль живий: старий запис «шлюз ковтає штрафи» виявився артефактом кешу
    # (`extra_body` не входить у ключ кешу шлюзу, тож девʼять плечей на одному пакеті вертали ту
    # саму суму `42fb002637e9ca2e`), а на розбитому кеші штраф міняє вивід. Не вмикаємо з двох
    # причин: на свіжих промптах значення 1.0-1.3 не міняють нічого взагалі (кусати починає з
    # 1.5), а на справжніх пакетах віча плата виходить один-до-одного — дослівного повернення
    # менше рівно настільки, наскільки гіршає сама репліка. Число сюди дописують ПІСЛЯ заміру,
    # який покаже виграш, а не обмін; поле входить у `sha256`, тож звіти до й після
    # непорівнянні. Скільки протоку є зараз — показує `протік` у звіті (`evalkit/dialogue.py`).
    # ★ `viche_chain_decay` тут теж НЕ виставлений, і теж за заміром. Згасання ланцюга відповідей
    # — найдешевший механізм із огляду поля (нуль тексту в пакет) і найкраще published-число:
    # 20 природних завершень із 20 проти 0 із 20. Гасити в нас є що — на восьми збережених
    # прогонах зі шпигуном ланцюгів завглибшки 3+ рівно 21.1% (16 із 76). Не вмикаємо з двох
    # причин, обидві заміряні офлайн: зчеплення пари за глибиною такту-відповіді дає 20.0% на
    # корені, 30.8% / 64.7% на першій і другій ланці й 28.6% на третій і глибше — тобто згасання
    # обміняло б найслабші ланки (28.6%) на почини наступної хвилі (20.0%), а не додало звʼязків;
    # і природного завершення воно не дає, бо цикл віча добирає такти до `mode.beats[1]`, а на 57
    # збережених прогонах віча розмову обриває стеля кроків (26 із 43), а не довгий ланцюг.
    # Глибину сюди дописують ПІСЛЯ заміру, який покаже виграш, а не обмін; поле входить у `sha256`.
    # ★ `viche_adjacency` тут теж НЕ виставлений, і теж за заміром — найповнішим з усіх трьох.
    # Суміжна пара (тип попередньої репліки обмежує набір ходів у відповідь) — єдиний механізм
    # огляду поля з чужою ЛЮДСЬКОЮ оцінкою (4.61 → 7.45) і найдешевший з наших: нуль викликів
    # моделі, нуль тексту в промпт. Заміряно 2026-08-30 живим шлюзом у цій самій умові, сіди 1 і
    # 2, теми «вовк» і «мито», по чотири віча в плечі: зчеплення 20 із 62 (32.3%) без важеля
    # проти 18 із 60 (30.0%) з ним, підхоплення 1 із 62 проти 1 із 60 — тобто ворота, поставлені
    # саме на підхопленні (`docs/research/dialogue-mechanics-ours.md`), стоять на місці. Ціна
    # мала (18 242 → 18 541 токена на віче, ремонтів 43 → 42), і дослівне повернення НЕ зросло
    # (протіків ≥ 4 слова нуль в обох плечах). Спрацювало правило 12 разів на ~76 тактів, тобто
    # зрушити 60 пар воно могло щонайбільше на 20 в. п. — заміру бракує сили, а не механізму.
    # Прапорець сюди виставляють ПІСЛЯ заміру, який покаже виграш; поле входить у `sha256`.
    # ★ `viche_bare_packet="mark"` тут ВИСТАВЛЕНИЙ — і за тим самим заміром, який доти тримав його
    # вимкненим. Голий пакет (у тілі пакета не лишається вільного тексту взагалі, адресат
    # позначений числом — місцем у пронумерованому переліку) заміряно 2026-08-30 живим шлюзом у
    # цій самій умові, сід 1, дві заморожені теми, кеш розбито однаковим `LINE_TOKENS` + 1 в усіх
    # трьох плечах (`docs/research/packet-strip.md`): пар без ознаки звʼязку 24 із 30 (80.0%) як
    # було, 24 із 34 (70.6%) у `bare`, 21 із 35 (60.0%) у `mark`; спроб мовця на такт 2.18 → 1.60
    # → 1.18, ремонтів 23 → 16 → 5, ескалацій 16 → 8 → 2, кроків 96 і 100 → 94 і 88 → 74 і 74 при
    # стелі 80, токенів на два віча 36 679 → 34 612 → 28 530, реплік 33 → 39 → 39, обривів нуль
    # скрізь, а частка реплік про тему НЕ впала, а зросла (42.4% → 61.5% / 48.7%).
    #
    # ЧОМУ ТЕПЕР, КОЛИ УМОВА ВМИКАННЯ БУЛА «ПІСЛЯ ЗАМІРУ НА ПІДХОПЛЕННІ». Замір зроблено, і він
    # зняв саму умову, а не виконав її (`docs/research/dialogue-tier-vs-content.md`, 82 074
    # токени). Підхоплення 1 пара в кожному плечі — це стеля МЕТРИКИ, не важеля: вона є функцією
    # довжини репліки (та сама пʼєса дає 6.7% без стелі, 4.1% при рядках ≤ 20 слів, 2.8% при ≤ 16,
    # 1.0% при ≤ 12), а на людському українському діалозі — «Бондарівна» і «Мина Мазайло», 663
    # репліки — при НАШІЙ довжині рядка вона дає 3.7% і 2.8%, тобто наші 2.9-3.3% і є людська
    # норма. І ярус її не рухає: той самий пакет, переграний на дорогій моделі, дав 1 пару з 65
    # проти 1 з 65 на дешевій при +60% ціни за виклик. Ворота, які чекали числа, чекали числа,
    # якого в розмові нашої довжини не буває.
    #
    # ЩО ЦИМ НЕ КУПЛЕНО, сказано так само прямо: точний двобічний тест на цих обсягах плече від
    # контролю не відрізняє (p = 0.11 для `mark`), і виграш головної метрики тримається на
    # зачинах-реакціях, до яких прилад свідомо сліпий. Тверде тут те, що лічиться викликами, а не
    # пропорцією пар: ремонт — це окремий виклик, ескалація — виклик на дорогому ярусі, і 23 → 5
    # та 16 → 2 тесту не потребують. Ціна названа: поле входить у `sha256` умови, тож звіти по цих
    # двох умовах до й після непорівнянні.
    #
    # В ОБОХ УМОВАХ, а не лише в `viche`, — з тієї самої причини, що охорона й суддя: `deploy.sh`
    # піднімає `viche`, але пара `viche` / `viche-notools` мусить різнитись ЛИШЕ набором
    # інструментів, інакше замір інструментів міряв би дві осі одразу.
    # ★ `viche_reply_lane` тут НЕ виставлений, і це замір, а не обережність — найповніший із усіх
    # у цьому ряду: не переграні такти, а ВІСІМ ЖИВИХ ВІЧ у цій самій умові (сіди 1 і 2, теми
    # «вовк» і «мито», кеш розбито `LINE_TOKENS` 423 і `VOTE_TOKENS` 161 однаково в обох плечах —
    # швидших за 250 мс реплік 1 із 95 і 1 із 80 при підписі кешу 85 мс; сирі дані
    # `docs/research/eval-runs/tier-live-{lapa,mamay}-{1,2}.json`, лічильники `tier-live-count.py`).
    #
    # Дорогий ярус промовляння виграє все, що лічиться ВИКЛИКАМИ, і програє все інше. Виграє:
    # спроб мовця на такт 1.19 → 1.00, ремонтів 15 → 0, ескалацій 6 → 0, дефектних причин голосу
    # 9 із 24 → 0 із 24, огризків ≤ 6 слів 17 → 0, протіків ≥ 4 слова 1 → 0, кроків 308 → 272
    # (обриви нуль в обох, голосування дійшло 4 віча з 4 в обох). Програє: 60 147 → 73 628 токенів
    # (+22.4%, і жодне дороге віче не дешевше за жодне дешеве), 223.2 → 342.9 с, а ГОЛОВНА метрика
    # погіршала — пар без жодної ознаки звʼязку 53.5% → 61.3%, бо зачинів-реакцій 28 → 21.
    #
    # Точний двобічний тест ділить ці два списки навпіл, і саме так їх і треба читати: усе про
    # РОЗМОВУ лежить у шумі (зчеплення p = 0.40, підхоплення p = 0.45, зачини p = 0.16, підхоплення
    # на зрізі до 12 слів p = 0.27), а все, що лічиться викликами, — ні (ремонт на такті 15 із 80
    # проти 0 із 80, p = 0.00003; дефектні причини голосу 9 із 24 проти 0 із 24, p = 0.0016).
    # Різниця в ціні тесту не потребує взагалі: дорожчий ярус — це та сама лічба викликів.
    #
    # ПРО ПІДХОПЛЕННЯ, ЧЕСНО, бо саме на ньому стоять ворота круга. Як є, воно зросло: 7 пар із 71
    # (9.9%) → 11 із 75 (14.7%). Але метрика є функцією довжини репліки (розділ 4 в
    # `dialogue-tier-vs-content.md`), а дорогий ярус говорить удвічі довше — 23.1 слова проти
    # 12.7. З вирівняною довжиною виграшу немає: пар, де ОБИДВІ репліки ≤ 16 слів, у дорогого
    # яруса 1 (проти 43), і підхоплень у них 0 в обох; на зрізі обох реплік до перших 12 слів
    # 7.0% → 2.7%, до 16 слів 8.5% → 6.7%. Тобто на однаковій довжині рядка дорогий ярус
    # підхоплює РІДШЕ, а не частіше.
    #
    # І це спростовує оцінку минулого круга, зроблену на переграних тактах: там дорогий ярус
    # виходив дешевшим (16 813 проти 17 539 на плечі `mark`), бо всі репліки проходять із першої
    # спроби. Перше справдилось, друге — ні: економія ремонту важить ≈ 7 700 токенів (15 зайвих
    # `speak` по 273 + 6 ескалацій по 607), а сам ярус бере 408 токенів за виклик замість 273 і
    # тягне сусідні слоти — суддя 5 969 → 9 821, партитура 15 666 → 16 553. Оцінка стелі економії
    # виявилась саме стелею, і повне віче її перекрило.
    #
    # Рядок сюди виставляють ПІСЛЯ заміру, який покаже виграш на ВИРІВНЯНІЙ довжині; поле входить
    # у `sha256`, тож звіти до й після непорівнянні.
    "viche": BASE.with_(mode="viche", max_width=6, max_steps=80, max_tokens=220,
                        toolset="lexis", verifier=False, temperature=0.8,
                        prompt_id="viche/v1", viche_guard=True, viche_sense=True,
                        viche_bare_packet="mark"),
    "viche-notools": BASE.with_(mode="viche", max_width=6, max_steps=80, max_tokens=220,
                                toolset="none", verifier=False, temperature=0.8,
                                prompt_id="viche/v1", viche_guard=True, viche_sense=True,
                                viche_bare_packet="mark"),
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
