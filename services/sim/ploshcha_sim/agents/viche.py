"""Віче: Mamay пише партитуру одним викликом, Lapa грає такти.

Чому окремий агент, а не гілка в `AgentGraph`. Граф — вимірювальний прилад, на ньому тримаються всі
попередні числа; додати в нього другий режим означало б зробити старі виміри непорівнянними. Тут
інша петля (партитура → репліки → втручання), інший критерій успіху (розмова відбулась) і немає
`abstain` як стану.

Розкладка яруса не декоративна, а сама архітектура:

    Mamay  — партитура, втручання на несподіванці, слово старости, сумнів попа   (це судження)
    Lapa   — кожна репліка селянина                                (це трансформація пакета в мову)
    код    — склад учасників, перебивки, чи йти до криниці            (визначено даними, не моделлю)

Вартість = `1 × Mamay + N × Lapa + рідкі втручання`: дорогий слот кличеться одиниці разів, а не N.
"""

import json
import random
import threading

from ..domain.task import Budget, TaskResult
from ..domain.viche import (
    BY_ROLE,
    GUEST,
    guest_beats,
    chronicle_schema,
    mood_view,
    plan_steps,
    DOUBT_MOVE,
    INTERRUPT_MOVE,
    MOVE_HINT,
    PIP,
    STAROSTA,
    SUMMARY_MOVE,
    Beat,
    Persona,
    cast_for,
    line_schema,
    repair_score,
    scatter,
    score_schema,
)
from ..ports.agent import AgentPort
from ..ports.tool import ToolCall
from ..ports.trace import StepRecord, TracePort

VICHE_NOTE = "viche"
# Ходи для ремонту повтору: інший хід — інший зміст. Обертаємо за індексом такту, тож
# відтворювано за сідом, як і решта збурень.
_CONTRAST = ("заперечити", "спитати_діло", "пожартувати", "пожалітись", "порахувати")
MAX_LINE_CHARS = 320
MIN_LINE_CHARS = 8
# Партитурі потрібен СВІЙ бюджет виводу. Спільна стеля `max_tokens` розрахована на репліку (~220),
# а дванадцять тактів це ~800 токенів: JSON обривався на півслові, парс падав, і мовчки вмикався
# запасний план «кожен реагує по разу». Тобто партитуру Mamay викидало щопрогону, а видно було лише
# «щось розмова коротка».
SCORE_TOKENS = 2200
CHRONICLE_TOKENS = 900

SCORE_SYSTEM = """Ти — Мамай, розпорядник сільського віча. Тобі дають новину й склад людей.
Ти НЕ пишеш їхніх слів. Ти розписуєш ПОРЯДОК: хто говорить, за ким, яким ходом і кому відповідає.
Розмова мусить бути ДОВГА і розвиватись: спершу реакції, потім суперечка з переходом на конкретику,
потім хтось згадує схожий випадок, і аж наприкінці — до чого дійшли. Не менше дванадцяти тактів.
Якщо для розмови бракує факту — постав такт із ходом «піти_питати» й запитом до довідника."""

LINE_SYSTEM = """Ти — селянин українського села. Кажеш ОДНУ репліку вголос, від першої особи.
Живою розмовною українською. Одне-два речення.
ЗАБОРОНЕНО: переказувати новину, повторювати чужі слова, називати себе на імʼя, писати свою роль
чи завдання. Тільки пряма мова — те, що людина справді сказала б уголос."""

SUMMARY_SYSTEM = """Ти — староста. Одним-двома реченнями зведи, до чого дійшло віче.
Не переказуй усіх по черзі: назви, у чому зійшлись і в чому ні."""

CHRONICLE_SYSTEM = """Ти — сільський Оповідач-літописець. Тобі дають розмову, що відбулась на вічі.
Напиши коротку хроніку: заголовок, одну-дві фрази оповіді, настрій села — і по ОДНІЙ думці кожного,
хто говорив: не переказ його слів, а те, з чим він лишився наодинці. Думка — від першої особи.
Окремо назви УХВАЛУ: чи село на чомусь зійшлось, що саме постановили зробити, кому це доручили і в
якому місці. Якщо згоди не було — так і скажи «ні», і не вигадуй рішення."""

DOUBT_SYSTEM = """Ти — сільський піп. Якщо в розмові прозвучало твердження без підстави — коротко
засумнівайся саме в ньому. Якщо підстав сумніватись немає, скажи це прямо одним реченням."""


class Viche(AgentPort):
    def __init__(self, router, effort, tools=None, *, trace: TracePort | None = None,
                 run_id: str = "viche", width: int = 5, system: str | None = None,
                 prompt_id: str = "viche/v1", prompt_sha: str = "",
                 score_system: str = SCORE_SYSTEM, line_system: str = LINE_SYSTEM,
                 summary_system: str = SUMMARY_SYSTEM, doubt_system: str = DOUBT_SYSTEM,
                 chronicle_system: str = CHRONICLE_SYSTEM):
        self.router = router
        self.effort = effort
        self.tools = tools
        self.trace = trace
        self.run_id = run_id
        self.width = width
        self.system = system
        self.prompt_id = prompt_id
        self.prompt_sha = prompt_sha
        # Промпти приходять ЗЗОВНІ (реєстр `evalkit/prompts`), константи лише дефолт: інакше
        # текст, від якого залежить якість голосів, не мав би ні версії, ні хеша в трасі.
        self.score_system = score_system
        self.line_system = line_system
        self.summary_system = summary_system
        self.doubt_system = doubt_system
        self.chronicle_system = chronicle_system
        # Скринька вхідних: сюди падає слово гостя й шепіт, поки віче ТРИВАЄ. Читається між
        # тактами, а не в середині — інакше репліка вклинювалась би в напівзгенерований такт.
        self._inbox: list[dict] = []
        self._lock = threading.Lock()
        self._whispers: dict[str, str] = {}

    def tell(self, message: dict) -> None:
        """Кинути слово в живе віче. Потокобезпечно: сервер і цикл — різні потоки."""
        with self._lock:
            self._inbox.append(dict(message))

    def _drain(self) -> list[dict]:
        with self._lock:
            out, self._inbox = self._inbox, []
        return out

    # ── публічний вхід ────────────────────────────────────────────────────────
    def run(self, task: str, seed: int = 0, budget: Budget | None = None,
            depth: int = 1) -> TaskResult:
        budget = budget or Budget()
        cast = cast_for(task, self.width)
        roles = [p.role for p in cast]
        said: list[tuple[Persona, str]] = []
        incidents: list[str] = []
        beats = scatter(self._plan(task, cast, seed, budget, incidents), roles, seed, task)
        self._emit_plan(beats)
        pending: list[Beat] = list(beats)
        index = 0
        while pending:
            index += 1
            if not budget.can_continue():
                incidents.append("viche_budget")
                break
            # Слово гостя вклинюється МІЖ тактами й тягне за собою відгуки, а не переписує план.
            extra = self._take_word(said, roles, seed, index)
            if extra:
                pending = extra + pending
                incidents.append("viche_guest")
            beat = pending.pop(0)
            said += self._play(task, beat, index, cast, said, seed, budget, incidents)

        if said:
            said.append(self._summary(task, said, seed, budget))
            doubt = self._doubt(task, said, seed, budget)
            if doubt is not None:
                said.append(doubt)
            self._chronicle(task, said, seed, budget, incidents)

        return self._result(task, said, beats, budget, incidents)

    def _take_word(self, said: list[tuple[Persona, str]], roles: list[str], seed: int,
                   index: int) -> list[Beat]:
        """Розбирає скриньку: слово гостя стає реплікою + відгуками, шепіт — пакетом одному."""
        out: list[Beat] = []
        for msg in self._drain():
            text = " ".join(str(msg.get("text") or "").split())[:MAX_LINE_CHARS]
            if not text:
                continue
            if msg.get("kind") == "whisper":
                who = str(msg.get("to") or "")
                if who in BY_ROLE:
                    self._whispers[who] = text
                continue
            said.append((GUEST, text))
            self._emit_line(self._span(GUEST, index), text, index)
            recent = [p.role for p, _ in said[-3:]]
            out += guest_beats(len(said), roles, recent, seed, text)
        return out

    # ── партитура ─────────────────────────────────────────────────────────────
    def _plan(self, task: str, cast: list[Persona], seed: int, budget: Budget,
              incidents: list[str]) -> list[Beat]:
        roles = [p.role for p in cast]
        tools = [s.name for s in self.tools.specs()] if self.tools is not None else []
        tools = [t for t in tools if t != "final_answer"]
        people = "\n".join(f"- {p.role} ({p.name}): {p.lens}" for p in cast)
        prompt = (f"НОВИНА ДЛЯ ВІЧА:\n{task}\n\nЛЮДИ:\n{people}\n\n"
                  f"ДОСТУПНІ ХОДИ: {', '.join(MOVE_HINT)}\n"
                  f"ДОВІДНИК: {', '.join(tools) if tools else 'немає'}\n\n"
                  "Розпиши такти віча. `у_відповідь` — номер попереднього такту або null.")
        schema = score_schema(roles, tools)
        beats = repair_score(_safe_json(
            self._call("decide", prompt, self.score_system, schema, seed, budget,
                       max_tokens=SCORE_TOKENS)), roles, tools)
        if not beats:
            # Збій структурованого виводу тут ПЕРЕРИВЧАСТИЙ, а не детермінований: та сама схема й
            # той самий промпт то проходять, то дають нерозбірний JSON. Один перепит із іншим сідом
            # рятує прогін; без нього єдиний невдалий виклик знецінював усю розмову.
            incidents.append("viche_score_retry")
            beats = repair_score(_safe_json(
                self._call("decide", prompt, self.score_system, schema, seed + 101, budget,
                           max_tokens=SCORE_TOKENS)), roles, tools)
        if not beats:
            # Запасний план мусить бути ГУЧНИЙ. Тихо він виглядав як «розмова коротка», а насправді
            # означав, що плану не було взагалі.
            incidents.append("viche_score_lost")
            beats = [Beat(хто=p.role, хід="згадати" if i else "спитати_діло")
                     for i, p in enumerate(cast)]
        return beats

    # ── такт ──────────────────────────────────────────────────────────────────
    def _play(self, task: str, beat: Beat, index: int, cast: list[Persona],
              said: list[tuple[Persona, str]], seed: int, budget: Budget,
              incidents: list[str]) -> list[tuple[Persona, str]]:
        who = BY_ROLE.get(beat.хто)
        if who is None:
            return []
        out: list[tuple[Persona, str]] = []
        fact: str | None = None

        if beat.інструмент and self.tools is not None:
            # Окремого рядка «йду спитаю» НЕМА. Він виходив пласким переказом підказки («Йду
            # дізнаюсь про «вовк»») та ще й тягнув сміттєвий запит від партитури («про «жена»»).
            # Те, що людина пішла, уже видно рухом у локацію (`agent.moved` перед `tool.result`),
            # тож рядок нічого не додавав, а один виклик коштував.
            fact = self._ask(beat, index, seed, budget, self._span(who, index))

        move = beat.хід if not beat.інструмент else "згадати"
        line = self._line(task, who, Beat(хто=beat.хто, хід=move, у_відповідь=beat.у_відповідь),
                          index, said + out, seed, budget, incidents, fact=fact)
        if line:
            out.append((who, line))
        return out

    def _ask(self, beat: Beat, index: int, seed: int, budget: Budget, span: str) -> str:
        """Інструмент = дія в локації. «Не знайшов» — теж факт для репліки, а не провал прогону."""
        call = ToolCall(tool=str(beat.інструмент), args=self._args(str(beat.інструмент), beat.запит))
        # Спершу ВИКЛИК, тоді результат. Доти віче емітило лише `tool.result`, тобто пара
        # `tool.called`/`tool.result` була зламана: результат є, виклику нема — і жоден спостерігач
        # не міг порахувати, скільки разів по довідник ходили.
        self._emit_call(call, index, seed, span)
        result = self.tools.call(call)
        self._emit_tool(call, result, index, seed, span)
        if not result.ok:
            return "інструмент не відповів"
        value = result.value if isinstance(result.value, dict) else {"результат": result.value}
        if result.found is False:
            return "у довіднику того немає"
        return json.dumps(value, ensure_ascii=False)[:400]

    def _args(self, tool: str, query: str | None) -> dict:
        spec = next((s for s in self.tools.specs() if s.name == tool), None)
        names = list((spec.params.get("properties") if spec else {}) or {})
        return {names[0]: query or ""} if names else {}

    def _line(self, task: str, who: Persona, beat: Beat, index: int,
              said: list[tuple[Persona, str]], seed: int, budget: Budget,
              incidents: list[str], *, fact: str | None) -> str:
        prompt = self._packet(task, who, beat, said, fact)
        system = self._persona_system(who)
        span = self._span(who, index)
        raw = self._call("speak", prompt, system, line_schema(), seed + index, budget,
                         span=span, voice=False)
        line = _text(raw)
        echoed = _echoes(line, task, prompt, who.lens)
        twin = _twin_of(line, said)
        stale = twin is not None
        if _drifted(line) or echoed or stale:
            # Втручання оркестратора — лише на несподіванці, не щотакту: інакше зникає вся економія.
            # Повтор і переказ — саме та несподіванка: обвал ентропії Lapa виміряний, тож
            # покладатись на «вона більше так не зробить» не можна.
            incidents.append(f"viche_{'echo' if echoed else 'same' if stale else 'drift'}:{who.role}")
            # Ремонт мусить бути КОНКРЕТНИЙ. Загальне «скажи інше» відкидалось удруге в половині
            # випадків, і такт гинув після трьох викликів. Тому: називаємо саме повторену фразу і
            # МІНЯЄМО ХІД — інший хід тягне інший зміст, а це вибір, визначений даними, отже код.
            swap = _CONTRAST[index % len(_CONTRAST)]
            nudge = self._packet(task, who, Beat(хто=beat.хто, хід=swap,
                                                 у_відповідь=beat.у_відповідь), said, fact)
            nudge += ("\n\nТИ ЩОЙНО ПОВТОРИВ ЧУЖЕ: «" + twin[:160] + "»\nСкажи про це ЗОВСІМ ІНШЕ."
                      if twin else "\n\nСКАЖИ ІНАКШЕ, СВОЇМИ СЛОВАМИ, не переказуючи новини.")
            # Драбина, а не стрибок на дорогий ярус. Перший живий замір: ремонт одразу через Mamay
            # дав 3532 його токени проти 1719 у Lapa — тобто дефект виконавця оплачувався
            # оркестратором, і вся економія задуму зникала.
            raw = self._call("speak", nudge, system, line_schema(), seed + index * 7 + 1,
                             budget, span=span, voice=False)
            line = _text(raw)
            if _drifted(line) or _too_similar(line, [t for _, t in said]):
                incidents.append(f"viche_escalate:{who.role}")
                raw = self._call("synthesize", nudge, system, line_schema(), seed + index,
                                 budget, span=span, voice=False)
                line = _text(raw)
            if _drifted(line) or _too_similar(line, [t for _, t in said]):
                return ""
        line = line[:MAX_LINE_CHARS]
        self._emit_line(span, line, index)
        return line

    def _persona_system(self, who: Persona) -> str:
        """Хто ти — у СИСТЕМІ, а не в тексті запиту.

        Перший живий прогін показав рівно це: Lapa дослівно переказала пакет разом із лінзою й
        підказкою ходу («дід Свирид: … Дід Свирид — памʼять: … Спитай, що робити практично»).
        Системне повідомлення вона так не копіює.
        """
        return f"{self.line_system}\n\nТИ: {who.name}. Дивишся на світ так: {who.lens}."

    def _packet(self, task: str, who: Persona, beat: Beat,
                said: list[tuple[Persona, str]], fact: str | None) -> str:
        parts = [f"НОВИНА: {task}"]
        target = None
        if beat.у_відповідь and 1 <= beat.у_відповідь <= len(said):
            target = said[beat.у_відповідь - 1]
        elif beat.хід == INTERRUPT_MOVE and said:
            target = said[-1]
        # Сам собі не відповідають: у живому прогоні Одарка звернулась «Та не все так просто,
        # Одарко». Партитура може вказати на власний такт, тож відсікає це код.
        if target is not None and target[0].role == who.role:
            target = None
        if target is not None:
            parts.append(f"ЩОЙНО СКАЗАВ {target[0].name}: «{target[1]}»")
        # ★ Профілактику «ось що вже казали, не повторюй» ПЕРЕВІРЕНО І ВІДКИНУТО: ремонтів лишилось
        # стільки ж (6 із 13), токенів стало більше (10262 → 11539), різність упала (0.967 → 0.864).
        # Список працює як праймінг — модель починає копіювати саме з нього. Тому в пакеті лише
        # останні репліки як КОНТЕКСТ відповіді, без заборонного списку.
        elif said:
            parts.append("ПЕРЕД ТИМ КАЗАЛИ: " + " | ".join(f"{p.name}: {t}" for p, t in said[-2:]))
        if fact is not None:
            parts.append(f"ЩО ТИ ДІЗНАВСЯ: {fact}")
        secret = self._whispers.pop(who.role, None)
        if secret:
            # Шепіт іде в розмову ЯК СВОЄ або з посиланням на гостя — вирішує кубик, не модель.
            told = random.Random(f"{who.role}:{secret[:30]}").random() < 0.5
            parts.append(f"ТОБІ ПОШЕПТАЛИ НА ВУХО: «{secret}»\n"
                         + ("Скажи це вголос, пославшись на приїжджого."
                            if told else "Скажи це вголос ЯК СВОЮ думку, не згадуючи, хто сказав."))
        if beat.хід == INTERRUPT_MOVE:
            parts.append("ТВІЙ ХІД: перебий і встав своє, коротко")
        elif beat.хід == "піти_питати":
            parts.append(f"ТВІЙ ХІД: скажи, що йдеш дізнатись про «{beat.запит or task[:40]}»")
        else:
            parts.append(f"ТВІЙ ХІД: {MOVE_HINT.get(beat.хід, beat.хід)}")
        return "\n".join(parts)

    # ── фінал ─────────────────────────────────────────────────────────────────
    def _summary(self, task: str, said: list[tuple[Persona, str]], seed: int,
                 budget: Budget) -> tuple[Persona, str]:
        prompt = (f"НОВИНА: {task}\n\nЩО КАЗАЛИ:\n"
                  + "\n".join(f"- {p.name}: {t}" for p, t in said))
        raw = self._call("synthesize", prompt, self.summary_system, line_schema(), seed, budget,
                         span=self._span(STAROSTA, 0))
        return STAROSTA, _text(raw)[:MAX_LINE_CHARS]

    def _doubt(self, task: str, said: list[tuple[Persona, str]], seed: int,
               budget: Budget) -> tuple[Persona, str] | None:
        """Верифікатор став реплікою, а не вироком: сумнів чути, але він не вбиває прогін."""
        prompt = (f"НОВИНА: {task}\n\nЩО КАЗАЛИ:\n"
                  + "\n".join(f"- {p.name}: {t}" for p, t in said))
        raw = self._call("judge", prompt, self.doubt_system, line_schema(), seed, budget,
                         span=self._span(PIP, 0))
        line = _text(raw)[:MAX_LINE_CHARS]
        return None if _drifted(line) else (PIP, line)

    def _emit_plan(self, beats: list[Beat]) -> None:
        """Партитура — предмет, а не лог: порядок віча висить на Дошці."""
        if self.trace is None or not beats:
            return
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=0, agent="planner", stage="plan", model="viche",
            lane=self.router.lane("decide"), prompt="", raw_output="",
            parsed={"agentId": STAROSTA.role,
                    "summary": f"порядок віча: {len(beats)} тактів",
                    "steps": plan_steps(beats)},
            schema_valid=True, world_valid=True))

    def _chronicle(self, task: str, said: list[tuple[Persona, str]], seed: int,
                   budget: Budget, incidents: list[str]) -> None:
        """Оповідач: хроніка дня + думка кожного — ОДНИМ викликом.

        Окремий виклик на думку коштував би як уся розмова. А вигадати думку кодом не можна: це
        єдине місце, де справді потрібне судження про сказане.
        """
        # Стеля кроків обмежує РОЗМОВУ, не її закриття. Літописець коштує рівно один виклик, як і
        # підсумок із сумнівом, — а вони цієї перевірки не мали. Через неї хроніка не запускалась
        # саме тоді, коли розмова вийшла довгою, тобто коли про неї було що написати.
        if self.trace is None:
            return
        roles = sorted({p.role for p, _ in said})
        prompt = (f"НОВИНА: {task}\n\nРОЗМОВА:\n"
                  + "\n".join(f"- {p.name} ({p.role}): {t}" for p, t in said))
        schema = chronicle_schema(roles)
        data = _safe_json(self._call("synthesize", prompt, self.chronicle_system, schema, seed,
                                     budget, max_tokens=CHRONICLE_TOKENS))
        if not data:
            # Той самий переривчастий збій, що й у партитури — один перепит іншим сідом.
            incidents.append("viche_chronicle_retry")
            data = _safe_json(self._call("synthesize", prompt, self.chronicle_system, schema,
                                         seed + 101, budget, max_tokens=CHRONICLE_TOKENS))
        if not data:
            incidents.append("viche_chronicle_lost")
            return
        for item in data.get("думки") or []:
            thought = str(item.get("думка") or "").strip()
            if item.get("хто") in roles and thought:
                self.trace.emit(StepRecord(
                    run_id=self.run_id, tick=0, agent="thinker", stage="reflect", model="viche",
                    lane=self.router.lane("synthesize"), prompt="", raw_output="",
                    parsed={"agentId": item["хто"], "thought": thought[:MAX_LINE_CHARS]},
                    schema_valid=True, world_valid=True))
        self._emit_decision(data.get("ухвала"), roles)
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=0, agent="chronicler", stage="report", model="viche",
            lane=self.router.lane("synthesize"), prompt="", raw_output="",
            parsed={"day": 1, "title": str(data.get("заголовок") or task)[:120],
                    "narration": str(data.get("оповідь") or "")[:600],
                    "mood": mood_view(str(data.get("настрій") or "спокій"),
                                      data.get("сила") or "помірно"),
                    "highlights": [t for _, t in said[:3]]},
            schema_valid=True, world_valid=True))

    def _emit_decision(self, raw, roles: list[str]) -> None:
        """Ухвала — рішення зі СЛІДОМ у світі, а не ще один рядок тексту.

        Тому вона й доручається комусь у конкретному місці: без цього «село вирішило» лишалось би
        фразою, а не станом, який видно на сцені й успадковує наступне віче.
        """
        if self.trace is None or not isinstance(raw, dict):
            return
        if str(raw.get("ухвалено")) != "так":
            return
        what = " ".join(str(raw.get("що") or "").split())
        who, where = str(raw.get("хто") or ""), str(raw.get("де") or "")
        if not what or who not in roles or not where:
            return
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=0, agent="council", stage="decide", model="viche",
            lane=self.router.lane("synthesize"), prompt="", raw_output="",
            parsed={"label": what[:90], "who": who, "poi": where},
            schema_valid=True, world_valid=True))

    def _result(self, task: str, said: list[tuple[Persona, str]], beats: list[Beat],
                budget: Budget, incidents: list[str]) -> TaskResult:
        """★ Немає `abstain`: у розмові відсутність даних — це репліка, а не термінальний стан."""
        answer = "\n".join(f"{p.name}: {t}" for p, t in said) if said else None
        voices = len({p.role for p, _ in said})
        return TaskResult(
            answer=answer,
            accepted=bool(said),
            outcome="answer" if said else "failure",
            evidence=None,
            degraded=not said,
            steps=budget.steps_used,
            tokens=budget.tokens_used,
            aux_tokens=budget.aux_tokens,
            tokens_by_lane=dict(sorted(budget.tokens_by_lane.items())),
            prompt_by_lane=dict(sorted(budget.prompt_by_lane.items())),
            tokens_by_stage=dict(sorted(budget.tokens_by_stage.items())),
            prompt_by_stage=dict(sorted(budget.prompt_by_stage.items())),
            tokens_by_stage_lane=dict(sorted(budget.tokens_by_stage_lane.items())),
            prompt_by_stage_lane=dict(sorted(budget.prompt_by_stage_lane.items())),
            incidents=incidents,
            notes=[VICHE_NOTE, f"beats={len(beats)}", f"lines={len(said)}", f"voices={voices}"],
            scratch=[],
        )

    # ── проводка ──────────────────────────────────────────────────────────────
    def _span(self, who: Persona, index: int) -> str:
        return f"{self.run_id}/viche/{who.role}/{index}"

    def _call(self, kind: str, prompt: str, system: str, schema: dict, seed: int,
              budget: Budget, *, span: str | None = None, voice: bool = True,
              max_tokens: int | None = None) -> str:
        llm = self.router.route(kind)
        cfg = self.effort.effort(kind)
        res = llm.generate_structured(prompt, schema, system=system,
                                      temperature=cfg.temperature,
                                      max_tokens=max_tokens or cfg.max_tokens,
                                      seed=seed)
        budget.spend(res.usage.total, self.router.lane(kind), res.usage.prompt_tokens, stage=kind)
        budget.steps_used += 1
        # ★ Відкинута спроба НЕ стає голосом. Перший живий прогін показав саме це: сцена
        # промовляла репліки, які ядро щойно забракувало як повтор, і чужим голосом.
        # Токени спроби все одно пораховані вище — платимо за неї чесно, але не озвучуємо.
        if self.trace is not None and voice:
            self.trace.emit(StepRecord(
                run_id=self.run_id, tick=budget.steps_used,
                agent="subagent" if span else "orchestrator", span=span,
                stage=kind, model=llm.model, lane=self.router.lane(kind),
                prompt=prompt, raw_output=_text(res.text) if span else res.text,
                parsed=None, schema_valid=True, world_valid=True,
                prompt_id=self.prompt_id, prompt_sha=self.prompt_sha, seed=seed))
        return res.text

    def _emit_line(self, span: str, line: str, index: int) -> None:
        """Голос — окремою подією, вже ПІСЛЯ прийняття: сцена промовляє лише те, що вціліло."""
        if self.trace is None or not line:
            return
        lane = self.router.lane("speak")
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=index, agent="subagent", span=span, stage="speak",
            model=self.router.route("speak").model, lane=lane, prompt="", raw_output=line,
            parsed=None, schema_valid=True, world_valid=True,
            prompt_id=self.prompt_id, prompt_sha=self.prompt_sha))

    def _emit_call(self, call: ToolCall, index: int, seed: int, span: str) -> None:
        if self.trace is None:
            return
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=index, agent="orchestrator", span=span, stage="select",
            model="viche", lane=self.router.lane("decide"), prompt="", raw_output="",
            parsed={"tool": call.tool, **dict(call.args)},
            schema_valid=True, world_valid=True, seed=seed))

    def _emit_tool(self, call: ToolCall, result, index: int, seed: int, span: str) -> None:
        if self.trace is None:
            return
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=index, agent="tool", stage="tool_result", span=span,
            model="tool", lane="none", prompt="", raw_output="",
            parsed={"tool": call.tool, "ok": result.ok, "found": result.found,
                    **({"error": result.error} if result.error else {})},
            schema_valid=True, world_valid=result.ok, seed=seed))


def _safe_json(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _text(raw: str) -> str:
    data = _safe_json(raw)
    line = str((data or {}).get("репліка") or "").strip()
    return " ".join((line or (raw or "").strip()).split())


def _grams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = [w.strip(".,!?«»\"'—-").lower() for w in text.split()]
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def _twin_of(line: str, said: list, threshold: float = 0.45) -> str | None:
    """Повертає САМУ повторену фразу, а не факт повтору: ремонт має бути адресний."""
    mine = _grams(line)
    if not mine:
        return None
    for _, other in said:
        theirs = _grams(other)
        if theirs and len(mine & theirs) / len(mine | theirs) >= threshold:
            return other
    return None


def _too_similar(line: str, earlier: list[str], threshold: float = 0.45) -> bool:
    """Дослівний повтор — не стиль, а дефект: у першому живому прогоні одна фраза прозвучала
    чотири рази від різних людей. Ловить це КОД, бо це визначено даними."""
    mine = _grams(line)
    if not mine:
        return False
    for other in earlier:
        theirs = _grams(other)
        if theirs and len(mine & theirs) / len(mine | theirs) >= threshold:
            return True
    return False


def _echoes(line: str, task: str, prompt: str, lens: str = "") -> bool:
    """Переказ завдання замість мови.

    Лінзу теж перевіряємо: перенесення її в системне повідомлення різко зменшило переказ, але не
    прибрало його — жива репліка починалась дослівно лінзою («чи бувало таке раніше, як тоді
    обійшлось»). Системне повідомлення не імунітет, а лише менша ймовірність.
    """
    mine = _grams(line, 5)
    return (bool(mine & _grams(task, 5))
            or bool(mine & _grams(prompt.split("ТВІЙ ХІД:")[-1], 5))
            or bool(lens and mine & _grams(lens, 5)))


def _drifted(line: str) -> bool:
    """Дрейф — це порожньо, надто коротко або взагалі не українською."""
    if len(line) < MIN_LINE_CHARS or line.startswith(("{", "[")):
        return True
    cyrillic = sum(1 for ch in line if "а" <= ch.lower() <= "я" or ch in "ґєіїҐЄІЇ")
    letters = sum(1 for ch in line if ch.isalpha())
    return letters == 0 or cyrillic / letters < 0.6


__all__ = ["Viche", "VICHE_NOTE", "SUMMARY_MOVE", "DOUBT_MOVE"]
