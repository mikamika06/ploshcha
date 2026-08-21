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

from ..domain.evidence import found_in
from ..domain.graph import child_budget
from ..domain.modes import Mode, mode_for
from ..domain.people import remembers
from ..domain.task import Budget, TaskResult
from ..domain.viche import (
    BY_ROLE,
    bonds_from,
    personas_from,
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
    stance_start, stance_after, stance_view, stance_label,
    vote_schema, tally, VOTES,
)
from ..ports.agent import AgentPort
from ..ports.tool import ToolCall
from ..ports.trace import StepRecord, TracePort

VICHE_NOTE = "viche"
# Скільки тактів планує одна хвиля і скільки хвиль щонайбільше.
#
# Компроміс між «одна партитура на прогін» (нічого не реагує) і «виклик на кожен такт» (дорого):
# оркестратор бачить стенограму й позиції кожні WAVE тактів.
WAVE = 5
MAX_WAVES = 5
# Ходи для ремонту повтору: інший хід — інший зміст. Обертаємо за індексом такту, тож
# відтворювано за сідом, як і решта збурень.
_CONTRAST = ("заперечити", "спитати_діло", "пожартувати", "пожалітись", "порахувати")
MAX_LINE_CHARS = 320
MIN_LINE_CHARS = 8
# Партитурі потрібен СВІЙ бюджет виводу. Спільна стеля `max_tokens` розрахована на репліку (~220),
# а дванадцять тактів це ~800 токенів: JSON обривався на півслові, парс падав, і мовчки вмикався
# запасний план «кожен реагує по разу». Тобто партитуру Mamay викидало щопрогону, а видно було лише
# «щось розмова коротка».
# Стеля виходу партитури рахується з РОЗМІРУ хвилі, а не береться однією великою цифрою.
# 2200 закладались під партитуру на ввесь прогін; хвиля з пʼяти тактів стільки не пише, а велика
# стеля лише запрошує модель писати більше.
SCORE_TOKENS = 2200


def score_cap(beats: int) -> int:
    return max(600, min(SCORE_TOKENS, 240 * beats + 200))
# На скільки ділиться бюджет для посланого. Не половина: він іде по одну річ, а віче
# мусить дожити до кінця.
SCOUT_PARTS = 6
SCOUT_SHOW = 3
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
                 chronicle_system: str = CHRONICLE_SYSTEM, village: list | None = None,
                 standing: dict[str, float] | None = None, rumours: list | None = None,
                 place: str | None = None, scout=None, memory=None):
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
        # Породжене село. Порожньо — сталі персони: віче мусить працювати й без ланки породження.
        # Скільки важить слово кожного (репутація) і які чутки ще ходять селом.
        # Місце розмови — це профіль ядра, а не підпис: інша ширина, інші такти, інша температура,
        # і подекуди взагалі немає старости, який зводить.
        self.mode: Mode = mode_for(place)
        # Кого посилають дізнатись. Це фабрика ПОВНОЦІННОГО агента зі своїм бюджетом: людина йде й
        # робить кілька кроків сама, а не смикає один інструмент. Порожньо — лишається один виклик.
        self.scout = scout
        # Памʼять села: минулі віча в пакеті, стосунки в партитурі, літопис по завершенні.
        self.memory = memory
        self._bonds = memory.bonds() if memory is not None else {}
        self.standing = dict(standing or {})
        # Чи знайшов довідник на останньому такті: `None` — інструмента не було взагалі.
        self._last_found: bool | None = None
        self.rumours = list(rumours or [])
        self.village = list(village or [])
        self._people = {x.role: x for x in self.village}
        self._recalled: list[str] = []
        # Скринька вхідних: сюди падає слово гостя й шепіт, поки віче ТРИВАЄ. Читається між
        # тактами, а не в середині — інакше репліка вклинювалась би в напівзгенерований такт.
        self._inbox: list[dict] = []
        self._lock = threading.Lock()
        self._whispers: dict[str, str] = {}
        # Вади самого КАНАЛУ (обрив на стелі, порожня відповідь) — окремо від вад змісту: їх
        # помічає `_call`, який про поточний список інцидентів нічого не знає.
        self._flaws: list[str] = []

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
        self._flaws = []
        self.width = min(self.width, self.mode.width)
        cast = self._cast(task)
        roles = [p.role for p in cast]
        said: list[tuple[Persona, str]] = []
        incidents: list[str] = []
        stances = stance_start(roles)
        played: list[Beat] = []
        target = self.mode.beats[1]
        index = 0
        wave = 0
        pending: list[Beat] = []
        while index < target and wave <= MAX_WAVES:
            if not pending:
                if wave >= MAX_WAVES or not budget.can_continue():
                    break
                want = min(WAVE, target - index)
                # Провал ПЕРШОЇ хвилі — це втрачена партитура (перепит + гучний запасний план).
                # Провал наступної — просто кінець розмови: план у неї вже був.
                fresh = self._plan(task, cast, seed + wave * 17, budget, incidents,
                                   said=said if wave else None,
                                   stances=stances if wave else None,
                                   want=want if wave else None,
                                   first=wave == 0)
                if not fresh:
                    break
                pending = scatter(fresh, roles, seed + wave, task, self._people,
                                  self.mode.interrupts if wave == 0 else 0, self._bonds)
                played += pending
                self._emit_plan(played)
                wave += 1
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
            before = len(said)
            self._last_found = None
            said += self._play(task, beat, index, cast, said, seed, budget, incidents)
            # Позицію рухає КОД за тим, що сталось у такті: інструмент дав факт чи ні, який хід.
            stances = stance_after(beat, stances, self.standing,
                                   self._last_found if len(said) > before else None)
        beats = played

        if said:
            # Староста зводить не всюди: у шинку модератора немає, і саме тому там кажуть інше.
            if self.mode.summary:
                summary = self._summary(task, said, seed, budget, incidents)
                if summary is not None:
                    said.append(summary)
            if self.mode.doubt:
                doubt = self._doubt(task, said, seed, budget, incidents)
                if doubt is not None:
                    said.append(doubt)
            votes = self._vote(task, cast, said, stances, seed, budget, incidents)
            said += votes.pop("репліки", [])
            self._chronicle(task, said, seed, budget, incidents, votes=votes)

        if self.memory is not None:
            for a, b, delta in bonds_from(beats):
                self.memory.bond(a, b, delta)
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

    def _past(self, task: str) -> str:
        """Що село вже пережило на споріднену тему.

        ★ Прийшлий цієї памʼяті НЕ має — але це не привід ховати її від партитури: розпорядник
        знає село, а не знає його той, кого щойно прийняли. Тому гейт стоїть у пакеті МОВЦЯ.
        """
        if self.memory is None:
            return ""
        past = self.memory.recall(task)
        if not past:
            return ""
        self._recalled = [f"{r['title']}: {r['narration']}" for r in past]
        if self.trace is not None:
            self.trace.emit(StepRecord(
                run_id=self.run_id, tick=0, agent="notebook", stage="mem_read", model="viche",
                lane="none", prompt="", raw_output="",
                parsed={"items": [r["title"] for r in past], "query": task[:120]},
                schema_valid=True, world_valid=True))
        return "СЕЛО ВЖЕ ПЕРЕЖИВАЛО: " + " | ".join(self._recalled) + "\n"

    def _cast(self, task: str) -> list[Persona]:
        if not self.village:
            return cast_for(task, self.width)
        folk = personas_from(self.village)
        chosen = cast_for(task, min(self.width, len(folk)))
        by_role = {p.role: p for p in folk}
        # Склад беремо з ПОРОДЖЕНОГО села: якщо роль у ньому не живе, її просто немає в цьому селі.
        picked = [by_role[p.role] for p in chosen if p.role in by_role]
        return picked or list(folk[:self.width])

    # ── партитура ─────────────────────────────────────────────────────────────
    def _plan(self, task: str, cast: list[Persona], seed: int, budget: Budget,
              incidents: list[str], *, said: list[tuple[Persona, str]] | None = None,
              stances: dict[str, float] | None = None, want: int | None = None,
              first: bool = True) -> list[Beat]:
        """Партитура — або вся наперед, або ХВИЛЯ під те, що вже сказано.

        Одна партитура на весь прогін означала, що аргументи ні на що не впливають: черга була
        написана до першого слова. Тепер Мамай планує по кілька тактів, бачачи стенограму й
        позиції; це між «один виклик на прогін» і «виклик на кожен такт» — реактивно, але не
        по ціні такту.
        """
        roles = [p.role for p in cast]
        tools = [s.name for s in self.tools.specs()] if self.tools is not None else []
        tools = [t for t in tools if t != "final_answer"]
        people = "\n".join(f"- {p.role} ({p.name}): {p.lens}" for p in cast)
        prompt = (f"НОВИНА ДЛЯ ВІЧА:\n{task}\n\nЛЮДИ:\n{people}\n\n"
                  f"ДОСТУПНІ ХОДИ: {', '.join(MOVE_HINT)}\n"
                  f"ДОВІДНИК: {', '.join(tools) if tools else 'немає'}\n"
                  + (("ЧУТКИ, ЩО ХОДЯТЬ СЕЛОМ: "
                      + " | ".join(f"{r['who']}: {r['claim']}" for r in self.rumours[:3]) + "\n")
                     if self.rumours else "")
                  + self._past(task)
                  + (("\nЩО ВЖЕ СКАЗАНО:\n"
                      + "\n".join(f"- {p.name} ({p.role}): {t}" for p, t in (said or [])[-8:])
                      + "\n" + stance_view(stances or {}, {p.role: p.name for p in cast}) + "\n")
                     if said else "")
                  + "\n"
                  + ("Розпиши НАСТУПНІ такти віча — відповідай на щойно сказане, "
                     "не повторюй уже сказаного. `у_відповідь` — номер попереднього такту або null."
                     if said else
                     "Розпиши такти віча. `у_відповідь` — номер попереднього такту або null."))
        span = (want, want) if want else self.mode.beats
        schema = score_schema(roles, tools, span)
        cap = score_cap(want or self.mode.beats[1])
        beats = repair_score(_safe_json(
            self._call("decide", prompt, self.score_system, schema, seed, budget,
                       max_tokens=cap)), roles, tools, self.standing)
        if not beats and not first:
            return []
        if not beats:
            # Збій структурованого виводу тут ПЕРЕРИВЧАСТИЙ, а не детермінований: та сама схема й
            # той самий промпт то проходять, то дають нерозбірний JSON. Один перепит із іншим сідом
            # рятує прогін; без нього єдиний невдалий виклик знецінював усю розмову.
            incidents.append("viche_score_retry")
            beats = repair_score(_safe_json(
                self._call("decide", prompt, self.score_system, schema, seed + 101, budget,
                           max_tokens=cap)), roles, tools, self.standing)
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
            span = self._span(who, index)
            fact = (self._send(beat, index, seed, budget, span, incidents)
                    if self.scout is not None else
                    self._ask(beat, index, seed, budget, span))

        self._emit_deed(beat.хто, beat.дія, index)
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
        self._last_found = result.found if result.ok else None
        if not result.ok:
            return "інструмент не відповів"
        value = result.value if isinstance(result.value, dict) else {"результат": result.value}
        if result.found is False:
            return "у довіднику того немає"
        return json.dumps(value, ensure_ascii=False)[:400]

    def _send(self, beat: Beat, index: int, seed: int, budget: Budget, span: str,
              incidents: list[str]) -> str:
        """Послати людину дізнатись — тобто породити ДИТИНУ-АГЕНТА зі своїм бюджетом.

        Різниця з одним викликом інструмента принципова: дитина робить власний багатокроковий цикл
        (вибрала інструмент → побачила результат → вирішила, що далі), і повертається з висновком,
        а не з сирим полем. Саме це «Мамай кличе себе як агента» й означає.

        Її кроки ми переграємо у СВОЮ трасу з проміжком тієї людини, яку послали: інакше на сцені
        це робив би хтось інший, і спостерігач бачив би не те, що сталось.
        """
        query = (beat.запит or "").strip()
        if not query:
            return self._ask(beat, index, seed, budget, span)
        child = child_budget(budget, SCOUT_PARTS)
        try:
            result = self.scout(child).run(query, seed=seed, budget=child)
        except Exception as exc:
            incidents.append(f"viche_scout_failed:{type(exc).__name__}")
            return "нічого не дізнався"

        for entry in list(getattr(result, "scratch", []) or [])[:SCOUT_SHOW]:
            call = entry.get("call") or {}
            if not call.get("tool"):
                continue
            self._emit_call(ToolCall(tool=str(call["tool"]),
                                     args={k: v for k, v in call.items() if k != "tool"}),
                            index, seed, span)
            # `found` у сліді оркестратора не лежить готовим — його виводять із результату. Без
            # цього посланий показував «незастосовно» там, де насправді знав: тризначність
            # («не знайшов» ≠ «зламався» ≠ «незастосовно») губилась саме на шляху спостереження.
            found = entry.get("found", found_in(entry.get("result")))
            self._last_found = found if isinstance(found, bool) else None
            self._emit_tool_record(str(call["tool"]), found, index, seed, span)

        # Витрати дитини лягають у НАШ бюджет: інакше стеля прогону нічого не обмежувала б.
        spent = int(getattr(result, "tokens", 0) or 0) + int(getattr(result, "aux_tokens", 0) or 0)
        if spent:
            budget.spend(spent, self.router.lane("decide"), 0, stage="decide")
        answer = " ".join((getattr(result, "answer", "") or "").split())
        if not answer or getattr(result, "outcome", "") == "abstain":
            incidents.append("viche_scout_empty")
            return "нічого не дізнався"
        return answer[:400]

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
        person = self._people.get(who.role)
        extra = ""
        if person is not None:
            marks = ", ".join(t.pole for t in person.marked)
            extra = (f"\nПро тебе: {person.bio}" if person.bio else "")
            extra += (f"\nТвоя примовка: «{person.saying}»" if person.saying else "")
            extra += (f"\nНоров: {marks}." if marks else "")
        # Прийшлий не має доступу до памʼяті села — і тому бачить те, чого свої вже не помічають.
        recalled = ""
        if self._recalled and (person is None or remembers(person)):
            recalled = "\nСело памʼятає: " + " | ".join(self._recalled[:2])
        manner = f"\n{self.mode.manner}" if self.mode.manner else ""
        return (f"{self.line_system}{manner}{recalled}\n\nТИ: {who.name}. "
                f"Дивишся на світ так: {who.lens}.{extra}")

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
                 budget: Budget, incidents: list[str]) -> tuple[Persona, str] | None:
        """Зведення старости — така сама спроба, як репліка, тож і перевіряється так само.

        Доти воно єдине йшло на сцену БЕЗ перевірки: обірваний вивід шлюзу ставав голосом дослівно,
        і староста промовляв `{"репліка": "Отак воно і`. Порожній вивід був гірший — німий мовець
        у стенограмі, який ще й накручував лічильник голосів.
        """
        prompt = (f"НОВИНА: {task}\n\nЩО КАЗАЛИ:\n"
                  + "\n".join(f"- {p.name}: {t}" for p, t in said))
        span = self._span(STAROSTA, 0)
        raw = self._call("synthesize", prompt, self.summary_system, line_schema(), seed, budget,
                         span=span, voice=False)
        line = _text(raw)[:MAX_LINE_CHARS]
        if _drifted(line):
            incidents.append("viche_summary_lost")
            return None
        self._emit_line(span, line, 0)
        return STAROSTA, line

    def _doubt(self, task: str, said: list[tuple[Persona, str]], seed: int,
               budget: Budget, incidents: list[str]) -> tuple[Persona, str] | None:
        """Верифікатор став реплікою, а не вироком: сумнів чути, але він не вбиває прогін.

        Голос емітиться ПІСЛЯ прийняття. Доти забракований сумнів уже прозвучав на сцені (`{}`
        вустами попа), а зі стенограми випадав — тобто сцена й підсумок розходились.
        """
        prompt = (f"НОВИНА: {task}\n\nЩО КАЗАЛИ:\n"
                  + "\n".join(f"- {p.name}: {t}" for p, t in said))
        span = self._span(PIP, 0)
        raw = self._call("judge", prompt, self.doubt_system, line_schema(), seed, budget,
                         span=span, voice=False)
        line = _text(raw)[:MAX_LINE_CHARS]
        if _drifted(line):
            incidents.append("viche_doubt_lost")
            return None
        self._emit_line(span, line, 0)
        return PIP, line

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

    def _vote(self, task: str, cast: list[Persona], said: list[tuple[Persona, str]],
              stances: dict[str, float], seed: int, budget: Budget,
              incidents: list[str] | None = None) -> dict:
        """Голос кожного — і ПІДРАХУНОК.

        Ухвала мусить бути числом, а не переказом: доти літописець сам вирішував, до чого дійшло
        віче, тобто модель судила про власну ж розмову. Тепер кожен каже «за/проти/утримуюсь» —
        це закритий енум, рівно те, що виконавець тягне надійно, — а рішення дає лічба.
        """
        spoke = [p for p in cast if any(q.role == p.role for q, _ in said)]
        if not spoke:
            return tally([])
        talk = "\n".join(f"- {p.name} ({p.role}): {t}" for p, t in said[-12:])
        votes: list[tuple[str, str]] = []
        reasons: list[tuple[str, str, str]] = []
        spoken: list[tuple[Persona, str]] = []
        for i, p in enumerate(spoke):
            if not budget.can_continue():
                break
            prompt = (f"НОВИНА: {task}\n\nЩО КАЗАЛИ:\n{talk}\n\n"
                      f"Ти — {p.name} ({p.role}). Твоя лінза: {p.lens}. "
                      f"Схиляєшся: {stance_label(stances.get(p.role, 0.0))}.\n"
                      "Як голосуєш і чому — коротко, своїми словами.")
            data = _safe_json(self._call("speak", prompt, self.line_system, vote_schema(),
                                         seed + 7 * i, budget, max_tokens=120))
            vote = str((data or {}).get("голос") or "")
            if vote not in VOTES:
                # Загублений голос МІНЯЄ ухвалу, тому мовчки його викидати не можна. Коли не
                # розібрався жоден, лічба чесно показувала нулі, але причина («шлюз віддав сміття»)
                # не лишалась ніде — і «віче не дійшло голосу» читалось як рішення села.
                if incidents is not None:
                    incidents.append(f"viche_vote_lost:{p.role}")
                continue
            why = " ".join(str((data or {}).get("чому") or "").split())[:90]
            votes.append((p.role, vote))
            reasons.append((p.role, vote, why))
            spoken.append((p, f"{vote}. {why}" if why else vote))
            self._emit_vote(p, vote, why, i + 1)
        return {**tally(votes), "голоси": reasons, "репліки": spoken}

    def _emit_vote(self, p: Persona, vote: str, why: str, index: int) -> None:
        """Голос звучить ВГОЛОС: інакше підрахунок був би ще одним прихованим числом."""
        if self.trace is None:
            return
        said = f"{vote}. {why}" if why else vote
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=index, agent="subagent", span=self._span(p, index),
            stage="speak", model=self.router.route("speak").model, lane=self.router.lane("speak"),
            prompt="", raw_output=said, parsed=None, schema_valid=True, world_valid=True,
            prompt_id=self.prompt_id, prompt_sha=self.prompt_sha))

    def _chronicle(self, task: str, said: list[tuple[Persona, str]], seed: int,
                   budget: Budget, incidents: list[str], votes: dict | None = None) -> None:
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
                  # Хвіст, а не вся стенограма: вхід літопису ріс разом із довжиною віча, а
                  # заголовок і настрій однаково пишуться з того, чим розмова СКІНЧИЛАСЬ.
                  + "\n".join(f"- {p.name} ({p.role}): {t}" for p, t in said[-16:])
                  + (f"\n\nГОЛОСУВАННЯ: {votes['підсумок']}\n"
                     + "\n".join(f"- {r}: {v} — {w}" for r, v, w in votes.get("голоси", []))
                     if votes and votes.get("лічба") else ""))
        schema = chronicle_schema(roles)
        data = _safe_json(self._call("synthesize", prompt, self.chronicle_system, schema, seed,
                                     budget, max_tokens=CHRONICLE_TOKENS))
        if not data:
            # Той самий переривчастий збій, що й у партитури — один перепит іншим сідом.
            incidents.append("viche_chronicle_retry")
            data = _safe_json(self._call("synthesize", prompt, self.chronicle_system, schema,
                                         seed + 101, budget, max_tokens=CHRONICLE_TOKENS))
        if not data:
            # ★ Втрачений літопис НЕ має лишати віче без кінця.
            #
            # Заміряно на живому прогоні: одна невдала відповідь шлюзу — і зникає геть усе
            # закриття: ухвала, чутка, думки, настрій, підсумок. Глядач дочитував останню репліку
            # й лишався ні з чим, ніби розмова обірвалась. А підрахунок голосів у нас ВЖЕ Є — він
            # порахований кодом і від моделі не залежить. Тому закриваємо тим, що знаємо напевно.
            incidents.append("viche_chronicle_lost")
            self._emit_closing(task, said, votes)
            return
        for item in data.get("думки") or []:
            thought = str(item.get("думка") or "").strip()
            if item.get("хто") in roles and thought:
                self.trace.emit(StepRecord(
                    run_id=self.run_id, tick=0, agent="thinker", stage="reflect", model="viche",
                    lane=self.router.lane("synthesize"), prompt="", raw_output="",
                    parsed={"agentId": item["хто"], "thought": thought[:MAX_LINE_CHARS]},
                    schema_valid=True, world_valid=True))
        if self.memory is not None:
            self.memory.remember(task, str(data.get("заголовок") or ""),
                                 str(data.get("оповідь") or ""), str(data.get("настрій") or ""))
        self._emit_rumour(data.get("чутка"), roles)
        decision = data.get("ухвала")
        if votes and votes.get("лічба") and isinstance(decision, dict):
            # Що ухвалили — вирішила лічба; літописець лише називає це людською мовою.
            decision = {**decision, "що": f"{votes['підсумок']} · {decision.get('що') or ''}"[:140]}
        self._emit_decision(decision, roles)
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=0, agent="chronicler", stage="report", model="viche",
            lane=self.router.lane("synthesize"), prompt="", raw_output="",
            parsed={"day": 1, "title": str(data.get("заголовок") or task)[:120],
                    "narration": str(data.get("оповідь") or "")[:600],
                    "mood": mood_view(str(data.get("настрій") or "спокій"),
                                      data.get("сила") or "помірно"),
                    "highlights": [t for _, t in said[:3]]},
            schema_valid=True, world_valid=True))

    def _emit_closing(self, task: str, said: list[tuple[Persona, str]], votes: dict | None) -> None:
        """Мінімальне закриття без літописця: ухвала з лічби і сухий підсумок замість оповіді."""
        if self.trace is None:
            return
        tally_line = (votes or {}).get("підсумок") or "віче розійшлось без ухвали"
        if votes and votes.get("лічба"):
            self.trace.emit(StepRecord(
                run_id=self.run_id, tick=0, agent="council", stage="report", model="viche",
                lane=self.router.lane("synthesize"), prompt="", raw_output="",
                parsed={"label": tally_line[:140], "who": (votes.get("голоси") or [("", "", "")])[0][0],
                        "poi": self.mode.place},
                schema_valid=True, world_valid=True))
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=0, agent="chronicler", stage="report", model="viche",
            lane=self.router.lane("synthesize"), prompt="", raw_output="",
            parsed={"day": 1, "title": task[:120], "narration": tally_line[:600],
                    "mood": mood_view("спокій", "помірно"),
                    "highlights": [t for _, t in said[-3:]]},
            schema_valid=True, world_valid=True))

    def _emit_rumour(self, raw, roles: list[str]) -> None:
        """Чутка — твердження без підстави, сказане вголос.

        Якщо підстава БУЛА, це не чутка, а просто слово — і в обіг воно не йде. Тому «є: так» самого
        по собі мало: розрізняє саме `підстава`.
        """
        if self.trace is None or not isinstance(raw, dict):
            return
        if not self.mode.rumours:
            return
        if str(raw.get("є")) != "так" or str(raw.get("підстава")) != "не було":
            return
        claim = " ".join(str(raw.get("що") or "").split())
        who = str(raw.get("хто") or "")
        if not claim or who not in roles:
            return
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=0, agent="rumour", stage="judge", model="viche",
            lane=self.router.lane("judge"), prompt="", raw_output="",
            parsed={"who": who, "claim": claim[:200]},
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
        # Вади каналу доїжджають ТИМ САМИМ шляхом, що й решта інцидентів (`task.outcome`), інакше
        # їх видно лише в цьому обʼєкті й ніде більше.
        incidents = incidents + self._flaws
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
                                      temperature=max(0.0, min(1.5,
                                                               cfg.temperature + self.mode.heat)),
                                      max_tokens=max_tokens or cfg.max_tokens,
                                      seed=seed)
        budget.spend(res.usage.total, self.router.lane(kind), res.usage.prompt_tokens, stage=kind)
        budget.steps_used += 1
        # ★ Розрізняємо «модель написала дурницю» і «шлюз не дописав». Обидва дають той самий
        # нерозбірний JSON, і доти вони були нерозрізненні: інцидент казав `score_lost`, а
        # справжня причина — замала стеля виводу або мовчання шлюзу — не лишалась ніде.
        if res.finish_reason == "length":
            self._flaws.append(f"viche_cut:{kind}")
        elif not (res.text or "").strip():
            self._flaws.append(f"viche_empty:{kind}")
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
                prompt_id=self.prompt_id, prompt_sha=self.prompt_sha, seed=seed,
                finish_reason=res.finish_reason))
        return res.text

    def _emit_deed(self, role: str, deed: str, index: int) -> None:
        """Дія тіла на такті — окремою подією.

        Сцена доти могла лише «ходити до POI»: люди на вічі тинялись, бо жодна подія не казала,
        що саме людина робить, поки говорить. Дію обирає Мамай у партитурі (закритий енум), а
        якщо не обрав — код виводить її з самого ходу.
        """
        if self.trace is None or not deed:
            return
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=index, agent="deed", span="", stage="speak",
            model="viche", lane=self.router.lane("speak"), prompt="", raw_output="",
            parsed={"agentId": role, "дія": deed}, schema_valid=True, world_valid=True))

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

    def _emit_tool_record(self, tool: str, found, index: int, seed: int, span: str) -> None:
        if self.trace is None:
            return
        self.trace.emit(StepRecord(
            run_id=self.run_id, tick=index, agent="tool", stage="tool_result", span=span,
            model="tool", lane="none", prompt="", raw_output="",
            parsed={"tool": tool, "ok": True, "found": found},
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
    """Розбір відповіді шлюзу з ВРЯТУВАННЯМ того, що вціліло.

    ★ Заміряно на живих прогонах: `viche_chronicle_lost` двічі з двох. Літопис — найбільша
    відповідь у прогоні (заголовок, оповідь, настрій, ухвала, чутка й думки кожного), і саме вона
    найчастіше приходить обрізаною або в обгортці з прози. Строгий `json.loads` викидав УСЕ через
    одну незакриту дужку в хвості, разом із заголовком і оповіддю, які лежали на самому початку.

    Тому: спершу чесний розбір; якщо не вийшло — беремо перший обʼєкт у тексті й доліплюємо
    незакриті дужки, відкинувши обірваний хвіст. Це не «пробачити модель», а не викидати те, що
    вона вже сказала правильно.
    """
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        pass
    raw = (text or "").strip()
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    cut = None
    for i, ch in enumerate(raw[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{" or ch == "[":
            depth += 1
        elif ch == "}" or ch == "]":
            depth -= 1
            if depth == 0:
                cut = i + 1
                break
    if cut is not None:
        try:
            value = json.loads(raw[start:cut])
            return value if isinstance(value, dict) else None
        except (ValueError, TypeError):
            return None
    # Обірвано посеред обʼєкта: пробуємо все коротші префікси, доліплюючи закриття. Перший, що
    # розібрався, і є тим, що модель встигла сказати цілим.
    body = raw[start:]
    for cutoff in _cut_points(body):
        head = body[:cutoff]
        for closing in _closings(head):
            try:
                value = json.loads(head + closing)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                return value
    return None


def _cut_points(body: str) -> list[int]:
    """Місця, де відповідь можна обрізати без втрати цілої пари: коми й закриття на верхніх рівнях."""
    points = [i for i, ch in enumerate(body) if ch in ",}]"]
    return list(reversed(points))[:40]


def _closings(head: str) -> list[str]:
    """Варіанти дужок, якими можна закрити обрізане: спершу без лапок, тоді з ними."""
    depth_curly = head.count("{") - head.count("}")
    depth_square = head.count("[") - head.count("]")
    tail = ""
    for _ in range(max(0, depth_square)):
        tail += "]"
    for _ in range(max(0, depth_curly)):
        tail += "}"
    stripped = head.rstrip().rstrip(",")
    return [tail, '"' + tail] if stripped is not None else [tail]


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
