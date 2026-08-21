"""Проєкція траси ядра в події контракту 1.1 — перший СПРАВЖНІЙ виробник контракту.

Доти контракт існував як опис і ручна фікстура, тобто був гіпотезою: ніхто не перевіряв, чи поля
взагалі складаються з того, що ядро віддає. Тут це перевіряється — і саме тут виявляється, чого в
трасі бракує (так з'явився `StepRecord.lane`).

Я4 переписав це на ПОТІК. Причина: `project_run` вимагав завершеного прогону (`result.scratch`), бо
результати інструментів у трасу не потрапляли. Тепер оркестратор емітить їх як
`StepRecord(agent="tool", stage="tool_result")`, отже траса повна, і проєкція можлива крок за кроком.

`project_run` лишився як **обгортка** над `StreamProjector`: для старих трас без `tool_result` він
синтезує ці записи зі `scratch`. Так мапінг існує в одному місці й дві реалізації не розʼїдуться.

Два інваріанти, які тут тримаються:
  • `abstain` доїжджає окремим станом (`task.outcome`), а не як помилка;
  • `found` доїжджає тризначним (`true|false|null`), бо «нема даних» ≠ «зламався» ≠ «незастосовно».
"""

from typing import Any

from ..domain.evidence import evidence_state, found_in
from ..domain.gate import FINAL_TOOL
from ..domain.state import phase_of
from ..ports.trace import StepRecord

PROTOCOL = "1.1.0"
LANES = ("lapa", "mamay", "unknown")

TOOL_STAGES = ("select", "act", "decide")

POI_OF_STAGE = {
    "recall": "well", "mem_read": "well", "mem_write": "well", "importance": "well",
    "judge": "church",
    "select": "forge", "act": "forge", "parse": "forge", "classify": "forge",
    "ground": "forge", "gate": "forge", "decide": "forge",
    "synthesize": "square", "generate": "square", "reflect": "square", "speak": "square",
}
DEFAULT_POI = "square"

# Голоси виводяться з ЯРУСУ й ролі, а не вигадуються: якщо в прогоні працював один ярус, голос
# буде один — і це видно. Імена збігаються з ростером фронта (apps/web/src/roles).
VOICE_OF_LANE = {"mamay": "starosta", "lapa": "koval"}
VOICE_VERIFIER = "pip"
VOICE_FALLBACK = "shynkar"
VILLAGERS = ("did", "sheptu", "mirosh", "parubok", "divchyna", "mati", "shynkar")
# `hist` — голос гостя (тебе). Без нього span із роллю падав у хеш-фолбек, і твоє слово
# промовляв чужий рот — той самий клас, що вже ловили зі старостою й попом.
ROLE_VOICES = frozenset(VILLAGERS) | {"koval", "starosta", "pip", "diak", "chumak",
                                      "hist"}

# Інструмент = дія в локації: людина йде туди, де про це можна дізнатись, а не «за стадією».
POI_OF_TOOL = {"довідка": "well", "словник": "church", "обчислити": "forge",
               "lookup_fact": "well", "check_date": "church", "calc": "forge"}


def _lane(value: str) -> str:
    return value if value in LANES else "unknown"


def poi_of_stage(stage: str) -> str:
    return POI_OF_STAGE.get(stage, DEFAULT_POI)


def villager_of_span(span: str) -> str:
    """Той самий span завжди дає того самого селянина — інакше голоси стрибали б між репліками.

    Якщо span НЕСЕ роль (віче: `run/viche/koval/3`), беремо її: там особу вже обрав код, і хеш
    затирав би саме те, що робить розмову розмовою.
    """
    for part in span.split("/"):
        if part in ROLE_VOICES:
            return part
    digits = "".join(ch for ch in span if ch.isdigit())
    index = int(digits[-3:]) if digits else sum(map(ord, span))
    return VILLAGERS[index % len(VILLAGERS)]


def is_prose(record: StepRecord) -> bool:
    """Мовленням стає лише проза. `select`/`parse`/`decide` віддають JSON-виклик, не мову."""
    text = (record.raw_output or "").strip()
    if not text:
        return False
    if record.agent == "subagent":
        return True
    if record.parsed and record.parsed.get("tool"):
        return False
    return not text.startswith(("{", "["))


class StreamProjector:
    """Стан: `seq`, поточний тік, останній POI. `feed` віддає готові конверти."""

    def __init__(self, run_id: str, ts: str, *, scene: dict | None = None,
                 started_at: str | None = None, max_ticks: int = 1,
                 agent_id: str = "orchestrator", emit_motion: bool = True,
                 emit_ticks: bool = True, emit_voices: bool = True,
                 cast: list[dict] | None = None):
        self.run_id = run_id
        self.ts = ts
        self.scene = scene
        self.started_at = started_at or ts
        self.max_ticks = max_ticks
        self.agent_id = agent_id
        self.emit_motion = emit_motion
        self.emit_ticks = emit_ticks
        self.emit_voices = emit_voices
        self.cast = cast
        self.seq = 0
        self._tick: int | None = None
        # POI на КОЖНОГО, а не один на сцену: у вічі ходить кілька людей, і спільний стан
        # гасив би рух усіх, крім першого.
        self._poi: dict[str, str] = {}
        self._last_tick = 0
        self._said = 0

    def _envelope(self, type_: str, payload: dict, tick: int) -> dict:
        out = {"protocol": PROTOCOL, "runId": self.run_id, "seq": self.seq, "ts": self.ts,
               "tick": tick, "type": type_, "payload": payload}
        self.seq += 1
        return out

    def start(self) -> list[dict]:
        if self.scene is None:
            return []
        out = [self._envelope("run.started", {
            "config": {"maxTicks": max(1, self.max_ticks), "castingMode": "library"},
            "scene": self.scene, "startedAt": self.started_at}, 0)]
        # Склад оголошує ЯДРО, а не записана фікстура: інакше сцена не знає про людей, яких ядро
        # справді має (староста, піп), а імена в тексті й на підписі можуть розійтись.
        if self.cast:
            out.append(self._envelope("casting.done", {"cast": list(self.cast)}, 0))
        return out

    def _tick_events(self, tick: int) -> list[dict]:
        self._last_tick = tick
        if not self.emit_ticks or tick == self._tick:
            self._tick = tick
            return []
        self._tick = tick
        return [self._envelope("tick.begin", {"timeOfDay": phase_of(tick)}, tick)]

    def _motion(self, stage: str, tick: int) -> list[dict]:
        return self._walk(self.agent_id, poi_of_stage(stage), tick)

    def _walk(self, agent: str, poi: str, tick: int) -> list[dict]:
        if not self.emit_motion or self._poi.get(agent) == poi:
            return []
        self._poi[agent] = poi
        return [self._envelope("agent.moved", {"agentId": agent, "to": {"poi": poi}}, tick)]

    def _voice(self, record: StepRecord, tick: int, *, speaker: str,
               place: str, text: str) -> list[dict]:
        if not self.emit_voices:
            return []
        said = " ".join((text or "").split())
        if not said:
            return []
        self._said += 1
        return [self._envelope("utterance.spoken",
                               {"agentId": speaker, "text": said[:600],
                                "place": {"poi": place}}, tick)]

    def feed(self, record: StepRecord) -> list[dict]:
        out: list[dict] = []
        # Тік не має йти назад: `verify.py` емітить `tick=0` без контексту кроку, і без цього
        # `run.done` віддавав би нуль, а сцена бачила б стрибок у минуле після третього тіку.
        tick = max(record.tick, self._last_tick)
        out += self._tick_events(tick)

        if record.agent == "tool":
            parsed = record.parsed or {}
            # Ритуал: спершу ЛЮДИНА йде туди, де про це можна дізнатись, і аж тоді питає. Без цього
            # виклик інструмента на сцені невидимий — його «бачить» лише лог.
            if record.span:
                out += self._walk(villager_of_span(record.span),
                                  POI_OF_TOOL.get(str(parsed.get("tool")), "well"), tick)
            out.append(self._envelope("tool.result", {
                "tool": parsed.get("tool", "?"),
                "ok": bool(parsed.get("ok", False)),
                "found": parsed.get("found"),
                **({"error": str(parsed["error"])} if parsed.get("error") else {}),
            }, tick))
            return out

        # Три типи, які фронт умів малювати від початку, а ядро ніколи не надсилало: план дня,
        # рефлексія, хроніка. Вони приходили ЛИШЕ з фікстури — та сама дірка, що з `casting.done`,
        # тільки з іншого боку. Тепер їх виробляє ядро.
        if record.agent == "deed":
            payload = record.parsed or {}
            who, deed = str(payload.get("agentId") or ""), str(payload.get("дія") or "")
            if who and deed and self.emit_motion:
                out.append(self._envelope("agent.moved", {
                    "agentId": who, "to": {"poi": self._poi.get(who, DEFAULT_POI)},
                    "activity": deed}, tick))
            return out

        if record.agent == "planner":
            payload = record.parsed or {}
            if payload.get("agentId") and payload.get("summary"):
                out.append(self._envelope("plan.formed", {
                    "agentId": str(payload["agentId"]), "summary": str(payload["summary"]),
                    **({"steps": [str(s) for s in payload["steps"]][:24]}
                       if payload.get("steps") else {}),
                }, tick))
            return out

        if record.agent == "thinker":
            payload = record.parsed or {}
            if payload.get("agentId") and str(payload.get("thought") or "").strip():
                out.append(self._envelope("reflection.formed", {
                    "agentId": str(payload["agentId"]),
                    "thought": str(payload["thought"]).strip()}, tick))
            return out

        if record.agent == "rumour":
            payload = record.parsed or {}
            claim, who = str(payload.get("claim") or ""), str(payload.get("who") or "")
            if claim:
                # `description` — ОБОВʼЯЗКОВЕ поле контракту. Без нього конверт не проходив
                # валідацію, а строгий парсер фронта мовчки викидав подію: чутки й ухвали просто
                # не доїжджали на Дошку. Заміряно на живому потоці: 33 з 33 `event.happened`
                # були невалідні.
                out.append(self._envelope("event.happened", {"event": {
                    "id": f"rumour-{self.run_id}", "kind": "rumour", "label": claim,
                    "description": f"чутку пустив {who}" if who else "чутка ходить селом",
                    **({"involves": [who]} if who else {}),
                }}, tick))
            return out

        if record.agent == "council":
            payload = record.parsed or {}
            label, who, poi = (str(payload.get("label") or ""), str(payload.get("who") or ""),
                               str(payload.get("poi") or ""))
            if label and poi:
                out.append(self._envelope("event.happened", {"event": {
                    "id": f"decision-{self.run_id}", "kind": "decision", "label": label,
                    "description": (f"ухвалено вічем, доручено: {who}" if who
                                    else "ухвалено вічем"),
                    "place": {"poi": poi},
                    **({"involves": [who]} if who else {}),
                }}, tick))
                # ★ Наслідок, а не лише напис: доручений СТАЄ на місце й лишається там.
                if who:
                    out += self._walk(who, poi, tick)
            return out

        if record.agent == "chronicler":
            payload = record.parsed or {}
            if payload.get("mood"):
                out.append(self._envelope("report.compiled", {"chronicle": {
                    "day": int(payload.get("day") or 1),
                    "title": str(payload.get("title") or ""),
                    "narration": str(payload.get("narration") or ""),
                    "mood": payload["mood"],
                    **({"highlights": [str(h) for h in payload["highlights"]][:5]}
                       if payload.get("highlights") else {}),
                }}, tick))
            return out

        if record.agent == "notebook":
            if record.stage != "mem_read":
                return out
            payload = record.parsed or {}
            items = payload.get("items") or payload.get("notes") or []
            out += self._motion(record.stage, tick)
            out.append(self._envelope("memory.recalled",
                                      {"items": [str(x) for x in items][:20]} | (
                                          {"query": str(payload["query"])}
                                          if payload.get("query") else {}), tick))
            return out

        if record.agent == "subagent":
            speaker = villager_of_span(record.span or record.run_id)
            # Маршрут ОГОЛОШУЄМО і для голосу. Без цього спостерігач бачив лише кроки оркестратора,
            # тобто розкладка ярусів на екрані показувала `{mamay: 3}` і нуль Lapa — при тому що
            # кожну репліку в вічі промовляє саме Lapa. Прилад, який недораховує виконавця, бреше.
            out.append(self._envelope("route.decided", {
                "stage": record.stage, "lane": _lane(record.lane), "model": record.model}, tick))
            out += self._walk(speaker, DEFAULT_POI, tick)
            out += self._voice(record, tick, speaker=speaker, place=DEFAULT_POI,
                               text=record.raw_output)
            return out

        if record.agent == "verifier":
            parsed = record.parsed or {}
            out += self._motion(record.stage, tick)
            out.append(self._envelope("verify.verdict", {
                "kind": parsed.get("kind", "supported"),
                "accepted": bool(parsed.get("accepted", False)),
                **({"reason": parsed["reason"]} if parsed.get("reason") else {}),
            }, tick))
            if parsed.get("reason"):
                out += self._voice(record, tick, speaker=VOICE_VERIFIER, place="church",
                                   text=str(parsed["reason"]))
            return out

        out += self._motion(record.stage, tick)
        out.append(self._envelope("route.decided", {
            "stage": record.stage, "lane": _lane(record.lane), "model": record.model,
            **({"tier": record.ablation["tier"]} if record.ablation.get("tier") else {}),
        }, tick))

        call = record.parsed or {}
        tool = call.get("tool")
        # `final_answer` — термінатор циклу, а не інструмент даних: подія про його «виклик» лише
        # засмічувала б потік і ламала парність tool.called/tool.result.
        if not tool or tool == FINAL_TOOL:
            if is_prose(record):
                out += self._voice(record, tick,
                                   speaker=VOICE_OF_LANE.get(_lane(record.lane), VOICE_FALLBACK),
                                   place=poi_of_stage(record.stage), text=record.raw_output)
            return out
        args = {k: v for k, v in call.items() if k != "tool"}
        out.append(self._envelope("tool.called",
                                  {"tool": tool} | ({"args": args} if args else {}), tick))
        return out

    def close(self, result: Any, *, done: bool = False) -> list[dict]:
        out: list[dict] = []
        tick = self._last_tick
        for note in getattr(result, "notes", []) or []:
            # `notes` має ДВА значення в різних виробниках: в оркестраторі це причини
            # перепланування (речення), у вічі — діагностика (`beats=17`, `voices=7`, тег `viche`).
            # Без розрізнення сцена писала «передумали: beats=17» і витісняла справжні події.
            # Правило: подія лише для НАРАТИВНОЇ нотатки, тобто такої, що є фразою.
            if " " in str(note).strip():
                out.append(self._envelope("plan.revised", {"reason": str(note)}, tick))
        scratch = list(getattr(result, "scratch", []) or [])
        out.append(self._envelope("task.outcome", {
            "outcome": getattr(result, "outcome", "answer"),
            "evidence": getattr(result, "evidence", evidence_state(scratch)),
            **({"verdictKind": result.verdict_kind}
               if getattr(result, "verdict_kind", None) else {}),
            **({"incidents": list(result.incidents)}
               if getattr(result, "incidents", None) else {}),
        }, tick))
        if done:
            # `events`/`reflections` — нуль не як заглушка: ядро не емітить `event.happened` і
            # `reflection.formed`, тож будь-яке інше число тут було б вигаданим.
            out.append(self._envelope("run.done", {
                "ticks": tick,
                "tokens": int(getattr(result, "tokens", 0) or 0)
                + int(getattr(result, "aux_tokens", 0) or 0),
                "counts": {"utterances": self._said, "events": 0, "reflections": 0},
            }, tick))
        return out


def _synthesize_tool_records(records: list[StepRecord], result: Any) -> list[StepRecord]:
    """Старі траси не мають `tool_result`; відтворюємо їх зі `scratch` у тому самому порядку."""
    if any(r.agent == "tool" for r in records):
        return list(records)
    scratch = list(getattr(result, "scratch", []) or [])
    out: list[StepRecord] = []
    used = 0
    for record in records:
        out.append(record)
        if record.agent in ("notebook", "verifier", "tool"):
            continue
        call = record.parsed or {}
        tool = call.get("tool")
        if not tool or tool == FINAL_TOOL:
            continue
        if used >= len(scratch):
            continue
        entry = scratch[used]
        used += 1
        value = entry.get("result")
        broken = isinstance(value, dict) and "error" in value
        out.append(StepRecord(
            run_id=record.run_id, tick=record.tick, agent="tool", stage="tool_result",
            model="tool", lane="none", prompt="", raw_output="",
            parsed={"tool": tool, "ok": not broken,
                    "found": entry.get("found", found_in(value)),
                    **({"error": str(value["error"])} if broken else {})},
            schema_valid=True, world_valid=not broken,
        ))
    return out


def project_run(records: list[StepRecord], result: Any, *, run_id: str, ts: str,
                scene: dict | None = None, started_at: str | None = None) -> list[dict]:
    """Пакетна проєкція — обгортка над `StreamProjector` для сумісності зі старими трасами."""
    ticks = max(1, len([r for r in records if r.agent != "tool"]))
    proj = StreamProjector(run_id, ts, scene=scene, started_at=started_at, max_ticks=ticks,
                           emit_motion=False, emit_ticks=False, emit_voices=False)
    events = proj.start()
    for record in _synthesize_tool_records(records, result):
        events += proj.feed(record)
    events += proj.close(result)
    return events
