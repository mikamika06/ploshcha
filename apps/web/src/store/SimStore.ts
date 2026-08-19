import type { ParsedEvent, RunOutcome, VerdictKind } from "@ploshcha/contract-ts";

export interface VillagerState {
  id: string;
  name: string;
  role: string;
  bio: string;
  activity?: string;
  location?: string; // поточний POI (з agent.moved) → зайнятість локацій
  plan?: { summary: string; steps: string[] };
  thoughts: string[]; // рефлексії (внутрішній монолог) — analytics
  said: string[]; // сказане вголос
}

export interface ViewState {
  runStatus: "idle" | "running" | "done" | "error";
  /** Термінальний стан задачі. `abstain` — ЧЕСНА ВІДМОВА, і це не помилка: ядро розрізняє
   *  «відповів» / «відмовився» / «зламався», тож UI не має права зливати друге з третім. */
  outcome?: RunOutcome;
  verdictKind?: VerdictKind;
  verdictReason?: string;
  /** Скільком подіям цей фронт не знає типу: контракт additive, тож це нормально, але видимо. */
  unknownEvents: number;
  sceneName?: string;
  villagers: Map<string, VillagerState>;
  /** Стенограма ОДНОГО прогону, НЕЗАЛЕЖНО від касту: у живому потоці немає `casting.*`, тому
   *  голос селянина, якого мапа ще не знає, інакше тихо зникав би. Скидається на `run.started`:
   *  накопичена між прогонами стенограма показувала б на нову тему репліки з минулої. */
  transcript: { agentId: string; text: string }[];
  runId?: string;
  /** Когнітивний шар. Ядро надсилало це від самого початку, а стор не мав жодного `case` — тобто
   *  видно було ЛИШЕ розмову й ходьбу, а чому вона така, не видно взагалі. */
  tools: { tool: string; ok?: boolean; found?: boolean | null }[];
  lanes: Record<string, number>;
  recalled: string[];
  degraded?: { stage: string; reason?: string };
  timeOfDay?: string;
  mood?: { valence: number; label: string };
  day?: number;
  narration?: string;
}

export type StoreListener = (ev: ParsedEvent, state: ViewState) => void;

/** Редьюсер подій контракту → queryable ViewState. Слухачі реагують імперативно й/або перечитують стан. */
export class SimStore {
  state: ViewState = { runStatus: "idle", villagers: new Map(), unknownEvents: 0,
                      transcript: [], tools: [], lanes: {}, recalled: [] };
  private listeners: StoreListener[] = [];

  on(l: StoreListener): void {
    this.listeners.push(l);
  }

  apply(ev: ParsedEvent): void {
    const s = this.state;
    if (!ev.known) {
      s.unknownEvents++;
      for (const l of this.listeners) l(ev, s);
      return;
    }
    switch (ev.type) {
      case "run.started":
        s.runStatus = "running";
        s.sceneName = ev.payload.scene.name;
        s.runId = ev.runId;
        s.transcript = [];
        s.tools = [];
        s.lanes = {};
        s.recalled = [];
        s.degraded = undefined;
        s.outcome = undefined;
        s.verdictKind = undefined;
        s.verdictReason = undefined;
        break;
      case "route.decided":
        s.lanes[ev.payload.lane] = (s.lanes[ev.payload.lane] ?? 0) + 1;
        break;
      case "tool.called":
        s.tools.push({ tool: ev.payload.tool });
        break;
      case "tool.result": {
        // Останній виклик того самого інструмента дістає свій результат; так пара
        // `called`/`result` лишається парою й у стані, а не лише в потоці.
        const open = [...s.tools].reverse().find((t) => t.tool === ev.payload.tool && t.ok === undefined);
        if (open) {
          open.ok = ev.payload.ok;
          open.found = ev.payload.found;
        } else {
          s.tools.push({ tool: ev.payload.tool, ok: ev.payload.ok, found: ev.payload.found });
        }
        break;
      }
      case "memory.recalled":
        s.recalled.push(...ev.payload.items);
        break;
      case "run.degraded":
        s.degraded = { stage: ev.payload.stage, reason: ev.payload.reason };
        break;
      case "casting.done":
        for (const v of ev.payload.cast) {
          s.villagers.set(v.id, { id: v.id, name: v.name, role: v.role, bio: v.bio, thoughts: [], said: [] });
        }
        break;
      case "agent.moved": {
        const v = s.villagers.get(ev.payload.agentId);
        if (v) {
          if (ev.payload.to.poi) v.location = ev.payload.to.poi; // де селянин зараз
          if (ev.payload.activity) v.activity = ev.payload.activity;
        }
        break;
      }
      case "plan.formed": {
        const v = s.villagers.get(ev.payload.agentId);
        if (v) v.plan = { summary: ev.payload.summary, steps: ev.payload.steps ?? [] };
        break;
      }
      case "reflection.formed": {
        const v = s.villagers.get(ev.payload.agentId);
        if (v) v.thoughts.push(ev.payload.thought);
        break;
      }
      case "utterance.spoken": {
        const v = s.villagers.get(ev.payload.agentId);
        if (v) v.said.push(ev.payload.text);
        s.transcript.push({ agentId: ev.payload.agentId, text: ev.payload.text });
        break;
      }
      case "tick.begin":
        s.timeOfDay = ev.payload.timeOfDay;
        if (ev.payload.mood) s.mood = ev.payload.mood;
        break;
      case "report.compiled":
        s.day = ev.payload.chronicle.day;
        s.narration = ev.payload.chronicle.narration;
        s.mood = ev.payload.chronicle.mood;
        break;
      case "run.done":
        s.runStatus = "done";
        break;
      case "run.error":
        s.runStatus = "error";
        break;
      case "verify.verdict":
        s.verdictKind = ev.payload.kind;
        s.verdictReason = ev.payload.reason;
        break;
      case "task.outcome":
        s.outcome = ev.payload.outcome;
        if (ev.payload.verdictKind) s.verdictKind = ev.payload.verdictKind;
        break;
      default:
        break;
    }
    for (const l of this.listeners) l(ev, s);
  }
}
