import type { PloshchaEvent } from "@ploshcha/contract-ts";

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
  sceneName?: string;
  villagers: Map<string, VillagerState>;
  timeOfDay?: string;
  mood?: { valence: number; label: string };
  day?: number;
  narration?: string;
}

export type StoreListener = (ev: PloshchaEvent, state: ViewState) => void;

/** Редьюсер подій контракту → queryable ViewState. Слухачі реагують імперативно й/або перечитують стан. */
export class SimStore {
  state: ViewState = { runStatus: "idle", villagers: new Map() };
  private listeners: StoreListener[] = [];

  on(l: StoreListener): void {
    this.listeners.push(l);
  }

  apply(ev: PloshchaEvent): void {
    const s = this.state;
    switch (ev.type) {
      case "run.started":
        s.runStatus = "running";
        s.sceneName = ev.payload.scene.name;
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
      default:
        break;
    }
    for (const l of this.listeners) l(ev, s);
  }
}
