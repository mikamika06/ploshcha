import type { PloshchaEvent } from "@ploshcha/contract-ts";

export interface VillagerState {
  id: string;
  name: string;
  role: string;
  bio: string;
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
          s.villagers.set(v.id, { id: v.id, name: v.name, role: v.role, bio: v.bio });
        }
        break;
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
