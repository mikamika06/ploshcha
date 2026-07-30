import type { ParsedEvent } from "@ploshcha/contract-ts";
import type { EventSourcePort } from "./types";
import { parseEnvelope } from "./validate";

/** Відтворює записаний прогін (*.jsonl) з фіксованою затримкою. Нуль токенів. */
export class FixtureDriver implements EventSourcePort {
  private stopped = false;
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(private lines: string[], private delayMs: number) {}

  subscribe(onEvent: (ev: ParsedEvent) => void, onEnd?: () => void): () => void {
    let i = 0;
    const step = () => {
      if (this.stopped) return;
      while (i < this.lines.length && !this.lines[i].trim()) i++;
      if (i >= this.lines.length) {
        onEnd?.();
        return;
      }
      const ev = parseEnvelope(this.lines[i++]);
      if (ev) onEvent(ev);
      this.timer = setTimeout(step, this.delayMs);
    };
    this.timer = setTimeout(step, 400);
    return () => {
      this.stopped = true;
      if (this.timer) clearTimeout(this.timer);
    };
  }
}
