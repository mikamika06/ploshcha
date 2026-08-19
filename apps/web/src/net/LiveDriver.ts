import type { ParsedEvent } from "@ploshcha/contract-ts";
import type { EventSourcePort } from "./types";
import { parseEnvelope } from "./validate";

/**
 * Живий потік із ядра через SSE. Друга реалізація того самого порту, що `FixtureDriver` —
 * саме для цього порт і існував: решта фронта не знає, звідки беруться події.
 *
 * Курсор тримаємо самі: браузер шле `Last-Event-ID` лише при АВТОМАТИЧНОМУ реконекті, а ми
 * перепідключаємось і вручну (після `stop`), тому позиція мусить жити в нашому стані.
 */
/** Ядро віддає всю розмову за секунди — прочитати неможливо. Гальмувати ЯДРО не можна: воно має
 *  лишатись швидким для замірів. Тому притримуємо тут: репліка раз на `PACE_MS`, а рух і службові
 *  події — одразу. Якщо черга виросла (наздоганяємо після паузи), темп прискорюється. */
const PACE_MS = 1400;
const PACE_MIN_MS = 260;
const PACE_BACKLOG = 6;
const PACED = new Set(["utterance.spoken"]);

export class LiveDriver implements EventSourcePort {
  private source: EventSource | undefined;
  private stopped = false;
  private cursor: number | undefined;
  private queue: ParsedEvent[] = [];
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(private url: string) {}

  private drain(onEvent: (ev: ParsedEvent) => void): void {
    if (this.timer !== undefined || this.stopped) return;
    const next = this.queue.shift();
    if (!next) return;
    onEvent(next);
    const delay = Math.max(PACE_MIN_MS, PACE_MS / Math.max(1, this.queue.length - PACE_BACKLOG + 1));
    this.timer = setTimeout(() => {
      this.timer = undefined;
      this.drain(onEvent);
    }, delay);
  }

  subscribe(onEvent: (ev: ParsedEvent) => void, onEnd?: () => void): () => void {
    const open = () => {
      if (this.stopped) return;
      const url = this.cursor === undefined ? this.url : `${this.url}?since=${this.cursor}`;
      const source = new EventSource(url);
      this.source = source;

      source.onmessage = (msg: MessageEvent<string>) => {
        if (msg.lastEventId) {
          const id = Number(msg.lastEventId);
          if (Number.isFinite(id)) this.cursor = id;
        }
        const ev = parseEnvelope(msg.data);
        if (!ev) return;
        if (ev.known && PACED.has(ev.type)) {
          this.queue.push(ev);
          this.drain(onEvent);
        } else {
          onEvent(ev);
        }
      };

      source.onerror = () => {
        source.close();
        if (this.stopped) return;
        // Браузер сам перепідключається лише поки зʼєднання «живе». Після close() — наша справа.
        setTimeout(open, 1000);
      };
    };

    open();
    return () => {
      this.stopped = true;
      if (this.timer !== undefined) clearTimeout(this.timer);
      this.timer = undefined;
      this.queue = [];
      this.source?.close();
      onEnd?.();
    };
  }
}

/** Команда назад у ядро. Джерело істини — черга ядра, локальний стан лише оптимістичний. */
export async function sendCommand(baseUrl: string, body: Record<string, unknown>): Promise<unknown> {
  const res = await fetch(`${baseUrl}/command`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`команда відхилена: ${res.status}`);
  return res.json();
}
