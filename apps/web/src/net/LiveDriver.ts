import type { ParsedEvent } from "@ploshcha/contract-ts";
import type { EventSourcePort } from "./types";
import { sessionId } from "./session";
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
/** ★ Кінець розмови СТОЇТЬ У ТІЙ САМІЙ ЧЕРЗІ, що й репліки.
 *
 *  Літопис їхав повз притримку, бо в `PACED` його не було, — і виходив на екран у ту мить, коли
 *  ядро його дописало. А закриття ядро робить пачкою: зведення старости, сумнів попа й голоси
 *  всього касту народжуються за секунди й лягають у чергу, яка грає по одній репліці на
 *  `PACE_MS`. Тому підсумок з ухвалою бачили ТОДІ, коли село ще казало «за» і «проти», — рівно
 *  те, на що скаржився гість. Пропустити його через чергу дешевше й чесніше за будь-який
 *  таймер: порядок подій тоді той самий, що в ядрі, і не залежить від швидкості мережі. */
const TAIL = new Set(["report.compiled"]);

export class LiveDriver implements EventSourcePort {
  private source: EventSource | undefined;
  private stopped = false;
  private cursor: number | undefined;
  private queue: ParsedEvent[] = [];
  private timer: ReturnType<typeof setTimeout> | undefined;
  /** Прогін, який іде ЗАРАЗ. Стара розмова не має доказувати себе поверх нової. */
  private run: string | undefined;

  /** `sid` — чиє село слухаємо. Ядро фільтрує потік ним, тож без нього прилетіли б чужі розмови. */
  constructor(private url: string, private sid: string = sessionId()) {
    // ★ Перше підключення бере СВОЮ історію з нуля, а не лише нові події.
    //
    // Село гостя переживає перезавантаження сторінки (сесія в localStorage, стан у ядрі), але
    // екран був порожній: потік починався з хвоста, тож усе сказане до перезавантаження зникало
    // разом із вкладкою. Ядро фільтрує потік сесією, тож із нуля прилітає рівно своє.
    if (this.sid) this.cursor = 0;
  }

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

  /**
   * Забути притримані репліки.
   *
   * ★ Розмова, яку гість завершив, не має доказувати себе ще півхвилини. Ядро віддає віче за
   * секунди, а сцена грає його по одній репліці на `PACE_MS`, тож у черзі стоять десятки чужих
   * уже реплік — і після «завершити» вони й далі спливали бульбашками над селом. Кінець прогону в
   * ядрі тут не поміч: черга вже на цьому боці, і забути її може лише цей бік.
   */
  drop(): void {
    this.queue = [];
    if (this.timer !== undefined) clearTimeout(this.timer);
    this.timer = undefined;
  }

  subscribe(onEvent: (ev: ParsedEvent) => void, onEnd?: () => void): () => void {
    const open = () => {
      if (this.stopped) return;
      const query = new URLSearchParams();
      if (this.sid) query.set("sid", this.sid);
      if (this.cursor !== undefined) query.set("since", String(this.cursor));
      const tail = query.toString();
      const url = tail ? `${this.url}?${tail}` : this.url;
      const source = new EventSource(url);
      this.source = source;

      source.onmessage = (msg: MessageEvent<string>) => {
        if (msg.lastEventId) {
          const id = Number(msg.lastEventId);
          if (Number.isFinite(id)) this.cursor = id;
        }
        const ev = parseEnvelope(msg.data);
        if (!ev) return;
        // ★ Нова розмова ЗМИВАЄ недограну стару.
        //
        // Фільтра за сесією тут не досить: обидва прогони належать тому самому гостю, тож ядро
        // чесно шле обидва. А притримка (`PACE_MS`) тримає в черзі до півхвилини реплік — після
        // нової теми вони доказувались уже під нею, і на екрані виходила одна розмова, зшита з
        // двох. Розрізняє їх номер прогону в конверті; чекати цього від сцени не можна, бо туди
        // подія доїде вже перемішаною.
        if (ev.known && ev.type === "run.started") {
          this.run = ev.runId;
          this.queue = this.queue.filter((q) => q.runId === ev.runId);
        }
        const held = PACED.has(ev.type) || TAIL.has(ev.type);
        if (this.run !== undefined && ev.runId !== this.run && held) return;
        if (ev.known && held) {
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

/**
 * Команда назад у ядро. Джерело істини — черга ядра, локальний стан лише оптимістичний.
 *
 * `sid` підставляється ТУТ, а не в кожному місці виклику: забути його означало б тихо змінити
 * спільне село замість свого, і помітно це стало б лише в чужій вкладці.
 */
export async function sendCommand(baseUrl: string, body: Record<string, unknown>): Promise<unknown> {
  const res = await fetch(`${baseUrl}/command`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sid: sessionId(), ...body }),
  });
  if (!res.ok) {
    // ★ Причину відмови несе САМЕ ядро, і вона осмислена («зараз віча немає», «занадто часто»).
    // Доти нагору їхав самий код статусу, а UI перекладав його в «ядро не відповідає» — тобто
    // казав, що звʼязку немає, тоді як звʼязок був, а слово відхилили з конкретної причини.
    let reason = "";
    try {
      reason = String(((await res.json()) as { error?: string }).error ?? "");
    } catch {
      reason = "";
    }
    throw new CommandRefused(res.status, reason || `команда відхилена: ${res.status}`);
  }
  return res.json();
}

/** Ядро відповіло і ВІДМОВИЛО. Це не те саме, що «ядро мовчить», і плутати їх не можна. */
export class CommandRefused extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = "CommandRefused";
  }
}

/** Стан ядра: чи воно взагалі працює. Без цього «село думає» триває вічно й мовчки. */
export interface Health {
  state: string;
  stoppedReason?: string | null;
  lastError?: string | null;
  queue?: { pending?: number; done?: number };
  spend?: { tokens?: number };
  caps?: { maxTokens?: number };
}

export async function fetchHealth(baseUrl: string): Promise<Health | null> {
  try {
    const res = await fetch(`${baseUrl}/health`);
    return res.ok ? ((await res.json()) as Health) : null;
  } catch {
    return null;
  }
}
