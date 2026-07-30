import { parseLine, type ParsedEvent } from "@ploshcha/contract-ts";

/**
 * Рантайм-гард: конверт валідується строго, невідомий тип ВИЖИВАЄ як `known: false`.
 *
 * Раніше будь-який новий тип падав у `null` і зникав. Тому додати подію означало зламати старий
 * фронт, тобто контракт лише називався additive. Тепер невідоме доїжджає до споживача, і рішення
 * «показувати чи ні» ухвалює сцена, а не парсер.
 */
export function parseEnvelope(line: string): ParsedEvent | null {
  const out = parseLine(line);
  if (out.ok) {
    return out.event;
  }
  if (out.reason === "protocol") {
    console.warn("[net] чужий мажор протоколу", out.detail);
  } else {
    console.warn(`[net] dropped event (${out.reason})`, out.detail ?? "");
  }
  return null;
}

/** Невідомий тип — не помилка, але й не дані для сцени: споживач мусить вирішувати явно. */
export function isKnown(ev: ParsedEvent): boolean {
  return ev.known;
}
