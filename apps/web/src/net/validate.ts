import { PloshchaEvent } from "@ploshcha/contract-ts";

/** Рантайм-гард: розбирає рядок і валідує проти zod-контракту. Невалідне → null (+лог). */
export function parseEnvelope(line: string): PloshchaEvent | null {
  let obj: unknown;
  try {
    obj = JSON.parse(line);
  } catch {
    console.warn("[net] invalid JSON line");
    return null;
  }
  const r = PloshchaEvent.safeParse(obj);
  if (!r.success) {
    const t = (obj as { type?: string }).type;
    console.warn("[net] dropped invalid event", t, r.error.issues[0]);
    return null;
  }
  return r.data;
}
