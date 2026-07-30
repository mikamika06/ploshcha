/**
 * Контракт-конформність: валідує кожен рядок фікстур проти zod-контракту,
 * перевіряє монотонність `seq` і рахує покриття 13 типів. Запуск: `pnpm validate`.
 */
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { EVENT_TYPES, COGNITION_TYPES, parseLine } from "./events";
import { SceneSpec } from "./scene";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..", "..");
const runsDir = join(root, "packages", "fixtures", "runs");
const scenesDir = join(root, "packages", "fixtures", "scenes");

let errors = 0;
let unknown = 0;
const seenTypes = new Set<string>();

// --- scenes ---
for (const f of readdirSync(scenesDir).filter((x) => x.endsWith(".json"))) {
  const data = JSON.parse(readFileSync(join(scenesDir, f), "utf8"));
  const r = SceneSpec.safeParse(data);
  if (!r.success) {
    errors++;
    console.error(`✖ scene ${f}:`, JSON.stringify(r.error.issues.slice(0, 3)));
  } else {
    console.log(`✓ scene ${f} — ${data.pois.length} POIs`);
  }
}

// --- runs (jsonl) ---
for (const f of readdirSync(runsDir).filter((x) => x.endsWith(".jsonl"))) {
  const lines = readFileSync(join(runsDir, f), "utf8").split("\n").filter((l) => l.trim());
  let expectSeq = 0;
  let ok = 0;
  let lineNo = 0;
  for (const line of lines) {
    lineNo++;
    let obj: unknown;
    try {
      obj = JSON.parse(line);
    } catch {
      errors++;
      console.error(`✖ ${f}:${lineNo} invalid JSON`);
      continue;
    }
    const parsed = parseLine(line);
    if (!parsed.ok) {
      errors++;
      console.error(`✖ ${f}:${lineNo} (${(obj as { type?: string }).type}): ${parsed.reason} ${parsed.detail ?? ""}`);
      continue;
    }
    const e = parsed.event;
    if (e.known) {
      seenTypes.add(e.type);
    } else {
      unknown++;
    }
    if (e.seq !== expectSeq) {
      errors++;
      console.error(`✖ ${f}:${lineNo} seq gap: expected ${expectSeq}, got ${e.seq}`);
    }
    expectSeq++;
    ok++;
  }
  console.log(`${ok === lines.length ? "✓" : "…"} run ${f} — ${ok}/${lines.length} events valid`);
}

// --- coverage ---
const missing = EVENT_TYPES.filter((t) => !seenTypes.has(t));
const cognition = COGNITION_TYPES.filter((t) => seenTypes.has(t));
console.log(`\nПокриття типів: ${seenTypes.size}/${EVENT_TYPES.length}${missing.length ? ` (нема: ${missing.join(", ")})` : " — усі"}`);
console.log(`Когнітивний ярус: ${cognition.length}/${COGNITION_TYPES.length}`);
console.log(`Невідомих типів пережито: ${unknown} (двошаровий розбір: конверт валідний, payload сирий)`);
console.log(errors ? `✖ ${errors} помилк(и)` : `✓ усі фікстури валідні`);
process.exit(errors ? 1 : 0);
