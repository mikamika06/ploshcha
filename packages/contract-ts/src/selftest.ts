/**
 * Самотест двошарового розбору. Не фікстури, а саме правила: що виживає, що падає й чому.
 *
 * Інваріант, який тут стережеться: НЕВІДОМИЙ тип мусить пережити розбір. Доки він падав у `null`,
 * додати подію означало зламати старий фронт — тобто контракт лише називався additive.
 * Запуск: `pnpm --filter @ploshcha/contract-ts test`
 */
import { PROTOCOL, PROTOCOL_MAJOR, RunOutcome, VerdictKind, parseEvent, parseLine } from "./events";

let failed = 0;

function check(name: string, cond: boolean, detail?: unknown): void {
  if (cond) {
    console.log(`✓ ${name}`);
  } else {
    failed++;
    console.error(`✖ ${name}`, detail ?? "");
  }
}

const shell = {
  protocol: PROTOCOL,
  runId: "selftest",
  seq: 0,
  ts: "2026-07-30T10:00:00Z",
  tick: 0,
};

// --- шар 1: конверт ---
check("рядок-не-JSON → reason=json", parseLine("{зламано").ok === false);
check(
  "конверт без runId → reason=envelope",
  (() => {
    const r = parseEvent({ protocol: PROTOCOL, seq: 0, ts: shell.ts, tick: 0, type: "run.error", payload: {} });
    return !r.ok && r.reason === "envelope";
  })(),
);
check(
  "чужий мажор протоколу → reason=protocol",
  (() => {
    const r = parseEvent({ ...shell, protocol: "9.0.0", type: "run.error", payload: { message: "x" } });
    return !r.ok && r.reason === "protocol";
  })(),
);
check(
  "той самий мажор, вищий мінор → приймається",
  parseEvent({ ...shell, protocol: `${PROTOCOL_MAJOR}.99.0`, type: "run.error", payload: { message: "x" } }).ok,
);

// --- шар 2: payload відомого типу ---
check(
  "відомий тип із поламаним payload → reason=payload",
  (() => {
    const r = parseEvent({ ...shell, type: "verify.verdict", payload: { kind: "щось", accepted: true } });
    return !r.ok && r.reason === "payload";
  })(),
);
check(
  "відомий тип, валідний payload → known=true",
  (() => {
    const r = parseEvent({ ...shell, type: "verify.verdict", payload: { kind: "abstain", accepted: true } });
    return r.ok && r.event.known === true;
  })(),
);

// --- ★ головне: невідомий тип виживає ---
const future = { ...shell, type: "belief.challenged", payload: { anything: [1, 2, 3] } };
check(
  "невідомий тип → known=false, а не null",
  (() => {
    const r = parseEvent(future);
    return r.ok && r.event.known === false;
  })(),
);
check(
  "у невідомого типу конверт розібраний, payload лишається сирим",
  (() => {
    const r = parseEvent(future);
    if (!r.ok || r.event.known) return false;
    return r.event.seq === 0 && r.event.type === "belief.challenged" && r.event.payload !== undefined;
  })(),
);
check(
  "невідомий тип із поламаним КОНВЕРТОМ усе одно падає",
  parseEvent({ type: "belief.challenged", payload: {} }).ok === false,
);

// --- стани, які не мають злитися ---
check("abstain існує окремо від failure", RunOutcome.options.includes("abstain") && RunOutcome.options.includes("failure"));
check("no_evidence існує окремо від supported", VerdictKind.options.includes("no_evidence"));
check(
  "abstain НЕ є run.error: це різні типи подій",
  (() => {
    const r = parseEvent({ ...shell, type: "task.outcome", payload: { outcome: "abstain", evidence: false } });
    return r.ok && r.event.known === true && r.event.type === "task.outcome";
  })(),
);

console.log(failed ? `\n✖ ${failed} перевірк(и) впали` : `\n✓ двошаровий розбір тримає всі інваріанти`);
process.exit(failed ? 1 : 0);
