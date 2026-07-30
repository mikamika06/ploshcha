import { z } from "zod";

/**
 * ПЛОЩА event contract — дзеркало contracts/ploshcha-events.schema.json.
 * Це рантайм-гард фронту (spectator-проєкція) і форма, яку «поважає» бек.
 *
 * 1.1.0 (когнітивний ярус). Додано СІМ типів, кожен під дані, які ядро вже віддає сьогодні —
 * нічого спекулятивного. Головне тут не список, а два інваріанти:
 *
 * 1. `outcome=abstain` — окремий СТАН, не помилка. Ядро вже розрізняє «відповів» / «чесно
 *    відмовився» / «зламався»; якщо контракт цього не несе, UI покаже чесну відмову як збій —
 *    рівно та помилка, що жила в метриці (K9: страта відмов читалась 0/8 при 8/8 правильних).
 * 2. Розбір ДВОШАРОВИЙ: конверт валідується строго, невідомий тип лишається raw і доїжджає до
 *    споживача. Інакше кожен новий тип ламає старий фронт і CI, і контракт перестає бути additive.
 */

export const PROTOCOL = "1.1.0";
export const PROTOCOL_MAJOR = 1;

// ---------- shared ----------
export const TimeOfDay = z.enum(["dawn", "morning", "noon", "evening", "dusk", "night"]);
export type TimeOfDay = z.infer<typeof TimeOfDay>;

export const CastingMode = z.enum(["library", "generate", "mixed"]);
export type CastingMode = z.infer<typeof CastingMode>;

export const MoodView = z.object({ valence: z.number().min(-1).max(1), label: z.string() }).strict();
export type MoodView = z.infer<typeof MoodView>;

export const PlaceRef = z
  .object({ poi: z.string().optional(), x: z.number().optional(), y: z.number().optional() })
  .strict()
  .refine((v) => v.poi !== undefined || (v.x !== undefined && v.y !== undefined), {
    message: "PlaceRef needs `poi` or (`x`,`y`)",
  });
export type PlaceRef = z.infer<typeof PlaceRef>;

export const VillagerPublic = z
  .object({
    id: z.string().regex(/^[a-z0-9_]+$/),
    name: z.string(),
    role: z.string(),
    bio: z.string(),
    traits: z.array(z.string()).optional(),
    home: z.string().optional(),
  })
  .strict();
export type VillagerPublic = z.infer<typeof VillagerPublic>;

export const SceneRef = z.object({ id: z.string(), name: z.string() }).strict();

export const PublicConfig = z
  .object({
    maxTicks: z.number().int().min(1),
    castingMode: CastingMode,
    season: z.string().optional(),
    seed: z.number().int().optional(),
  })
  .strict();

export const VillageEvent = z
  .object({
    id: z.string(),
    kind: z.string(),
    label: z.string(),
    description: z.string(),
    place: PlaceRef.optional(),
    involves: z.array(z.string()).optional(),
  })
  .strict();
export type VillageEvent = z.infer<typeof VillageEvent>;

export const RelationDelta = z
  .object({ a: z.string(), b: z.string(), change: z.enum(["closer", "apart", "neutral"]), note: z.string().optional() })
  .strict();

export const DayChronicle = z
  .object({
    day: z.number().int().min(1),
    title: z.string(),
    narration: z.string(),
    highlights: z.array(z.string()).optional(),
    mood: MoodView,
    relationships: z.array(RelationDelta).optional(),
  })
  .strict();
export type DayChronicle = z.infer<typeof DayChronicle>;

export const RunCounts = z
  .object({ utterances: z.number().int().min(0), events: z.number().int().min(0), reflections: z.number().int().min(0) })
  .strict();

// ---------- когнітивний ярус (1.1.0) ----------
/** Дзеркало `VerdictKind` з ploshcha_sim/agents/verify.py — паритет під тестом. */
export const VerdictKind = z.enum([
  "supported", "abstain", "unsupported", "contradicted", "parse_fail", "no_evidence",
]);
export type VerdictKind = z.infer<typeof VerdictKind>;

/** Дзеркало `Outcome` з ploshcha_sim/domain/evidence.py. `abstain` — стан, а не помилка. */
export const RunOutcome = z.enum(["answer", "abstain", "failure"]);
export type RunOutcome = z.infer<typeof RunOutcome>;

export const ModelLane = z.enum(["lapa", "mamay", "unknown"]);
export type ModelLane = z.infer<typeof ModelLane>;

/** `null` = питання незастосовне (інструмент не шукає, а обчислює) — те саме, що `ToolResult.found`. */
export const Found = z.boolean().nullable();

// ---------- envelope ----------
const envelope = {
  protocol: z.string().regex(/^\d+\.\d+\.\d+$/),
  runId: z.string().min(1).max(64).regex(/^[A-Za-z0-9._-]+$/),
  seq: z.number().int().min(0),
  ts: z.string(),
  tick: z.number().int().min(0),
  note: z.string().max(500).optional(),
};

const ev = <T extends string, P extends z.ZodTypeAny>(type: T, payload: P) =>
  z.object({ ...envelope, type: z.literal(type), payload }).strict();

// ---------- 13 events ----------
export const RunStarted = ev(
  "run.started",
  z.object({ config: PublicConfig, scene: SceneRef, startedAt: z.string() }).strict(),
);
export const CastingBegin = ev("casting.begin", z.object({ mode: CastingMode }).strict());
export const CastingDone = ev("casting.done", z.object({ cast: z.array(VillagerPublic).min(1) }).strict());
export const TickBegin = ev("tick.begin", z.object({ timeOfDay: TimeOfDay, mood: MoodView.optional() }).strict());
export const PlanFormed = ev(
  "plan.formed",
  z.object({ agentId: z.string(), summary: z.string(), steps: z.array(z.string()).optional() }).strict(),
);
export const AgentMoved = ev(
  "agent.moved",
  z.object({ agentId: z.string(), to: PlaceRef, activity: z.string().optional() }).strict(),
);
export const UtteranceSpoken = ev(
  "utterance.spoken",
  z
    .object({
      agentId: z.string(),
      to: z.array(z.string()).optional(),
      text: z.string().min(1),
      place: PlaceRef.optional(),
      tone: z.string().optional(),
    })
    .strict(),
);
export const EventHappened = ev("event.happened", z.object({ event: VillageEvent }).strict());
export const ReflectionFormed = ev(
  "reflection.formed",
  z.object({ agentId: z.string(), thought: z.string().min(1) }).strict(),
);
export const ReportCompiled = ev("report.compiled", z.object({ chronicle: DayChronicle }).strict());
export const RunDegraded = ev("run.degraded", z.object({ stage: z.string(), reason: z.string().optional() }).strict());
export const RunDone = ev(
  "run.done",
  z.object({ ticks: z.number().int().min(0), tokens: z.number().int().min(0), counts: RunCounts }).strict(),
);
export const RunError = ev("run.error", z.object({ message: z.string() }).strict());

export const RouteDecided = ev(
  "route.decided",
  z.object({ stage: z.string(), lane: ModelLane, model: z.string().optional(), tier: z.string().optional() }).strict(),
);
export const ToolCalled = ev(
  "tool.called",
  z.object({ agentId: z.string().optional(), tool: z.string(), args: z.record(z.unknown()).optional() }).strict(),
);
export const ToolResulted = ev(
  "tool.result",
  z.object({ tool: z.string(), ok: z.boolean(), found: Found.optional(), error: z.string().optional() }).strict(),
);
export const MemoryRecalled = ev(
  "memory.recalled",
  z.object({ agentId: z.string().optional(), items: z.array(z.string()), query: z.string().optional() }).strict(),
);
export const VerifyVerdict = ev(
  "verify.verdict",
  z
    .object({
      kind: VerdictKind,
      accepted: z.boolean(),
      reason: z.string().optional(),
      grounding: z.enum(["required", "optional"]).optional(),
    })
    .strict(),
);
export const PlanRevised = ev(
  "plan.revised",
  z.object({ agentId: z.string().optional(), reason: z.string(), rung: z.string().optional() }).strict(),
);
/** Термінальний стан задачі. `evidence`: true знайдено / false шукали й нема / null не шукали. */
export const TaskOutcome = ev(
  "task.outcome",
  z
    .object({
      outcome: RunOutcome,
      evidence: Found.optional(),
      verdictKind: VerdictKind.optional(),
      incidents: z.array(z.string()).optional(),
    })
    .strict(),
);

export const PloshchaEvent = z.discriminatedUnion("type", [
  RunStarted,
  CastingBegin,
  CastingDone,
  TickBegin,
  PlanFormed,
  AgentMoved,
  UtteranceSpoken,
  EventHappened,
  ReflectionFormed,
  ReportCompiled,
  RunDegraded,
  RunDone,
  RunError,
  RouteDecided,
  ToolCalled,
  ToolResulted,
  MemoryRecalled,
  VerifyVerdict,
  PlanRevised,
  TaskOutcome,
]);
export type PloshchaEvent = z.infer<typeof PloshchaEvent>;
export type EventType = PloshchaEvent["type"];

/** Усі типи — для перевірки покриття у фікстурах/тестах. */
export const EVENT_TYPES = [
  "run.started", "casting.begin", "casting.done", "tick.begin", "plan.formed", "agent.moved",
  "utterance.spoken", "event.happened", "reflection.formed", "report.compiled", "run.degraded",
  "run.done", "run.error",
  "route.decided", "tool.called", "tool.result", "memory.recalled", "verify.verdict",
  "plan.revised", "task.outcome",
] as const;

/** Типи когнітивного яруса — v1 їх не знав, тож старий споживач мусить їх ПЕРЕЖИТИ, а не впасти. */
export const COGNITION_TYPES = [
  "route.decided", "tool.called", "tool.result", "memory.recalled", "verify.verdict",
  "plan.revised", "task.outcome",
] as const;

// ---------- двошаровий розбір ----------
export const Envelope = z.object({ ...envelope, type: z.string().min(1), payload: z.unknown() });
export type Envelope = z.infer<typeof Envelope>;

/** Невідомий тип виживає: конверт розібраний, payload лишається сирим. */
export interface RawEvent extends Envelope {
  known: false;
}

export type ParsedEvent = (PloshchaEvent & { known: true }) | RawEvent;

export type ParseFailure = { ok: false; reason: "json" | "envelope" | "protocol" | "payload"; detail?: string };
export type ParseSuccess = { ok: true; event: ParsedEvent };
export type ParseOutcome = ParseSuccess | ParseFailure;

const KNOWN = new Set<string>(EVENT_TYPES);

/**
 * Шар 1 — конверт і мажор протоколу (строго). Шар 2 — payload відомого типу (строго);
 * невідомий тип віддається як `known: false`, бо additive-контракт не має ламати старого споживача.
 */
export function parseEvent(input: unknown): ParseOutcome {
  const shell = Envelope.safeParse(input);
  if (!shell.success) {
    return { ok: false, reason: "envelope", detail: shell.error.issues[0]?.message };
  }
  const major = Number(shell.data.protocol.split(".")[0]);
  if (major !== PROTOCOL_MAJOR) {
    return { ok: false, reason: "protocol", detail: shell.data.protocol };
  }
  if (!KNOWN.has(shell.data.type)) {
    return { ok: true, event: { ...shell.data, known: false } };
  }
  const full = PloshchaEvent.safeParse(input);
  if (!full.success) {
    return { ok: false, reason: "payload", detail: full.error.issues[0]?.message };
  }
  return { ok: true, event: { ...full.data, known: true } };
}

export function parseLine(line: string): ParseOutcome {
  try {
    return parseEvent(JSON.parse(line));
  } catch {
    return { ok: false, reason: "json" };
  }
}
