import { z } from "zod";

/**
 * ПЛОЩА event contract (v1) — дзеркало contracts/ploshcha-events.schema.json.
 * Це рантайм-гард фронту (spectator-проєкція) і форма, яку «поважає» бек.
 */

export const PROTOCOL = "1.0.0";

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
]);
export type PloshchaEvent = z.infer<typeof PloshchaEvent>;
export type EventType = PloshchaEvent["type"];

/** Усі 13 типів — для перевірки покриття у фікстурах/тестах. */
export const EVENT_TYPES = [
  "run.started", "casting.begin", "casting.done", "tick.begin", "plan.formed", "agent.moved",
  "utterance.spoken", "event.happened", "reflection.formed", "report.compiled", "run.degraded",
  "run.done", "run.error",
] as const;
