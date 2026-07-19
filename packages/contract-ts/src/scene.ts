import { z } from "zod";

/** SceneSpec — опис статичної сцени-села. Дзеркало contracts/scene.schema.json. */

export const POIKind = z.enum([
  "well", "church", "mill", "forge", "tavern", "board", "bell", "home", "field", "pond", "square", "gate",
]);
export type POIKind = z.infer<typeof POIKind>;

export const POI = z
  .object({
    id: z.string().regex(/^[a-z0-9_]+$/),
    name: z.string(),
    kind: POIKind,
    x: z.number(),
    y: z.number(),
    meaning: z.string().optional(),
    provisional: z.boolean().optional(),
  })
  .strict();
export type POI = z.infer<typeof POI>;

/** Декоративний спрайт на карті (assets/nb/<sprite>.png), розставлений редактором розкладки. */
export const SceneProp = z
  .object({
    sprite: z.string(),
    x: z.number(),
    y: z.number(),
    scale: z.number().default(1),
    rot: z.number().default(0), // оберт у градусах
    anchor: z.tuple([z.number(), z.number()]).default([0.5, 1]),
    z: z.number().optional(), // ключ глибини (нативний y); без нього — рахується з низу спрайта
  })
  .strict();
export type SceneProp = z.infer<typeof SceneProp>;

export const SceneSpec = z
  .object({
    id: z.string(),
    name: z.string(),
    size: z.object({ w: z.number().int().min(1), h: z.number().int().min(1) }).strict(),
    background: z.string(),
    masks: z
      .object({
        walk: z.string(),
        zone: z.string().optional(),
        keepout: z.string().optional(),
        flow: z.string().optional(),
        space: z.object({ w: z.number().int(), h: z.number().int() }).strict(),
      })
      .strict(),
    pois: z.array(POI).min(1),
    props: z.array(SceneProp).optional(),
  })
  .strict();
export type SceneSpec = z.infer<typeof SceneSpec>;
