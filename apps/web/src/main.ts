import "./style.css";
import { Texture } from "pixi.js";
import { SceneSpec } from "@ploshcha/contract-ts";
import type { POI } from "@ploshcha/contract-ts";
import sceneJson from "@fixtures/scenes/verbolozy.scene.json";
import quietDayRaw from "@fixtures/runs/quiet-day.jsonl?raw";
import { SceneRenderer } from "./scene/SceneRenderer";
import type { VegTextures, VegType } from "./scene/Vegetation";
import { WalkGrid } from "./agents/WalkGrid";
import { AgentDirector } from "./agents/AgentDirector";
import { SimStore } from "./store/SimStore";
import { Narrator } from "./roles/Narrator";
import { ChatLog } from "./hud/ChatLog";
import { FixtureDriver } from "./net/FixtureDriver";
import { loadGraded, makeShadowTexture } from "./util/gfx";
import { REPLAY_MS } from "./config";

const TIME_UA: Record<string, string> = {
  dawn: "світанок",
  morning: "ранок",
  noon: "полудень",
  evening: "вечір",
  dusk: "сутінки",
  night: "ніч",
};
const clockLabel = (t: string): string => TIME_UA[t] ?? t;

function tuftUrls(dir: string, count: number): string[] {
  return Array.from({ length: count }, (_, i) => `/assets/nb/${dir}/0${i}.png`);
}

const VEG_SRC: Record<VegType, string[]> = {
  wheat: tuftUrls("tuft_wheat2", 6),
  flower: tuftUrls("tuft_flower2", 8),
  reed: tuftUrls("tuft_reed2", 6),
  grass: tuftUrls("tuft_grass", 6),
  tree: ["1_trees/00", "1_trees/01", "1_trees/02", "v2_trees/00", "v2_trees/01", "v2_trees/02"].map((s) => `/assets/nb/${s}.png`),
  bush: ["1_trees/03", "v2_trees/03", "v2_trees/04"].map((s) => `/assets/nb/${s}.png`),
};
const VEG_CAP: Partial<Record<VegType, number>> = { tree: 220, bush: 180 };

async function loadVegTextures(): Promise<VegTextures> {
  const out = {} as VegTextures;
  for (const key of Object.keys(VEG_SRC) as VegType[]) {
    const loaded = await Promise.all(VEG_SRC[key].map((u) => loadGraded(u, VEG_CAP[key] ?? 150).catch(() => null)));
    out[key] = loaded.filter((t): t is Texture => t !== null);
  }
  return out;
}

async function loadCharTextures(): Promise<Texture[]> {
  const urls: string[] = [];
  for (let i = 0; i < 6; i++) urls.push(`/assets/nb/c1_chars_a/0${i}.png`);
  for (let i = 0; i < 6; i++) urls.push(`/assets/nb/c2_chars_b/0${i}.png`);
  const loaded = await Promise.all(urls.map((u) => loadGraded(u, 130).catch(() => null)));
  return loaded.filter((t): t is Texture => t !== null);
}

async function boot(): Promise<void> {
  const scene = SceneSpec.parse(sceneJson);
  const SCL = scene.size.w / scene.masks.space.w;
  const shadowTex = makeShadowTexture();

  const renderer = new SceneRenderer(scene);
  renderer.mount(document.getElementById("frame")!);
  await renderer.loadGround();

  const grid = new WalkGrid(scene.masks.space.w, scene.masks.space.h, SCL);
  // NB: keepout маска = «не саджати» (стежки+хати+вода), а не «тут хата» — тому для
  // walk-grid її НЕ використовуємо (інакше вирізає всю ходьбу). Ходьба лише за walk2.
  await grid.load(scene.masks.walk);

  const [vegTex, chars] = await Promise.all([loadVegTextures(), loadCharTextures()]);
  await renderer.buildVegetation(vegTex, shadowTex);
  renderer.initWeather();
  renderer.playIntro();

  const pois = new Map<string, POI>();
  for (const p of scene.pois) pois.set(p.id, p);

  const director = new AgentDirector(renderer.world, grid, pois, chars, shadowTex, SCL);
  const narrator = new Narrator();
  const chat = new ChatLog();
  const store = new SimStore();

  store.on((ev, state) => {
    switch (ev.type) {
      case "run.started":
        narrator.say(`Ранок у селі ${ev.payload.scene.name}. Село прокидається…`);
        chat.setClock(ev.payload.scene.name);
        break;
      case "casting.done":
        director.spawn(ev.payload.cast);
        break;
      case "tick.begin":
        chat.setClock(clockLabel(ev.payload.timeOfDay));
        if (ev.payload.mood) renderer.weather?.setMood(ev.payload.mood.valence);
        break;
      case "agent.moved":
        director.moveTo(ev.payload.agentId, ev.payload.to);
        break;
      case "utterance.spoken": {
        const name = state.villagers.get(ev.payload.agentId)?.name ?? ev.payload.agentId;
        director.speak(ev.payload.agentId, ev.payload.text);
        chat.line(name, ev.payload.text);
        break;
      }
      case "event.happened":
        chat.sys(`${ev.payload.event.label}: ${ev.payload.event.description}`);
        break;
      case "reflection.formed": {
        const name = state.villagers.get(ev.payload.agentId)?.name ?? ev.payload.agentId;
        chat.line(`${name} (думка)`, ev.payload.thought);
        break;
      }
      case "report.compiled":
        narrator.say(ev.payload.chronicle.narration);
        chat.chronicle(ev.payload.chronicle.title);
        chat.setClock(`день ${ev.payload.chronicle.day}`);
        renderer.weather?.setMood(ev.payload.chronicle.mood.valence);
        break;
      case "run.done":
        chat.sys("Село засинає. Кінець дня.");
        break;
      case "run.degraded":
        chat.sys(`… деградація: ${ev.payload.stage}`);
        break;
      case "run.error":
        chat.sys(`⚠ Помилка: ${ev.payload.message}`);
        break;
      default:
        break;
    }
  });

  const hudEl = document.getElementById("hud");
  document.getElementById("hud-toggle")?.addEventListener("click", () => hudEl?.classList.toggle("collapsed"));

  renderer.app.ticker.add(() => {
    const dt = Math.min(renderer.app.ticker.deltaMS / 1000, 0.05);
    renderer.update(dt);
    director.update(dt);
  });

  const fpsEl = document.getElementById("fps");
  let fpsT = 0;
  renderer.app.ticker.add(() => {
    fpsT += renderer.app.ticker.deltaMS;
    if (fpsT > 500 && fpsEl) {
      fpsEl.textContent = String(Math.round(renderer.app.ticker.FPS));
      fpsT = 0;
    }
  });

  const lines = quietDayRaw.split("\n");
  const driver = new FixtureDriver(lines, REPLAY_MS);
  driver.subscribe(
    (ev) => store.apply(ev),
    () => console.log("[fixture] done"),
  );

  (window as unknown as { __ploshcha: unknown }).__ploshcha = { renderer, store, director };
}

boot().catch((e: unknown) => {
  console.error("[boot] failed", e);
  const msg = e instanceof Error ? e.message : String(e);
  document.body.innerHTML = `<pre style="color:#f88;padding:20px;font:14px monospace">boot failed: ${msg}</pre>`;
});
