import "./style.css";
import { SceneSpec } from "@ploshcha/contract-ts";
import type { POI } from "@ploshcha/contract-ts";
import sceneJson from "@fixtures/scenes/verbolozy.scene.json";
import quietDayRaw from "@fixtures/runs/quiet-day.jsonl?raw";
import { SceneRenderer } from "./scene/SceneRenderer";
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

async function boot(): Promise<void> {
  const scene = SceneSpec.parse(sceneJson);
  const SCL = scene.size.w / scene.masks.space.w;

  const renderer = new SceneRenderer(scene);
  renderer.mount(document.getElementById("frame")!);
  await renderer.loadGround();

  const grid = new WalkGrid(scene.masks.space.w, scene.masks.space.h, SCL);
  await grid.load(scene.masks.walk);

  const charUrls: string[] = [];
  for (let i = 0; i < 6; i++) charUrls.push(`/assets/nb/c1_chars_a/0${i}.png`);
  for (let i = 0; i < 6; i++) charUrls.push(`/assets/nb/c2_chars_b/0${i}.png`);
  const loaded = await Promise.all(charUrls.map((u) => loadGraded(u, 130).catch(() => null)));
  const chars = loaded.filter((t): t is NonNullable<typeof t> => t !== null);

  const pois = new Map<string, POI>();
  for (const p of scene.pois) pois.set(p.id, p);

  const director = new AgentDirector(renderer.world, grid, pois, chars, makeShadowTexture(), SCL);
  const narrator = new Narrator();
  const chat = new ChatLog();
  const store = new SimStore();

  store.on((ev, state) => {
    switch (ev.type) {
      case "run.started":
        narrator.say(`Ранок у селі ${ev.payload.scene.name}. Село прокидається…`);
        chat.sys(`Прогін «${ev.payload.scene.name}» почався`);
        break;
      case "casting.done":
        director.spawn(ev.payload.cast);
        break;
      case "tick.begin":
        chat.setDay(TIME_UA[ev.payload.timeOfDay] ?? ev.payload.timeOfDay);
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
        chat.sys(`✦ ${ev.payload.event.label}: ${ev.payload.event.description}`);
        break;
      case "reflection.formed": {
        const name = state.villagers.get(ev.payload.agentId)?.name ?? ev.payload.agentId;
        chat.line(`${name} (думка)`, ev.payload.thought);
        break;
      }
      case "report.compiled":
        narrator.say(ev.payload.chronicle.narration, "Літописець", 12000);
        chat.chronicle(ev.payload.chronicle.title);
        chat.setDay(`день ${ev.payload.chronicle.day}`);
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

  renderer.app.ticker.add(() => {
    const dt = Math.min(renderer.app.ticker.deltaMS / 1000, 0.05);
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
