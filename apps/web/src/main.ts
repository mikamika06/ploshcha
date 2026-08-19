import "./style.css";
import { Texture } from "pixi.js";
import { SceneSpec } from "@ploshcha/contract-ts";
import type { POI } from "@ploshcha/contract-ts";
import sceneJson from "@fixtures/scenes/verbolozy.scene.json";
import quietDayRaw from "@fixtures/runs/quiet-day.jsonl?raw";
import { SceneRenderer, type ObjectSpec } from "./scene/SceneRenderer";
import type { VegTextures, VegType } from "./scene/Vegetation";
import { WalkGrid } from "./agents/WalkGrid";
import { AgentDirector } from "./agents/AgentDirector";
import { SimStore } from "./store/SimStore";
import { Ports, radiusFor } from "./interact/Ports";
import { Board } from "./interact/Board";
import { GroupTalk } from "./interact/GroupTalk";
import { Chronicle } from "./interact/Chronicle";
import { discussionFor, type TalkPart } from "./interact/discussion";
import { LivingRoom, type RoomCast, type Pt } from "./interact/LivingRoom";
import { Inspector } from "./interact/Inspector";

/** «Чорний ящик»: декоративні відкриті iso-сцени духів + нефункціональний токен-тул (флейвор). */
const BLACKBOX: Record<string, { verb: string; lines: string[] }> = {
  pond: { verb: "Придивитись до ставка", lines: ["Ряска ледь колишеться.", "Плюснула риба — і тиша.", "Очерет шепоче на вітрі.", "Хмара пропливла у воді."] },
  bell: { verb: "Торкнутись дзвона", lines: ["Дзвін теплий від сонця.", "Мідь гуде ледь чутно.", "Мотузка шорстка в руці.", "Луна ще спить у металі."] },
};

/** Соціальні кімнати → жива кімната (iso-box + наші спрайти ходять по МАСЦІ-зоні з Nano Banana).
 *  mask = згенерована walk-зона (зелене=прохідне, вирізає стільці/меблі); floor = полігон-фолбек. */
// Усі локації тепер уніфіковані: одна канва 1792×1008 на одному фоні #21222F (без лого),
// тому — один масштаб фігур для всіх (нормальні однакові розміри людей).
const FIG = 0.8;
// Фолбек-підлога (діамант по центру) діє лише мить до завантаження маски; далі головна — маска.
const FB: Pt[] = [
  [0.5, 0.8],
  [0.78, 0.56],
  [0.5, 0.32],
  [0.22, 0.56],
];
const SOCIAL_ROOM: Record<string, { bg: string; name: string; floor: Pt[]; mask?: string }> = {
  tavern: { bg: "/assets/rooms/tavern.webp", name: "Шинок", mask: "/assets/rooms/tavern_mask.png", floor: FB },
  mill: { bg: "/assets/rooms/mill.webp", name: "Млин", mask: "/assets/rooms/mill_mask.png", floor: FB },
  church: { bg: "/assets/rooms/church.webp", name: "Церква", mask: "/assets/rooms/church_mask.png", floor: FB },
  forge: { bg: "/assets/rooms/forge.webp", name: "Кузня", mask: "/assets/rooms/forge_mask.png", floor: FB },
  pond: { bg: "/assets/rooms/pond.webp", name: "Ставок", mask: "/assets/rooms/pond_mask.png", floor: FB },
  bell: { bg: "/assets/rooms/bell.webp", name: "Дзвіниця", mask: "/assets/rooms/bell_mask.png", floor: FB },
  square: { bg: "/assets/rooms/square.webp", name: "Площа", mask: "/assets/rooms/square_mask.png", floor: FB },
};

import { FixtureDriver } from "./net/FixtureDriver";
import { LiveDriver, sendCommand } from "./net/LiveDriver";
import { parseEnvelope } from "./net/validate";
import { GRADE_MUTED, loadGraded } from "./util/gfx";
import { IS_LIVE, LIVE_URL, REPLAY_MS } from "./config";

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

const ROLE_IDS = [
  "koval", "mirosh", "pip", "sheptu", "starosta", "shynkar",
  "chumak", "diak", "did", "mati", "parubok", "divchyna",
];

async function loadRoleFrames(): Promise<Map<string, Texture[]>> {
  const map = new Map<string, Texture[]>();
  await Promise.all(
    ROLE_IDS.map(async (id) => {
      const fr = await Promise.all([0, 1, 2].map((n) => loadGraded(`/assets/roles/${id}/${n}.png`, undefined, GRADE_MUTED).catch(() => null)));
      const valid = fr.filter((t): t is Texture => t !== null);
      if (valid.length === 3) map.set(id, valid);
    }),
  );
  return map;
}

async function boot(): Promise<void> {
  const scene = SceneSpec.parse(sceneJson);
  const SCL = scene.size.w / scene.masks.space.w;

  const objects = await fetch("/assets/objects/objects.json")
    .then((r) => (r.ok ? r.json() : { objects: [] }))
    .then((d: { objects: ObjectSpec[] }) => d.objects)
    .catch(() => [] as ObjectSpec[]);

  const renderer = new SceneRenderer(scene);
  renderer.mount(document.getElementById("frame")!);
  // Хмарна завіса ОДРАЗУ — ховає сцену, поки все вантажиться й рендериться (з людьми).
  renderer.playIntro();
  await renderer.loadGround();
  await renderer.loadObjects(objects);
  await renderer.loadProps();

  const grid = new WalkGrid(scene.masks.space.w, scene.masks.space.h, SCL);
  // NB: keepout маска = «не саджати» (стежки+хати+вода), а не «тут хата» — тому для
  // walk-grid її НЕ використовуємо (інакше вирізає всю ходьбу). Ходьба лише за walk2.
  await grid.load(scene.masks.walk);
  // футпринти хат (реальні bbox зі спрайтів) — щоб селяни не заходили ЗА/ПІД них
  grid.blockObjects(objects);

  const [vegTex, chars, roleFrames] = await Promise.all([loadVegTextures(), loadCharTextures(), loadRoleFrames()]);
  await renderer.buildVegetation(vegTex);
  renderer.initWeather();

  const pois = new Map<string, POI>();
  for (const p of scene.pois) pois.set(p.id, p);

  const director = new AgentDirector(renderer.world, grid, pois, chars, roleFrames, SCL);
  const store = new SimStore();

  // ── діегетична оболонка (без HUD-бару): порти → занурення → Дошка / розмова ──
  const stageEl = document.getElementById("stage")!;
  const whisper = document.createElement("div");
  whisper.className = "whisper";
  stageEl.appendChild(whisper);
  const vign = document.createElement("div");
  vign.className = "vign";
  stageEl.appendChild(vign);
  const loc = document.createElement("div");
  loc.className = "loccard";
  loc.innerHTML = `<div class="loc-name"></div><div class="loc-mean"></div><button class="loc-back" type="button">← до села</button>`;
  stageEl.appendChild(loc);

  const squarePoi = scene.pois.find((p) => p.kind === "square");
  let resumeSkips = 0; // після рестарту тікера (вкладка/вихід з кімнати) — відкинути перші «биті» кадри
  let talkOpen = false; // чи чекає відкрите вікно розмови на репліки з живого потоку

  const exitToVillage = (): void => {
    talkOpen = false;
    board.close();
    groupTalk.close();
    room.close();
    inspector.close();
    loc.classList.remove("on");
    vign.classList.remove("on");
    renderer.camera?.back();
    ports.setEnabled(true);
    // кімната стопить тікер (опукла оболонка ховає діораму) → на виході прокидаємо сцену
    if (!renderer.app.ticker.started) {
      resumeSkips = 3;
      renderer.app.ticker.start();
    }
  };
  const enterDive = (p: POI): void => {
    inspector.close();
    ports.setEnabled(false);
    whisper.classList.remove("on");
    vign.classList.add("on");
    renderer.camera?.diveTo(p.x, p.y);
  };

  const groupTalk = new GroupTalk(
    () => exitToVillage(),
    (text, whisperTo) => {
      if (!IS_LIVE) return;
      void sendCommand(LIVE_URL, {
        kind: whisperTo ? "whisper" : "say", text,
        ...(whisperTo ? { to: whisperTo } : {}),
      }).catch(() => groupTalk.setStatus("слово не доїхало — віче вже скінчилось"));
    },
  );
  const board = new Board(
    (t) => {
      board.close();
      // учасники — реальні селяни зі стору; якщо ще не «наспавнились» — запасний гурт
      const villagers = [...store.state.villagers.values()];
      const pool: TalkPart[] =
        villagers.length >= 2
          ? villagers.map((v) => ({ id: v.id, name: v.name, role: v.role }))
          : [
              { id: "parubok", name: "Іван", role: "parubok" },
              { id: "divchyna", name: "Оксана", role: "divchyna" },
              { id: "did", name: "дід Свирид", role: "did" },
              { id: "sheptu", name: "баба Горпина", role: "sheptu" },
            ];
      const parts = pool.sort(() => Math.random() - 0.5).slice(0, Math.min(5, pool.length));
      // усі сходяться на Площу + камера пірнає туди (видно, як зібрались коло криниці)
      if (squarePoi) {
        for (const p of parts) director.moveTo(p.id, { poi: squarePoi.id });
        renderer.camera?.diveTo(squarePoi.x, squarePoi.y);
      }
      // У живому режимі репліки беруться ЛИШЕ з реального потоку. Генератор тут дав би фікцію:
      // тема щойно поставлена в чергу, ядро над нею думає десятки секунд, тож `transcript`
      // порожній — і колишня умова `live.length` тихо падала в заготовані репліки.
      if (IS_LIVE) {
        talkOpen = true;
        groupTalk.openLive(t.text, "Тему передано в ядро. Село сходиться, Мамай думає…");
        // Ключ мусить бути свіжий: черга ядра ідемпотентна за ключем, тож стабільний хеш тексту
        // означав би, що другий клік по тій самій темі тихо НЕ запускає прогін.
        void sendCommand(LIVE_URL, {
          kind: "topic",
          text: t.text,
          key: `${t.id}-${Date.now().toString(36)}`,
        }).catch((err: unknown) => {
          console.warn("[board] тема не доїхала в ядро", err);
          groupTalk.finish("Тема не доїхала в ядро — воно не відповідає.");
        });
      } else {
        talkOpen = false;
        groupTalk.open(t.text, parts, discussionFor(t.text, parts));
      }
    },
    () => exitToVillage(),
  );
  const chron = new Chronicle();
  const inspector = new Inspector(() => inspector.close());
  const room = new LivingRoom(
    () => exitToVillage(),
    (vid) => {
      const v = store.state.villagers.get(vid);
      if (v) inspector.open(v); // аналітика на людину і всередині локації
    },
  );

  const ports = new Ports(renderer.world, scene.pois, {
    onHover: (p, x, y) => {
      if (!p) {
        whisper.classList.remove("on");
        return;
      }
      whisper.textContent = p.name;
      whisper.style.left = `${x}px`;
      whisper.style.top = `${y}px`;
      whisper.classList.add("on");
    },
    onSelect: (p) => {
      enterDive(p);
      if (p.kind === "board") {
        board.open();
      } else if (SOCIAL_ROOM[p.kind]) {
        const r = SOCIAL_ROOM[p.kind];
        // ЖИВА зайнятість: у локації ті селяни, чий поточний POI = цей (з agent.moved), а не фікс-каст
        const here = [...store.state.villagers.values()].filter((v) => v.location === p.id);
        const cast: RoomCast[] = here.map((v) => ({ id: v.role, name: v.name, vid: v.id }));
        room.open(r.bg, r.name, cast, r.floor, r.mask, { figScale: FIG, token: BLACKBOX[p.kind] });
        renderer.app.ticker.stop(); // опукла кімната ховає діораму → не рендеримо/не колишемо її поки там
      } else {
        (loc.querySelector(".loc-name") as HTMLElement).textContent = p.name;
        (loc.querySelector(".loc-mean") as HTMLElement).textContent = p.meaning ?? "";
        loc.classList.add("on");
      }
    },
  });
  loc.querySelector(".loc-back")!.addEventListener("click", () => exitToVillage());
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") exitToVillage();
  });

  // клік по селянину на діорамі → інспектор когніції (ручний хіт-тест поза Pixi)
  const canvasEl = renderer.app.view as unknown as HTMLCanvasElement;
  canvasEl.addEventListener("click", (e) => {
    const cam = renderer.camera;
    if (!cam || cam.consumeDrag()) return; // це був пан, не клік
    const w = cam.clientToWorld(e.clientX, e.clientY);
    // порти (POI) мають пріоритет — їхні тапи обробляє Pixi окремо (guard = точний радіус порту)
    for (const p of scene.pois) if (Math.hypot(w.x - p.x, w.y - p.y) < radiusFor(p.kind)) return;
    const id = director.nearestAt(w.x, w.y, 60);
    if (id) {
      const v = store.state.villagers.get(id);
      if (v) inspector.open(v);
    } else {
      inspector.close();
    }
  });

  store.on((ev) => {
    // Невідомий тип доїжджає сюди сирим (контракт additive) — сцена свідомо його не малює.
    if (!ev.known) return;
    switch (ev.type) {
      case "casting.done":
        director.spawn(ev.payload.cast);
        break;
      case "tick.begin":
        if (ev.payload.mood) renderer.weather?.setMood(ev.payload.mood.valence);
        break;
      case "agent.moved":
        director.moveTo(ev.payload.agentId, ev.payload.to);
        break;
      case "utterance.spoken": {
        const who = ev.payload.agentId;
        director.speak(who, ev.payload.text);
        if (!talkOpen) break;
        const v = store.state.villagers.get(who);
        // Імʼя беремо ЛИШЕ зі стору: каст оголошує ядро (`casting.done`), тож розійтись із текстом
        // репліки неможливо. Мапа-латка `ROLE_NAME_UA` існувала саме тому, що каст був фікстурний.
        const part: TalkPart = { id: who, name: v?.name ?? who, role: v?.role ?? who };
        if (squarePoi) director.moveTo(who, { poi: squarePoi.id });
        groupTalk.push(part, { ...part, text: ev.payload.text });
        break;
      }
      case "task.outcome":
        chron.outcome(ev.payload.outcome);
        if (talkOpen) {
          // Порожня розмова називається порожньою: вигадати репліку тут = та сама фікція.
          groupTalk.finish(
            ev.payload.outcome === "abstain"
              ? "Село не дійшло згоди: у довіднику нема підтвердження."
              : "Прогін завершено, реплік не було.",
          );
        }
        break;
      case "run.error":
        if (talkOpen) groupTalk.finish("Ядро впало — розмова не відбулась.");
        break;
      case "tool.called":
        chron.called(ev.payload.tool);
        break;
      case "tool.result":
        chron.resulted(ev.payload.tool, ev.payload.ok, ev.payload.found);
        break;
      case "memory.recalled":
        chron.recalled(ev.payload.items.length);
        break;
      case "plan.revised":
        chron.revised(ev.payload.reason);
        break;
      case "run.degraded":
        chron.degraded(ev.payload.stage, ev.payload.reason);
        break;
      case "run.started":
        chron.clear();
        break;
      case "event.happened": {
        // Ухвала — не чергова тема, а СКРІПЛЕНЕ рішення: інший вигляд, і по ній не запускається
        // новий прогін (клікабельні лише «гарячі»).
        const decision = ev.payload.event.kind === "decision";
        board.addTopic({
          text: ev.payload.event.label,
          heat: decision ? "sealed" : "warm",
          author: decision ? store.state.villagers.get(ev.payload.event.involves?.[0] ?? "")?.name : undefined,
        });
        if (decision) chron.decided(ev.payload.event.label);
        break;
      }
      case "report.compiled":
        renderer.weather?.setMood(ev.payload.chronicle.mood.valence);
        chron.chronicle(ev.payload.chronicle.title ?? "", ev.payload.chronicle.narration ?? "");
        break;
      case "plan.formed":
        board.pinPlan(ev.payload.summary, ev.payload.steps ?? []);
        break;
      case "reflection.formed": {
        const who = store.state.villagers.get(ev.payload.agentId);
        chron.thought(who?.name ?? ev.payload.agentId);
        break;
      }
      default:
        break;
    }
  });

  renderer.app.ticker.add(() => {
    const rawMs = renderer.app.ticker.deltaMS;
    if (resumeSkips > 0 && rawMs > 34) {
      resumeSkips--; // кадр із ненормально великим delta одразу після рестарту тікера → пропустити (без стрибка вітру)
      return;
    }
    resumeSkips = 0;
    const dt = Math.min(rawMs / 1000, 0.05);
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

  // Вкладка схована → стопимо тікер (щоб не було повільних/нерівних кадрів і смикання вітру
  // на поверненні); показана → чистий рестарт.
  document.addEventListener("visibilitychange", () => {
    const t = renderer.app.ticker;
    if (document.hidden) {
      t.stop();
    } else {
      resumeSkips = 3; // перші кадри після рестарту мають стале/велике deltaMS → відкидаємо
      t.start();
    }
  });

  // Люди мають стояти на місцях ще ПІД завісою: стартові події (каст) застосовуємо одразу,
  // а таймлайн-реплей — з першого tick.begin. spawn ідемпотентний → без дублів.
  const allLines = quietDayRaw.split("\n").filter((l) => l.trim());
  const firstTick = allLines.findIndex((l) => l.includes('"type":"tick.begin"'));
  const head = firstTick > 0 ? allLines.slice(0, firstTick) : allLines;
  const tail = firstTick > 0 ? allLines.slice(firstTick) : [];
  for (const l of head) {
    const ev = parseEnvelope(l);
    if (ev) store.apply(ev); // run.started + casting.* → селяни спавняться зараз, під хмарами
  }
  const driver = IS_LIVE
    ? new LiveDriver(`${LIVE_URL}/stream`)
    : new FixtureDriver(tail.length ? tail : allLines, REPLAY_MS);
  driver.subscribe(
    (ev) => store.apply(ev),
    () => console.log(IS_LIVE ? "[live] закрито" : "[fixture] done"),
  );

  (window as unknown as { __ploshcha: unknown }).__ploshcha = { renderer, store, director };

  // Усе відрендерено й люди наспавнились → даємо сцені кілька кадрів проступити під завісою
  // (не залежимо від rAF у фоновій вкладці — таймер-fallback), і аж тоді розводимо хмари.
  await Promise.race([
    new Promise<void>((res) => requestAnimationFrame(() => requestAnimationFrame(() => res()))),
    new Promise<void>((res) => setTimeout(res, 250)),
  ]);
  renderer.dissipateIntro();
}

boot().catch((e: unknown) => {
  console.error("[boot] failed", e);
  const msg = e instanceof Error ? e.message : String(e);
  document.body.innerHTML = `<pre style="color:#f88;padding:20px;font:14px monospace">boot failed: ${msg}</pre>`;
});
