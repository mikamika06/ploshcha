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
import { discussionFor, type TalkPart } from "./interact/discussion";
import { LivingRoom, type RoomCast, type Pt } from "./interact/LivingRoom";
import { Curtain } from "./scene/Curtain";
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
// Скільки щонайдовше стоїмо на мапі, поки село думає, перш ніж однаково брати хмари.
const PONDER_CAP_MS = 90000;
// Наскільки близько треба стояти до місця, щоб рахуватись «тут».
const NEAR_POI = 260;
// Як часто, поки чекаємо, питаємо ядро, чи воно ще живе.
const HEALTH_EVERY_MS = 4000;
// Фолбек-підлога (діамант по центру) діє лише мить до завантаження маски; далі головна — маска.
const FB: Pt[] = [
  [0.5, 0.8],
  [0.78, 0.56],
  [0.5, 0.32],
  [0.22, 0.56],
];
/** Місце віча (словник ядра) → намальована локація. Ядро каже `shynok`, сцена знає `tavern`. */
const PLACE_ROOM: Record<string, string> = {
  ploshcha: "square", shynok: "tavern", tserkva: "church",
  kuznya: "forge", mlyn: "mill", dzvin: "bell", stavok: "pond",
};
const SOCIAL_ROOM: Record<string, { bg: string; name: string; floor: Pt[]; mask?: string }> = {
  tavern: { bg: assetUrl("/assets/rooms/tavern.webp"), name: "Шинок", mask: assetUrl("/assets/rooms/tavern_mask.png"), floor: FB },
  mill: { bg: assetUrl("/assets/rooms/mill.webp"), name: "Млин", mask: assetUrl("/assets/rooms/mill_mask.png"), floor: FB },
  church: { bg: assetUrl("/assets/rooms/church.webp"), name: "Церква", mask: assetUrl("/assets/rooms/church_mask.png"), floor: FB },
  forge: { bg: assetUrl("/assets/rooms/forge.webp"), name: "Кузня", mask: assetUrl("/assets/rooms/forge_mask.png"), floor: FB },
  pond: { bg: assetUrl("/assets/rooms/pond.webp"), name: "Ставок", mask: assetUrl("/assets/rooms/pond_mask.png"), floor: FB },
  bell: { bg: assetUrl("/assets/rooms/bell.webp"), name: "Дзвіниця", mask: assetUrl("/assets/rooms/bell_mask.png"), floor: FB },
  square: { bg: assetUrl("/assets/rooms/square.webp"), name: "Площа", mask: assetUrl("/assets/rooms/square_mask.png"), floor: FB },
};

import { FixtureDriver } from "./net/FixtureDriver";
import { LiveDriver, sendCommand, fetchHealth } from "./net/LiveDriver";
import { parseEnvelope } from "./net/validate";
import { GRADE_MUTED, loadGraded, assetUrl } from "./util/gfx";
import { IS_LIVE, LIVE_URL, REPLAY_MS } from "./config";

function tuftUrls(dir: string, count: number): string[] {
  return Array.from({ length: count }, (_, i) => assetUrl(`/assets/nb/${dir}/0${i}.png`));
}

const VEG_SRC: Record<VegType, string[]> = {
  wheat: tuftUrls("tuft_wheat2", 6),
  flower: tuftUrls("tuft_flower2", 8),
  reed: tuftUrls("tuft_reed2", 6),
  grass: tuftUrls("tuft_grass", 6),
  tree: ["1_trees/00", "1_trees/01", "1_trees/02", "v2_trees/00", "v2_trees/01", "v2_trees/02"].map((s) => assetUrl(`/assets/nb/${s}.png`)),
  bush: ["1_trees/03", "v2_trees/03", "v2_trees/04"].map((s) => assetUrl(`/assets/nb/${s}.png`)),
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
  for (let i = 0; i < 6; i++) urls.push(assetUrl(`/assets/nb/c1_chars_a/0${i}.png`));
  for (let i = 0; i < 6; i++) urls.push(assetUrl(`/assets/nb/c2_chars_b/0${i}.png`));
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
      const fr = await Promise.all([0, 1, 2].map((n) => loadGraded(assetUrl(`/assets/roles/${id}/${n}.png`), undefined, GRADE_MUTED).catch(() => null)));
      const valid = fr.filter((t): t is Texture => t !== null);
      if (valid.length === 3) map.set(id, valid);
    }),
  );
  return map;
}

async function boot(): Promise<void> {
  const scene = SceneSpec.parse(sceneJson);
  const SCL = scene.size.w / scene.masks.space.w;

  const objects = await fetch(assetUrl("/assets/objects/objects.json"))
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
  await grid.load(scene.masks.walk, undefined, scene.masks.zone);
  // футпринти хат (реальні bbox зі спрайтів) — щоб селяни не заходили ЗА/ПІД них
  grid.blockObjects(objects);

  const [vegTex, chars, roleFrames] = await Promise.all([
    loadVegTextures(), loadCharTextures(), loadRoleFrames(),
  ]);
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
  // Поки ядро складає порядок, на мапі мусить бути слово: інакше тицьнув тему — і хвилину нічого.
  const думка = document.createElement("div");
  думка.className = "think-map plaq";
  stageEl.appendChild(думка);
  const mapThink = (text: string): void => {
    думка.textContent = text;
    думка.classList.toggle("on", Boolean(text));
  };

  const loc = document.createElement("div");
  loc.className = "loccard";
  loc.innerHTML = `<div class="loc-name"></div><div class="loc-mean"></div><button class="loc-back" type="button">← до села</button>`;
  stageEl.appendChild(loc);

  const curtain = new Curtain();
  const squarePoi = scene.pois.find((p) => p.kind === "square");
  /** Куди пішло віче: поки не порожнє — репліки ядра лунають У ЛОКАЦІЇ, а не над мапою. */
  let talkRoom: string | null = null;
  /** Де саме глядач зараз стоїть (словник ядра): звідси береться місце для нової теми. */
  let openPlace: string | null = null;
  /**
   * Хто це говорить — для локації.
   *
   * ★ `hist` — це ТИ. Ядро віддає твоє слово під цією роллю, а серед мальованих ролей її немає:
   * кімната просила assetUrl(`/assets/roles/hist/0.png`), діставала 404 і ставила побиту картинку. Твоє
   * слово в село виглядало як поламка саме через це.
   */
  const castOf = (who: string): RoomCast => {
    const v = store.state.villagers.get(who);
    if (v) return { id: v.role, name: v.name, vid: v.id };
    return who === "hist"
      ? { id: "chumak", name: "ти", vid: who }
      : { id: who, name: who, vid: who };
  };

  /** Порядок готовий — можна брати хмари й переносити в локацію. */
  let planReady = false;
  /** Хто говорив останнім — до нього підходять і від нього відступають. */
  let lastSpeaker: string | undefined;
  /** Дія приходить ОКРЕМОЮ подією перед реплікою — тримаємо її до самої репліки. */
  const pendingDeed = new Map<string, string>();
  /** Репліки, що встигли прозвучати, ПОКИ село ще сходиться: чекають відкриття локації. */
  const preRoom: { who: string; text: string; deed?: string; toward?: string }[] = [];
  let resumeSkips = 0; // після рестарту тікера (вкладка/вихід з кімнати) — відкинути перші «биті» кадри
  let talkOpen = false; // чи чекає відкрите вікно розмови на репліки з живого потоку

  const exitToVillage = (): void => {
    preRoom.length = 0;
    talkOpen = false;
    talkRoom = null;
    director.hold([...store.state.villagers.keys()], false);
    board.close();
    groupTalk.close();
    room.close();
    inspector.close();
    loc.classList.remove("on");
    renderer.camera?.back();
    ports.setEnabled(true);
    // кімната стопить тікер (опукла оболонка ховає діораму) → на виході прокидаємо сцену
    if (!renderer.app.ticker.started) {
      resumeSkips = 3;
      renderer.app.ticker.start();
    }
  };
  /**
   * Вхід у порт. Наближення тут БІЛЬШЕ НЕМАЄ.
   *
   * Камера пірнала й затемнювала кадр перед кожним відкриттям — зайвий кадр-переріз, який нічого
   * не пояснював: локація однаково відкривається своєю сценою, а на мапі люди говорять
   * бульбашками. Лишається тільки те, що справді треба: порти не ловлять клік, підказка гасне.
   */
  const enterDive = (_p: POI): void => {
    inspector.close();
    ports.setEnabled(false);
    whisper.classList.remove("on");
  };

  const groupTalk = new GroupTalk(
    () => exitToVillage(),
    (text) => {
      if (!IS_LIVE) return;
      void sendCommand(LIVE_URL, { kind: "say", text })
        .catch(() => groupTalk.setStatus("слово не доїхало — віче вже скінчилось"));
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
      // Усі СПРАВДІ йдуть на місце віча — своїми ногами, по мапі. Доти це було наближення камери
      // до площі, тобто «зібрались» лише на словах: люди лишались там, де стояли.
      const talkKind = PLACE_ROOM[board.where];
      const talkPoi = scene.pois.find((q) => q.kind === talkKind) ?? squarePoi;
      if (talkPoi) {
        // Громада ОДРАЗУ на місці: тицьнув тему — і всі там. Хода через усе село була хвилиною
        // порожнього екрана, а після віча вони й лишаються стояти там, де говорили.
        director.hold(parts.map((q) => q.id), true);
        const spots = grid.spotsNear(talkPoi.x, talkPoi.y, parts.length);
        parts.forEach((q, i) => {
          const spot = spots[i] ? grid.cellCenter(spots[i][0], spots[i][1])
            : { x: talkPoi.x, y: talkPoi.y };
          director.placeAt(q.id, spot);
        });
      }
      // У живому режимі репліки беруться ЛИШЕ з реального потоку. Генератор тут дав би фікцію:
      // тема щойно поставлена в чергу, ядро над нею думає десятки секунд, тож `transcript`
      // порожній — і колишня умова `live.length` тихо падала в заготовані репліки.
      if (IS_LIVE) {
        talkOpen = true;
        // Якщо місце має намальовану локацію — розмова йде ТАМ; VN-накладка лишається запасним
        // шляхом (Дошка як місце, або невідоме місце з ядра).
        if (talkKind) void enterTalkRoom(board.where, parts);
        else groupTalk.openLive(t.text, "Тему передано в ядро. Село сходиться, Мамай думає…");
        // Ключ мусить бути свіжий: черга ядра ідемпотентна за ключем, тож стабільний хеш тексту
        // означав би, що другий клік по тій самій темі тихо НЕ запускає прогін.
        void sendCommand(LIVE_URL, {
          kind: "topic",
          text: t.text,
          place: board.where,
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
  // Літопис прибрано зі сцени: стос службових плашок праворуч заступав село й нічого не додавав.
  // Гучними лишились тільки поламки — вони йдуть у підпис знизу, де й «Село думу думає…».
  const inspector = new Inspector(() => inspector.close());
  const room = new LivingRoom(
    () => exitToVillage(),
    (text) => {
      if (!IS_LIVE) return;
      const live = talkRoom !== null;
      // Слово завжди ВГОЛОС. Якщо віче йде — воно вклинюється в нього; якщо ні — стає темою
      // тут-таки, інакше поле в тихій локації нічого не робило й читалось як зламане.
      void sendCommand(LIVE_URL, live
        ? { kind: "say", text }
        : { kind: "topic", text, place: openPlace ?? board.where,
            key: `room-${Date.now().toString(36)}` })
        .then(() => {
          if (!live) room.notice("Тему кинуто селу — зараз почнуть.");
        })
        .catch(() => room.notice("Слово не доїхало — ядро не відповідає."));
    },
  );

  /**
   * Село затягує хмарами й розводить їх уже над локацією, де сходиться віче.
   *
   * Доти місце розмови було лише написом на бирці: ядро справді вело інший процес у шинку й у
   * церкві, а на екрані нічого не мінялось. Тепер видно, КУДИ пішли люди.
   */
  const enterTalkRoom = async (placeId: string, parts: TalkPart[]): Promise<void> => {
    const kind = PLACE_ROOM[placeId];
    const r = kind ? SOCIAL_ROOM[kind] : undefined;
    if (!r) return;
    talkRoom = kind;
    openPlace = placeId;
    // Спершу СЕЛО ДУМАЄ — на мапі, при людях, що вже стоять на місці. Хмари беремо аж тоді, коли
    // порядок складено: один структурований виклик оркестратора йде десятки секунд, і кидати
    // глядача в порожню локацію на цей час немає сенсу.
    planReady = false;
    mapThink("Село думу думає…");
    const until = performance.now() + PONDER_CAP_MS;
    // ★ Лічильник, а не залишок від різниці часу. `(now - until) % 4000 < 220` тут завжди істина:
    // різниця відʼємна аж до кінця очікування, а відʼємний залишок у JS теж відʼємний, тобто
    // менший за 220. Через це стан ядра питали НА КОЖНОМУ оберті циклу (заміряно 7 із 7), а не
    // раз на чотири секунди — до сотень зайвих запитів за одне «село думає».
    let askAt = performance.now() + HEALTH_EVERY_MS;
    while (!planReady && performance.now() < until) {
      await new Promise((res) => window.setTimeout(res, 200));
      // ★ Ядро могло СПИНИТИСЬ (стеля токенів, падіння) — тоді теми просто лежать у черзі, а
      // «село думає» висить вічно. Мовчазне очікування тут — та сама давня поламка: механізм
      // стоїть, а шлях спостереження бреше. Тому раз на кілька секунд питаємо стан.
      if (!planReady && performance.now() >= askAt) {
        askAt = performance.now() + HEALTH_EVERY_MS;
        const h = await fetchHealth(LIVE_URL);
        if (h && h.state !== "running" && !planReady) {
          // Розмови не буде — тож і слухати нема чого. Доти `talkOpen` лишався піднятим, і пізні
          // репліки йшли у вікно, яке ніколи не відкривалось: подія доїхала, а зникла безслідно.
          talkRoom = null;
          talkOpen = false;
          director.hold([...store.state.villagers.keys()], false);
          mapThink(`Ядро спинилось: ${h.stoppedReason ?? h.lastError ?? h.state}`);
          window.setTimeout(() => mapThink(""), 9000);
          return;
        }
      }
    }
    mapThink("");
    if (talkRoom !== kind) return;
    // Спершу ХОДЬБА: доки Мамай думає, село сходиться на місце — це видно на мапі. Хмари беремо
    // лише коли всі дійшли; стеля потрібна, бо непрохідна клітинка не має вішати сцену назавжди.
    if (talkRoom !== kind) return; // віче вже скінчилось або перебите, поки сходились
    await curtain.sweep(() => {
      inspector.close();
      board.close();
      groupTalk.close();
      ports.setEnabled(false);
      whisper.classList.remove("on"); // підказка порту лишалась висіти вже над локацією
      const cast: RoomCast[] = parts.map((p) => ({ id: p.role, name: p.name, vid: p.id }));
      room.open(r.bg, r.name, cast, r.floor, r.mask, { figScale: FIG });
      room.setLive(true);
      // Якщо ядро вже щось віддало, доки відкривалась локація — воно звучить тут, по кліку.
      for (const w of preRoom.splice(0)) {
        room.addPerson(castOf(w.who));
        room.enqueue(w.who, w.text, w.deed, w.toward);
      }
      renderer.app.ticker.stop();
    });
  };

  const ports = new Ports(renderer.world, scene.pois, {
    wasDrag: () => Boolean(renderer.camera?.wasDrag()),
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
        openPlace = Object.keys(PLACE_ROOM).find((k) => PLACE_ROOM[k] === p.kind) ?? null;
        // ЖИВА зайнятість: спершу ті, кого ядро саме туди відправило (`agent.moved` → `location`),
        // а якщо таких ще немає — ті, хто фізично СТОЇТЬ коло цього місця на мапі. Інакше на
        // холодному старті всі сім локацій порожні, хоч люди видно коло самих дверей.
        const byEvent = [...store.state.villagers.values()].filter((v) => v.location === p.id);
        const here = byEvent.length
          ? byEvent
          : (director.occupancy(scene.pois, NEAR_POI).get(p.id) ?? [])
              .map((id) => store.state.villagers.get(id))
              .filter((v): v is NonNullable<typeof v> => Boolean(v))
              .slice(0, 5);
        // Хто зайшов у локацію — той ТАМ і числиться. Інакше стан двоїться: подія каже одне,
        // положення на мапі інше, і та сама людина знову опиняється в двох місцях.
        for (const v of here) v.location = p.id;
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
    // Радіус влучання менший для пальця: 60 світових пікселів на телефоні — це пів долоні, і
    // кожен дотик по мапі відкривав інспектора на випадковій людині поблизу.
    const id = director.nearestAt(w.x, w.y, matchMedia("(pointer: coarse)").matches ? 30 : 60);
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
      case "agent.moved": {
        // Дія тіла з партитури приходить у `activity`: у локації її грає кімната, на мапі —
        // це просто хода до POI, як і було.
        const deed = ev.payload.activity;
        if (talkRoom && room.isOpen && deed) {
          // Дію не грає одразу: вона мусить піти РАЗОМ зі своєю реплікою, коли глядач гортає.
          pendingDeed.set(ev.payload.agentId, deed);
          break;
        }
        director.moveTo(ev.payload.agentId, ev.payload.to);
        break;
      }
      case "utterance.spoken": {
        const who = ev.payload.agentId;
        // Три голоси — три матеріали: сумнів попа не має виглядати як звичайна репліка, інакше
        // верифікатор знову зливається з рештою, як зливалось усе до цього.
        lastSpeaker = who;
        director.speak(who, ev.payload.text, who === "pip" ? "doubt" : "voice");
        // Віче переїхало в локацію → репліка лунає над головою ТАМ, а не над мапою за хмарами.
        planReady = true; // слово вже пролунало — далі чекати порядку нема сенсу
        if (talkRoom) {
          if (room.isOpen) {
            room.addPerson(castOf(who));
            room.enqueue(who, ev.payload.text, pendingDeed.get(who), lastSpeaker);
          } else {
            // Село ще йде — слово чекає локації, а не гине в порожнечі.
            preRoom.push({ who, text: ev.payload.text, deed: pendingDeed.get(who),
                           toward: lastSpeaker });
          }
          pendingDeed.delete(who);
          break;
        }
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
        // Локацію НЕ закриваємо: люди лишаються стояти там, де говорили, а глядач дочитує чергу
        // й виходить «до села», коли захоче. Напису тут немає: `task.outcome` приходить раніше,
        // ніж дочитано чергу, і «віче скінчилось» брехало посеред розмови.
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
        mapThink("Ядро впало — розмова не відбулась.");
        window.setTimeout(() => mapThink(""), 9000);
        if (talkOpen) groupTalk.finish("Ядро впало — розмова не відбулась.");
        break;
      case "tool.called":
      case "tool.result":
        break; // похід по довідник видно на сцені; рядок про нього в літописі нічого не додавав
      case "memory.recalled":
      case "plan.revised":
        break; // це внутрішнє життя ядра; його місце в інспекторі, а не поверх села
      case "run.degraded":
        // Зупинка — не дрібниця: її мусить бути видно, тому вона йде в той самий підпис знизу.
        mapThink(`Віче стало: ${ev.payload.reason ?? ev.payload.stage}`);
        window.setTimeout(() => mapThink(""), 9000);
        break;
      case "event.happened": {
        // Ухвала — не чергова тема, а СКРІПЛЕНЕ рішення: інший вигляд, і по ній не запускається
        // новий прогін (клікабельні лише «гарячі»).
        const decision = ev.payload.event.kind === "decision";
        const rumour = ev.payload.event.kind === "rumour";
        board.addTopic({
          text: ev.payload.event.label,
          heat: decision ? "sealed" : rumour ? "warm" : "cold",
          author: decision ? store.state.villagers.get(ev.payload.event.involves?.[0] ?? "")?.name : undefined,
        });
        break;
      }
      case "report.compiled":
        renderer.weather?.setMood(ev.payload.chronicle.mood.valence);
        break;
      case "reflection.formed":
        // Рефлексія — це ВНУТРІШНЄ; її місце в інспекторі людини, а не в літописі. «Хтось
        // лишився при своєму» нічого не означало для того, хто дивиться на село.
        break;
      default:
        break;
    }
  });

  // Директор мусить знати зум: бульбашка тримає сталий розмір НА ЕКРАНІ, а не у світі.
  renderer.app.ticker.add(() => director.setZoom(renderer.camera?.zoom ?? 1));

  renderer.app.ticker.add(() => {
    const rawMs = renderer.app.ticker.deltaMS;
    if (resumeSkips > 0 && rawMs > 34) {
      resumeSkips--; // кадр із ненормально великим delta одразу після рестарту тікера → пропустити (без стрибка вітру)
      return;
    }
    resumeSkips = 0;
    const dt = Math.min(rawMs / 1000, 0.05);
    // ★ Рух ведемо ЗГЛАДЖЕНИМ кроком часу, а не миттєвим.
    //
    // Заміряно: сама сторінка з часом не важчає (купа 17МБ рівно, обʼєктів 3516, слухачів 4), але
    // час кадру йде за завантаженням машини — 58мс на самоті проти 143мс, коли поруч працює ще
    // одна вкладка чи саме ядро. Нерівні кадри дають ривок вітру, бо фаза стрибає разом із dt.
    // Середнє за кількома кадрами прибирає саме стрибок, не змінюючи швидкості руху.
    smoothDt = smoothDt ? smoothDt * 0.82 + dt * 0.18 : dt;
    renderer.update(smoothDt);
    director.update(smoothDt);
  });

  /**
   * Вкладка в тлі — сцену СТОП.
   *
   * Браузер душить `requestAnimationFrame` до ~1 кадру на секунду, коли вкладка не активна. Тікер
   * далі йде, кожен кадр робить свій крок — і замість руху виходить ривок раз на секунду, тим
   * помітніший, чим довше сторінка провисіла. Тому на час невидимості просто не рахуємо нічого, а
   * повертаючись, відкидаємо перші кадри з ненормальним `deltaMS`.
   *
   * ★ `wasRunning` тут не для краси: у локації тікер спинено НАВМИСНЕ (опукла кімната ховає
   * діораму). Другий такий самий слухач, що стояв нижче, вмикав тікер безумовно — тож досить було
   * перемкнути вкладку туди-сюди, і мапа знову малювалась під кімнатою. Слухач мусить бути ОДИН.
   */
  let wasRunning = true;
  let smoothDt = 0;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      wasRunning = renderer.app.ticker.started;
      renderer.app.ticker.stop();
    } else if (wasRunning) {
      resumeSkips = 3;
      renderer.app.ticker.start();
    }
  });

  const fpsEl = document.getElementById("fps");
  // Лічильник кадрів — інструмент, не декорація села: вмикається лише прапорцем `?fps=1`.
  if (new URLSearchParams(location.search).has("fps")) {
    fpsEl?.parentElement?.classList.add("on");
  }
  let fpsT = 0;

  renderer.app.ticker.add(() => {
    fpsT += renderer.app.ticker.deltaMS;
    if (fpsT > 500 && fpsEl) {
      fpsEl.textContent = String(Math.round(renderer.app.ticker.FPS));
      fpsT = 0;
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

  (window as unknown as { __ploshcha: unknown }).__ploshcha = { renderer, store, director, room, curtain, board, groupTalk };

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
