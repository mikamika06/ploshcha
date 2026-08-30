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
import { FIXTURE_CAST, discussionFor, type TalkPart } from "./interact/discussion";
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

/**
 * Куди йти говорити. Дошка — не місце для віча, це стіна з темами.
 *
 * ★ Без запасного варіанта віче, призначене на Дошку, лишалось НА МАПІ: люди стояли коло криниці,
 * репліки висіли над селом, а локація не відкривалась узагалі. Виглядало як «розмова мимо
 * локації». Місця без своєї кімнати тепер ведуть на Площу — туди, де село й збирається.
 */
function roomFor(placeId: string): string {
  return PLACE_ROOM[placeId] ?? "square";
}
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
import { LiveDriver, sendCommand, fetchHealth, CommandRefused } from "./net/LiveDriver";
import type { EventSourcePort } from "./net/types";
import { Gripe } from "./interact/Gripe";
import { parseEnvelope } from "./net/validate";
import { GRADE_MUTED, loadGraded, assetUrl } from "./util/gfx";
import { IS_LIVE, LIVE_URL, REPLAY_MS } from "./config";

function tuftUrls(dir: string, count: number): string[] {
  return Array.from({ length: count }, (_, i) => assetUrl(`/assets/nb/${dir}/0${i}.webp`));
}

const VEG_SRC: Record<VegType, string[]> = {
  wheat: tuftUrls("tuft_wheat2", 6),
  flower: tuftUrls("tuft_flower2", 8),
  reed: tuftUrls("tuft_reed2", 6),
  grass: tuftUrls("tuft_grass", 6),
  tree: ["1_trees/00", "1_trees/01", "1_trees/02", "v2_trees/00", "v2_trees/01", "v2_trees/02"].map((s) => assetUrl(`/assets/nb/${s}.webp`)),
  bush: ["1_trees/03", "v2_trees/03", "v2_trees/04"].map((s) => assetUrl(`/assets/nb/${s}.webp`)),
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
  for (let i = 0; i < 6; i++) urls.push(assetUrl(`/assets/nb/c1_chars_a/0${i}.webp`));
  for (let i = 0; i < 6; i++) urls.push(assetUrl(`/assets/nb/c2_chars_b/0${i}.webp`));
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
      const fr = await Promise.all([0, 1, 2].map((n) => loadGraded(assetUrl(`/assets/roles/${id}/${n}.webp`), undefined, GRADE_MUTED).catch(() => null)));
      const valid = fr.filter((t): t is Texture => t !== null);
      if (valid.length === 3) map.set(id, valid);
    }),
  );
  return map;
}

/** Замір етапу старту: `performance.measure` + рядок у консоль. */
async function phase<T>(name: string, run: () => Promise<T>): Promise<T> {
  const t0 = performance.now();
  const out = await run();
  const ms = performance.now() - t0;
  performance.measure(`старт: ${name}`, { start: t0, duration: ms });
  console.info(`[старт] ${name}: ${Math.round(ms)} мс`);
  return out;
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
  // ★ Кожен етап старту позначений: «довго вантажиться» без розкладу — це здогад, а не діагноз.
  // Читається з консолі або `performance.getEntriesByType("measure")`.
  await phase("земля", () => renderer.loadGround());
  await phase("будівлі", () => renderer.loadObjects(objects));
  await phase("реквізит", () => renderer.loadProps());

  const grid = new WalkGrid(scene.masks.space.w, scene.masks.space.h, SCL);
  // NB: keepout маска = «не саджати» (стежки+хати+вода), а не «тут хата» — тому для
  // walk-grid її НЕ використовуємо (інакше вирізає всю ходьбу). Ходьба лише за walk2.
  await phase("сітка ходьби", () => grid.load(scene.masks.walk, undefined, scene.masks.zone));
  // футпринти хат (реальні bbox зі спрайтів) — щоб селяни не заходили ЗА/ПІД них
  grid.blockObjects(objects);

  const [vegTex, chars, roleFrames] = await phase("текстури зелені й людей", () => Promise.all([
    loadVegTextures(), loadCharTextures(), loadRoleFrames(),
  ]));
  await phase("зелень у сцену", () => renderer.buildVegetation(vegTex));
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

  /**
   * Питання з вибором — тією самою хмаркою, що й підписи села.
   *
   * ★ Діалогів браузера тут немає й не буде: `confirm()` — це системне вікно з чужим шрифтом
   * поверх села, тобто рівно той вихід зі сцени, якого ця оболонка уникає всюди. Матеріал той
   * самий (`plaq`), слова людські, вибір із двох — і жодного «ви впевнені?».
   */
  const ask = document.createElement("div");
  ask.className = "ask plaq";
  ask.innerHTML = `<div class="ask-say"></div>
                   <div class="ask-row">
                     <button class="tag ask-yes" type="button"></button>
                     <button class="tag ask-no" type="button">Хай говорять</button>
                   </div>`;
  stageEl.appendChild(ask);
  let asked: (() => void) | null = null;
  const closeAsk = (): void => {
    asked = null;
    ask.classList.remove("on");
  };
  const askThen = (say: string, yes: string, run: () => void): void => {
    (ask.querySelector(".ask-say") as HTMLElement).textContent = say;
    (ask.querySelector(".ask-yes") as HTMLElement).textContent = yes;
    asked = run;
    ask.classList.add("on");
  };
  ask.addEventListener("click", (e) => e.stopPropagation());
  ask.querySelector(".ask-yes")!.addEventListener("click", () => {
    const go = asked;
    closeAsk();
    go?.();
  });
  ask.querySelector(".ask-no")!.addEventListener("click", () => closeAsk());

  // Перша підказка: стрілка на Дошку. Показуємо рівно один раз на пристрій — правило про
  // «предмети замість кнопок» треба сказати вголос лише доти, доки воно невідоме.
  const HINT_SEEN = "ploshcha.hint.board";
  const firstHint = document.createElement("div");
  firstHint.className = "first-hint";
  firstHint.innerHTML = `<div class="first-hint-plaq">Тисни на Дошку-вісник</div>
                         <div class="first-hint-arrow"></div>`;
  stageEl.appendChild(firstHint);
  let hintPoi: POI | null = null;
  // ★ Показ підказки відкладений, і цей таймер треба вміти скасувати.
  //
  // Доти гість, який тицьнув Дошку раніше за 2.6 с, діставав підказку ПІСЛЯ того, як вона вже
  // сховалась: `hideHint` знімав клас, а відкладений показ вішав його назад. І позбутись її вже
  // не було як — `hintPoi` порожній, тож стеження за камерою (`hintPoi` у циклі) мовчить, і
  // табличка лишалась висіти поверх відкритої Дошки до кінця сесії.
  let hintTimer: number | undefined;
  const hideHint = (): void => {
    hintPoi = null;
    if (hintTimer !== undefined) window.clearTimeout(hintTimer);
    hintTimer = undefined;
    firstHint.classList.remove("on");
    try {
      localStorage.setItem(HINT_SEEN, "1");
    } catch {
      /* приватний режим: підказка просто зʼявиться ще раз */
    }
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
  let stoppedAt = 0;   // коли тікер спинили: на цю паузу зсуваємо годинник сцени
  let talkOpen = false; // чи чекає відкрите вікно розмови на репліки з живого потоку
  /**
   * Прогін, який ЗАРАЗ коштує грошей: `runId` живого віча або `null`.
   *
   * Береться з подій ядра, а не з наших намірів: тема могла ще лежати в черзі, віче могло
   * скінчитись саме, поки глядач дочитує притримані репліки. Питати «завершити?» про розмову,
   * якої вже немає, — та сама брехня інтерфейсу, що й «Село думу думає» над мертвим ядром.
   */
  let liveRun: string | null = null;
  const finishRun = (): void => {
    if (!IS_LIVE) return;
    // ★ Рішення гостя — уже ФАКТ, а не намір, і питати про нього вдруге не можна.
    //
    // Подія кінця приходить із ядра, і між «Завершити» та нею лежить ціле віче: тиша береться на
    // межі такту, а такт — це виклик Мамая (медіана 3.2 с, максимум 15.9 с на цьому шлюзі).
    // Заміряно браузером на живому ядрі 2026-08-28: **8416 мс** від кліку до `task.outcome`. Усе
    // це вікно `liveRun` лишався піднятим, тож наступний вихід або нова тема діставали ту саму
    // табличку вдруге — «Віче ще триває» про віче, яке гість щойно спинив сам. Тому знімаємо тут:
    // подія ядра лишається підтвердженням, а не єдиним джерелом.
    liveRun = null;
    // Притримані репліки — це та сама розмова, яку щойно спинили. Лишити їх грати означало б
    // сперечатись із власним питанням: гість сказав «завершити», а село говорить далі.
    driver.drop?.();
    void sendCommand(LIVE_URL, { kind: "finish" }).catch((err: unknown) =>
      console.warn("[live] віче не спинилось", err));
  };

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
      // Годинник сцени не має «наздоганяти» те, що глядач простояв у локації: інакше вітер
      // перестрибує на хвилину вперед одним кадром.
      if (stoppedAt) renderer.resumeClock(performance.now() - stoppedAt);
      stoppedAt = 0;
      renderer.app.ticker.start();
    }
  };
  /**
   * Вийти з розмови. Поки віче йде — спершу питання, і воно ж єдине місце, де село спиняють.
   *
   * ★ Вкладку, яку закривають, тут не перехопити: `beforeunload` уміє лише системне вікно, а
   * `pagehide` не відрізняє «пішов» від «перезавантажив» — і слухняний F5 коштував би гостю
   * всієї розмови. Тому закриту вкладку ловить ядро (пільга без слухача), а фронт відповідає за
   * ті виходи, які видно: «до села», Escape і нова тема.
   */
  const leaveTalk = (): void => {
    if (!IS_LIVE || !liveRun) {
      exitToVillage();
      return;
    }
    askThen("Віче ще триває. Завершити його?", "Завершити", () => {
      finishRun();
      exitToVillage();
    });
  };
  /**
   * Вхід у порт. Наближення тут БІЛЬШЕ НЕМАЄ.
   *
   * Камера пірнала й затемнювала кадр перед кожним відкриттям — зайвий кадр-переріз, який нічого
   * не пояснював: локація однаково відкривається своєю сценою, а на мапі люди говорять
   * бульбашками. Лишається тільки те, що справді треба: порти не ловлять клік, підказка гасне.
   */
  const enterDive = (_p: POI): void => {
    // Підказка своє сказала: гість уже кудись зайшов, а висіла вона й над локацією.
    hideHint();
    inspector.close();
    ports.setEnabled(false);
    whisper.classList.remove("on");
  };

  const groupTalk = new GroupTalk(
    () => leaveTalk(),
    (text) => {
      if (!IS_LIVE) return;
      void sendCommand(LIVE_URL, { kind: "say", text })
        .catch((err: unknown) => groupTalk.setStatus(err instanceof CommandRefused
          ? `ядро не взяло слово: ${err.message}`
          : "слово не доїхало — ядро не відповідає"));
    },
  );
  const board = new Board(
    (t) => {
      // ★ Нова тема закриває стару розмову, тож питаємо ДО того, як ядро це зробить.
      //
      // Гість, що кинув другу тему, доти не знав, куди поділась перша: ядро мусить її згорнути
      // (інакше вони змішаються в одному потоці), а на екрані це виглядало б як обрив. Питання
      // тут — не осторога, а те саме рішення, названо вголос.
      if (IS_LIVE && liveRun) {
        askThen("Віче ще триває. Завершити його й почати нове?", "Почати нове",
                () => startTopic(t));
        return;
      }
      startTopic(t);
    },
    () => exitToVillage(),
  );
  function startTopic(t: { id: string; text: string }): void {
    board.close();
    // ★ Імена на сцені називає ЛИШЕ ядро (`casting.done`).
    //
    // Тут стояв запасний ростер із «Оксаною» — і в живому режимі саме він доїжджав на сцену
    // першим: першу тему вкладки глядач кидає, коли складу ще нема (його оголошує прогін, а
    // прогону ще нема), локація відкривалась на цих іменах, а `LivingRoom.addPerson` на вже
    // посадженому `vid` мовчки виходив. `id` фікстури дорівнює ролі, тобто збігається з `id`
    // справжнього касту, тож підпис лишався фікстурним НАЗАВЖДИ: у базі власника «Пилип
    // Завзятко», на екрані «Іван» (аудит 2026-08-29, чотири розбіжні підписи з восьми).
    // Тепер із ядром гурт порожній: кожен приходить у локацію на своїй першій репліці, під
    // імʼям із `casting.done` (`castOf`). Без ядра грає фікстура — вона для того й названа так.
    const villagers = [...store.state.villagers.values()];
    const pool: TalkPart[] =
      villagers.length >= 2
        ? villagers.map((v) => ({ id: v.id, name: v.name, role: v.role }))
        : IS_LIVE ? [] : FIXTURE_CAST;
    const parts = pool.sort(() => Math.random() - 0.5).slice(0, Math.min(5, pool.length));
    // Усі СПРАВДІ йдуть на місце віча — своїми ногами, по мапі. Доти це було наближення камери
    // до площі, тобто «зібрались» лише на словах: люди лишались там, де стояли.
    const talkKind = roomFor(board.where);
    const talkPoi = scene.pois.find((q) => q.kind === talkKind) ?? squarePoi;
    if (talkPoi && parts.length) {
      // ★ Громада СХОДИТЬСЯ на місце — ногами, поспішаючи. Не ставиться туди одразу.
      //
      // Миттєве розставляння тут стояло навмисно: колись локація відкривалась аж тоді, як усі
      // зійшлись, і хода через усе село була хвилиною чекання. Тепер хмари беруться по готовності
      // порядку (`enterTalkRoom`), тобто хода не затримує НІЧОГО, — а стрибок лишався видним:
      // замір у браузері 2026-08-29 показав чотирьох, що перелетіли 507, 662, 837 і 930 світових
      // пікселів за один кадр у 2 мс. Поспіх (`HURRY`) заміряний окремо, у самому директорі.
      director.hold(parts.map((q) => q.id), true);
      const spots = grid.spotsNear(talkPoi.x, talkPoi.y, parts.length);
      parts.forEach((q, i) => {
        const spot = spots[i] ? grid.cellCenter(spots[i][0], spots[i][1])
          : { x: talkPoi.x, y: talkPoi.y };
        director.moveTo(q.id, spot);
      });
    }
    // У живому режимі репліки беруться ЛИШЕ з реального потоку. Генератор тут дав би фікцію:
    // тема щойно поставлена в чергу, ядро над нею думає десятки секунд, тож `transcript`
    // порожній — і колишня умова `live.length` тихо падала в заготовані репліки.
    if (IS_LIVE) {
      talkOpen = true;
      // Стара розмова тут і кінчається. Ядро згорне її саме (`topic` кличе `finish`), але подія
      // про це прийде через секунди, а притримані репліки грають щосекунди — і нова тема
      // починалась би під голоси минулої.
      driver.drop?.();
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
        groupTalk.finish(err instanceof CommandRefused
          ? `Ядро не взяло тему: ${err.message}`
          : "Тема не доїхала в ядро — воно не відповідає.");
      });
    } else {
      talkOpen = false;
      groupTalk.open(t.text, parts, discussionFor(t.text, parts));
    }
  }
  // Літопис прибрано зі сцени: стос службових плашок праворуч заступав село й нічого не додавав.
  // Гучними лишились тільки поламки — вони йдуть у підпис знизу, де й «Село думу думає…».
  const inspector = new Inspector(() => inspector.close());
  const room = new LivingRoom(
    () => leaveTalk(),
    (text) => {
      if (!IS_LIVE) return;
      const live = talkRoom !== null;
      // Слово завжди ВГОЛОС. Якщо віче йде — воно вклинюється в нього; якщо ні — стає темою
      // тут-таки, інакше поле в тихій локації нічого не робило й читалось як зламане.
      const asTopic = (): Promise<unknown> => sendCommand(LIVE_URL, {
        kind: "topic", text, place: openPlace ?? board.where,
        key: `room-${Date.now().toString(36)}`,
      }).then(async (r) => {
        // ★ Кажемо, СКІЛЬКИ чекати. Виконавець один, і поки він веде чуже віче, слово чесно лежить
        // у черзі — а на екрані це виглядало як «нічого не сталось». Заміряно: прогін ~2 хв, коли
        // шлюз повільний (медіана Mamay 6.8 с проти 1.8 с у спокійний час).
        const h = await fetchHealth(LIVE_URL);
        const ahead = h?.queue?.pending ?? 0;
        room.notice(ahead > 0
          ? `Слово в черзі: перед ним ${ahead}. Село візьметься, щойно звільниться.`
          : "Тему кинуто селу — зараз почнуть.");
        return r;
      });
      // ★ Слово не має пропадати через те, що віче саме скінчилось.
      //
      // Доти рішення «вклинитись у віче чи кинути тему» ухвалював фронт за своєю змінною, а ядро
      // тим часом уже закрило віче й чесно відповідало 409 «зараз віча немає». Слово гинуло, і
      // глядач бачив «ядро не відповідає» — хоч ядро відповіло. Тепер відмова саме з цієї
      // причини означає «те саме слово, але темою».
      const sent = live
        ? sendCommand(LIVE_URL, { kind: "say", text })
            .catch((err) => {
              if (err instanceof CommandRefused && err.status === 409) return asTopic();
              throw err;
            })
        : asTopic();
      void sent.catch((err) => room.notice(
        err instanceof CommandRefused
          ? `Ядро не взяло слово: ${err.message}`
          : "Слово не доїхало — ядро не відповідає."));
    },
  );
  // Маски прохідності всіх локацій (разом 100 КБ) тягнемо наперед: без них перший вхід у кімнату
  // мусив би чекати на завантаження, доки люди ще не сіли на підлогу.
  const warmMasks = (): void => LivingRoom.warm(
    Object.values(SOCIAL_ROOM).map((r) => r.mask).filter((m): m is string => Boolean(m)));
  // ★ Гріємо маски ПІСЛЯ хмар, а не «коли буде вільно».
  //
  // `requestIdleCallback` спрацьовував рівно тоді, коли головний потік звільнявся, — тобто саме
  // під час розходження завіси. Сім масок декодуються не миттєво, і ця робота лягала на єдині
  // секунди, які глядач бачить як рух. Тепер вони чекають, поки хмари догорнуться.
  window.setTimeout(warmMasks, 5000);

  /**
   * Село затягує хмарами й розводить їх уже над локацією, де сходиться віче.
   *
   * Доти місце розмови було лише написом на бирці: ядро справді вело інший процес у шинку й у
   * церкві, а на екрані нічого не мінялось. Тепер видно, КУДИ пішли люди.
   */
  const enterTalkRoom = async (placeId: string, parts: TalkPart[]): Promise<void> => {
    const kind = roomFor(placeId);
    const r = SOCIAL_ROOM[kind];
    if (!r) return;
    talkRoom = kind;
    openPlace = placeId;
    // Спершу СЕЛО ДУМАЄ — на мапі, при людях, що вже стоять на місці. Хмари беремо аж тоді, коли
    // порядок складено: один структурований виклик оркестратора йде десятки секунд, і кидати
    // глядача в порожню локацію на цей час немає сенсу.
    planReady = false;
    // ★ ЛОКАЦІЯ ВІДКРИВАЄТЬСЯ ОДРАЗУ, а чекання переїхало всередину неї.
    //
    // Доти глядач дивився на мапу з плашкою «Село думу думає…», доки ядро не віддасть перше слово,
    // і лише тоді бачив завісу й кімнату. Заміряно в браузері: плашка висіла 6.5 с, кімната
    // відкривалась на 7.1-й — при тому що ядро віддає перше слово за 2.5 с. Чотири секунди з семи
    // були накладними витратами показу, а не думання.
    if (talkRoom !== kind) return;
    await curtain.sweep(() => {
      inspector.close();
      board.close();
      groupTalk.close();
      ports.setEnabled(false);
      whisper.classList.remove("on");
      const cast: RoomCast[] = parts.map((p) => ({ id: p.role, name: p.name, vid: p.id }));
      room.open(r.bg, r.name, cast, r.floor, r.mask, { figScale: FIG });
      room.setLive(true);
      for (const w of preRoom.splice(0)) {
        room.addPerson(castOf(w.who));
        room.enqueue(w.who, w.text, w.deed, w.toward);
      }
      stoppedAt = performance.now();
      renderer.app.ticker.stop();
    });
    // Чекання тепер видно В КІМНАТІ: люди вже на місці, і зрозуміло, що саме триває.
    room.waiting(true);
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
          // Кімната вже відкрита, тож причину кажемо в ній, а не на мапі за спиною глядача.
          room.waiting(false);
          room.notice(`Ядро спинилось: ${h.stoppedReason ?? h.lastError ?? h.state}`);
          return;
        }
      }
    }
    mapThink("");
    room.waiting(false);
    // Слово, що прийшло, доки глядач чекав у кімнаті, звучить тут — черга кімнати вже відкрита.
    for (const w of preRoom.splice(0)) {
      room.addPerson(castOf(w.who));
      room.enqueue(w.who, w.text, w.deed, w.toward);
    }
  };

  // Скарга — завжди під рукою, у кутку: людина, що впіймала баг, не має шукати, куди про нього
  // сказати. Місце («де саме») підставляється саме, бо гість його не назве.
  new Gripe(() => (talkRoom ? `локація:${talkRoom}` : "мапа"));

  const ports = new Ports(renderer.world, scene.pois, {
    wasDrag: () => Boolean(renderer.camera?.wasDrag()),
    onHover: (p, x, y) => {
      if (!p) {
        whisper.classList.remove("on");
        return;
      }
      // Підпис місця — лише поки над ним нікого не чути: доти він накривав саму репліку, і
      // «Площа» лежала поверх слів. Підпис — підказка, репліка — зміст.
      if (director.speaking()) {
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
        hideHint();
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
        stoppedAt = performance.now();
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
    if (e.key !== "Escape") return;
    // Escape на відкритому питанні — це відповідь «ні», а не другий вихід поверх першого.
    if (ask.classList.contains("on")) {
      closeAsk();
      return;
    }
    leaveTalk();
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
      case "run.started":
        // Ядро взялось за тему — відтепер розмова коштує грошей, і є що завершувати.
        liveRun = ev.runId;
        break;
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
        // ★ Поки віче йде в ЛОКАЦІЇ, над мапою не звучить нічого.
        //
        // Доти кожна репліка створювала ще й бульбашку на діорамі — невидиму під кімнатою, але
        // живу свої сім секунд. Вийшовши й наблизивши камеру, глядач ловив на селянинові чужий
        // голос із щойно закритої розмови («проти. …») — репліку, якої ніхто на мапі не казав.
        if (!talkRoom) director.speak(who, ev.payload.text, who === "pip" ? "doubt" : "voice");
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
        // Розмова догоріла сама: питати про завершення більше нема про що, хай навіть глядач ще
        // дочитує притримані репліки — ядро на них уже не витрачається.
        liveRun = null;
        closeAsk();
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
        liveRun = null;
        closeAsk();
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
        // ★ Кінець на прохання — не поламка, і читатись мусить інакше.
        //
        // Тим самим типом події їдуть дві різні речі: стеля витрат чи смерть робітника (стан
        // ядра, стосується всіх) і згорнуте віче цього гостя. Перше — «Віче стало: <причина>»,
        // друге — рівно те, про що просили, без службового слова.
        if (ev.payload.stage === "viche") {
          liveRun = null;
          closeAsk();
          mapThink(ev.payload.reason ?? "Віче завершено");
        } else {
          // Зупинка — не дрібниця: її мусить бути видно, тому вона йде в той самий підпис знизу.
          mapThink(`Віче стало: ${ev.payload.reason ?? ev.payload.stage}`);
        }
        window.setTimeout(() => mapThink(""), 9000);
        break;
      case "event.happened": {
        // ★ Дошка — це стіна ТЕМ, а не звіт про прогони.
        //
        // Ухвала носить у собі текст теми, з якою прийшов гість, тож на стіні опинялись «ухвалили:
        // л ри лри» й таке інше. Наслідок ухвали й так видно в селі — доручений стоїть на своєму
        // місці, — а сам підсумок лишається в хроніці. На Дошку йде лише чутка, і лише така, що
        // читається як тема: кілька слів, а не набір літер.
        if (ev.payload.event.kind !== "rumour") break;
        const label = (ev.payload.event.label ?? "").trim();
        const words = label.split(/\s+/).filter((w) => /[а-яіїєґА-ЯІЇЄҐa-zA-Z]/.test(w));
        if (words.length < 3 || label.length < 12) break;
        board.addTopic({ text: label, heat: "warm" });
        break;
      }
      case "report.compiled": {
        renderer.weather?.setMood(ev.payload.chronicle.mood.valence);
        // Підсумок Оповідача — це і є кінець розмови. Тримаємо його в кімнаті, доки глядач не
        // дочитає чергу: ядро завершує прогін раніше, ніж прочитано останню репліку.
        const ch = ev.payload.chronicle;
        if (talkRoom && room.isOpen) {
          room.finale(ch.title || "Віче скінчилось", ch.narration || "");
        }
        break;
      }
      case "reflection.formed":
        // Рефлексія — це ВНУТРІШНЄ; її місце в інспекторі людини, а не в літописі. «Хтось
        // лишився при своєму» нічого не означало для того, хто дивиться на село.
        break;
      default:
        break;
    }
  });

  // Директор мусить знати зум: бульбашка тримає сталий розмір НА ЕКРАНІ, а не у світі.
  renderer.app.ticker.add(() => {
    director.setZoom(renderer.camera?.zoom ?? 1);
    const v = renderer.camera?.visibleWorldRect(0);
    if (v) director.setView(v);
  });

  // Підказка тримається САМОГО предмета: камера рухається, стрілка лишається на Дошці.
  renderer.app.ticker.add(() => {
    if (!hintPoi || !renderer.camera) return;
    // Вістря лягає майже на сам предмет: із запасом 70 воно висіло над порожньою травою.
    const c = renderer.camera.worldToClient(hintPoi.x, hintPoi.y - 18);
    firstHint.style.left = `${c.x}px`;
    firstHint.style.top = `${c.y}px`;
    // Плашку тримаємо В КАДРІ, а стрілку — на предметі. На вузькому екрані Дошка часто збоку, і
    // підказка вилазила за край (заміряно: left −52 при ширині 213). Тому зсуваємо саме плашку,
    // а вістря лишається там, куди треба тиснути.
    const plaq = firstHint.firstElementChild as HTMLElement | null;
    if (!plaq) return;
    const half = plaq.offsetWidth / 2;
    const lo = half + 10 - c.x;
    const hi = window.innerWidth - half - 10 - c.x;
    const shift = Math.round(Math.min(Math.max(0, lo), Math.max(0, hi)));
    // Зсуваємо ЛИШЕ плашку; вістря лишається на предметі, бо живе окремим елементом від точки.
    plaq.style.transform = `translateX(calc(-50% + ${shift}px))`;
  });

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
      stoppedAt = performance.now();
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
    if (!ev) continue;
    // ★ ЗАПИСАНИЙ `run.started` — не живий прогін, і в живому режимі його брати не можна.
    //
    // Голова фікстури тут потрібна рівно задля касту: люди мусять стояти на місцях ще під
    // хмарами. Але першим рядком запису йде `run.started` (`runId: "quiet-day"`), і стор чесно
    // ставив його як поточний прогін — тобто `liveRun` був піднятий із першого кадру КОЖНОГО
    // завантаження, ще до того, як гість щось кинув. Наслідок видно було на екрані: перша ж тема
    // з Дошки діставала табличку «Віче ще триває. Завершити його й почати нове?» про розмову,
    // якої ніколи не було, а Escape на тихій мапі — «Завершити його?». Заміряно на порожньому
    // ядрі (нуль подій у шині): 2 теми — 2 таблички.
    if (IS_LIVE && ev.known && ev.type === "run.started") continue;
    store.apply(ev); // casting.* → селяни спавняться зараз, під хмарами
  }
  // Тип — ПОРТ, а не котрась із двох реалізацій: решта фронта не має знати, звідки беруться
  // події, і саме тому притримка вміє забути чергу через необовʼязковий `drop`.
  const driver: EventSourcePort = IS_LIVE
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
  // ★ Хмари розходяться, коли село СПРАВДІ готове показатись.
  //
  // Текстури Pixi вивантажує на GPU лінькувато — під час ПЕРШОГО малювання. Тож завіса починала
  // розходитись рівно тоді, коли головний потік ще заливав кілька десятків текстур: хмари йшли
  // ривками, а з-під них проступала недомальована сцена. Тепер спершу мовчки малюємо кадр під
  // завісою (це й змушує вивантажити все), даємо два кадри на спокій — і аж тоді розводимо.
  renderer.app.renderer.render(renderer.app.stage);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(null))));
  renderer.dissipateIntro();

  // Підказку показуємо ПІСЛЯ хмар: під завісою її однаково не видно, а поява разом із селом
  // читається як частина відкриття, а не як спливне вікно.
  let seen = "1";
  try {
    seen = localStorage.getItem(HINT_SEEN) ?? "";
  } catch {
    seen = "";
  }
  if (!seen) {
    const board = scene.pois.find((p) => p.kind === "board");
    if (board) {
      hintPoi = board;
      hintTimer = window.setTimeout(() => {
        // Гість міг устигнути тицьнути сам — тоді підказці вже нема чого казати.
        if (hintPoi) firstHint.classList.add("on");
      }, 2600);
      // Підказка не має жити вічно: якщо гість пішов гуляти селом, вона своє сказала.
      // Довше, ніж було: 22 секунди не вистачало навіть роздивитись село.
      window.setTimeout(hideHint, 60000);
    }
  }
}

boot().catch((e: unknown) => {
  console.error("[boot] failed", e);
  const msg = e instanceof Error ? e.message : String(e);
  document.body.innerHTML = `<pre style="color:#f88;padding:20px;font:14px monospace">boot failed: ${msg}</pre>`;
});
