import { Container, Sprite, Texture } from "pixi.js";
import type { PlaceRef, POI, VillagerPublic } from "@ploshcha/contract-ts";
import type { WalkGrid } from "./WalkGrid";
import { ovalShadow } from "../util/gfx";

type Cell = [number, number];

/** Ознака → колір. Стримано: село має лишатись цілісним, а не набором різнокольорових фігурок. */
function tintOf(traits: string[]): number {
  // Ознаки — це ПОЛЮСИ («молодий»/«старий»), не назви осей. Ядро колись слало назву осі, і молода
  // дівчина приїжджала з міткою «старий», тобто фарбувалась сивиною за протилежною ознакою.
  let r = 1, g = 1, b = 1;
  if (traits.includes("старий")) { r *= 0.94; g *= 0.94; b *= 0.97; }
  if (traits.includes("молодий")) { r *= 1.03; g *= 1.02; b *= 1.0; }
  if (traits.includes("заможний")) { r *= 1.05; g *= 1.0; b *= 0.92; }
  if (traits.includes("бідний")) { r *= 0.97; g *= 0.96; b *= 0.95; }
  if (traits.includes("прийшлий")) { r *= 0.95; g *= 0.99; b *= 1.06; }
  const to255 = (v: number): number => Math.max(0, Math.min(255, Math.round(v * 255)));
  return (to255(r) << 16) | (to255(g) << 8) | to255(b);
}

/** Три голоси, один матеріал: сказане вголос, пошептане, сумнів. */
export type BubbleKind = "voice" | "whisper" | "doubt";



// Ширина рядка й скільки рядків показуємо. Довше — у вікні розмови й у літописі, де є місце.
// Скільки бульбашка висить. Довгу треба встигнути прочитати, коротка не має стирчати.
const BUBBLE_MIN_ZOOM = 0.95;
/**
 * Наскільки швидше йде той, кого покликали на віче. Число ЗАМІРЯНЕ, а не вибране на смак.
 *
 * Замір 2026-08-29 у браузері на справжній сітці ходьби (`bfs` від клітинки кожного селянина до
 * Площі, сума відстаней між центрами клітинок): 239, 1313, 1413, 1748, 2347, 2762, 2762 і 3573
 * світових пікселі на вісьмох людей, медіана 2047. Буденний крок — `walk` × `SCL`, тобто 20-27
 * пікселів на секунду, тож при старому множнику 2.6 медіанна хода тривала 33 с, а найдовша 68 с.
 * Стільки глядач на мапі не стоїть: він іде в локацію на ПЕРШІЙ репліці, а дорога ядра до неї
 * міряється поодинокими викликами Мамая (медіана 5.3 с, максимум 22.1 с — `/health` живого ядра),
 * тобто нікого з них не було б видно на місці. При 6 медіанна хода — 14 с, найближчі доходять за
 * 1.5-6 с, найдальший лишається в дорозі 30 с і доходить уже без глядача. Затримки це не додає
 * НІКОМУ: хмари беруться по готовності порядку, а не по тому, чи всі зійшлись.
 */
const HURRY = 6;


interface Rec {
  id: string;
  name: string;
  sprite: Sprite;
  shadow: Sprite;
  tex: Texture;
  frames: Texture[] | null;
  sc: number;
  x: number;
  y: number;
  face: number;
  path: Cell[] | null;
  pi: number;
  speed: number;
  walk: number; // буденний крок, до якого повертаємось після віча
  bob: number;
  state: "idle" | "walk";
  cell: Cell;
  bubble?: Container;
  bubbleT: number;
  wanderTimer: number;
}

/** Звʼязує селян контракту зі спрайтами: спавн, рух по BFS, бульбашки реплік. */
export class AgentDirector {
  private recs = new Map<string, Rec>();
  /** Кого зараз тримає віче: така людина не тиняється, доки розмова йде. */
  private held = new Set<string>();
  private zoom = 1;
  /** Що зараз у кадрі (світові координати). Потрібно, щоб репліка не вилазила за екран. */
  private view: { x0: number; y0: number; x1: number; y1: number } | null = null;
  private nextTex = 0;

  constructor(
    private world: Container,
    private grid: WalkGrid,
    private pois: Map<string, POI>,
    private charTex: Texture[],
    private roleFrames: Map<string, Texture[]>,
    private SCL: number,
  ) {}

  spawn(villagers: VillagerPublic[]): void {
    for (const v of villagers) {
      if (this.recs.has(v.id)) continue;
      const walk = 10 + Math.random() * 4;
      const cell = this.grid.randCell();
      if (!cell) continue;
      const p = this.grid.cellCenter(cell[0], cell[1]);
      const frames = this.roleFrames.get(v.role) ?? null;
      const tex = frames ? frames[0] : this.charTex[this.nextTex++ % Math.max(1, this.charTex.length)];
      const sprite = new Sprite(tex);
      sprite.anchor.set(0.5, 1);
      const sc = (30 * this.SCL) / (tex.height || 1);
      const shadow = new Sprite(ovalShadow()); // м'яка контактна калюжа під ногами
      shadow.anchor.set(0.5, 0.5);
      const dh = tex.height * sc;
      shadow.width = dh * 0.42;
      shadow.height = dh * 0.15;
      shadow.alpha = 0.32;
      this.world.addChild(shadow);
      // ★ Вигляд рахується з ОЗНАК, які надіслало ядро: старший сивіший і тьмяніший, заможніший
      // тепліший. Ядро не вигадує кольорів, сцена не вигадує норову — кожен робить своє.
      sprite.tint = tintOf(v.traits ?? []);
      this.world.addChild(sprite);
      this.recs.set(v.id, {
        id: v.id, name: v.name, sprite, shadow, tex, frames, sc,
        x: p.x, y: p.y, face: Math.random() < 0.5 ? -1 : 1,
        path: null, pi: 0, speed: walk, walk, bob: 0, state: "idle",
        cell, bubbleT: 0, wanderTimer: 1 + Math.random() * 3,
      });
    }
  }

  /** Найближчий селянин до світ-точки в межах maxDist (native) — для кліку-інспекту. */
  /**
   * Чи всі ці люди вже дійшли.
   *
   * Віче не має починатись, поки село ще сходиться: інакше «зібрались на площі» — це напис, а не
   * подія. Той, кого сцена не знає, не рахується — інакше чекання ніколи не скінчиться.
   */
  /**
   * Тримати цих людей при вічі або відпустити назад у їхнє блукання.
   *
   * Покликані на віче ще й ПОСПІШАЮТЬ: буденний крок через усе село — це під хвилину чекання
   * порожнього екрана, а поспіх сам собою читається як «сталося щось важливе».
   */
  hold(ids: string[], on: boolean): void {
    for (const id of ids) {
      const r = this.recs.get(id);
      if (on) {
        this.held.add(id);
        if (r) r.speed = r.walk * HURRY;
      } else {
        this.held.delete(id);
        if (r) r.speed = r.walk;
      }
    }
  }

  // ★ Миттєвого розставляння (`placeAt`) тут БІЛЬШЕ НЕМАЄ, і метод прибрано, а не лишено про
  // запас. Він з'явився, коли хмари бралися аж по тому, як усі зійшлись, — тоді хода справді
  // тримала глядача перед порожнім селом. Відтоді локація відкривається по готовності порядку
  // (`enterTalkRoom`), тобто хода вже нікого не затримує, а стрибок через півсела лишався видним
  // як поламка: замір у браузері — четверо перелетіли 507, 662, 837 і 930 пікселів за один кадр
  // у 2 мс. Живий метод «постав одразу» повернув би це першою ж правкою, тож його немає.

  /** Середина юрби — камері є за чим іти. */
  centroid(ids: string[]): { x: number; y: number } | null {
    let n = 0, x = 0, y = 0;
    for (const id of ids) {
      const r = this.recs.get(id);
      if (!r) continue;
      x += r.x;
      y += r.y;
      n++;
    }
    return n ? { x: x / n, y: y / n } : null;
  }

  /** Чи всі вже БІЛЯ місця (останній підходить), а не тільки коли зовсім спинились. */
  allNear(ids: string[], wx: number, wy: number, dist: number): boolean {
    return ids.every((id) => {
      const r = this.recs.get(id);
      return !r || Math.hypot(r.x - wx, r.y - wy) <= dist;
    });
  }

  gathered(ids: string[]): boolean {
    return ids.every((id) => {
      const r = this.recs.get(id);
      return !r || r.path === null;
    });
  }

  /**
   * Хто де стоїть — РОЗПОДІЛ, а не «хто поруч».
   *
   * ★ Людина може бути лише в одному місці. Пошук «усі в радіусі» цього не гарантував: площа,
   * церква й дзвіниця стоять близько, тож той самий селянин потрапляв у каст трьох локацій
   * одночасно — і на мапі, і в кожній із них. Тому кожного приписуємо до ОДНОГО місця —
   * найближчого в межах `maxDist`, — і локація бере лише своїх.
   */
  occupancy(pois: { id: string; x: number; y: number }[], maxDist: number): Map<string, string[]> {
    const out = new Map<string, string[]>();
    for (const p of pois) out.set(p.id, []);
    for (const r of this.recs.values()) {
      let best: string | null = null;
      let bd = maxDist;
      for (const p of pois) {
        const d = Math.hypot(r.x - p.x, r.y - p.y);
        if (d < bd) {
          bd = d;
          best = p.id;
        }
      }
      if (best) out.get(best)!.push(r.id);
    }
    return out;
  }

  nearestAt(wx: number, wy: number, maxDist: number): string | null {
    let best: string | null = null;
    let bd = maxDist;
    for (const r of this.recs.values()) {
      const d = Math.hypot(r.x - wx, r.y - wy);
      if (d < bd) {
        bd = d;
        best = r.id;
      }
    }
    return best;
  }

  moveTo(id: string, to: PlaceRef): void {
    const r = this.recs.get(id);
    if (!r) return;
    let nx: number;
    let ny: number;
    if (to.poi && this.pois.has(to.poi)) {
      const poi = this.pois.get(to.poi)!;
      nx = poi.x;
      ny = poi.y;
    } else if (to.x !== undefined && to.y !== undefined) {
      nx = to.x;
      ny = to.y;
    } else {
      return;
    }
    const goal = this.grid.nearWalk(nx, ny);
    if (!goal) return;
    // Ціль може бути в ізольованій кишені сітки — тоді йдемо настільки близько, наскільки можна.
    const path = this.grid.bfs(r.cell, goal) ?? this.grid.pathToward(r.cell, goal);
    if (path && path.length > 1) {
      r.path = path;
      r.pi = 1;
      r.state = "walk";
    }
  }

  speak(_id: string, _text: string, _kind: BubbleKind = "voice"): void {
    // ★ НА МАПІ РЕПЛІК НЕМАЄ ВЗАГАЛІ.
    //
    // Бульбашка над селом породила три різні поламки поспіль: репліки, що спливали при наближенні
    // вже після завершення розмови; хвостик, притиснутий до краю кадру й тому наведений на
    // порожню землю (заміряно відрив 897, 1342 і 2405 пікселів); і слова мовця, який пішов, що
    // лишались висіти над місцем, де його вже немає. Щоразу лікували наслідок.
    //
    // Розмова має ОДНЕ місце — намальовану локацію, де видно, хто говорить і кому. Мапа лишається
    // мапою: по ній видно, хто куди пішов. Слово при цьому не гине — воно є в стенограмі, у
    // хроніці й у самій кімнаті.
    return;
  }

  setView(rect: { x0: number; y0: number; x1: number; y1: number }): void {
    this.view = rect;
  }

  /** Чи цей селянин зараз у кадрі. Репліку малюємо лише над тим, кого видно. */
  private inView(r: Rec): boolean {
    const v = this.view;
    return !v || (r.x >= v.x0 && r.x <= v.x1 && r.y >= v.y0 && r.y <= v.y1);
  }

  /** Чи хтось саме зараз говорить (є жива бульбашка). */
  speaking(): boolean {
    for (const r of this.recs.values()) if (r.bubble) return true;
    return false;
  }

  setZoom(zoom: number): void {
    this.zoom = zoom;
  }

  update(dt: number): void {
    for (const r of this.recs.values()) {
      // амбієнтне блукання: коли стоїть — час від часу йде до близької точки
      if (r.state === "idle" && !this.held.has(r.id)) {
        r.wanderTimer -= dt;
        if (r.wanderTimer <= 0) {
          r.wanderTimer = 2 + Math.random() * 4;
          const t = this.grid.randCellNear(r.cell, 14);
          const path = t ? this.grid.bfs(r.cell, t) : null;
          if (path && path.length > 1) {
            r.path = path;
            r.pi = 1;
            r.state = "walk";
          }
        }
      }
      if (r.state === "walk" && r.path) {
        if (r.pi >= r.path.length) {
          r.state = "idle";
          r.path = null;
        } else {
          const c = this.grid.cellCenter(r.path[r.pi][0], r.path[r.pi][1]);
          const dx = c.x - r.x;
          const dy = c.y - r.y;
          const d = Math.hypot(dx, dy);
          const step = r.speed * this.SCL * dt;
          if (d <= step) {
            r.x = c.x;
            r.y = c.y;
            r.cell = r.path[r.pi];
            r.pi++;
          } else {
            r.x += (dx / d) * step;
            r.y += (dy / d) * step;
            if (Math.abs(dx) > 0.5) r.face = dx < 0 ? -1 : 1;
            r.bob += dt * 6.5;
          }
        }
      }
      const walking = r.state === "walk" && r.path !== null;
      if (!walking) r.bob += dt * 2.3; // повільне «дихання» у спокої
      const baseH = r.tex.height * r.sc;
      const s = Math.sin(r.bob);
      // кадрова анімація: спокій = кадр 0, хода чергує кадри 1/2
      if (r.frames) {
        const f = walking ? (s >= 0 ? r.frames[1] : r.frames[2] ?? r.frames[1]) : r.frames[0];
        if (f && r.sprite.texture !== f) r.sprite.texture = f;
      }
      const bounce = walking ? Math.abs(s) * baseH * 0.02 : 0;
      r.sprite.x = r.x;
      r.sprite.y = r.y - bounce;
      r.sprite.rotation = 0;
      r.sprite.scale.set(r.sc * r.face, r.sc);
      r.sprite.zIndex = r.y | 0; // ціле → мікрорух не «бруднить» сортування щокадру
      r.shadow.x = r.x;
      r.shadow.y = r.y; // калюжа під ногами (anchor 0.5,0.5)
      r.shadow.alpha = walking ? 0.32 - Math.abs(s) * 0.05 : 0.32;
      r.shadow.zIndex = (r.y | 0) - 1;
      if (r.bubble) {
        r.bubbleT -= dt;
        // ★ Репліка над УСІМ, а не серед спрайтів.
        //
        // Глибина рахувалась як `y + 2`, тобто бульбашка стояла в тій самій черзі, що й люди,
        // хати й дерева: усе, що нижче по екрану, лізло на текст. Репліки живуть окремим ярусом
        // над сценою (нижче лише невидимі хіт-зони портів на 2e9), а між собою — за глибиною.
        r.bubble.zIndex = 1_500_000_000 + (r.y | 0);
        // Розмір бульбашки НЕ фіксований: вона мусить лишатись сталою на екрані, а не рости
        // разом зі світом, коли камера віддаляється.
        const k = Math.max(0.55, Math.min(1.25, 1 / Math.max(0.35, this.zoom)));
        r.bubble.scale.set(k);
        // ★ Хвостик НЕ ПОКИДАЄ мовця — ні на піксель убік.
        //
        // Доти репліку заганяли в кадр цілком: `bubble.x` ставав межею видимого прямокутника, хоч
        // би де стояв мовець. Але позиція бульбашки — це вістря хвостика (`pivot` на `h + tail`),
        // тобто зсув на два світові екрани лишав хвостик стирчати над порожньою дорогою. Заміряно
        // в браузері 2026-08-29 на живому вічі при зумі 1.7: **1391 кадр** із відірваною
        // бульбашкою, відрив до **2405** світових пікселів, і рівно ці дві репліки власник і
        // сфотографував над порожнім полем.
        //
        // У кадр репліку заганяє тепер не зсув, а правило: мовця не видно — репліки немає. Тому по
        // горизонталі — рівно над головою, а по вертикалі лишається один-єдиний доводчик: якщо над
        // головою немає місця (людина при верхньому краю), бульбашка ОПУСКАЄТЬСЯ на самого мовця,
        // а не відлітає вбік. Хвостик у найгіршому разі лежить на його ж фігурі.
        const b = r.bubble.getLocalBounds();
        const v = this.view;
        r.bubble.x = r.x;
        r.bubble.y = v
          ? Math.max(r.y - r.tex.height * r.sc - 6 * this.SCL, v.y0 + b.height * k + 8)
          : r.y - r.tex.height * r.sc - 6 * this.SCL;
        // Відʼїхали або мовець вийшов із кадру — репліку прибираємо назовсім, щоб вона не
        // «вигулькнула» при поверненні й не лишилась висіти там, де вже нікого немає.
        if (this.zoom < BUBBLE_MIN_ZOOM || !this.inView(r)) {
          this.clearBubble(r);
          continue;
        }
        if (r.bubbleT <= 0) this.clearBubble(r);
      }
    }
  }

  private clearBubble(r: Rec): void {
    if (r.bubble) {
      this.world.removeChild(r.bubble);
      r.bubble.destroy({ children: true });
      r.bubble = undefined;
    }
  }
}
