import { Container, Graphics, Sprite, Text, Texture } from "pixi.js";
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

const SKIN: Record<BubbleKind, {
  paper: number; ink: number; edge: number; italic: boolean; alpha: number;
}> = {
  // Один вигляд на весь застосунок: тепла хмарка з мʼяким кутом і темним обідком. Доти кожна
  // поверхня мала свій матеріал (рваний папір, деревина, темна плашка), і те саме речення
  // виглядало трьома різними речами залежно від того, звідки на нього дивишся.
  voice: { paper: 0xf7ecd2, ink: 0x2e2009, edge: 0x6f4f22, italic: false, alpha: 1 },
  whisper: { paper: 0xe6eef7, ink: 0x22303f, edge: 0x4f6a86, italic: true, alpha: 0.95 },
  doubt: { paper: 0xf3e3bd, ink: 0x4a3520, edge: 0x7d5a2a, italic: true, alpha: 1 },
};

/** Обрис хмарки: заокруглений прямокутник і хвіст ОДНИМ контуром, щоб обідок ішов і по хвосту. */
function cloudPath(g: Graphics, w: number, h: number, tail: number, rad: number): void {
  g.moveTo(rad, 0);
  g.lineTo(w - rad, 0);
  g.quadraticCurveTo(w, 0, w, rad);
  g.lineTo(w, h - rad);
  g.quadraticCurveTo(w, h, w - rad, h);
  g.lineTo(w / 2 + 13, h);
  g.lineTo(w / 2 + 1, h + tail);
  g.lineTo(w / 2 - 13, h);
  g.lineTo(rad, h);
  g.quadraticCurveTo(0, h, 0, h - rad);
  g.lineTo(0, rad);
  g.quadraticCurveTo(0, 0, rad, 0);
  g.closePath();
}

// Ширина рядка й скільки рядків показуємо. Довше — у вікні розмови й у літописі, де є місце.
const BUBBLE_WIDTH = 260;
const BUBBLE_CHARS_PER_LINE = 38;
const BUBBLE_LINES = 3;
// Скільки бульбашка висить. Довгу треба встигнути прочитати, коротка не має стирчати.
const BUBBLE_BASE_S = 1.1;
const BUBBLE_PER_CHAR_S = 0.045;
const BUBBLE_MAX_S = 7;
// Більше двох одночасно — стіна тексту над селом; шість штук ми вже бачили.
const BUBBLE_AT_ONCE = 2;
// Нижче цього зуму людей не розібрати, тож і підписувати нема кого.
const BUBBLE_MIN_ZOOM = 0.95;
// Наскільки швидше йде той, кого покликали на віче.
const HURRY = 2.6;

/** Обрізає ЦІЛИМИ словами до N рядків: інакше репліка кінчалась на півслові. */
function clampLines(text: string, perLine: number, lines: number): string {
  const words = text.split(/\s+/);
  const out: string[] = [];
  let row = "";
  for (const w of words) {
    const next = row ? `${row} ${w}` : w;
    if (next.length <= perLine) {
      row = next;
      continue;
    }
    out.push(row);
    row = w;
    if (out.length === lines) break;
  }
  if (out.length < lines && row) out.push(row);
  const cut = out.length === lines && out.join(" ").length < text.length;
  return out.join(" ") + (cut ? "…" : "");
}

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

  /**
   * Поставити людину НА місце одразу, без ходьби.
   *
   * Віче більше не збирають ходою через усе село: тицьнув тему — і громада вже там. Хода лишалась
   * хвилиною порожнього екрана, а сенсу не додавала: рішення однаково ухвалюють у локації.
   */
  placeAt(id: string, to: PlaceRef): void {
    const r = this.recs.get(id);
    if (!r) return;
    let nx = to.x;
    let ny = to.y;
    if (to.poi && this.pois.has(to.poi)) {
      const poi = this.pois.get(to.poi)!;
      nx = poi.x;
      ny = poi.y;
    }
    if (nx === undefined || ny === undefined) return;
    const cell = this.grid.nearWalk(nx, ny);
    if (!cell) return;
    const at = this.grid.cellCenter(cell[0], cell[1]);
    r.x = at.x;
    r.y = at.y;
    r.cell = cell;
    r.path = null;
    r.state = "idle";
  }

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

  speak(id: string, text: string, kind: BubbleKind = "voice"): void {
    const r = this.recs.get(id);
    if (!r) return;
    // Найстарішу бульбашку прибираємо самі: шість одночасно вже давали стіну тексту над селом.
    const live = [...this.recs.values()].filter((x) => x.bubble && x.id !== id);
    while (live.length >= BUBBLE_AT_ONCE) {
      const oldest = live.reduce((a, b) => (a.bubbleT <= b.bubbleT ? a : b));
      this.clearBubble(oldest);
      live.splice(live.indexOf(oldest), 1);
    }
    this.clearBubble(r);
    r.bubble = this.makeBubble(text, kind);
    // Час життя від ДОВЖИНИ: сталі 3.6 с не давали дочитати довгу й тримали коротку без потреби.
    r.bubbleT = Math.min(BUBBLE_MAX_S, BUBBLE_BASE_S + text.length * BUBBLE_PER_CHAR_S);
    this.world.addChild(r.bubble);
  }

  /** На загальному плані селянин ~20 px: бульбашка там вища за хату й нічого не пояснює. */
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
        r.bubble.x = r.x;
        r.bubble.y = r.y - r.tex.height * r.sc - 6 * this.SCL;
        r.bubble.zIndex = r.y + 2;
        // Розмір бульбашки НЕ фіксований: вона мусить лишатись сталою на екрані, а не рости
        // разом зі світом, коли камера віддаляється.
        const k = Math.max(0.55, Math.min(1.25, 1 / Math.max(0.35, this.zoom)));
        r.bubble.scale.set(k);
        r.bubble.visible = this.zoom >= BUBBLE_MIN_ZOOM;
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

  /**
   * Бульбашка — ГОЛОС: світлий папір із чорнилом, а не темна плашка.
   *
   * Доти вона була тим самим темним прямокутником, що й службовий літопис, підпис імені й кнопки:
   * вісім різних значень одним виглядом. Тут голос дістає власний матеріал, і його вже не сплутати
   * зі службовим рядком.
   *
   * Хвостик був і раніше — 9 пікселів, тобто невидимий на тлі 40-піксельного селянина. Через це
   * репліка читалась як «висить сама по собі», хоча привʼязка працювала. Тепер хвіст масштабується
   * разом із рештою.
   */
  private makeBubble(text: string, kind: BubbleKind = "voice"): Container {
    const skin = SKIN[kind];
    const c = new Container();
    // Не ріжемо на півслові: переносимо й обрізаємо ЦІЛИМИ рядками, крапки лише в кінці.
    const label = clampLines(text, BUBBLE_CHARS_PER_LINE, BUBBLE_LINES);
    const t = new Text(label, {
      fontFamily: "Georgia, 'Times New Roman', serif",
      fontSize: 15,
      fontWeight: "500",
      fontStyle: skin.italic ? "italic" : "normal",
      fill: skin.ink,
      wordWrap: true,
      wordWrapWidth: BUBBLE_WIDTH,
      align: "left",
      lineHeight: 20,
    });
    const padX = 20;
    const padY = 13;
    const tail = 17;
    const RAD = 18;
    const w = t.width + padX * 2;
    const h = t.height + padY * 2;

    const shade = new Graphics();
    shade.beginFill(0x2b1c0c, 0.28);
    cloudPath(shade, w, h, tail, RAD);
    shade.endFill();
    shade.x = 2;
    shade.y = 5;

    const bg = new Graphics();
    bg.beginFill(skin.paper, 1);
    bg.lineStyle(3.6, skin.edge, 1);
    cloudPath(bg, w, h, tail, RAD);
    bg.endFill();

    t.x = padX;
    t.y = padY;
    c.addChild(shade, bg, t);
    c.alpha = skin.alpha;
    c.pivot.set(w / 2, h + tail);
    return c;
  }

}
