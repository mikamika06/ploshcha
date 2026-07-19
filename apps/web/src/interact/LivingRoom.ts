export interface RoomCast {
  id: string; // роль (для спрайта /assets/roles/<id>/)
  name: string;
  vid?: string; // id селянина у сторі → відкриття аналітики-інспектора
}
export type Pt = [number, number];

interface RV {
  cast: RoomCast;
  el: HTMLImageElement;
  frames: string[];
  cur: number;
  x: number;
  y: number;
  tx: number;
  ty: number;
  state: "idle" | "walk";
  timer: number;
  face: number;
  bob: number;
  bubble: HTMLElement | null;
  bubbleT: number;
}

const AMBIENT = [
  "Гарна нині погода.",
  "Чув, чумак приїхав.",
  "Ой, кумо, і не кажи!",
  "Та де ж воно поділося…",
  "Хліб сьогодні вдався.",
  "На вечорниці підеш?",
  "Ех, натомився за день.",
  "Будьмо, люди добрі.",
];

const rand = (a: number, b: number): number => a + Math.random() * (b - a);

/** Точка в опуклому багатокутнику (кільце вершин). */
function inPoly(x: number, y: number, poly: Pt[]): boolean {
  let sign = 0;
  for (let i = 0; i < poly.length; i++) {
    const [ax, ay] = poly[i];
    const [bx, by] = poly[(i + 1) % poly.length];
    const cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
    if (cross !== 0) {
      const s = cross > 0 ? 1 : -1;
      if (sign === 0) sign = s;
      else if (s !== sign) return false;
    }
  }
  return true;
}

/** Діамант відкритої підлоги шинку (частки): ближній/правий/дальній/лівий. */
const DEFAULT_FLOOR: Pt[] = [
  [0.55, 0.93],
  [0.73, 0.62],
  [0.5, 0.53],
  [0.34, 0.72],
];

/**
 * Жива кімната: iso-box-сцена (contain) + наші спрайти-селяни ходять по ДІАМАНТУ підлоги,
 * тиняються й перекидаються репліками. Все DOM, поза Pixi-камерою.
 */
export class LivingRoom {
  private root: HTMLElement;
  private bg: HTMLImageElement;
  private field: HTMLElement;
  private nameEl: HTMLElement;
  private vs: RV[] = [];
  private raf = 0;
  private last = 0;
  private floor: Pt[] = DEFAULT_FLOOR;
  private figScale = 1; // масштаб фігур (outdoor-сцени великі → люди дрібніші)
  private bbox = { x0: 0, x1: 1, y0: 0, y1: 1 };
  private mask: Uint8Array | null = null; // прохідна маска з Nano Banana (зелене=1); null → полігон
  private mgw = 0;
  private mgh = 0;

  constructor(
    private onClose: () => void,
    private onInspect?: (vid: string) => void,
  ) {
    this.root = document.createElement("div");
    this.root.className = "room";
    this.root.innerHTML = `
      <div class="room-stage">
        <img class="room-bg" alt="">
        <div class="room-field"></div>
      </div>
      <button class="room-back" type="button">← до села</button>
      <div class="room-name"></div>
      <div class="room-empty">— тут зараз нікого —</div>`;
    this.bg = this.root.querySelector(".room-bg") as HTMLImageElement;
    this.field = this.root.querySelector(".room-field") as HTMLElement;
    this.nameEl = this.root.querySelector(".room-name") as HTMLElement;
    this.root.querySelector(".room-back")!.addEventListener("click", () => this.onClose());
    document.getElementById("stage")!.appendChild(this.root);
  }

  open(
    bgUrl: string,
    name: string,
    cast: RoomCast[],
    floor: Pt[] = DEFAULT_FLOOR,
    maskUrl?: string,
    opts?: { cover?: boolean; token?: { verb: string; lines: string[] }; figScale?: number },
  ): void {
    this.figScale = opts?.figScale ?? 1;
    // стейдж бере аспект самої картинки → field 0-1 == image 0-1 (маска не з'їжджає)
    const stage = this.root.querySelector(".room-stage") as HTMLElement;
    this.bg.onload = (): void => {
      if (this.bg.naturalWidth) stage.style.aspectRatio = `${this.bg.naturalWidth} / ${this.bg.naturalHeight}`;
    };
    this.bg.src = bgUrl;
    this.root.classList.toggle("room--cover", !!opts?.cover);
    this.nameEl.textContent = name;
    this.floor = floor;
    const xs = floor.map((p) => p[0]);
    const ys = floor.map((p) => p[1]);
    this.bbox = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
    this.mask = null;
    if (maskUrl) this.loadMask(maskUrl);
    this.field.innerHTML = "";
    this.vs = cast.map((c) => {
      const el = document.createElement("img");
      el.className = "rv";
      el.draggable = false;
      const frames = [0, 1, 2].map((n) => `/assets/roles/${c.id}/${n}.png`);
      el.src = frames[0];
      const [x, y] = this.randFloor();
      const rv: RV = {
        cast: c, el, frames, cur: 0, x, y, tx: x, ty: y,
        state: "idle", timer: rand(0.5, 3), face: 1, bob: 0, bubble: null, bubbleT: 0,
      };
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        // клік по селянину в локації → аналітика (як на головній карті); без vid — амбієнтна репліка
        if (rv.cast.vid && this.onInspect) this.onInspect(rv.cast.vid);
        else this.say(rv, AMBIENT[(Math.random() * AMBIENT.length) | 0]);
      });
      this.field.appendChild(el);
      return rv;
    });
    if (opts?.token) this.placeToken(opts.token);
    (this.root.querySelector(".room-empty") as HTMLElement).style.opacity = this.vs.length ? "0" : "1";
    this.root.classList.add("on");
    this.last = performance.now();
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(this.loop);
  }

  /** Завантажує walk-маску з Nano Banana у сітку (зелене=прохідне). Поки не завантажилась — полігон. */
  private loadMask(url: string): void {
    const img = new Image();
    img.onload = (): void => {
      const gw = 160;
      const gh = Math.max(1, Math.round((gw * img.height) / img.width));
      const cv = document.createElement("canvas");
      cv.width = gw;
      cv.height = gh;
      const ctx = cv.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, gw, gh);
      const d = ctx.getImageData(0, 0, gw, gh).data;
      const grid = new Uint8Array(gw * gh);
      for (let i = 0; i < gw * gh; i++) {
        const r = d[i * 4];
        const g = d[i * 4 + 1];
        const b = d[i * 4 + 2];
        grid[i] = g > 110 && r < 130 && b < 130 ? 1 : 0; // зелене = прохідне
      }
      // приймаємо маску, якщо в ній є бодай трохи зеленого (маленькі легітимні зони, як-от кузня,
      // теж валідні); поріг лише відсіює зовсім биту/незавантажену маску (~0%)
      let green = 0;
      for (let i = 0; i < grid.length; i++) green += grid[i];
      if (green > gw * gh * 0.012) {
        this.mask = grid;
        this.mgw = gw;
        this.mgh = gh;
        this.bbox = { x0: 0, x1: 1, y0: 0, y1: 1 };
        // маска вантажиться асинхронно — селян встигли розставити по полігону (fallback),
        // і хтось міг стати поза зоною (напр. на воді). Строга поstorкрокова перевірка їх
        // би там і замкнула. Тому пересаджуємо всіх, хто опинився не на масці, на валідну клітину.
        for (const v of this.vs) {
          if (!this.inFloor(v.x, v.y)) {
            const [x, y] = this.randFloor();
            v.x = x;
            v.y = y;
            v.tx = x;
            v.ty = y;
            v.state = "idle";
            v.timer = rand(0.3, 2);
          }
        }
      }
    };
    img.src = url;
  }

  private inFloor(x: number, y: number): boolean {
    if (this.mask) {
      const gx = Math.min(this.mgw - 1, Math.max(0, (x * this.mgw) | 0));
      const gy = Math.min(this.mgh - 1, Math.max(0, (y * this.mgh) | 0));
      return this.mask[gy * this.mgw + gx] === 1;
    }
    return inPoly(x, y, this.floor);
  }

  private randFloor(): Pt {
    for (let k = 0; k < 80; k++) {
      const x = rand(this.bbox.x0, this.bbox.x1);
      const y = rand(this.bbox.y0, this.bbox.y1);
      if (this.inFloor(x, y)) return [x, y];
    }
    return [(this.bbox.x0 + this.bbox.x1) / 2, (this.bbox.y0 + this.bbox.y1) / 2];
  }

  private loop = (): void => {
    if (!this.root.classList.contains("on")) return; // кімната закрита — зупиняємо цикл
    const now = performance.now();
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    const W = this.field.clientWidth;
    const H = this.field.clientHeight;
    for (const v of this.vs) {
      if (v.state === "idle") {
        v.timer -= dt;
        if (v.timer <= 0) {
          const [tx, ty] = this.randFloor();
          v.tx = tx;
          v.ty = ty;
          v.state = "walk";
        }
        if (!v.bubble && Math.random() < dt * 0.05) this.say(v, AMBIENT[(Math.random() * AMBIENT.length) | 0]);
      } else {
        const dx = v.tx - v.x;
        const dy = v.ty - v.y;
        const d = Math.hypot(dx, dy);
        const step = 0.05 * dt;
        if (d <= step) {
          v.x = v.tx;
          v.y = v.ty;
          v.state = "idle";
          v.timer = rand(1, 4);
        } else {
          const nx = v.x + (dx / d) * step;
          const ny = v.y + (dy / d) * step;
          if (this.inFloor(nx, ny)) {
            // крок дозволений лише якщо лишаємось на масці (СТРОГО по зоні)
            v.x = nx;
            v.y = ny;
            if (Math.abs(dx) > 0.0008) v.face = dx < 0 ? -1 : 1;
            v.bob += dt * 8;
          } else {
            // впершись у межу зони — зупиняємось і беремо нову ціль
            v.state = "idle";
            v.timer = rand(0.2, 0.8);
          }
        }
      }
      const walking = v.state === "walk";
      const fr = walking ? (Math.sin(v.bob) >= 0 ? 1 : 2) : 0;
      if (fr !== v.cur) {
        v.cur = fr;
        v.el.src = v.frames[fr];
      }
      const px = v.x * W;
      const py = v.y * H;
      const sprH = H * (0.18 + 0.06 * v.y) * this.figScale; // глибинний масштаб × масштаб сцени
      v.el.style.height = `${sprH}px`;
      v.el.style.left = `${px}px`;
      v.el.style.top = `${py}px`;
      v.el.style.transform = `translate(-50%,-100%) scaleX(${v.face})`;
      v.el.style.zIndex = String(Math.round(v.y * 1000));
      if (v.bubble) {
        v.bubbleT -= dt;
        v.bubble.style.left = `${px}px`;
        v.bubble.style.top = `${py - sprH}px`;
        v.bubble.style.zIndex = String(Math.round(v.y * 1000) + 1);
        if (v.bubbleT <= 0) {
          v.bubble.remove();
          v.bubble = null;
        }
      }
    }
    this.raf = requestAnimationFrame(this.loop);
  };

  /** Нефункціональний токен-тул (флейвор) для «чорний ящик»-локацій. */
  private placeToken(token: { verb: string; lines: string[] }): void {
    let i = -1;
    const btn = document.createElement("button");
    btn.className = "loc-token";
    btn.type = "button";
    btn.textContent = token.verb;
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      i = (i + 1) % token.lines.length;
      const t = document.createElement("div");
      t.className = "loc-token-line";
      t.textContent = token.lines[i];
      this.field.appendChild(t);
      requestAnimationFrame(() => (t.style.opacity = "1"));
      window.setTimeout(() => (t.style.opacity = "0"), 2600);
      window.setTimeout(() => t.remove(), 3400);
    });
    this.field.appendChild(btn);
  }

  private say(v: RV, text: string): void {
    if (v.bubble) v.bubble.remove();
    const b = document.createElement("div");
    b.className = "rv-bubble";
    b.textContent = `${v.cast.name}: ${text}`;
    this.field.appendChild(b);
    v.bubble = b;
    v.bubbleT = 3;
  }

  close(): void {
    this.root.classList.remove("on");
    cancelAnimationFrame(this.raf);
    this.field.innerHTML = "";
    this.vs = [];
  }
}
