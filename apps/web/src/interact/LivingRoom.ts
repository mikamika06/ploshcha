import { assetUrl } from "../util/gfx";
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
  /** Розмір бульбашки, зміряний ОДИН раз при появі: щокадрове читання розкладки — це смикання. */
  bw: number;
  bh: number;
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
/** Скільки тримати перше слово після відкриття: рівно стільки, скільки розходяться хмари. */
const OPEN_HOLD_MS = 1500;
/** Пів ширини фігури в частках кадру: спрайт ~0.42 своєї висоти, висота ~0.2 кадру. */
const BODY_HALF = 0.022;

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
interface MaskGrid {
  grid: Uint8Array;
  gw: number;
  gh: number;
}

/** Картинка маски → сітка прохідності. `null` = маски фактично немає (бита, 404, порожня). */
function parseMask(img: HTMLImageElement): MaskGrid | null {
  const gw = 160;
  const gh = Math.max(1, Math.round((gw * img.height) / img.width));
  const cv = document.createElement("canvas");
  cv.width = gw;
  cv.height = gh;
  const ctx = cv.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(img, 0, 0, gw, gh);
  const d = ctx.getImageData(0, 0, gw, gh).data;
  const grid = new Uint8Array(gw * gh);
  for (let i = 0; i < gw * gh; i++) {
    const r = d[i * 4];
    const g = d[i * 4 + 1];
    const b = d[i * 4 + 2];
    grid[i] = g > 110 && r < 130 && b < 130 ? 1 : 0; // зелене = прохідне
  }
  // приймаємо маску, якщо в ній є бодай трохи зеленого (маленькі легітимні зони, як-от кузня, теж
  // валідні); поріг лише відсіює зовсім биту/незавантажену маску (~0%)
  let green = 0;
  for (let i = 0; i < grid.length; i++) green += grid[i];
  return green > gw * gh * 0.012 ? { grid, gw, gh } : null;
}

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
  /** Прохідні клітини маски парами (x, y) у частках кадру. `null` — маски немає, ходить полігон. */
  private cells: Float32Array | null = null;
  /** Маски живуть довше за кімнату: локації відкривають по колу, а вантажити щоразу — це чекання. */
  private static MASKS = new Map<string, MaskGrid | null>();
  private static LOADING = new Map<string, Promise<MaskGrid | null>>();
  /** Чи вже розсаджено каст. Доти прибульці чекають у `pending`, а не сідають на полігон-запасник. */
  private seated = false;
  private pending: RoomCast[] = [];

  /** Живе віче в цій локації: гомін-заповнювач мовчить, бо говорять справжні репліки ядра. */
  private live = false;
  private say2: HTMLInputElement | null = null;
  private sayForm: HTMLElement | null = null;
  /** Черга реплік ядра. У живому режимі глядач гортає їх сам — клік за кліком. */
  private queue: { vid: string; text: string; deed?: string; toward?: string }[] = [];
  /** Де поле кімнати лежить НА ЕКРАНІ. Оновлюємо зрідка: читання розкладки щокадру — це смикання. */
  private view = { left: 0, top: 0, t: 0 };
  /** Чим скінчилось віче: тримаємо, доки глядач не дочитає чергу. */
  private end: { title: string; body: string } | null = null;
  private shown = 0;
  /** Коли кімнату відкрито: перше слово тримаємо, поки розходяться хмари. */
  private openedAt = 0;

  constructor(
    private onClose: () => void,
    private onSay?: (text: string) => void,
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
      <div class="room-notice plaq"></div>
      <div class="room-wait plaq">Село думу думає…</div>
      <div class="room-end plaq">
        <div class="room-end-title"></div>
        <div class="room-end-body"></div>
        <button class="tag room-end-go" type="button">← до села</button>
      </div>
      <form class="room-say gtalk-say">
        <input class="room-input gtalk-input" type="text" maxlength="300" autocomplete="off"
               placeholder="сказати своє селу вголос…">
        <button class="tag room-send gtalk-send" type="button">Сказати</button>
      </form>`;
    this.bg = this.root.querySelector(".room-bg") as HTMLImageElement;
    this.field = this.root.querySelector(".room-field") as HTMLElement;
    this.nameEl = this.root.querySelector(".room-name") as HTMLElement;
    this.root.querySelector(".room-back")!.addEventListener("click", () => this.onClose());
    // Клік по кімнаті = «далі». Форма й кнопки свій клік з'їдають самі, клік по людині —
    // це прицілювання шепоту, тож гортання туди не лізе.
    this.root.addEventListener("click", () => this.advance());
    this.sayForm = this.root.querySelector(".room-say") as HTMLElement;
    this.say2 = this.root.querySelector(".room-input") as HTMLInputElement;
    this.sayForm.addEventListener("submit", (e) => e.preventDefault());
    this.sayForm.addEventListener("click", (e) => e.stopPropagation());
    this.root.querySelector(".room-back")!.addEventListener("click", (e) => e.stopPropagation());
    const go = this.root.querySelector(".room-end-go") as HTMLElement;
    go.addEventListener("click", (e) => {
      e.stopPropagation();
      this.onClose();
    });
    this.root.querySelector(".room-send")!.addEventListener("click", () => this.send());
    this.say2.addEventListener("keydown", (e) => {
      // Те саме, що на Дошці: Escape із набраним словом лише відпускає поле (слово лишається),
      // а порожнє поле його не тримає — інакше з локації взагалі не вийти клавіатурою.
      if (e.key === "Escape") {
        if (!this.say2?.value.trim()) return;
        e.stopPropagation();
        this.say2?.blur();
        return;
      }
      e.stopPropagation();
      if (e.key === "Enter") {
        e.preventDefault();
        this.send();
      }
    });
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
    // ★ Нова локація — нова розмова. Чергу й лічильник показаного скидали лише в `close()`, тож
    // після відкриття кімнати поверх кімнати (віче слідом за віче) свіжа репліка ставала в хвіст
    // за старими: `shown` уже 1, і `enqueue` мовчки нічого не показував. Заміряно: 0 бульбашок
    // замість 1 і 3 чужі репліки в черзі.
    this.queue = [];
    this.shown = 0;
    this.openedAt = performance.now();
    // стейдж бере аспект самої картинки → field 0-1 == image 0-1 (маска не з'їжджає)
    const stage = this.root.querySelector(".room-stage") as HTMLElement;
    this.bg.onload = (): void => {
      this.bg.style.opacity = "1";
      // числом, а не `aspect-ratio`: тим самим `--ar` CSS рахує ще й граничну ширину, щоб у
      // низькому вікні низ кімнати не зрізало (див. `.room-stage`)
      if (this.bg.naturalWidth) stage.style.setProperty("--ar", String(this.bg.naturalWidth / this.bg.naturalHeight));
    };
    // ★ Стару картинку прибираємо ДО того, як почне вантажитись нова.
    //
    // `<img>` тримає попередній кадр, доки не приїде наступний, тож на пів секунди в новій
    // локації світилась попередня — а виглядало це як «кімната не та». Ховаємо, і показуємо
    // назад аж коли нова готова.
    //
    // Ховаємо ПРОЗОРІСТЮ, а не `visibility`: інлайновий `visibility: visible` на картинці
    // перебивав успадковане `visibility: hidden` закритої кімнати, і невидимий `<img>` на весь
    // екран далі ловив кліки — по локаціях на мапі після першого ж виходу неможливо було влучити.
    this.bg.style.opacity = "0";
    this.bg.removeAttribute("src");
    this.bg.src = bgUrl;
    if (this.bg.complete && this.bg.naturalWidth) this.bg.style.opacity = "1";
    this.root.classList.toggle("room--cover", !!opts?.cover);
    this.nameEl.textContent = name;
    this.floor = floor;
    const xs = floor.map((p) => p[0]);
    const ys = floor.map((p) => p[1]);
    this.bbox = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
    this.mask = null;
    this.cells = null;
    this.field.innerHTML = "";
    this.vs = [];
    this.pending = [];
    this.seated = false;
    // ★ Розсаджуємо людей ЛИШЕ коли відома прохідна зона.
    //
    // Маска вантажиться асинхронно, і доти єдина відома підлога — грубий полігон-запасник. Селян
    // ставили по ньому одразу, а коли маска приїздила, кожного, хто опинився поза зеленим, код
    // пересаджував на валідну клітину — тобто на очах у глядача людину смикало через півкімнати.
    // Заміряно: чотири фігури, стрибки 0.08-0.50 частки кадру (до 606px) в одному кадрі на 126-й
    // мілісекунді — рівно тоді, коли завершувалось завантаження маски. На повільному звʼязку
    // маска приїздить уже посеред розмови, і це виглядає як телепорт «за локацію й назад».
    //
    // Тому: маска в кеші — садимо одразу; немає — чекаємо на неї (і на помилку теж), а поле доти
    // прозоре. Кеш статичний, бо локації відкривають по колу: телепорт мусив би зникнути з
    // першого ж разу, а не «здебільшого».
    const seat = (): void => {
      this.spawn(cast);
      if (opts?.token) this.placeToken(opts.token);
      this.field.style.opacity = "1";
    };
    const cached = maskUrl ? LivingRoom.MASKS.get(maskUrl) : undefined;
    if (!maskUrl || cached !== undefined) {
      if (cached) this.useMask(cached);
      seat();
    } else {
      this.field.style.opacity = "0";
      this.loadMask(maskUrl, seat);
    }
    this.root.classList.add("on");
    this.last = performance.now();
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(this.loop);
  }

  /**
   * Маска локації з кешу; чого немає — вантажимо один раз.
   *
   * `warm` кличеться після старту села: маски всіх локацій разом важать 100 КБ, а без них перший
   * вхід у кімнату мусить чекати на завантаження, доки люди ще не сіли.
   */
  static mask(url: string): Promise<MaskGrid | null> {
    const have = LivingRoom.MASKS.get(url);
    if (have !== undefined) return Promise.resolve(have);
    const inflight = LivingRoom.LOADING.get(url);
    if (inflight) return inflight;
    const job = new Promise<MaskGrid | null>((resolve) => {
      const img = new Image();
      const finish = (grid: MaskGrid | null): void => {
        LivingRoom.MASKS.set(url, grid);
        LivingRoom.LOADING.delete(url);
        resolve(grid);
      };
      img.onerror = (): void => finish(null);
      img.onload = (): void => finish(parseMask(img));
      img.src = url;
    });
    LivingRoom.LOADING.set(url, job);
    return job;
  }

  static warm(urls: string[]): void {
    for (const url of urls) void LivingRoom.mask(url);
  }

  /** Спрайти касту на вже відомій підлозі. Окремо від `open`, бо чекає на маску. */
  private spawn(cast: RoomCast[]): void {
    this.seated = true;
    for (const c of cast) this.addPerson(c);
    for (const c of this.pending.splice(0)) this.addPerson(c);
    if (this.shown === 0 && this.queue.length) this.advance(); // слово, що чекало підлоги
  }

  /**
   * Завантажує walk-маску з Nano Banana у сітку (зелене=прохідне) і кладе в кеш.
   *
   * `done` кличеться в БУДЬ-ЯКОМУ разі — і на битій масці, і на 404: інакше локація, чия маска не
   * доїхала, лишилась би назавжди порожньою, а це та сама давня поламка «механізм працює, а на
   * екрані нічого».
   */
  private loadMask(url: string, done: () => void): void {
    void LivingRoom.mask(url).then((grid) => {
      if (grid) this.useMask(grid);
      done();
    });
  }

  private useMask(m: MaskGrid): void {
    this.mask = m.grid;
    this.mgw = m.gw;
    this.mgh = m.gh;
    this.bbox = { x0: 0, x1: 1, y0: 0, y1: 1 };
    // Список прохідних клітин — щоб посадка була вибором із того, що є, а не лотереєю.
    const out: number[] = [];
    for (let gy = 0; gy < m.gh; gy++) {
      for (let gx = 0; gx < m.gw; gx++) {
        if (m.grid[gy * m.gw + gx] === 1) {
          out.push((gx + 0.5) / m.gw, (gy + 0.5) / m.gh);
        }
      }
    }
    this.cells = out.length ? Float32Array.from(out) : null;
  }

  /**
   * Чи стане тут ФІГУРА, а не точка.
   *
   * Ноги стоять у точці, але спрайт росте вбік: на телефоні кімната вужча за екран у пікселях, і
   * пів фігури звисало з бруківки в темряву, хоч ноги формально були в зоні. Тому перевіряємо ще
   * й плечі — ліворуч і праворуч на пів ширини спрайта в частках кадру.
   */
  private fits(x: number, y: number): boolean {
    return this.inFloor(x, y) && this.inFloor(x - BODY_HALF, y) && this.inFloor(x + BODY_HALF, y);
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
    // ★ З МАСКИ беремо клітину, а не тичемо навмання.
    //
    // Вісімдесят спроб по всьому кадру — це ставка на те, що прохідного багато. У тісній кімнаті
    // (кузня — два відсотки кадру) спроби вигоряли, і код повертав ЦЕНТР кадру, тобто ставив
    // людину просто в стіну. Саме так вона й «телепортувалась за зону».
    if (this.cells && this.cells.length) {
      // Кілька спроб: беремо клітину, куди влазить ціла фігура, а не лише ноги.
      for (let k = 0; k < 24; k++) {
        const i = (Math.random() * (this.cells.length / 2)) | 0;
        const x = this.cells[i * 2];
        const y = this.cells[i * 2 + 1];
        if (this.fits(x, y)) return [x, y];
      }
      const i = (Math.random() * (this.cells.length / 2)) | 0;
      return [this.cells[i * 2], this.cells[i * 2 + 1]];
    }
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
    // Положення поля на екрані читаємо двічі на секунду: воно міняється лише від зміни вікна.
    if (now - this.view.t > 500) {
      const r = this.field.getBoundingClientRect();
      this.view = { left: r.left, top: r.top, t: now };
    }
    // Жодних поправок на розмір екрана: людина мусить бути тієї частки кімнати, якої вона є на
    // малюнку. Доти фігури «підтягувались» під висоту коробки — і на телефоні виходили велетнями,
    // бо кімната лежала смужкою. Тепер смужки немає (сцена заповнює кадр), тож і поправка зайва.
    for (const v of this.vs) {
      if (v.state === "idle") {
        v.timer -= dt;
        // Під час ЖИВОЇ розмови ніхто не тиняється сам: людина рухається тільки тоді, коли того
        // вимагає такт («підходить», «відступає», «ходить»). Доти всі снували підлогою на
        // власному таймері, і будь-яка постановка тонула в цьому броунівському русі.
        if (v.timer <= 0 && !this.live) {
          const [tx, ty] = this.randFloor();
          v.tx = tx;
          v.ty = ty;
          v.state = "walk";
        }
        if (!this.live && !v.bubble && Math.random() < dt * 0.05) this.say(v, AMBIENT[(Math.random() * AMBIENT.length) | 0]);
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
          if (this.fits(nx, ny)) {
            // крок дозволений лише якщо на масці лишається ВСЯ фігура, а не самі ноги
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
      const sprH = H * (0.18 + 0.06 * v.y) * this.figScale; // глибина × масштаб сцени
      v.el.style.height = `${sprH}px`;
      // ★ Тримаємо фігуру В МЕЖАХ намальованої кімнати.
      //
      // Ноги стоять на масці підлоги, а тіло росте ВГОРУ й убік: коло самого краю маски людина
      // вилазила за стіну. Маска цього не ловить — вона про те, де можна СТАТИ, а не про те, де
      // видно спрайт.
      const halfW = (sprH * 0.42) / 2;
      v.el.style.left = `${Math.min(Math.max(px, halfW), Math.max(halfW, W - halfW))}px`;
      // Ноги стоять на `py`, тіло росте ВГОРУ на всю висоту спрайта — тож нижня межа для ніг це
      // сама висота фігури, інакше голова виходить за верх кімнати (заміряно: 18px над краєм).
      v.el.style.top = `${Math.min(Math.max(py, sprH), H)}px`;
      v.el.style.transform = `translate(-50%,-100%) scaleX(${v.face})`;
      v.el.style.zIndex = String(Math.round(v.y * 1000));
      if (v.bubble) {
        // ★ Репліку тримаємо У ВИДИМОМУ кадрі, а не просто в межах кімнати.
        //
        // На телефоні кімната ШИРША за екран (боки навмисно під обрізом), тож «усередині поля» і
        // «видно» — різні речі: бульбашка людини скраю чесно лежала в кімнаті й при цьому за
        // краєм екрана. Прочитати її було нічим — камери в локації немає, як на мапі. Тому межі
        // рахуємо від вікна, перетнутого з полем.
        const half = v.bw / 2;
        const loX = Math.max(half + 6, -this.view.left + half + 6);
        const hiX = Math.min(W - half - 6, window.innerWidth - this.view.left - half - 6);
        const loY = Math.max(v.bh + 6, -this.view.top + v.bh + 6);
        const hiY = Math.min(H - 6, window.innerHeight - this.view.top - 6);
        const bx = hiX > loX ? Math.min(Math.max(px, loX), hiX) : px;
        const by = hiY > loY ? Math.min(Math.max(py - sprH, loY), hiY) : py - sprH;
        v.bubble.style.left = `${bx}px`;
        v.bubble.style.top = `${by}px`;
        // ★ Репліка над УСІМА людьми, а не лише над своїм господарем.
        //
        // Глибина в кімнаті рахується з `y`, тож той, хто стоїть ближче до глядача, перекривав
        // чужу бульбашку: людина буквально залазила на текст. Репліки живуть окремим ярусом
        // вище за будь-який спрайт, а між собою впорядковані так само за глибиною.
        v.bubble.style.zIndex = String(500_000 + Math.round(v.y * 1000));
        // `Infinity` = репліка з черги: висить, доки не клікнеш далі. Гомін-заповнювач і твій
        // відгомін мають скінченний час і гаснуть самі.
        if (v.bubbleT !== Infinity) {
          v.bubbleT -= dt;
          if (v.bubbleT <= 0) this.clearBubble(v);
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

  /**
   * Що зараз робиться, коли на екрані тихо.
   *
   * Глядач дочитав чергу, тицяє — і нічого не рухається, бо оркестратор саме складає наступну
   * хвилю тактів. Без слова це читається як поламка, і саме так його й прочитали.
   */
  waiting(on: boolean): void {
    const el = this.root.querySelector(".room-wait") as HTMLElement | null;
    if (!el) return;
    el.classList.toggle("on", on && this.live && this.shown >= this.queue.length && !this.end);
  }

  /**
   * Чим скінчилось віче — окремою карткою, коли черга дочитана.
   *
   * Доти остання репліка просто обривалась: ані ухвали, ані підсумку, ані знаку, що розмова
   * взагалі скінчилась. Картку показуємо не за подією ядра, а коли глядач ДОЧИТАВ — інакше вона
   * вискакує посеред розмови, бо ядро завершує прогін раніше, ніж прочитано чергу.
   */
  finale(title: string, body: string): void {
    this.end = { title, body };
    this.showEndIfRead();
  }

  private showEndIfRead(): void {
    this.waiting(false);
    const el = this.root.querySelector(".room-end") as HTMLElement | null;
    if (!el || !this.end) return;
    if (this.shown < this.queue.length) return; // ще є що читати
    (el.querySelector(".room-end-title") as HTMLElement).textContent = this.end.title;
    (el.querySelector(".room-end-body") as HTMLElement).textContent = this.end.body;
    el.classList.add("on");
  }

  /**
   * Повідомлення сцені — видимою плашкою, а не тишею.
   *
   * Слово, послане в скінчене віче, поверталось помилкою в консоль: на екрані не мінялось нічого,
   * і виглядало це як «шепіт не працює».
   */
  notice(text: string): void {
    const el = this.root.querySelector(".room-notice") as HTMLElement | null;
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("on", Boolean(text));
    if (text) window.setTimeout(() => el.classList.remove("on"), 5200);
  }

  /** Живий режим: гомін вимикається, зʼявляється смуга «сказати своє». */
  setLive(on: boolean): void {
    this.live = on;
    this.root.classList.toggle("room--live", on);
  }

  /**
   * Хтось заговорив, а його в кімнаті ще немає — він ПРИХОДИТЬ.
   *
   * Мовчазний `return` тут був би тією самою давньою поламкою: подія доїхала, механізм працює,
   * а на екрані нічого — і виглядає, ніби ядро мовчить.
   */
  addPerson(c: RoomCast, text?: string): void {
    const seated = this.vs.find((r) => r.cast.vid === c.vid);
    if (seated) {
      // ★ Імʼя ПЕРЕПИСУЄМО тим, яке прийшло від ядра.
      //
      // Доти хто сів першим, той і лишався своїм підписом до кінця віча — а склад ядро оголошує
      // ОКРЕМОЮ подією, яка може доїхати вже після того, як людина в кімнаті. Разом із фікстурним
      // гуртом фронта це й давало підпис, що не збігається з тим, кого мало на увазі ядро (аудит
      // 2026-08-29: «Оксана» замість «Олени Завійної»). Гурт прибрано, звірка лишається.
      if (c.name && c.name !== seated.cast.name) {
        seated.cast = { ...seated.cast, name: c.name };
        // Бульбашка вже могла висіти зі старим підписом — правимо й її, інакше вірне імʼя
        // побачить лише той, хто дочекається наступної репліки.
        const who = seated.bubble?.querySelector(".rv-who");
        if (who) who.textContent = c.name;
      }
      return;
    }
    // Поки не знаємо підлоги, прибулець чекає: посадити його на запасний полігон означало б
    // смикнути через півкімнати, щойно приїде маска.
    if (!this.seated) {
      const waits = this.pending.findIndex((q) => q.vid === c.vid);
      if (waits < 0) this.pending.push(c);
      else if (c.name) this.pending[waits] = { ...this.pending[waits], name: c.name };
      if (text) this.enqueue(c.vid ?? c.id, text);
      return;
    }
    const el = document.createElement("img");
    el.className = "rv";
    el.draggable = false;
    const frames = [0, 1, 2].map((n) => assetUrl(`/assets/roles/${c.id}/${n}.webp`));
    el.src = frames[0];
    // ★ Людина ВХОДИТЬ, а не виникає.
    //
    // Каст росте по ходу віча: кожен мовець додається в кімнату аж на своїй першій репліці, і доти
    // він просто зʼявлявся у випадковій точці підлоги — вісім селян означало вісім таких появ
    // просто під час розмови, і читалось це як телепорт. Тепер ціль лишається тією самою випадковою
    // точкою, а старт береться біля нижнього краю зони, тобто «від дверей»: людина проходить свій
    // шлях тими самими кроками, що й усі інші.
    const [tx, ty] = this.randFloor();
    let x = tx;
    let y = this.bbox.y1;
    if (!this.fits(x, y)) {
      // Низ зони може бути зайнятий меблями — тоді входимо з того боку, з якого взагалі можна.
      const side = tx < (this.bbox.x0 + this.bbox.x1) / 2 ? this.bbox.x0 : this.bbox.x1;
      if (this.fits(side, ty)) { x = side; y = ty; } else { x = tx; y = ty; }
    }
    const rv: RV = {
      cast: c, el, frames, cur: 0, x, y, tx, ty,
      state: x === tx && y === ty ? "idle" : "walk",
      timer: rand(0.5, 3), face: 1, bob: 0, bubble: null, bubbleT: 0, bw: 0, bh: 0,
    };
    // Поява без ривка: спрайт проявляється, поки робить перші кроки.
    el.style.opacity = "0";
    requestAnimationFrame(() => { el.style.opacity = "1"; });
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      this.advance();
    });
    this.field.appendChild(el);
    this.vs.push(rv);
    if (text) this.sayBy(c.vid ?? c.id, text);
  }

  /**
   * Дія тіла з партитури: людина не тиняється, а РОБИТЬ те, що каже такт.
   *
   * Ходьба сама по собі нічого не означала — усі просто снували підлогою. Тепер крок «підходить»
   * веде до співрозмовника, «відступає» — від нього, і в кадрі видно, хто наступає, а хто здає.
   */
  deed(vid: string, deed: string, towardVid?: string): void {
    const v = this.vs.find((r) => r.cast.vid === vid || r.cast.id === vid);
    if (!v) return;
    const other = towardVid
      ? this.vs.find((r) => r.cast.vid === towardVid || r.cast.id === towardVid)
      : undefined;
    const stand = (): void => {
      v.tx = v.x;
      v.ty = v.y;
      v.state = "idle";
    };
    const step = (kx: number, ky: number): boolean => {
      const nx = Math.max(this.bbox.x0, Math.min(this.bbox.x1, v.x + kx));
      const ny = Math.max(this.bbox.y0, Math.min(this.bbox.y1, v.y + ky));
      if (!this.fits(nx, ny)) return false;
      v.tx = nx;
      v.ty = ny;
      v.state = "walk";
      return true;
    };
    /**
     * Крок до співрозмовника (sign=1) або від нього (−1).
     *
     * Якщо в лоба не вийшло (стіна, лава), пробуємо збоку; а як і так нікуди — СТАЄМО. Мовчазне
     * «нічого не сталось» тут гірше за все: людина далі йшла зі старою ціллю, тобто «відступив»
     * виглядало як «підійшов». Заміряно: 0.186 → 0.166 → 0.146 замість 0.186 → 0.166 → 0.21.
     */
    const toward = (sign: number): void => {
      if (!other) return stand();
      const dx = other.x - v.x;
      const dy = other.y - v.y;
      const len = Math.hypot(dx, dy) || 1;
      const d = Math.min(0.16, Math.max(0.06, len * 0.35));
      const base = Math.atan2(dy, dx) + (sign < 0 ? Math.PI : 0);
      for (const turn of [0, 0.6, -0.6, 1.2, -1.2]) {
        if (step(Math.cos(base + turn) * d, Math.sin(base + turn) * d * 0.75)) {
          if (Math.abs(dx) > 0.005) v.face = dx * sign < 0 ? -1 : 1;
          return;
        }
      }
      stand();
    };
    switch (deed) {
      case "підходить":
        toward(1);
        break;
      case "відступає":
        toward(-1);
        break;
      case "ходить": {
        const a = Math.random() * Math.PI * 2;
        step(Math.cos(a) * 0.12, Math.sin(a) * 0.08);
        break;
      }
      case "відвертається":
        if (other) v.face = other.x > v.x ? -1 : 1;
        v.timer = 2.5;
        break;
      case "розводить_руками":
        v.el.animate(
          [{ transform: "scaleX(1)" }, { transform: "scaleX(1.16)" }, { transform: "scaleX(1)" }],
          { duration: 420, easing: "ease-in-out" },
        );
        break;
      default: // «стоїть» — саме це й треба: спинитись і слухати
        stand();
        v.timer = 2.5;
        if (other && Math.abs(other.x - v.x) > 0.005) v.face = other.x < v.x ? -1 : 1;
    }
  }

  /**
   * Репліка ядра стає в ЧЕРГУ, а не вискакує сама.
   *
   * Ядро віддає такти пачками й швидше, ніж їх встигаєш прочитати: без черги розмова
   * пробігала повз, і в кадрі лишався тільки хвіст. Тепер темп задає глядач кліком.
   */
  enqueue(vid: string, text: string, deed?: string, toward?: string): void {
    this.queue.push({ vid, text, deed, toward });
    this.waiting(false); // приїхала репліка — чекати більше нема чого
    if (this.shown === 0 && this.queue.length === 1) this.advance();
  }

  /** Показати наступну репліку черги. */
  advance(): void {
    // Доки люди не сіли (чекаємо маску), показувати нікому: репліка мовчки зникала б, бо мовця
    // ще немає в кімнаті, а лічильник показаного вже зрушив би.
    if (!this.seated) return;
    // ★ Перше слово чекає, поки розійдуться хмари.
    //
    // Ядро тепер віддає його за секунду з невеликим, і репліка вискакувала ще під завісою — тобто
    // повз глядача. Тримаємо її, доки завіса не догорнулась; наступні йдуть по кліку, як і доти.
    if (this.shown === 0 && performance.now() < this.openedAt + OPEN_HOLD_MS) {
      window.setTimeout(() => this.advance(), OPEN_HOLD_MS / 3);
      return;
    }
    const next = this.queue[this.shown];
    if (!next) {
      this.showEndIfRead(); // дочитав усе — саме час показати, чим скінчилось
      this.waiting(true);   // а якщо кінця ще немає — сказати, що ядро думає
      return;
    }
    this.shown++;
    // Попередню репліку прибираємо тут-таки: місце звільняє наступна, а не годинник.
    for (const v of this.vs) if (v.bubble) this.clearBubble(v);
    if (next.deed) this.deed(next.vid, next.deed, next.toward);
    this.sayBy(next.vid, next.text);
    if (this.shown >= this.queue.length) this.showEndIfRead();
  }

  /** Справжня репліка ядра — над головою того, хто її сказав. */
  sayBy(vid: string, text: string): boolean {
    const v = this.vs.find((r) => r.cast.vid === vid || r.cast.id === vid);
    if (!v) return false;
    // ★ Репліка НЕ згасає сама: вона висить, доки не клікнеш далі. Таймер тут означав «не встиг
    // прочитати — сам винен», хоч темп розмови задає глядач.
    this.say(v, text);
    v.bubbleT = Infinity;
    return true;
  }

  private clearBubble(v: RV): void {
    if (!v.bubble) return;
    v.bubble.remove();
    v.bubble = null;
    v.bubbleT = 0;
  }

  get isOpen(): boolean {
    return this.root.classList.contains("on");
  }

  private send(): void {
    const text = this.say2?.value.trim();
    if (!text) return;
    if (this.say2) this.say2.value = "";
    this.onSay?.(text);
  }

  private say(v: RV, text: string, kind = ""): void {
    if (v.bubble) v.bubble.remove();
    const b = document.createElement("div");
    b.className = `rv-bubble${kind ? ` rv-${kind}` : ""}`;
    // Імʼя — ОКРЕМИМ рядком над реплікою, а не «Імʼя: текст» усередині. Вставлене в текст, воно
    // читалось як частина сказаного, і кожна репліка починалась зі службового слова.
    const who = document.createElement("div");
    who.className = "rv-who";
    who.textContent = v.cast.name;
    const line = document.createElement("div");
    line.className = "rv-line";
    line.textContent = text;
    b.append(who, line);
    this.field.appendChild(b);
    // Розмір читаємо раз, одразу після вставлення: далі кадр працює самою арифметикою.
    v.bw = b.offsetWidth;
    v.bh = b.offsetHeight;
    v.bubble = b;
    v.bubbleT = 3;
  }

  close(): void {
    this.waiting(false);
    this.end = null;
    (this.root.querySelector(".room-end") as HTMLElement | null)?.classList.remove("on");
    this.queue = [];
    this.shown = 0;
    this.openedAt = performance.now();
    this.setLive(false);
    this.root.classList.remove("on");
    cancelAnimationFrame(this.raf);
    this.field.innerHTML = "";
    this.vs = [];
  }
}
