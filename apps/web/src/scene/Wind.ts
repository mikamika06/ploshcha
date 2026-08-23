import type { VegItem } from "./Vegetation";
import { vnoise } from "./noise";

const WIND = 0.5;
const FREQ = 0.35;
const GUST_SCALE = 0.3 + 1.4 * FREQ; // константа — винесена з циклу
const CELL = 88; // крок грубої сітки поривів (native px) — << довжини хвилі поля, тож інтерполяція невидима

export interface ViewRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * ★ Знос поля — це ІНТЕГРАЛ напрямку, а не «напрямок × час».
 *
 * Тут була справжня причина того, що листя «з часом сіпається все швидше». Зсув рахувався як
 * `cos(wa(t)) · t`: напрямок сам крутиться з часом, тож похідна зсуву мала доданок `t · wa'(t)` —
 * тобто швидкість зносу росла ЛІНІЙНО з віком сцени. Заміряно крок координати за кадр:
 *
 *     вік 10 с     0.00168
 *     вік 3600 с   0.47169     — у 280 разів швидше, звідси й дрижання
 *
 * Правильно — накопичувати зсув по кроках: `ox += cos(wa)·швидкість·dt`. Тоді швидкість зносу
 * стала на будь-якому віці, а поле лишається тим самим полем.
 */
function gustAt(x: number, y: number, ox: number, oy: number): number {
  const gx = x * 0.0022 - ox;
  const gy = y * 0.0022 - oy;
  const n = vnoise(gx, gy) * 0.62 + vnoise(gx * 2.1 + 5, gy * 2.1 + 3) * 0.38;
  const g = Math.max(0, n - 0.5) / 0.5;
  return g * g;
}

/** Колихання рослин через sprite.skew.x за полем вітру + пориви (blast). */
export class Wind {
  private blastStart = -99;
  /** Накопичений знос поля (одиниці шуму). Росте зі сталою швидкістю, хоч би скільки жила сцена. */
  private ox = 0;
  private oy = 0;
  private last = -1;
  private gw: number;
  private gh: number;
  private grid: Float32Array; // поривне поле на грубій сітці (рахуємо раз/кадр замість 2×vnoise на спрайт)

  constructor(private list: VegItem[], private Wn: number, Hn: number) {
    this.gw = Math.ceil(this.Wn / CELL) + 2;
    this.gh = Math.ceil(Hn / CELL) + 2;
    this.grid = new Float32Array(this.gw * this.gh);
  }

  blast(t: number): void {
    this.blastStart = t;
  }

  update(t: number, view?: ViewRect, frameS = 0.016): void {
    // ★ Дрібне тремтіння листя має сенс лише поки кадри часті.
    //
    // `rustle` коливається 2.6 рад/с: при 60 кадрах це шелест, при 8 — вибірка потрапляє в
    // випадкові фази, і замість шелесту виходить сіпання. Тому на рідких кадрах амплітуду гасимо
    // до нуля: краще спокійний листок, ніж смикання, яке око читає як поламку.
    const fine = Math.max(0, Math.min(1, (0.055 - frameS) / 0.03));
    const wa = 0.9 * Math.sin(t * 0.06) + 0.5 * Math.sin(t * 0.017 + 2);
    const wdx = Math.cos(wa);
    const wdy = Math.sin(wa);
    // Крок часу беремо з самого годинника сцени й обрізаємо: після паузи (глядач був у локації)
    // один кадр не має права зсунути поле на хвилину вперед.
    const step = this.last < 0 ? 0 : Math.max(0, Math.min(0.1, t - this.last));
    this.last = t;
    this.ox += wdx * 0.35 * step;
    this.oy += wdy * 0.35 * step;
    const dtb = t - this.blastStart;
    const blastAmt = dtb >= 0 && dtb < 1.7 ? Math.sin((Math.PI * dtb) / 1.7) : 0;
    const blastPh = Math.min(Math.max(dtb / 1.7, 0), 1);
    // 1) поривне поле на грубій сітці — один прохід за кадр
    const gw = this.gw;
    for (let gy = 0; gy < this.gh; gy++) {
      for (let gx = 0; gx < gw; gx++) {
        this.grid[gy * gw + gx] = gustAt(gx * CELL, gy * CELL, this.ox, this.oy);
      }
    }
    for (const w of this.list) {
      if (w.removed) continue;
      // 2) кулінг: поза видимою рамкою — ховаємо (Pixi пропускає updateTransform + рендер)
      if (view && (w.x < view.x0 || w.x > view.x1 || w.y < view.y0 || w.y > view.y1)) {
        if (w.sprite.visible) w.sprite.visible = false;
        continue;
      }
      if (!w.sprite.visible) w.sprite.visible = true;
      // 3) білінійна вибірка пориву з сітки — нуль тригонометрії на спрайт
      const fx = w.x / CELL;
      const fy = w.y / CELL;
      const ix = Math.min(gw - 2, fx | 0);
      const iy = Math.min(this.gh - 2, fy | 0);
      const rx = fx - ix;
      const ry = fy - iy;
      const i00 = iy * gw + ix;
      const g =
        this.grid[i00] * (1 - rx) * (1 - ry) +
        this.grid[i00 + 1] * rx * (1 - ry) +
        this.grid[i00 + gw] * (1 - rx) * ry +
        this.grid[i00 + gw + 1] * rx * ry;
      let gust = g * GUST_SCALE;
      if (blastAmt > 0) {
        const along = w.x / this.Wn;
        gust += blastAmt * Math.exp(-Math.pow((along - blastPh) * 2.6, 2)) * 1.3;
      }
      const rustle = 0.03 * fine * Math.sin(t * 2.6 + w.phase);
      const skew = -((gust * WIND) / w.stiff) * wdx - rustle * (gust > 0.02 ? 1 : 0.25);
      // 4) епсилон-гейт: не бруднимо transform, якщо зміна невидима
      if (Math.abs(skew - w.lastSkew) > 0.004) {
        w.sprite.skew.x = skew;
        w.lastSkew = skew;
      }
    }
  }
}
